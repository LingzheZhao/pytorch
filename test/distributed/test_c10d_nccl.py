# Owner(s): ["oncall: distributed"]

import gzip
import json
import os
import sys
import unittest
from unittest import mock

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)


import test_c10d_common

import torch.distributed as dist
import torch.testing._internal.common_utils as common
from torch.profiler import (
    CuspyConfig,
    profile,
    ProfilerActivity,
    ProfilerActivityConfig,
)
from torch.testing._internal.common_cuda import (
    TEST_CUDA_GRAPH_TOOLS_ID,
    TEST_CUPTI_V13_5,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
    sm_is_or_higher_than,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
    TemporaryFileName,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)

NCCL_BACKEND = "nccl"


class RendezvousEnvTest(TestCase):
    hw_classification = HardwareClassification.CUDA

    @retry_on_connect_failures
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    def test_common_errors(self):
        # retry_on_connect_failures re-runs this whole body on a RuntimeError,
        # so the default process group must not survive a failed attempt.
        try:
            vars = {
                "WORLD_SIZE": "1",
                "RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(common.find_free_port()),
            }

            class Env:
                def __init__(self, vars):
                    self.env_patcher = mock.patch.dict(os.environ, vars, clear=True)

                def __enter__(self):
                    self.env_patcher.start()

                def __exit__(self, type, value, traceback):
                    self.env_patcher.stop()

            def without(d, key):
                d = d.copy()
                d.pop(key)
                return d

            def withouts(d, keys):
                d = d.copy()
                for key in keys:
                    d.pop(key)
                return d

            with Env(without(vars, "WORLD_SIZE")):
                self.assertEqual(None, os.environ.get("WORLD_SIZE"))
                with self.assertRaisesRegex(ValueError, "WORLD_SIZE expected"):
                    gen = c10d.rendezvous("env://")
                    next(gen)
                c10d.init_process_group(backend=NCCL_BACKEND, world_size=1)
                self.assertEqual(c10d.get_rank(), 0)
                self.assertEqual(c10d.get_world_size(), 1)
                c10d.destroy_process_group()

            with Env(without(vars, "RANK")):
                self.assertEqual(None, os.environ.get("RANK"))
                with self.assertRaisesRegex(ValueError, "RANK expected"):
                    gen = c10d.rendezvous("env://")
                    next(gen)
                c10d.init_process_group(backend=NCCL_BACKEND, rank=0)
                self.assertEqual(c10d.get_rank(), 0)
                self.assertEqual(c10d.get_world_size(), 1)
                c10d.destroy_process_group()

            with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
                self.assertEqual(None, os.environ.get("RANK"))
                self.assertEqual(None, os.environ.get("WORLD_SIZE"))
                c10d.init_process_group(backend=NCCL_BACKEND, rank=0, world_size=1)
                self.assertEqual(c10d.get_rank(), 0)
                self.assertEqual(c10d.get_world_size(), 1)
                c10d.destroy_process_group()

            with Env(vars):
                c10d.init_process_group(backend=NCCL_BACKEND)
                self.assertEqual(c10d.get_rank(), 0)
                self.assertEqual(c10d.get_world_size(), 1)
                c10d.destroy_process_group()

            with Env(without(vars, "MASTER_ADDR")):
                self.assertEqual(None, os.environ.get("MASTER_ADDR"))
                with self.assertRaisesRegex(ValueError, "MASTER_ADDR expected"):
                    gen = c10d.rendezvous("env://")
                    next(gen)

            with Env(without(vars, "MASTER_PORT")):
                self.assertEqual(None, os.environ.get("MASTER_PORT"))
                with self.assertRaisesRegex(ValueError, "MASTER_PORT expected"):
                    gen = c10d.rendezvous("env://")
                    next(gen)

            with Env(without(vars, "WORLD_SIZE")):
                self.assertEqual(None, os.environ.get("WORLD_SIZE"))
                gen = c10d.rendezvous(f"env://?world_size={1}")
                _, _, size = next(gen)
                self.assertEqual(size, 1)

            with Env(without(vars, "RANK")):
                self.assertEqual(None, os.environ.get("RANK"))
                gen = c10d.rendezvous(f"env://?rank={0}")
                _, rank, _ = next(gen)
                self.assertEqual(rank, 0)

            with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
                self.assertEqual(None, os.environ.get("RANK"))
                self.assertEqual(None, os.environ.get("WORLD_SIZE"))
                gen = c10d.rendezvous(f"env://?rank={0}&world_size={1}")
                _, rank, size = next(gen)
                self.assertEqual(rank, 0)
                self.assertEqual(size, 1)
        finally:
            if c10d.is_initialized():
                c10d.destroy_process_group()


class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    hw_classification = HardwareClassification.CUDA

    @requires_nccl()
    @retry_on_connect_failures
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    def test_default_store_timeout_nccl(self):
        self._test_default_store_timeout(NCCL_BACKEND)


class ProcessGroupNCCLInitTest(MultiProcessTestCase):
    hw_classification = HardwareClassification.CUDA
    device_type = "cuda"

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        dm = torch.get_device_module(self.device_type)
        return dm.device_count()

    @property
    def device(self):
        return torch.device(self.device_type, self.rank % self.world_size)

    # A helper with the must-needed init args for test infra.
    # kwargs can be filled in by individual init tests.
    def _init_process_group(self, **kwargs):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            **kwargs,
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_init_wo_backend_str(self):
        self._init_process_group(device_id=self.device)
        x = torch.empty(1, device=self.device)
        c10d.all_reduce(x)

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_scalable_init(self):
        os.environ["TORCH_NCCL_RANKS_PER_ROOT"] = "1"
        self._init_process_group(backend=NCCL_BACKEND, device_id=self.device)
        x = torch.empty(1, device=self.device)
        c10d.all_reduce(x)
        os.environ["TORCH_NCCL_RANKS_PER_ROOT"] = "0"


class NcclProcessGroupWithDispatchedCollectivesTests(
    test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
):
    hw_classification = HardwareClassification.CUDA

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_collectives(self):
        self._test_collectives(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_allreduce_coalesced(self):
        self._test_allreduce_coalesced(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_all_to_all_single(self):
        self._test_all_to_all_single(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_allgather_base(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            NCCL_BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "cuda"
        tensor = torch.ones(10, 10, device=torch.device(device))
        output_tensor = torch.zeros(10, 10, device=torch.device(device))
        dist.all_gather_single(output_tensor, tensor)
        self.assertEqual(output_tensor, tensor)

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    @parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_allgather_float8(self, float8_dtype):
        device = torch.device(f"cuda:{self.rank:d}")
        if not sm_is_or_higher_than(device, 9, 0):
            self.skipTest("FP8 reduction support begins with sm90 capable devices")
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            NCCL_BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "cuda"
        tensor = torch.ones(10, 16, device=torch.device(device)).to(float8_dtype)
        output_tensor = torch.zeros(10, 16, device=torch.device(device)).to(
            float8_dtype
        )
        dist.all_gather_single(output_tensor, tensor)
        self.assertEqual(output_tensor.view(torch.float32), tensor.view(torch.float32))


instantiate_parametrized_tests(NcclProcessGroupWithDispatchedCollectivesTests)


@unittest.skipIf(torch.cuda.device_count() < 2, "2 CUDA GPUs required")
class CollectiveAnnotationTest(MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA
    world_size = 2

    @classmethod
    def backend_str(cls):
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @skipIfRocm
    def test_captured_collective_is_annotated(self):
        if not TEST_CUDA_GRAPH_TOOLS_ID:
            self.skipTest("CUDA graph annotations are unavailable")
        from torch.cuda.graph_annotations import get_kernel_annotations

        torch.cuda.set_device(self.device)
        x = torch.ones(1024, device=self.device)
        dist.all_reduce(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, enable_annotations=True):
            dist.all_reduce(x)
        annotations = [
            a
            for kernel_annotations in get_kernel_annotations().values()
            for a in kernel_annotations
            if "Collective name" in a
        ]
        self.assertTrue(annotations, "no kernel carries collective metadata")
        for annotation in annotations:
            self.assertEqual(annotation["Collective name"], "all_reduce")
            self.assertEqual(annotation["In msg nelems"], 1024)

    @skipIfRocm
    def test_cuspy_collective_is_annotated(self):
        if not TEST_CUPTI_V13_5:
            self.skipTest("requires CUDA >= 13.5 + libcupti >= 13.5")
        torch.cuda.set_device(self.device)
        x = torch.ones(1024, device=self.device)
        dist.all_reduce(x)
        torch.cuda.synchronize()
        cuda_config = ProfilerActivityConfig(profiler_configs=[CuspyConfig()])
        with TemporaryFileName(mode="w+") as trace_path:
            with profile(
                activities=[ProfilerActivity.CPU, {ProfilerActivity.CUDA: cuda_config}]
            ) as prof:
                dist.all_reduce(x)
                torch.cuda.synchronize()
            prof.export_chrome_trace(trace_path)
            if os.path.exists(trace_path + ".gz"):
                with gzip.open(trace_path + ".gz", "rt") as f:
                    events = json.load(f)["traceEvents"]
            else:
                with open(trace_path) as f:
                    events = json.load(f)["traceEvents"]
        kernels = [
            e
            for e in events
            if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower()
        ]
        self.assertTrue(kernels, "no NCCL kernel in the trace")
        for kernel in kernels:
            self.assertEqual(kernel["args"]["Collective name"], "all_reduce")
            self.assertEqual(kernel["args"]["In msg nelems"], 1024)


if __name__ == "__main__":
    if torch.cuda._initialized:
        raise AssertionError(
            "test_distributed must not have initialized CUDA context on main process"
        )

    run_tests()
