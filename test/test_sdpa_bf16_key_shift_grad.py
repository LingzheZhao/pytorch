# Owner(s): ["module: sdpa"]
"""The gradient of scaled_dot_product_attention along a shift shared by all keys.

Adding one vector b to every key adds q_i . b to every logit of row i. The softmax removes a
constant per row, so the output does not depend on b, and for any loss L

    dL/db = sum_j dL/dk_j = scale * sum_i (sum_j dS_ij) q_i = 0,

since every row of dS = P * (dP - rowsum(P * dP)) sums to zero. This holds for whatever keys the
kernel receives. A backward that forms dS from its own P and dP keeps the row sums at zero up to the
rounding of the sum itself, which does not depend on the sign of q_i . u, so the component of dL/db
along u = b / |b| averages to zero over independent batches. A non-zero mean, resolved by many
standard errors, is a bias. Under an optimiser that normalises its updates (Adam), a consistent bias
in a direction whose true gradient is zero moves that parameter steadily.

A backward that takes the row term from the stored output, D_i = rowsum(dO_i * O_i) (the dot_do_o
step of the flash-attention backward), gets instead sum_j dS_ij = dO_i . (Obar_i - O_i), where
Obar = P V with P in float32 and O is the stored, rounded output. For the "mse" loss below, with
dO = O - t and dlt = Obar - O, this is (Obar_i - t_i) . dlt_i - |dlt_i|^2, so
    E[dL/db . u] ~ -scale * sum_{n,i} (q_i . u) E|dlt_i|^2:
negative when the queries share a positive component along u, present with b = 0, and absent for the
"linear" loss, whose dO does not depend on the rounding. Per batch the sweep records this prediction,
x_pred = scale * sum_{n,i} (q_i . u) dO_i . (O_ref_i - O_i), from the dO the kernel received and its
output, with O_ref (below) in place of Obar. It applies only to a backward that takes D from the
stored output; the math back-end is its control. O_ref differs from the kernel's own Obar by the
float32 rounding of the logits, which grows with the row offset: with dO = O - O_ref - noise the mean
of x_pred carries -E|O_ref - O|^2 rather than -E|Obar - O|^2, so x_pred overstates the bias where the
offsets are large. The cells with key_shift = 0 (small offsets) are the clean comparison.

Inputs (per head h, a fixed unit vector u_h; z are standard normal draws, s = sqrt(logit_std)):
    q = s * z_q + query_shift * u        common component of every query along u
    k = s * (z_k - (z_k . u) u)          keys orthogonal to u: b alone sets the row offsets
    v = z_v
    keys passed to the kernel: k + key_shift * u, added in float32 or float64, then cast
Before the cast, the token part of the logits has standard deviation logit_std and each row is
offset by (q_i . u) * key_shift / sqrt(head_dim). In bf16 the casts of q and of k + b add
key-dependent rounding to the logits (from u . dk alone about query_shift * rms(u . dk) / sqrt(head_dim),
near 1 at the default cell), so at small logit_std the within-row spread is not logit_std; each run
records the measured within-row standard deviation of the logits of the tensors the kernel receives.
The losses are "linear", sum(out * c), with c a standard-normal tensor drawn independently for each
batch, and "mse", 0.5 * |out - t|^2 with t = O_ref + residual * rms(O_ref) * noise, where O_ref is
the attention written out in float32 (float64 for float64 inputs) on the tensors the kernel receives.

Per batch: x = mean over heads of dL/db_h . u_h, and the size of the terms that cancel,
terms = mean over heads of sum_{n,j} |dL/dk_{n,h,j} . u_h|.
Rule (the test's assertion): over the batches, fail when |mean x| > Z standard errors AND
|mean x| > TOL * mean(terms), with Z = 5 and TOL = 1e-6. The second clause excuses a rounding bias
far below the gradient's own scale (float64 can resolve one); the first excuses noise.

Usage:
    python test/test_sdpa_bf16_key_shift_grad.py                  # the unittest
    python test/test_sdpa_bf16_key_shift_grad.py --sweep [--json FILE] [--backends math,flash,...]
                                                    [--modes bf16,fp32,fp64,autocast] [--quick]
The sweep prints one table row per cell, precision and back-end, and records the attention kernels
of one profiled step per precision and back-end.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
import re
import sys
import time
import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from torch.testing._internal.common_utils import run_tests, TestCase
except ImportError:  # the internal helpers import expecttest and numpy
    TestCase = unittest.TestCase

    def run_tests():
        unittest.main()


Z = 5.0
TOL = 1e-6
BACKENDS = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}
# "autocast": float32 inputs, the attention called under torch.autocast("cuda", torch.bfloat16)
MODES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64, "autocast": torch.float32}
# kernel-name patterns that show a fused back-end ran (forward, backward); math must match none
KERNEL_PATTERNS = {
    "flash": (r"flash_fwd", r"flash_bwd"),
    "efficient": (r"fmha_cutlassF", r"fmha_cutlassB"),
    "cudnn": (r"cudnn.*(fprop|fwd)|(fprop|fwd).*cudnn", r"cudnn.*(bprop|bwd)|(bprop|bwd).*cudnn"),
}
FUSED_NAMES = re.compile(r"flash_fwd|flash_bwd|fmha|cudnn.*sdpa|sdpa.*cudnn|fprop|bprop", re.I)
ATTENTION_NAMES = re.compile(r"attn|attention|fmha|flash|sdp|softmax|cudnn|mha|fprop|bprop", re.I)


@dataclasses.dataclass(frozen=True)
class Cell:
    batch: int = 512
    heads: int = 8
    seq: int = 21
    head_dim: int = 64
    logit_std: float = 4.0
    query_shift: float = 512.0
    key_shift: float = 64.0
    loss: str = "mse"
    residual: float = 0.1


DEFAULT = Cell()


@contextlib.contextmanager
def _tf32_off():
    # float32 products of the math back-end and the reference in full float32 (cuDNN flags are not
    # touched: the test helpers freeze them, and no convolution runs here)
    m = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = m


def _directions(cell, device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    u = torch.randn(cell.heads, cell.head_dim, generator=g, device=device)
    return u / u.norm(dim=-1, keepdim=True)


def _inputs(cell, i, u, device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed * 1_000_003 + i + 1)
    shape = (cell.batch, cell.heads, cell.seq, cell.head_dim)
    zq, zk, v, noise = (torch.randn(shape, generator=g, device=device) for _ in range(4))
    u4 = u[None, :, None, :]
    s = math.sqrt(cell.logit_std)
    q = s * zq + cell.query_shift * u4
    k = s * (zk - (zk * u4).sum(-1, keepdim=True) * u4)
    return q, k, v, noise


def _written_out(q, k, v):
    """The attention written out, and the mean over rows of the within-row standard deviation of the logits."""
    logits = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.softmax(logits, dim=-1) @ v, logits.std(dim=-1).mean()


def _batch(cell, backend, mode, i, u, device, seed):
    q, k, v, noise = _inputs(cell, i, u, device, seed)
    if mode == "fp64":
        q, k, v, noise, u = (t.double() for t in (q, k, v, noise, u))
    work = MODES[mode]
    b = torch.nn.Parameter(cell.key_shift * u)
    kb = k + b[None, :, None, :]
    kc = kb.to(work)
    kc.retain_grad()
    qc, vc = q.to(work), v.to(work)
    cast = torch.autocast("cuda", dtype=torch.bfloat16) if mode == "autocast" else contextlib.nullcontext()
    with sdpa_kernel([BACKENDS[backend]]), cast:
        raw = F.scaled_dot_product_attention(qc, kc, vc)
    raw.retain_grad()  # the dO the kernel's backward receives
    out_dtype = raw.dtype
    out = raw.to(q.dtype)
    with torch.no_grad():  # the tensors the kernel receives, upcast, and their attention written out
        seen = torch.bfloat16 if mode == "autocast" else work
        qs, ks, vs = (t.detach().to(seen).to(q.dtype) for t in (qc, kc, vc))
        ref, logit_std_seen = _written_out(qs, ks, vs)
    if cell.loss == "linear":
        loss = (out * noise).sum()
    else:
        target = ref + cell.residual * ref.pow(2).mean().sqrt() * noise
        loss = 0.5 * (out - target).pow(2).sum()
    loss.backward()
    gb = b.grad
    dk = kc.grad.to(gb.dtype)
    u4 = u[None, :, None, :]
    x = (gb * u).sum(-1).mean()
    terms = (dk * u4).sum(-1).abs().sum(dim=(0, 2)).mean()
    error = None
    try:  # the dot_do_o prediction; a diagnostic column, so a failure here leaves x intact
        with torch.no_grad():
            d_o = raw.grad.to(q.dtype)
            o = raw.detach().to(q.dtype)
            qu = (qs * u4).sum(-1)  # (B, H, L)
            rows = (d_o * (ref - o)).sum(-1)  # (B, H, L): dO_i . (O_ref_i - O_i)
            x_pred = ((qu * rows).sum(dim=(0, 2)) / math.sqrt(q.shape[-1])).mean()
    except Exception as e:  # noqa: BLE001 - recorded
        x_pred, error = torch.full((), float("nan"), device=x.device), repr(e)[:300]
    row = torch.stack([t.double() for t in (x, gb.norm(), terms, dk.norm(), loss.detach(), x_pred,
                                            logit_std_seen)])
    return row, out_dtype, error


def _stats(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((a - m) ** 2 for a in v) / (n - 1)) if n > 1 else float("nan")
    return {"mean": m, "se": sd / math.sqrt(n) if n > 1 else float("nan"), "sd": sd, "n": n}


def measure(cell, backend, mode, n_batches, device="cuda", seed=0, deadline=None):
    """Mean and standard error over n_batches of the component of dL/db along u, and the scales."""
    u = _directions(cell, device, seed)
    rows, out_dtype, pred_error = [], None, None
    with _tf32_off():
        for i in range(n_batches):
            if deadline is not None and time.time() > deadline:
                break
            row, out_dtype, error = _batch(cell, backend, mode, i, u, device, seed)
            pred_error = pred_error or error
            rows.append(row)
    cols = list(zip(*torch.stack(rows).cpu().tolist())) if rows else [[]] * 7
    x, gnorm, terms, dknorm, loss, x_pred, logit_std_seen = (list(c) for c in cols)
    nan = float("nan")

    def stats(v):
        return _stats(v) if len(v) > 1 else {"mean": nan, "se": nan, "sd": nan, "n": len(v)}

    def mean(v):
        return sum(v) / len(v) if v else nan

    s, p = stats(x), stats(x_pred)
    d = stats([a - c for a, c in zip(x, x_pred)])
    t = mean(terms)
    res = {
        "backend": backend, "mode": mode, "attention_output_dtype": str(out_dtype), "n": s["n"],
        "mean": s["mean"], "se": s["se"], "sd": s["sd"],
        "z": s["mean"] / s["se"] if s["se"] and s["se"] == s["se"] else nan,
        "terms": t, "mean_over_terms": s["mean"] / t if t else nan,
        "gradient_norm": mean(gnorm), "key_gradient_norm": mean(dknorm), "loss": mean(loss),
        "within_row_logit_std": mean(logit_std_seen),
        # the dot_do_o prediction (module docstring); expected to match only a backward that takes D from O
        "dot_do_o_prediction": {"mean": p["mean"], "se": p["se"], "error": pred_error},
        "measured_minus_prediction": {"mean": d["mean"], "se": d["se"]},
        "correlation_with_prediction": _correlation(x, x_pred),
    }
    res["gradient_norm_over_terms"] = res["gradient_norm"] / t if t else nan
    res["fails"] = bool(abs(s["mean"]) > Z * s["se"] and abs(s["mean"]) > TOL * t) if s["n"] > 1 else None
    return res


def _correlation(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa, sbb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return sab / math.sqrt(saa * sbb) if saa > 0 and sbb > 0 else float("nan")


def kernel_record(cell, backend, mode, device="cuda", seed=0):
    """The CUDA kernels of one profiled forward and backward, and whether the back-end's kernels ran."""
    from torch.profiler import profile, ProfilerActivity

    u = _directions(cell, device, seed)
    with _tf32_off(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _batch(cell, backend, mode, 0, u, device, seed)
        torch.cuda.synchronize()
    ops, kernels = {}, {}
    for e in prof.key_averages():
        cpu = e.device_type == torch.autograd.DeviceType.CPU
        if cpu and ATTENTION_NAMES.search(e.key):
            ops[e.key] = e.count
        if not cpu:
            kernels[e.key] = e.count
    attention = {k: c for k, c in kernels.items() if ATTENTION_NAMES.search(k)}
    if backend == "math":
        ran = not any(FUSED_NAMES.search(k) for k in kernels)
    else:
        fwd, bwd = KERNEL_PATTERNS[backend]
        ran = any(re.search(fwd, k, re.I) for k in kernels) and any(re.search(bwd, k, re.I) for k in kernels)
    q = torch.zeros(cell.batch, cell.heads, cell.seq, cell.head_dim, device=device,
                    dtype=torch.bfloat16 if mode == "autocast" else MODES[mode])
    try:
        with sdpa_kernel([BACKENDS[backend]]):
            choice = int(torch._fused_sdp_choice(q, q, q))
    except Exception as e:  # noqa: BLE001 - a record
        choice = repr(e)
    # the element type is read from the full names; cuDNN's bf16 attention kernels are named "..._wmma_f16_..."
    return {"backend": backend, "mode": mode, "backend_kernels_ran": ran, "fused_sdp_choice": choice,
            "attention_ops": ops, "attention_kernels": attention}


def _run_or_refuse(fn):
    try:
        return "ok", fn()
    except Exception as e:  # noqa: BLE001 - recorded; the other cases still run
        if isinstance(e, RuntimeError) and ("No available kernel" in str(e) or "not supported" in str(e).lower()):
            return "refused", (str(e).splitlines() or [""])[0][:300]
        return "error", repr(e)[:1000]


# ---------------------------------------------------------------- unittest


CASES = [("math", "bf16"), ("flash", "bf16"), ("efficient", "bf16"), ("math", "fp32"), ("efficient", "fp32"),
         ("math", "fp64"), ("cudnn", "bf16")]
# Known failures (unittest.expectedFailure): the fused bf16 backwards. Flash takes the row term
# D_i = rowsum(dO_i * O_i) from the stored bf16 output (flash_bwd_dot_do_o_kernel) while it recomputes P in
# float32, so the rows of dS do not sum to zero; cuDNN also launches a dot_do_o kernel. The means follow the
# prediction of a backward that takes its row term from the stored output (module docstring), negative when the
# queries share a component along u; this was read in the sources for flash only.
EXPECTED_TO_FAIL = {("flash", "bf16"), ("efficient", "bf16"), ("cudnn", "bf16")}
TEST_BATCHES = {"bf16": 300, "fp32": 300, "fp64": 32}


class TestSDPAKeyShiftGradient(TestCase):
    def _check(self, backend, mode):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if mode == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            self.skipTest("bf16 is not supported on this device")
        status, r = _run_or_refuse(lambda: measure(DEFAULT, backend, mode, TEST_BATCHES[mode]))
        if status == "refused":
            self.skipTest(f"{backend} refuses {mode} inputs: {r}")
        if status == "error":
            raise RuntimeError(r)
        msg = (f"{backend} {mode}: mean dL/db . u = {r['mean']:.3e} +- {r['se']:.3e} (z {r['z']:.1f}) over "
               f"{r['n']} batches; the exact value is 0; summed |dL/dk . u| = {r['terms']:.3e}")
        print(msg, file=sys.stderr)  # an expected failure hides its message
        self.assertGreater(r["se"], 0.0)
        self.assertLessEqual(abs(r["mean"]), max(Z * r["se"], TOL * r["terms"]), msg)


for _backend, _mode in CASES:
    def _test(self, backend=_backend, mode=_mode):
        self._check(backend, mode)
    if (_backend, _mode) in EXPECTED_TO_FAIL:
        _test = unittest.expectedFailure(_test)
    setattr(TestSDPAKeyShiftGradient, f"test_key_shift_gradient_{_backend}_{_mode}", _test)
del _test


# ---------------------------------------------------------------- sweep


def default_grid():
    # the default cell, then its controls: a linear readout, no common query component, no key shift
    cells = [DEFAULT, dataclasses.replace(DEFAULT, loss="linear", residual=0.0),
             dataclasses.replace(DEFAULT, query_shift=0.0), dataclasses.replace(DEFAULT, key_shift=0.0)]
    for loss, residual in (("mse", 0.1), ("mse", 1.0), ("linear", 0.0)):
        for logit_std in (4.0, 1.0, 16.0):
            for query_shift in (512.0, 64.0, 0.0):
                for key_shift in (64.0, 8.0, 0.0):
                    c = Cell(logit_std=logit_std, query_shift=query_shift, key_shift=key_shift, loss=loss,
                             residual=residual)
                    if c not in cells:
                        cells.append(c)
    for loss, residual in (("mse", 0.1), ("linear", 0.0)):
        for query_shift in (512.0, 0.0):
            for key_shift in (64.0, 0.0):
                cells.append(Cell(batch=4, seq=1024, query_shift=query_shift, key_shift=key_shift, loss=loss,
                                  residual=residual))
    return cells


RUN_ORDER = [("flash", "bf16"), ("efficient", "bf16"), ("cudnn", "bf16"), ("math", "bf16"),
             ("efficient", "fp32"), ("math", "fp32"), ("math", "fp64")]
DEFAULT_ONLY = [("flash", "autocast"), ("efficient", "autocast"), ("cudnn", "autocast"), ("math", "autocast"),
                ("flash", "fp32"), ("cudnn", "fp32")]  # the last two record a refusal, if any


def _versions():
    p = torch.cuda.get_device_properties(0)
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "multiprocessors": p.multi_processor_count,
            "math_sdp_fp16_bf16_reduction_allowed": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}


def _row(cell, r):
    head = (f"{cell.batch:4d} {cell.heads:2d} {cell.seq:5d} {cell.head_dim:3d} {cell.logit_std:5g} "
            f"{cell.query_shift:5g} {cell.key_shift:4g} {cell.loss:6s} {cell.residual:4g} {r['mode']:8s} "
            f"{r['backend']:9s}")
    if r.get("status") != "ok":
        return f"{head} {r.get('status')}: {r.get('reason', '')[:80]}"
    rule = "n/a" if r["fails"] is None else ("FAIL" if r["fails"] else "ok")
    return (f"{head} {r['n']:4d} {r['mean']:+.3e} {r['se']:.2e} {r['z']:+7.1f} {r['mean_over_terms']:+.2e} "
            f"{r['gradient_norm_over_terms']:.2e} {r['dot_do_o_prediction']['mean']:+.3e} "
            f"{r['within_row_logit_std']:6.2f} {rule}")


HEADER = ("   B  H     L   D  lstd  qsh  |b| loss   rho  mode     backend      n  mean x      SE        z     "
          "mean/terms |g|/terms x_pred     seen_lstd rule")


def sweep(args):
    if not torch.cuda.is_available():
        sys.exit("the sweep needs a CUDA device")
    start = time.time()
    deadline = start + args.budget_seconds
    backends, modes = set(args.backends.split(",")), set(args.modes.split(","))
    grid = [DEFAULT] if args.quick else default_grid()
    out = {"versions": _versions(), "rule": {"Z": Z, "TOL": TOL},
           "definition": __doc__, "batches": args.batches, "fp64_batches": args.fp64_batches, "seed": args.seed,
           "kernels": [], "cells": [], "not_run": []}

    def write():
        if args.json:
            tmp = args.json + ".tmp"
            with open(tmp, "w") as f:
                json.dump(out, f, indent=1)
            os.replace(tmp, args.json)

    for backend, mode in RUN_ORDER + DEFAULT_ONLY:
        if backend in backends and mode in modes:
            status, rec = _run_or_refuse(lambda: kernel_record(DEFAULT, backend, mode, seed=args.seed))
            out["kernels"].append(rec if status == "ok" else {"backend": backend, "mode": mode, "status": status,
                                                              "reason": rec})
            k = out["kernels"][-1]
            print("kernels", backend, mode, k.get("backend_kernels_ran", status),
                  sorted(k.get("attention_kernels", {})), flush=True)
    write()
    print(HEADER, flush=True)
    for ci, cell in enumerate(grid):
        pairs = RUN_ORDER + (DEFAULT_ONLY if cell == DEFAULT else [])
        pairs = [(b, m) for b, m in pairs if b in backends and m in modes]
        if time.time() > deadline:
            out["not_run"].append(dataclasses.asdict(cell))
            continue
        rec = {"cell": dataclasses.asdict(cell), "runs": []}
        out["cells"].append(rec)
        for backend, mode in pairs:
            if time.time() > deadline:
                rec["runs"].append({"backend": backend, "mode": mode, "status": "not run (budget)"})
                continue
            n = args.fp64_batches if mode == "fp64" else args.batches
            status, r = _run_or_refuse(lambda: measure(cell, backend, mode, n, seed=args.seed, deadline=deadline))
            r = dict(r, status="ok") if status == "ok" else {"backend": backend, "mode": mode, "status": status,
                                                              "reason": r}
            rec["runs"].append(r)
            print(_row(cell, r), flush=True)
        write()
    out["seconds"] = time.time() - start
    write()
    if args.table:
        with open(args.table, "w") as f:
            f.write(HEADER + "\n")
            for rec in out["cells"]:
                cell = Cell(**rec["cell"])
                for r in rec["runs"]:
                    f.write(_row(cell, r) + "\n")
    # refusals and errors of single runs are recorded, not raised; the exit code shows only a sweep in
    # which the first (default) cell has no completed run
    if not out["cells"] or not any(r.get("status") == "ok" for r in out["cells"][0]["runs"]):
        sys.exit("no run of the default cell completed")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--sweep", action="store_true")
        p.add_argument("--json")
        p.add_argument("--table")
        p.add_argument("--batches", type=int, default=256)
        p.add_argument("--fp64-batches", type=int, default=16)
        p.add_argument("--backends", default="math,flash,efficient,cudnn")
        p.add_argument("--modes", default="bf16,fp32,fp64,autocast")
        p.add_argument("--budget-seconds", type=float, default=900.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--quick", action="store_true", help="the default cell only")
        sweep(p.parse_args())
    else:
        run_tests()
