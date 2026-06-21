#!/usr/bin/env python3
"""Sweep batch sizes: FlashInfer old vs new vs vLLM Triton SSU kernel.

Patches the FlashInfer kernel between old (multi-stage TMA) and new (single-TMA)
and runs each config in a subprocess for clean module state.
"""

import subprocess, sys, os, json, shutil, re, textwrap, tempfile

H, D, N, G = 256, 64, 128, 8
CACHE = 256
BATCH_SIZES = [16, 32, 64, 128, 256, 512, 1024]

KERNEL_PATH = "/opt/vllm/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/mamba/kernel_selective_state_update_stp.cuh"
JIT_CACHE = "/home/vllm/.cache/flashinfer/"

KERNEL_OLD = "/workspace/kernel_ssu_old.cuh"
KERNEL_NEW = "/workspace/kernel_ssu_new.cuh"
KERNEL_NOTMA = "/workspace/kernel_ssu_notma.cuh"

WORKER = textwrap.dedent(r'''
import torch, sys, json
from flashinfer.testing import bench_gpu_time

B    = int(sys.argv[1])
mode = sys.argv[2]   # "flashinfer" or "triton"
H, D, N, G, CACHE = 256, 64, 128, 8, 256

state = torch.randn(CACHE, H, D, N, dtype=torch.float16, device="cuda")
x     = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
dt    = torch.randn(B, H, dtype=torch.float32, device="cuda").as_strided((B, H, D), (H, 1, 0))
A     = (-torch.rand(H, dtype=torch.float32, device="cuda") - 1.0).as_strided((H, D, N), (1, 0, 0))
Bm    = torch.randn(B, G, N, dtype=torch.float16, device="cuda")
C     = torch.randn(B, G, N, dtype=torch.float16, device="cuda")
Dm    = torch.randn(H, dtype=torch.float32, device="cuda").as_strided((H, D), (1, 0))
dtb   = (torch.rand(H, dtype=torch.float32, device="cuda") - 4.0).as_strided((H, D), (1, 0))
z     = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
idx   = torch.arange(B, dtype=torch.int64, device="cuda")

def report(times_ms):
    times_us = sorted([t * 1000.0 for t in times_ms])
    n = len(times_us)
    print(json.dumps({"min": times_us[0], "p50": times_us[n//2],
                       "mean": sum(times_us)/n, "p95": times_us[int(n*0.95)],
                       "n": n}))

if mode == "flashinfer":
    import flashinfer.mamba
    def run():
        return flashinfer.mamba.selective_state_update(
            state, x, dt, A, Bm, C, Dm, z=z, dt_bias=dtb,
            dt_softplus=True, state_batch_indices=idx, algorithm="horizontal")
    run()
    times = bench_gpu_time(fn=run, enable_cupti=True, repeat_iters=100, cold_l2_cache=True)
    report(times)

elif mode == "triton":
    import triton, triton.language as tl
    from packaging import version
    TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

    @triton.jit
    def fast_exp(x):
        LOG2E = tl.constexpr(1.4426950408889634)
        return tl.math.exp2(LOG2E * x)

    if TRITON3:
        @triton.jit
        def softplus(dt):
            dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
            return dt
    else:
        @triton.jit
        def softplus(dt):
            dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
            return dt

    @triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
    @triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
    @triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
    @triton.heuristics({"HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"] is not None})
    @triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
    @triton.jit(do_not_specialize=["N_seqs"])
    def _selective_scan_update_kernel(
        state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr,
        state_batch_indices_ptr,
        N_seqs, nheads, dim, dstate, nheads_ngroups_ratio,
        stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
        stride_x_batch, stride_x_head, stride_x_dim,
        stride_dt_batch, stride_dt_head, stride_dt_dim,
        stride_dt_bias_head, stride_dt_bias_dim,
        stride_A_head, stride_A_dim, stride_A_dstate,
        stride_B_batch, stride_B_group, stride_B_dstate,
        stride_C_batch, stride_C_group, stride_C_dstate,
        stride_D_head, stride_D_dim,
        stride_z_batch, stride_z_head, stride_z_dim,
        stride_out_batch, stride_out_head, stride_out_dim,
        DT_SOFTPLUS: tl.constexpr,
        TIE_HDIM: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        HAS_DT_BIAS: tl.constexpr,
        HAS_D: tl.constexpr,
        HAS_Z: tl.constexpr,
        HAS_STATE_BATCH_INDICES: tl.constexpr,
        BLOCK_SIZE_DSTATE: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_b = tl.program_id(axis=1)
        pid_h = tl.program_id(axis=2)
        if HAS_STATE_BATCH_INDICES:
            state_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
            state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
        else:
            state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
        x_ptr  += pid_b * stride_x_batch  + pid_h * stride_x_head
        dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
        if HAS_DT_BIAS:
            dt_bias_ptr += pid_h * stride_dt_bias_head
        A_ptr += pid_h * stride_A_head
        B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
        C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
        if HAS_Z:
            z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
        out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
        state_ptrs = state_ptr + offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        mask  = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
        state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs_m * stride_x_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_ptr + offs_m * stride_dt_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr + offs_m * stride_dt_bias_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A  = tl.load(A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate, mask=mask, other=0.0).to(tl.float32)
            dA = fast_exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A  = tl.load(A_ptr).to(tl.float32)
            dA = fast_exp(A * dt)
        B = tl.load(B_ptr + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptr + offs_n * stride_C_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)
        dB    = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
        state = state * dA + dB * x[:, None]
        tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)
        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            D_val = tl.load(D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
            out += x * D_val
        if HAS_Z:
            z = tl.load(z_ptr + offs_m * stride_z_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
            out *= z * tl.sigmoid(z)
        tl.store(out_ptr + offs_m * stride_out_dim, out, mask=offs_m < dim)

    out = torch.empty_like(x)
    tie_hdim = True
    grid = (triton.cdiv(D, 16), B, H)
    def run():
        _selective_scan_update_kernel[grid](
            state, x, dt, dtb, A, Bm, C, Dm, z, out, idx,
            B, H, D, N, H // G,
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            x.stride(0), x.stride(1), x.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2),
            dtb.stride(0), dtb.stride(1),
            A.stride(0), A.stride(1), A.stride(2),
            Bm.stride(0), Bm.stride(1), Bm.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            Dm.stride(0), Dm.stride(1),
            z.stride(0), z.stride(1), z.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            True, tie_hdim, 16,
            num_warps=2,
        )
    run()
    times = bench_gpu_time(fn=run, enable_cupti=True, repeat_iters=100, cold_l2_cache=True)
    report(times)
''')


def patch_kernel(version):
    src = {"old": KERNEL_OLD, "new": KERNEL_NEW, "notma": KERNEL_NOTMA}[version]
    shutil.copy(src, KERNEL_PATH)
    shutil.rmtree(JIT_CACHE, ignore_errors=True)


def run_worker(batch_size, mode):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(WORKER)
        worker_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, worker_path, str(batch_size), mode],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  ERROR (BS={batch_size}, {mode}): {result.stderr[:200]}", file=sys.stderr)
            return None
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('{'):
                return json.loads(line)
        print(f"  ERROR: no JSON in output: {result.stdout[:200]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (BS={batch_size}, {mode})", file=sys.stderr)
        return None
    finally:
        os.unlink(worker_path)


def state_bytes(batch_size):
    return batch_size * H * D * N * 2 * 2  # read + write, fp16


def main():
    import torch
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Config: H={H} D={D} N={N} G={G} state=fp16")
    print()

    results = {}

    # 1. Triton (no kernel patching needed)
    print("=== Benchmarking Triton ===")
    for bs in BATCH_SIZES:
        r = run_worker(bs, "triton")
        results[("triton", bs)] = r
        if r:
            print(f"  BS={bs:4d}  p50={r['p50']:7.1f} us")

    # 2. FlashInfer OLD (multi-stage TMA)
    print("\n=== Benchmarking FlashInfer OLD (multi-stage TMA) ===")
    patch_kernel("old")
    for bs in BATCH_SIZES:
        r = run_worker(bs, "flashinfer")
        results[("fi_old", bs)] = r
        if r:
            print(f"  BS={bs:4d}  p50={r['p50']:7.1f} us")

    # 3. FlashInfer NEW (single-TMA)
    print("\n=== Benchmarking FlashInfer NEW (single-TMA) ===")
    patch_kernel("new")
    for bs in BATCH_SIZES:
        r = run_worker(bs, "flashinfer")
        results[("fi_new", bs)] = r
        if r:
            print(f"  BS={bs:4d}  p50={r['p50']:7.1f} us")

    # 4. FlashInfer NOTMA (cooperative vectorized loads)
    print("\n=== Benchmarking FlashInfer NOTMA (cooperative loads, no TMA) ===")
    patch_kernel("notma")
    for bs in BATCH_SIZES:
        r = run_worker(bs, "flashinfer")
        results[("fi_notma", bs)] = r
        if r:
            print(f"  BS={bs:4d}  p50={r['p50']:7.1f} us")

    # Summary table
    print("\n" + "=" * 110)
    print(f"{'BS':>5s}  {'FI old p50':>10s}  {'FI new p50':>10s}  {'FI notma':>10s}  {'Triton p50':>10s}  "
          f"{'new/old':>8s}  {'notma/new':>9s}  {'notma/tri':>9s}  {'BW notma':>8s}  {'floor':>7s}")
    print("-" * 110)

    for bs in BATCH_SIZES:
        fi_old   = results.get(("fi_old", bs))
        fi_new   = results.get(("fi_new", bs))
        fi_notma = results.get(("fi_notma", bs))
        tri      = results.get(("triton", bs))

        old_p50   = fi_old['p50']   if fi_old   else float('nan')
        new_p50   = fi_new['p50']   if fi_new   else float('nan')
        notma_p50 = fi_notma['p50'] if fi_notma else float('nan')
        tri_p50   = tri['p50']      if tri      else float('nan')

        sb = state_bytes(bs)
        bw_notma = sb / (notma_p50 / 1e6) / 1e9 if fi_notma else float('nan')
        floor_us = sb / 8e12 * 1e6  # 8 TB/s

        ratio_new_old   = new_p50 / old_p50     if (fi_old and fi_new)     else float('nan')
        ratio_notma_new = notma_p50 / new_p50   if (fi_notma and fi_new)   else float('nan')
        ratio_notma_tri = notma_p50 / tri_p50   if (fi_notma and tri)      else float('nan')

        print(f"{bs:5d}  {old_p50:10.1f}  {new_p50:10.1f}  {notma_p50:10.1f}  {tri_p50:10.1f}  "
              f"{ratio_new_old:8.3f}  {ratio_notma_new:9.3f}  {ratio_notma_tri:9.3f}  {bw_notma:8.1f}  {floor_us:7.1f}")


if __name__ == "__main__":
    main()
