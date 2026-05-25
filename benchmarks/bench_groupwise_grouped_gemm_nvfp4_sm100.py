"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from itertools import product

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability, is_sm100a_supported


def bench_nvfp4(group_size, m, n, k, out_dtype):
    torch.random.manual_seed(0)
    tile_size = 16
    alignment_sf = 128
    a = torch.randint(
        0, 256, (group_size * m, k // 2), dtype=torch.uint8, device="cuda:0"
    )
    b = torch.randint(
        0, 256, (group_size, n, k // 2), dtype=torch.uint8, device="cuda:0"
    )
    out = torch.empty(group_size * m, n, dtype=out_dtype, device="cuda:0")

    a_scale = torch.randint(
        0,
        256,
        (
            (group_size * m + (alignment_sf - 1) * group_size)
            // alignment_sf
            * alignment_sf,
            k // tile_size,
        ),
        dtype=torch.uint8,
        device="cuda:0",
    )
    b_scale = torch.randint(
        0,
        256,
        (
            group_size,
            (n + alignment_sf - 1) // alignment_sf * alignment_sf,
            k // tile_size,
        ),
        dtype=torch.uint8,
        device="cuda:0",
    )

    segment_offsets = torch.arange(
        0, (group_size + 1) * m, m, device="cuda:0", dtype=torch.int32
    )

    tile_k_list = [128, 256]

    ms_best = float("inf")
    config_best = None
    for tile_k in tile_k_list:
        measurements = bench_gpu_time(
            lambda: flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
                a,
                b,
                a_scale,
                b_scale,
                segment_offsets,
                out=out,
                tile_m=128,
                tile_n=128,
                tile_k=tile_k,
            ),
            dry_run_time_ms=10,
            repeat_time_ms=100,
        )
        ms = np.median(measurements)
        if ms < ms_best:
            ms_best = ms
            config_best = {"tile_k": tile_k}

    tflops = 2 * group_size * m * n * k * 1e-9 / ms_best
    return tflops, ms_best, config_best


def bench_mxfp4(group_size, m, n, k, in_dtype, out_dtype):
    torch.random.manual_seed(0)
    tile_size = 32
    alignment_sf = 128
    fp8_info = torch.finfo(in_dtype)
    a = (
        torch.empty(group_size * m, k, dtype=torch.float32, device="cuda:0")
        .uniform_(-fp8_info.max, fp8_info.max)
        .to(in_dtype)
    )
    b = torch.randint(
        0, 256, (group_size, n, k // 2), dtype=torch.uint8, device="cuda:0"
    )
    out = torch.empty(group_size * m, n, dtype=out_dtype, device="cuda:0")

    a_scale = torch.randint(
        0,
        256,
        (
            (group_size * m + (alignment_sf - 1) * group_size)
            // alignment_sf
            * alignment_sf,
            k // tile_size,
        ),
        dtype=torch.uint8,
        device="cuda:0",
    )
    b_scale = torch.randint(
        0,
        256,
        (
            group_size,
            (n + alignment_sf - 1) // alignment_sf * alignment_sf,
            k // tile_size,
        ),
        dtype=torch.uint8,
        device="cuda:0",
    )

    segment_offsets = torch.arange(
        0, (group_size + 1) * m, m, device="cuda:0", dtype=torch.int32
    )

    mma_sm_list = [1, 2]
    tile_m_list = [128]
    tile_n_list = [64, 128, 192, 256]
    tile_k_list = [128, 256]
    swap_ab_list = [True, False]

    ms_best = float("inf")
    config_best = None
    for mma_sm, tile_m, tile_n, tile_k, swap_ab in product(
        mma_sm_list, tile_m_list, tile_n_list, tile_k_list, swap_ab_list
    ):
        measurements = bench_gpu_time(
            lambda: flashinfer.gemm.group_gemm_mxfp4_nt_groupwise(
                a,
                b,
                a_scale,
                b_scale,
                segment_offsets,
                out=out,
                mma_sm=mma_sm,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                swap_ab=swap_ab,
            ),
            dry_run_time_ms=10,
            repeat_time_ms=100,
        )
        ms = np.median(measurements)
        if ms < ms_best:
            ms_best = ms
            config_best = {
                "mma_sm": mma_sm,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "swap_ab": swap_ab,
            }

    tflops = 2 * group_size * m * n * k * 1e-9 / ms_best
    return tflops, ms_best, config_best


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 vs MXFP4 group GEMM on SM100"
    )
    parser.add_argument(
        "--group-size", type=int, nargs="+", default=[1, 3, 8, 16]
    )
    parser.add_argument(
        "--m", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[1024, 2048, 4096, 7168, 8192]
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1024, 2048, 4096, 7168, 8192]
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    compute_capability = get_compute_capability(device)
    if not is_sm100a_supported(device):
        print(
            f"This benchmark requires SM100/SM103 GPU, got SM{compute_capability[0]}{compute_capability[1]}."
        )
        return

    out_dtype = torch.bfloat16
    in_dtype_mxfp4 = torch.float8_e4m3fn

    print(
        f"{'group_size':>10} {'m':>6} {'n':>6} {'k':>6} | "
        f"{'NVFP4 TFLOPS':>13} {'config':>12} | "
        f"{'MXFP4 TFLOPS':>13} {'config':>30} | "
        f"{'ratio':>6}"
    )
    print("-" * 130)

    for group_size in args.group_size:
        for m in args.m:
            for n in args.n:
                for k in args.k:
                    nvfp4_tflops, nvfp4_ms, nvfp4_cfg = bench_nvfp4(
                        group_size, m, n, k, out_dtype
                    )
                    mxfp4_tflops, mxfp4_ms, mxfp4_cfg = bench_mxfp4(
                        group_size, m, n, k, in_dtype_mxfp4, out_dtype
                    )
                    ratio = nvfp4_tflops / mxfp4_tflops if mxfp4_tflops > 0 else 0
                    nvfp4_cfg_str = f"tk={nvfp4_cfg['tile_k']}"
                    mxfp4_cfg_str = (
                        f"sm={mxfp4_cfg['mma_sm']} tn={mxfp4_cfg['tile_n']} "
                        f"tk={mxfp4_cfg['tile_k']} sw={mxfp4_cfg['swap_ab']}"
                    )
                    print(
                        f"{group_size:>10} {m:>6} {n:>6} {k:>6} | "
                        f"{nvfp4_tflops:>10.2f}    {nvfp4_cfg_str:>12} | "
                        f"{mxfp4_tflops:>10.2f}    {mxfp4_cfg_str:>30} | "
                        f"{ratio:>5.2f}x"
                    )


if __name__ == "__main__":
    main()
