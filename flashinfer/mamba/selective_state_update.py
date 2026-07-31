"""
Copyright (c) 2025 by FlashInfer team.

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

import functools
from typing import Any, List, Optional

import torch

from ..api_logging import flashinfer_api
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ..trace.templates.mamba import selective_state_update_trace
from ..jit.mamba import (
    gen_selective_state_update_module,
    gen_selective_state_update_sm100_module,
    gen_selective_state_update_sm90_module,
)
from ..utils import get_compute_capability, register_custom_op, register_fake_op


@functools.cache
def _get_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    cu_seqlens_dtype: torch.dtype,
    num_accepted_tokens_dtype: torch.dtype,
    sm_major: int,
    state_scale_dtype: Optional[torch.dtype] = None,
    philox_rounds: int = 0,
    force_num_stages: int = 0,
    force_permutation_type: int = 0,
    force_stage_cols: int = 0,
):
    args = (
        state_dtype,
        input_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        state_scale_dtype,
        dim,
        dstate,
        ntokens_mtp,
        cu_seqlens_dtype,
        num_accepted_tokens_dtype,
        philox_rounds,
        force_num_stages,
        force_permutation_type,
        force_stage_cols,
    )
    if sm_major >= 10:
        return gen_selective_state_update_sm100_module(*args).build_and_load()
    elif sm_major >= 9:
        return gen_selective_state_update_sm90_module(*args).build_and_load()
    else:
        return gen_selective_state_update_module(*args).build_and_load()


class SSUHorizontalRunner(TunableRunner):
    TACTIC_CONFIGS = {
        0: (0, 0, 0),
        1: (1, 1, 0),
        2: (1, 2, 0),
        3: (4, 1, 0),
        4: (4, 2, 0),
        5: (1, 1, 64),
        6: (1, 2, 64),
        7: (2, 1, 64),
        8: (2, 2, 64),
        9: (1, 1, 32),
        10: (1, 2, 32),
        11: (4, 1, 32),
        12: (4, 2, 32),
    }

    def __init__(
        self,
        module_base_args,
        dt_softplus,
        pad_slot_id,
        disable_state_update,
        cache_steps,
        algorithm_int,
    ):
        self._module_base_args = module_base_args
        self._dt_softplus = dt_softplus
        self._pad_slot_id = pad_slot_id
        self._disable_state_update = disable_state_update
        self._cache_steps = cache_steps
        self._algorithm_int = algorithm_int

    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile
    ) -> List[int]:
        return list(self.TACTIC_CONFIGS.keys())

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        return (
            self._module_base_args,
            self._algorithm_int,
            inputs[7] is not None,
            inputs[8] is not None,
            inputs[12] is not None,
        )

    def forward(
        self,
        inputs: List[Any],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> None:
        if do_preparation:
            for stages, perm, sc in self.TACTIC_CONFIGS.values():
                _get_module(*self._module_base_args, stages, perm, sc)
            return

        if tactic < 0:
            force_num_stages, force_permutation_type, force_stage_cols = 0, 0, 0
        else:
            force_num_stages, force_permutation_type, force_stage_cols = self.TACTIC_CONFIGS[tactic]

        module = _get_module(
            *self._module_base_args, force_num_stages, force_permutation_type, force_stage_cols
        )

        (
            state, x, dt, A, B, C, D,
            z, dt_bias, output,
            state_batch_indices, dst_state_batch_indices,
            state_scale,
            intermediate_states_buffer, intermediate_state_indices,
            intermediate_state_scales, rand_seed,
            cu_seqlens, num_accepted_tokens,
        ) = inputs

        module.selective_state_update(
            state, x, dt, A, B, C, D,
            z, dt_bias, self._dt_softplus,
            state_batch_indices, dst_state_batch_indices,
            self._pad_slot_id, state_scale, output,
            self._disable_state_update,
            intermediate_states_buffer, intermediate_state_indices,
            intermediate_state_scales, rand_seed,
            self._cache_steps, cu_seqlens, num_accepted_tokens,
            self._algorithm_int,
        )


@flashinfer_api(trace=selective_state_update_trace)
def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    state_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    intermediate_state_scales: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    cache_steps: int = 0,
    algorithm: str = "auto",
    dst_state_batch_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Selective state update operation for Mamba layers (the generation phase).

    Parameters
    ----------
    state : torch.Tensor
        State tensor with shape (state_cache_size, dim, dstate) or (state_cache_size, nheads, dim, dstate)
    x : torch.Tensor
        Input tensor with shape (batch, dim) or (batch, nheads, dim) for single-token,
        (batch, T, nheads, dim) for multi-token,
        or (total_tokens, nheads, dim) for varlen multi-token (with cu_seqlens)
    dt : torch.Tensor
        Delta time tensor, same layout as x
    A : torch.Tensor
        A matrix with shape (dim, dstate) or (nheads, dim, dstate)
    B : torch.Tensor
        B matrix with shape (batch, dstate) or (batch, ngroups, dstate) for single-token,
        (batch, T, ngroups, dstate) for multi-token,
        or (total_tokens, ngroups, dstate) for varlen multi-token
    C : torch.Tensor
        C matrix, same layout as B
    D : torch.Tensor
        D vector with shape (dim,) or (nheads, dim)
    z : Optional[torch.Tensor]
        Optional z tensor, same layout as x
    dt_bias : Optional[torch.Tensor]
        Optional dt bias with shape (dim,) or (nheads, dim)
    dt_softplus : bool
        Whether to apply softplus to dt
    state_batch_indices : Optional[torch.Tensor]
        Batch indices for state cache reading. Shape (batch,) or (N, max_seqlen).
        For speculative decoding with num_accepted_tokens, must be 2D.
    dst_state_batch_indices : Optional[torch.Tensor]
        Destination indices for state cache writing. Shape (batch,) or (N, max_seqlen).
        When provided, state is read from state_batch_indices and written to
        dst_state_batch_indices (enables separate read/write state slots).
    pad_slot_id : int
        Sentinel value for padded entries in state_batch_indices
    state_scale : Optional[torch.Tensor]
        Optional float32 scale tensor with shape (state_cache_size, nheads, dim)
        for int16 state quantization with block scaling. Listed in the custom
        op's ``mutates_args``: when ``state`` is quantized (int16), the kernel
        writes the new per-block scales here in place — the caller dequantizes
        ``state`` against this tensor on read-back (mirrors the
        ``intermediate_state_scales`` contract below).
    out : Optional[torch.Tensor]
        Optional output tensor (same shape as x)
    disable_state_update : bool
        If True, skip updating the state tensor (useful for speculative decoding verification)
    intermediate_states_buffer : Optional[torch.Tensor]
        Optional buffer for caching intermediate states during speculative decoding
        with shape (batch, cache_steps, nheads, dim, dstate). Also listed in
        ``mutates_args`` — the kernel writes intermediate states into it
        in place.
    intermediate_state_indices : Optional[torch.Tensor]
        Optional indices mapping batch elements to intermediate state buffer positions
        with shape (batch,)
    intermediate_state_scales : Optional[torch.Tensor]
        Optional per-block float32 scale tensor matching ``intermediate_states_buffer``.
        When provided alongside an int16 ``intermediate_states_buffer``, the kernel
        writes the computed scales into this tensor (it is listed in the custom
        op's ``mutates_args``), and the caller is responsible for *dequantizing*
        the intermediate states using these scales when reading them back.
        Mirrors the ``state_scale`` layout but for the speculative-decoding
        intermediate buffer.
    rand_seed : Optional[torch.Tensor]
        Optional single-element int64 CUDA tensor for stochastic rounding seed.
    philox_rounds : int
        Number of Philox-4x32 PRNG rounds for stochastic rounding (default 10).
    cache_steps : int
        Number of steps/tokens to cache for speculative decoding.
        For varlen mode (cu_seqlens provided), this specifies max_seqlen.
    cu_seqlens : Optional[torch.Tensor]
        Cumulative sequence lengths with shape (N + 1,), integer dtype
        (the JIT specializes on the actual dtype, so int32 or int64 is fine;
        int32 is the default when omitted). When provided, inputs are in
        packed-token varlen format: ``x`` / ``dt`` are 3-D
        ``(total_tokens, nheads, dim)``, ``B`` / ``C`` are 3-D
        ``(total_tokens, ngroups, dstate)``, with sequence boundaries given
        by ``cu_seqlens``.
    num_accepted_tokens : Optional[torch.Tensor]
        Number of accepted tokens per sequence with shape (N,).
        Determines which state to read as initial state for each sequence.
    algorithm : str
        Algorithm to use: "auto", "simple", "vertical", "horizontal"

    Returns
    -------
    output : torch.Tensor
        Output tensor with same shape as x
    """
    is_varlen = cu_seqlens is not None and x.dim() == 3
    is_mtp = cache_steps >= 1 and not is_varlen

    if state.dim() == 3:
        state = state.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if D.dim() == 1:
        D = D.unsqueeze(0)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    if not is_varlen:
        # Handle x, dt, B, C, z dimensions based on mode
        # For single-token: 2D -> 3D (batch, nheads, dim)
        # For multi-token: 3D -> 4D (batch, T, nheads, dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if is_mtp and x.dim() == 3:
            # Add T dimension for MTP mode: (batch, nheads, dim) -> (batch, T, nheads, dim)
            x = x.unsqueeze(1)

        if dt.dim() == 2:
            dt = dt.unsqueeze(1)
        if is_mtp and dt.dim() == 3:
            dt = dt.unsqueeze(1)

        if B.dim() == 2:
            B = B.unsqueeze(1)
        if is_mtp and B.dim() == 3:
            B = B.unsqueeze(1)

        if C.dim() == 2:
            C = C.unsqueeze(1)
        if is_mtp and C.dim() == 3:
            C = C.unsqueeze(1)

        if z is not None:
            if z.dim() == 2:
                z = z.unsqueeze(1)
            if is_mtp and z.dim() == 3:
                z = z.unsqueeze(1)

    # Normalize state_scale to 3D: (state_cache_size, nheads, dim)
    if state_scale is not None and state_scale.dim() == 4 and state_scale.size(-1) == 1:
        state_scale = state_scale.squeeze(-1)

    # Validate rand_seed and philox_rounds
    if rand_seed is not None:
        if not isinstance(rand_seed, torch.Tensor):
            raise TypeError(
                f"rand_seed must be a CUDA int64 tensor, got {type(rand_seed).__name__}"
            )
        if rand_seed.numel() != 1:
            raise ValueError(
                f"rand_seed must be a single-element tensor, got numel={rand_seed.numel()}"
            )
        if rand_seed.dtype != torch.int64:
            raise ValueError(f"rand_seed must have dtype int64, got {rand_seed.dtype}")
        if not rand_seed.is_cuda:
            raise ValueError("rand_seed must be a CUDA tensor")
        if state_scale is not None:
            raise ValueError("rand_seed and state_scale cannot both be provided")
        if philox_rounds <= 0:
            raise ValueError(
                f"philox_rounds must be > 0 when rand_seed is provided, got {philox_rounds}"
            )
    else:
        # No stochastic rounding when rand_seed is None
        philox_rounds = 0

    if intermediate_states_buffer is not None and dst_state_batch_indices is not None:
        raise ValueError(
            "intermediate_states_buffer and dst_state_batch_indices are mutually exclusive"
        )

    if out is None:
        output = torch.empty_like(x)
    else:
        output = out

    # Determine stateIndex dtype from index tensors, default to int32
    stateIndex_dtype = torch.int32
    if state_batch_indices is not None:
        stateIndex_dtype = state_batch_indices.dtype
    elif dst_state_batch_indices is not None:
        stateIndex_dtype = dst_state_batch_indices.dtype
    elif intermediate_state_indices is not None:
        stateIndex_dtype = intermediate_state_indices.dtype

    # Extract dim/dstate/ntokens for JIT specialization
    dim = state.size(2)
    dstate = state.size(3)
    if is_varlen:
        ntokens_mtp = cache_steps
    elif x.dim() == 4:
        ntokens_mtp = x.size(1)
    else:
        ntokens_mtp = 1

    # Parse algorithm parameter for stage/permutation hints
    force_num_stages = 0
    force_permutation_type = 0

    if algorithm == "auto":
        algorithm_int = 0
    elif algorithm == "simple":
        algorithm_int = 1
    elif algorithm == "vertical":
        algorithm_int = 2
    elif algorithm == "horizontal":
        algorithm_int = 3
    elif algorithm == "horizontal_1stage":
        algorithm_int = 3
        force_num_stages = 1
    elif algorithm == "horizontal_4stage":
        algorithm_int = 3
        force_num_stages = 4
    elif algorithm == "horizontal_bankcycle":
        algorithm_int = 3
        force_permutation_type = 1
    elif algorithm == "horizontal_slot":
        algorithm_int = 3
        force_permutation_type = 2
    elif algorithm == "async_horizontal":
        # Backward compat: async_horizontal is now merged into simple
        algorithm_int = 1
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    _selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        dst_state_batch_indices,
        pad_slot_id,
        state_scale,
        output,
        disable_state_update,
        intermediate_states_buffer,
        intermediate_state_indices,
        intermediate_state_scales,
        rand_seed,
        cache_steps,
        cu_seqlens,
        num_accepted_tokens,
        algorithm_int,
        philox_rounds,
        state.dtype,
        x.dtype,
        dt.dtype,
        A.dtype,
        stateIndex_dtype,
        dim,
        dstate,
        ntokens_mtp,
        force_num_stages,
        force_permutation_type,
        0,
    )
    return output


@register_custom_op(
    "flashinfer::selective_state_update",
    mutates_args=(
        "state",
        "output",
        "intermediate_states_buffer",
        "state_scale",
        "intermediate_state_scales",
    ),
)
def _selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    state_scale: Optional[torch.Tensor],
    output: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    algorithm: int,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    force_num_stages: int = 0,
    force_permutation_type: int = 0,
    force_stage_cols: int = 0,
) -> None:
    """Internal function registered with torch.library for torch.compile() support."""
    major, _ = get_compute_capability(state.device)
    cu_seqlens_dtype = cu_seqlens.dtype if cu_seqlens is not None else torch.int32
    na_dtype = (
        num_accepted_tokens.dtype if num_accepted_tokens is not None else torch.int64
    )
    state_scale_dtype = state_scale.dtype if state_scale is not None else None

    module_base_args = (
        state_dtype, input_dtype, weight_dtype, matrixA_dtype,
        stateIndex_dtype, dim, dstate, ntokens_mtp,
        cu_seqlens_dtype, na_dtype, major,
        state_scale_dtype, philox_rounds,
    )

    if force_num_stages > 0 or force_permutation_type > 0 or force_stage_cols > 0:
        module = _get_module(
            *module_base_args, force_num_stages, force_permutation_type, force_stage_cols
        )
        module.selective_state_update(
            state, x, dt, A, B, C, D,
            z, dt_bias, dt_softplus,
            state_batch_indices, dst_state_batch_indices,
            pad_slot_id, state_scale, output,
            disable_state_update,
            intermediate_states_buffer, intermediate_state_indices,
            intermediate_state_scales, rand_seed,
            cache_steps, cu_seqlens, num_accepted_tokens,
            algorithm,
        )
        return

    tuner = AutoTuner.get()
    inputs = [
        state, x, dt, A, B, C, D,
        z, dt_bias, output,
        state_batch_indices, dst_state_batch_indices,
        state_scale,
        intermediate_states_buffer, intermediate_state_indices,
        intermediate_state_scales, rand_seed,
        cu_seqlens, num_accepted_tokens,
    ]

    runner = SSUHorizontalRunner(
        module_base_args=module_base_args,
        dt_softplus=dt_softplus,
        pad_slot_id=pad_slot_id,
        disable_state_update=disable_state_update,
        cache_steps=cache_steps,
        algorithm_int=algorithm,
    )

    _default_init = lambda shapes, dtype, device: (
        torch.rand(shapes, device=device) * 10 - 5
    ).to(dtype)

    def _init_dt(shapes, dtype, device):
        bs, H_dim, D_dim = shapes
        base = (torch.rand((bs, H_dim), device=device) * 10 - 5).to(dtype)
        return base.as_strided((bs, H_dim, D_dim), (H_dim, 1, 0))

    def _init_indices(shapes, dtype, device):
        return torch.arange(shapes[0], dtype=dtype, device=device)

    batch_input_idx = [1, 2, 4, 5]
    batch_dim_idx = [0, 0, 0, 0]
    tensor_inits = [_default_init, _init_dt, _default_init, _default_init]
    if z is not None:
        batch_input_idx.append(7)
        batch_dim_idx.append(0)
        tensor_inits.append(_default_init)
    if state_batch_indices is not None:
        batch_input_idx.append(10)
        batch_dim_idx.append(0)
        tensor_inits.append(_init_indices)
    if dst_state_batch_indices is not None:
        batch_input_idx.append(11)
        batch_dim_idx.append(0)
        tensor_inits.append(_init_indices)

    constraint_specs = [
        ConstraintSpec(9, 0, lambda shapes: shapes[1][0]),
    ]

    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=tuple(batch_input_idx),
                dim_idx=tuple(batch_dim_idx),
                gen_tuning_buckets=get_hybrid_num_tokens_buckets,
                map_to_tuning_buckets=map_to_hybrid_bucket_uncapped,
                tensor_initializers=tensor_inits,
            ),
        ),
        constraint_specs=tuple(constraint_specs),
    )

    runner, tactic = tuner.choose_one(
        "ssu_horizontal", [runner], tuning_config, inputs
    )
    runner(inputs=inputs, tactic=tactic)


@register_fake_op("flashinfer::selective_state_update")
def _selective_state_update_fake(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    state_scale: Optional[torch.Tensor],
    output: torch.Tensor,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    algorithm: int,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    ntokens_mtp: int,
    force_num_stages: int = 0,
    force_permutation_type: int = 0,
    force_stage_cols: int = 0,
) -> None:
    """Fake implementation for torch.compile() meta tensor propagation."""
    pass
