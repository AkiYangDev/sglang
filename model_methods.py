from __future__ import annotations
def _forward_prepare(self, x: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch, attn_backend, q_out: Optional[torch.Tensor]=None, x_quant=None, q_rope_out: Optional[torch.Tensor]=None, k_nope_out: Optional[torch.Tensor]=None, k_rope_out: Optional[torch.Tensor]=None, defer_npu_q_projection: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    x_linear = x_quant if x_quant is not None else x
    if self.fuse_wqa_wkv:
        qkv_a, _ = self.wqkv_a(x_linear)
        q_lora = qkv_a[..., :self.q_lora_rank]
    else:
        q_lora, _ = self.wq_a(x_linear)
        qkv_a = None
    use_cp = is_cp_active(forward_batch)
    kv: Optional[torch.Tensor]
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_fp8, is_unified_kv_triton
    unified = is_unified_kv_triton()
    fp8_2buff = is_unified_kv_fp8()
    is_decode = forward_batch.forward_mode.is_decode_or_idle()
    fuse_verify = envs.SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY.get() and forward_batch.forward_mode.is_target_verify()
    fuse_verify_fp8 = fuse_verify and unified and fp8_2buff and (k_nope_out is not None) and (k_rope_out is not None)
    fuse_prefill = unified and fp8_2buff and (k_nope_out is not None) and (k_rope_out is not None) and (not is_decode) and (not forward_batch.forward_mode.is_target_verify())
    do_fused_qk_norm_rope = unified and (is_decode or fuse_verify or fuse_prefill) or (not unified and self.use_fused_qk_norm_rope)
    q_lora_quant = None
    if do_fused_qk_norm_rope:
        if _is_gfx95_supported or _is_gfx1250_supported:
            q_for_wqb, q_lora = _fused_rmsnorm_fp8_quant(q_lora, self.q_norm.weight, self.q_norm.variance_epsilon, e8m0=self._q_lora_quant_e8m0(with_indexer=True))
            q, _ = self.wq_b(q_for_wqb)
            q_lora_quant = q_for_wqb
        else:
            q_lora, q_for_wqb = self._normalize_q_lora(q_lora)
            q, _ = self.wq_b(q_for_wqb)
        kv = qkv_a[..., self.q_lora_rank:] if qkv_a is not None else self.wkv(x_linear)[0]
        token_to_kv_pool = get_token_to_kv_pool()
        swa_rope_cache = None
        if unified and fuse_verify and (not fuse_verify_fp8):
            kv = kv.contiguous()
            swa_cache, swa_loc = (None, None)
            swa_page_size, bf16_store = (1, True)
        elif unified and fuse_prefill:
            swa_cache, swa_loc = (None, None)
            swa_page_size, bf16_store = (1, False)
        elif unified:
            swa_cache = token_to_kv_pool.get_unified_kv(self.layer_id)
            swa_loc = attn_backend.get_unified_swa_loc(forward_batch)
            swa_page_size, bf16_store = (1, not fp8_2buff)
            if fp8_2buff:
                swa_rope_cache = token_to_kv_pool.get_unified_kv_rope(self.layer_id)
        else:
            swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
            swa_cache = token_to_kv_pool.get_swa_raw_buffer(self.layer_id)
            swa_page_size, bf16_store = (token_to_kv_pool.swa_page_size, False)
            if use_cp:
                swa_cache, swa_loc = (None, None)
        from sglang.kernels.ops.attention.fused_qk_norm_rope_store import fused_qk_norm_rope_swa_store
        q = fused_qk_norm_rope_swa_store(q=q, kv=kv, q_norm_weight=None, kv_norm_weight=self.kv_norm.weight, q_rms_eps=self.eps, kv_rms_eps=self.eps, rope_head_dim=self.qk_rope_head_dim, cos_cache=self.cos_cache, sin_cache=self.sin_cache, positions=positions, swa_cache=swa_cache, swa_loc=swa_loc, swa_page_size=swa_page_size, q_out=q_out, dtype=x.dtype, bf16_store=bf16_store, fp8_2buff=fp8_2buff, swa_rope_cache=swa_rope_cache, k_nope_out=k_nope_out if fuse_prefill else None, k_rope_out=k_rope_out if fuse_prefill else None, q_rope_out=q_rope_out)
        if unified and fuse_prefill:
            kv = k_nope_out
        elif not unified and use_cp:
            kv = cp_materialize_global_token_order(kv.contiguous(), forward_batch, torch.cuda.current_stream())
            attn_backend.store_cache(layer_id=self.layer_id, swa_k=kv, forward_batch=forward_batch)
        elif not (unified and fuse_verify and (not fuse_verify_fp8)):
            kv = None
    elif _is_npu:
        q_lora = self.q_norm(q_lora)
        if defer_npu_q_projection:
            q = q_lora
        else:
            q, _ = self.wq_b(q_lora)
            q = q.view(-1, self.n_local_heads, self.head_dim)
            q = torch_npu.npu_rms_norm(q, self.q_rms_norm_ones, self.eps)[0]
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank:]
        else:
            kv, _ = self.wkv(x)
        kv = self.kv_norm(kv)
        cos4, sin4 = self._get_npu_rope_position_cache(forward_batch, positions, q.dtype, inverse=False)
        Dsv4NpuRoPE.apply_rotary_mul_inplace(kv.unsqueeze(1) if defer_npu_q_projection else q, None if defer_npu_q_projection else kv.unsqueeze(1), cos4, sin4, qk_nope_dim=self.qk_nope_head_dim)
        kv_for_cache = kv
        if use_cp:
            kv_for_cache = cp_gather_full_sequence_states(kv.contiguous(), forward_batch, torch.cuda.current_stream())
        attn_backend.store_cache(layer_id=self.layer_id, swa_k=kv_for_cache, forward_batch=forward_batch)
        kv = None
        if q_out is not None:
            q_out.copy_(q)
    else:
        q_lora, q_for_wqb = self._normalize_q_lora(q_lora)
        if _is_hip:
            q, q_lora, fuse_q_rope = _hip.compute_q_b(self, q_lora, q_for_wqb, positions, q_out, unified, use_cp)
        else:
            q = self._compute_q_b(q_for_wqb, positions, q_out)
            fuse_q_rope = False
        if unified:
            kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)
            if use_cp:
                kv = cp_materialize_global_token_order(kv.contiguous(), forward_batch, torch.cuda.current_stream())
        elif use_cp and (not self.is_dsv41):
            kv = self._compute_kv_bf16(x_linear, positions, qkv_a=qkv_a)
            kv = cp_materialize_global_token_order(kv.contiguous(), forward_batch, torch.cuda.current_stream())
            attn_backend.store_cache(layer_id=self.layer_id, swa_k=kv, forward_batch=forward_batch)
        elif use_cp:
            if qkv_a is not None:
                kv = qkv_a[..., self.q_lora_rank:]
            else:
                kv, _ = self.wkv(x_linear)
            kv = cp_materialize_global_token_order(kv.contiguous(), forward_batch, torch.cuda.current_stream())
            tail = attn_backend.forward_metadata.late_layer_tail
            global_positions = tail.pos_global if tail is not None else forward_batch.positions[:kv.shape[0]]
            get_token_to_kv_pool().set_swa_key_buffer_radix_fused_norm_rope(layer_id=self.layer_id, swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch), kv=kv, kv_weight=self.kv_norm.weight.data, eps=self.eps, freqs_cis=self.freqs_cis, positions=global_positions)
            kv = None
        elif in_replay_tail_graph():
            assert not fuse_q_rope
            bcg_tail_kv_store(self, x_linear, positions, qkv_a)
            kv = None
        else:
            self._compute_kv_to_cache(x_linear, positions, forward_batch, attn_backend, qkv_a=qkv_a, q_rope=q if fuse_q_rope else None)
            kv = None
    del qkv_a
    if self.compress_ratio in (1, 2) and (self.compressor is not None or self.indexer is not None):
        if _low_ratio_sources_break(attn_backend, forward_batch):
            bcg_deepseek_v4_low_ratio_sources(self, x, q_lora, positions)
        else:
            attn_backend.forward_low_ratio_sources(layer=self, x=x, q_lora=q_lora, positions=positions, forward_batch=forward_batch)
    else:
        use_npu_cp_full_metadata = use_cp and _is_npu
        if self.indexer is not None:
            if use_npu_cp_full_metadata:
                with attn_backend.use_dsv4_cp_full_metadata(forward_batch):
                    attn_backend.forward_indexer_compressor(x, forward_batch, self.indexer.layer_id, self.indexer.compressor)
                self.indexer(x=x, q_lora=q_lora, forward_batch=forward_batch, attn_backend=attn_backend, skip_compressor=True)
            else:
                self.indexer(x=x, q_lora=q_lora, forward_batch=forward_batch, attn_backend=attn_backend, q_lora_quant=q_lora_quant)
        if self.compressor is not None:
            if use_npu_cp_full_metadata:
                with attn_backend.use_dsv4_cp_full_metadata(forward_batch):
                    attn_backend.forward_core_compressor(x, forward_batch, self.layer_id, self.compressor)
            else:
                attn_backend.forward_core_compressor(x, forward_batch, self.layer_id, self.compressor)
    return (q, kv)

def _forward_npu_chunked_prefill(self, x, positions, forward_batch, attn_backend, x_quant, chunk_size, defer_all_reduce):
    q_lora, _ = self._forward_prepare(x, positions, forward_batch, attn_backend, x_quant=x_quant, defer_npu_q_projection=True)
    cos, sin = self._get_npu_rope_position_cache(forward_batch, positions, q_lora.dtype, inverse=False)
    inv_cos, inv_sin = self._get_npu_rope_position_cache(forward_batch, positions, q_lora.dtype, inverse=True)
    output = x.new_empty(x.shape[0], self.n_local_groups, self.o_lora_rank)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    sink = self._local_attn_sink(self.n_local_heads)
    for begin in range(0, x.shape[0], chunk_size):
        end = min(begin + chunk_size, x.shape[0])
        q, _ = self.wq_b(q_lora[begin:end])
        q = q.view(-1, self.n_local_heads, self.head_dim)
        q = torch_npu.npu_rms_norm(q, self.q_rms_norm_ones, self.eps)[0]
        Dsv4NpuRoPE.apply_rotary_mul_inplace(q, None, cos[begin:end], sin[begin:end], qk_nope_dim=self.qk_nope_head_dim)
        with attn_backend.prefill_query_chunk(begin, end, x.shape[0], self.compress_ratio):
            o = attn_backend.forward(q=q, k=q, v=q, layer=self.attn_mqa, forward_batch=forward_batch, compress_ratio=self.compress_ratio, attn_sink=sink, save_kv_cache=False)
        del q
        Dsv4NpuRoPE.apply_rotary_mul_inplace(o, None, inv_cos[begin:end], inv_sin[begin:end], qk_nope_dim=self.qk_nope_head_dim)
        projected = _apply_wo_a_bf16_matmul(o.view(end - begin, self.n_local_groups, -1), wo_a, is_decode=False, is_prefill=True)
        output[begin:end].copy_(projected)
        del o, projected
    return self._project_wo_b(output, defer_all_reduce)

def _apply_wo_a_bf16_matmul(o: torch.Tensor, wo_a: torch.Tensor, is_decode: bool, is_target_verify: bool=False, fuse_mxfp8_quant: bool=False, is_prefill: bool=False, fast_path: bool=False, fp8_grid: bool=False, emit_fp8: bool=False) -> torch.Tensor | Mxfp8SwizzledInput | Fp8GridActivation | Mxfp8Activation:
    global _wo_a_aiter_batched_gemm_disabled
    hip_fast_path = fast_path and _is_hip and _is_gfx95_supported
    hip_decode_verify = hip_fast_path and (is_decode and o.shape[0] == 1 and o.is_contiguous() or (is_target_verify and 2 <= o.shape[0] <= 384)) and _hip.wo_a_split_k_allowed()
    if (fast_path and _is_cuda and (is_decode and o.shape[0] == 1 and (get_platform().is_blackwell or get_platform().is_sm90) or (is_target_verify and 0 < o.shape[0] <= 384 and get_platform().is_blackwell) or (is_prefill and 4096 <= o.shape[0] <= 65536 and get_platform().is_blackwell)) or hip_decode_verify or (hip_fast_path and is_prefill and (4096 <= o.shape[0] <= 65536))) and (o.shape[1:] == (2, 4096) and wo_a.shape == (2, 1024, 4096) and (o.dtype == wo_a.dtype == torch.bfloat16) and (o.stride(2) == 1) and (o.stride(1) == 4096) and (o.stride(0) >= 8192) and wo_a.is_contiguous()):
        if is_decode and o.shape[0] == 1:
            return wo_a_bf16_gemv(o, wo_a)
        if 2 <= o.shape[0] <= 8:
            if fuse_mxfp8_quant:
                return Mxfp8SwizzledInput(*wo_a_bf16_small_batch_mxfp8(o, wo_a))
            return wo_a_bf16_small_batch(o, wo_a)
        return _wo_a_bf16_bmm_token_major(o, wo_a)
    if fast_path and _is_cuda and is_target_verify and (o.shape[0] > 0) and get_platform().is_blackwell and (o.dtype == wo_a.dtype == torch.bfloat16) and (o.stride(2) == 1) and wo_a.is_contiguous():
        return _wo_a_bf16_bmm_token_major(o, wo_a)
    if (is_decode or is_target_verify) and hip_fast_path:
        y = _hip.wo_a_fp8_grid_matmul(o, wo_a, fp8_grid, emit_fp8=emit_fp8)
        if y is not None:
            return y
    if is_decode and _wo_a_aiter_batched_gemm_enabled and (not _wo_a_aiter_batched_gemm_disabled):
        try:
            xq = o.transpose(0, 1).contiguous()
            y = _wo_a_batched_gemm_bf16(xq, wo_a, dtype=torch.bfloat16)
            return y.transpose(0, 1).contiguous()
        except Exception as err:
            _wo_a_aiter_batched_gemm_disabled = True
            logger.warning('aiter wo_a batched_gemm_bf16 failed; disabling the reroute and falling back to einsum for the rest of this process: %s', err)
    return torch.einsum('tgd,grd->tgr', o, wo_a)

def _project_wo_b(self, o, defer_all_reduce: bool=False):
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    defer_all_reduce = defer_all_reduce and (self.attn_tp_size > 1 and (_is_cuda or (self.attn_tp_size == get_parallel().tp_size and self.wo_b.reduce_results and (not self.wo_b.use_decode_attn_tp))) and (not should_skip_mlp_all_reduce()))
    o, _ = self.wo_b(o.flatten(1) if isinstance(o, torch.Tensor) else o, skip_all_reduce=defer_all_reduce)
    if defer_all_reduce:
        return mhc.AttnOutput(o)
    if self.attn_tp_size > 1 and self.attn_tp_size < get_parallel().tp_size:
        o = attn_tp_all_reduce(o)
    return o