from __future__ import annotations
@contextmanager
def prefill_query_chunk(self, begin: int, end: int, num_tokens: int, compress_ratio: int):
    """Read a query tile against the already populated full prefill caches."""
    original = self.forward_metadata
    assert original.actual_seq_lengths_kv.numel() == 1
    assert 0 <= begin < end <= num_tokens
    chunk = copy(original)
    cache = getattr(original, 'npu_prefill_chunk_metadata', None)
    if cache is None:
        cache = original.npu_prefill_chunk_metadata = {}
    key = (begin, end, num_tokens, compress_ratio)
    cached = cache.get(key)
    if cached is None:
        chunk.actual_seq_lengths_q_pa = original.actual_seq_lengths_q_pa.new_tensor([0, end - begin])
        chunk.actual_seq_lengths_kv = original.actual_seq_lengths_kv - (num_tokens - end)
    else:
        chunk.actual_seq_lengths_q_pa, chunk.actual_seq_lengths_kv, chunk.kernel_metadata = cached
    if getattr(original, 'c4_topk_indices', None) is not None:
        chunk.c4_topk_indices = original.c4_topk_indices[begin:end]
    self.forward_metadata = chunk
    try:
        if cached is None:
            chunk.kernel_metadata = self._kernel_metadata_from_parts(bs=1, actual_seq_lengths_q_pa=chunk.actual_seq_lengths_q_pa, actual_seq_lengths_kv=chunk.actual_seq_lengths_kv, attention_only_ratio=compress_ratio)
            cache[key] = (chunk.actual_seq_lengths_q_pa, chunk.actual_seq_lengths_kv, chunk.kernel_metadata)
        yield
    finally:
        self.forward_metadata = original

def _kernel_metadata_from_parts(self, *, bs: int, actual_seq_lengths_q_pa: torch.Tensor, actual_seq_lengths_kv: torch.Tensor, attention_only_ratio: Optional[int]=None) -> dict:
    fm = self.forward_metadata
    common = {**_sparse_attn_kv_quant_kwargs(), 'cu_seqlens_q': actual_seq_lengths_q_pa, 'seqused_kv': actual_seq_lengths_kv, 'cmp_ratio': 1, 'ori_mask_mode': 4, 'cmp_mask_mode': 3, 'ori_win_left': getattr(fm, 'ori_win_left', self._dsv4_sliding_window_size - 1), 'ori_win_right': getattr(fm, 'ori_win_right', 0), 'layout_q': 'TND', 'layout_kv': 'PA_ND'}
    base_kwargs = {'batch_size': bs, 'num_heads_q': self._dsv4_q_head_num, 'num_heads_kv': self._dsv4_kv_head_num, 'head_dim': self._dsv4_head_dim, 'has_ori_kv': True, 'has_cmp_kv': False}
    c1a_kwargs = base_kwargs | common
    if self._is_dspark_draft_worker:
        cu_q_cpu = fm.actual_seq_lengths_q_pa_cpu
        seq_kv_cpu = fm.seq_lens_cpu_int
        if cu_q_cpu is None or seq_kv_cpu is None or cu_q_cpu.device.type != 'cpu' or (seq_kv_cpu.device.type != 'cpu'):
            raise RuntimeError('DSpark host metadata requires CPU query and KV length mirrors')
        if cu_q_cpu.numel() > bs + 1:
            cu_q_cpu = cu_q_cpu[:bs + 1]
        host_inputs = {'seqused_kv': seq_kv_cpu[:bs].to(dtype=torch.int32).contiguous()}
        host_inputs['cu_seqlens_q'] = cu_q_cpu.to(dtype=torch.int32).contiguous()
        metadata_op, _ = _sparse_attn_ops(is_dspark=True)
        c1a_metadata = metadata_op(**c1a_kwargs | host_inputs)
    else:
        metadata_op, _ = _sparse_attn_ops()
        c1a_metadata = metadata_op(**c1a_kwargs) if attention_only_ratio in (None, 0) else None
    kernel_metadata = {'c1a_metadata': c1a_metadata}
    if self._dsv4_has_c4 and attention_only_ratio in (None, 4):
        c4a_overrides = {'cmp_ratio': 4, 'has_cmp_kv': True, 'cmp_topk': self._dsv4_index_topk}
        c4a_kwargs = c1a_kwargs | c4a_overrides
        metadata_op, _ = _sparse_attn_ops()
        kernel_metadata['c4a_metadata'] = metadata_op(**c4a_kwargs)
        if attention_only_ratio == 4:
            return kernel_metadata
        if actual_seq_lengths_q_pa is not None:
            actual_q = actual_seq_lengths_q_pa[1:].clone()
        else:
            actual_q = actual_seq_lengths_kv
        kernel_metadata['li_quant_metadata'] = torch.ops.custom.npu_quant_lightning_indexer_metadata(device=str(actual_q.device), actual_seq_lengths_query=actual_q, actual_seq_lengths_key=actual_seq_lengths_kv, layout_key='PA_BSND', sparse_count=self._dsv4_index_topk, sparse_mode=3, layout_query='TND', cmp_ratio=4, key_quant_mode=0, query_quant_mode=0, num_heads_q=self._dsv4_index_n_heads, num_heads_k=1, head_dim=self._dsv4_index_head_dim)
    if self._dsv4_has_c128 and attention_only_ratio in (None, 128):
        c128a_overrides = {'cmp_ratio': 128, 'has_cmp_kv': True}
        c128a_kwargs = c1a_kwargs | c128a_overrides
        metadata_op, _ = _sparse_attn_ops()
        kernel_metadata['c128a_metadata'] = metadata_op(**c128a_kwargs)
    return kernel_metadata

def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: RadixAttention, forward_batch: ForwardBatch, *, compress_ratio: int=0, attn_sink: Optional[torch.Tensor]=None, save_kv_cache: bool=True) -> torch.Tensor:
    if compress_ratio not in (0, 4, 128):
        raise ValueError(f'V4 attention expects compress_ratio in (0, 4, 128); got {compress_ratio}')
    if forward_batch.forward_mode.is_idle():
        return torch.zeros_like(q)
    if save_kv_cache:
        self.store_cache(layer_id=layer.layer_id, swa_k=k, forward_batch=forward_batch)
    if compress_ratio == 0:
        return self._forward_swa(q, layer, forward_batch, attn_sink)
    return self._forward_compressed(q, layer, forward_batch, attn_sink, compress_ratio)

def _forward_swa(self, q: torch.Tensor, layer: RadixAttention, forward_batch: ForwardBatch, attn_sink: Optional[torch.Tensor]) -> torch.Tensor:
    fm = self.forward_metadata
    pool = self.token_to_kv_pool
    ori_kv = pool.get_swa_buffer(layer.layer_id)
    attn_kwargs = dict(**_sparse_attn_kv_quant_kwargs(), cu_seqlens_q=fm.actual_seq_lengths_q_pa, seqused_kv=fm.actual_seq_lengths_kv, ori_mask_mode=4, ori_win_left=getattr(fm, 'ori_win_left', self._dsv4_sliding_window_size - 1), ori_win_right=getattr(fm, 'ori_win_right', 0), layout_q='TND', layout_kv='PA_ND', q=q, ori_kv=ori_kv, ori_block_table=fm.swa_page_table, sinks=attn_sink, metadata=fm.kernel_metadata['c1a_metadata'], softmax_scale=layer.scaling, cmp_ratio=1)
    ori_sparse_indices = getattr(fm, 'ori_sparse_indices', None)
    if ori_sparse_indices is not None:
        attn_kwargs['ori_sparse_indices'] = ori_sparse_indices
    q_arg = attn_kwargs.pop('q')
    _, attn_op = _sparse_attn_ops(self._is_dspark_draft_worker)
    out, _ = attn_op(q_arg, **attn_kwargs)
    return out

def _forward_compressed(self, q: torch.Tensor, layer: RadixAttention, forward_batch: ForwardBatch, attn_sink: Optional[torch.Tensor], compress_ratio: int) -> torch.Tensor:
    fm = self.forward_metadata
    pool = self.token_to_kv_pool
    metadata = fm.kernel_metadata.get(f'c{compress_ratio}a_metadata')
    cmp_kv = pool.get_compress_buffer(layer.layer_id, False)
    if metadata is None or cmp_kv is None:
        raise RuntimeError(f"DeepseekV4AscendAttnBackend._forward_compressed: missing required state for layer_id={layer.layer_id} compress_ratio={compress_ratio}. metadata({('present' if metadata is not None else 'MISSING')}), cmp_kv({('present' if cmp_kv is not None else 'MISSING')}). Available kernel_metadata keys: {list(fm.kernel_metadata.keys())}. This indicates a configuration / pool-init bug — silently returning zeros would corrupt model output.")
    ori_kv = pool.get_swa_buffer(layer.layer_id)
    ori_page_size = ori_kv.shape[1]
    cmp_native_page_size = cmp_kv.shape[1]
    cmp_block_table = getattr(fm, f'c{compress_ratio}_page_table')
    expected_cmp_page_size = ori_page_size // 4 if compress_ratio == 4 else pool.c128_kv_pool.kernel_page_size
    assert cmp_native_page_size == expected_cmp_page_size, f'c{compress_ratio} page_size={cmp_native_page_size} != expected={expected_cmp_page_size} for ori page_size={ori_page_size}; C4 and C128 must use their configured native page layouts (see NPUDeepSeekV4SingleKVPool.kernel_page_size)'
    attn_kwargs = dict(**_sparse_attn_kv_quant_kwargs(), cu_seqlens_q=fm.actual_seq_lengths_q_pa, seqused_kv=fm.actual_seq_lengths_kv, ori_mask_mode=4, ori_win_left=self._dsv4_sliding_window_size - 1, ori_win_right=0, layout_q='TND', layout_kv='PA_ND', q=q, ori_kv=ori_kv, ori_block_table=fm.swa_page_table, sinks=attn_sink, metadata=metadata, softmax_scale=layer.scaling, cmp_ratio=compress_ratio, cmp_mask_mode=3, cmp_kv=cmp_kv, cmp_block_table=cmp_block_table)
    if compress_ratio == 4:
        topk = fm.c4_topk_indices
        attn_kwargs['cmp_sparse_indices'] = topk.view(-1, 1, topk.shape[-1])
    else:
        attn_kwargs['cmp_sparse_indices'] = None
    q_arg = attn_kwargs.pop('q')
    _, attn_op = _sparse_attn_ops()
    out, _ = attn_op(q_arg, **attn_kwargs)
    return out