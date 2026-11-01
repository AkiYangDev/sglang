def initialize(self, config, rope_original_seq_len=None):
    rope_theta, rope_scaling = get_rope_config(config)
    self.rope_scaling = dict(rope_scaling) if rope_scaling else None
    scaling = self.rope_scaling or {}
    self.rope_base = config.compress_rope_theta if self.compress_ratio else rope_theta
    original_seq_len: int = rope_original_seq_len if rope_original_seq_len is not None else scaling['original_max_position_embeddings'] if self.compress_ratio else 0
    self.rope_kwargs = dict(dim=self.qk_rope_head_dim, seqlen=get_npu_rope_cache_length(config) if _is_npu else config.max_position_embeddings, original_seq_len=original_seq_len, base=self.rope_base, factor=scaling.get('factor', 1.0), beta_fast=scaling.get('beta_fast', 32), beta_slow=scaling.get('beta_slow', 1))
    freqs_cis = precompute_freqs_cis(**self.rope_kwargs)
    self.register_buffer('freqs_cis', freqs_cis, persistent=False)
    self.freqs_cis: torch.Tensor