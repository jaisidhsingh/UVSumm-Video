from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# model configs

# scoring transformer configs
cfg.trans_cfg = SimpleNamespace(**{})
cfg.trans_cfg.embedding_dim = 512
cfg.trans_cfg.num_blocks = 6
cfg.trans_cfg.num_heads = 8

