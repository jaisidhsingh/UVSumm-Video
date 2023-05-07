from types import SimpleNamespace


cfg = SimpleNamespace(**{})
"""
Scoring Transformer configs
"""
cfg.trans_cfg = SimpleNamespace(**{})
# CLIP embedding dimensions for our features
cfg.trans_cfg.embedding_dim = 512
# num_blocks count of encoder blocks and num_blocks count of decoder blocks
cfg.trans_cfg.num_blocks = 6 
# number of attention heads for the MHSA function
cfg.trans_cfg.num_heads = 16
# type of positional encoding, choices: [learnable, cosine]
cfg.trans_cfg.encoding_type = "cosine"
# length of the sequence of frames
cfg.trans_cfg.seq_len = 320

