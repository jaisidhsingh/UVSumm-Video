from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# training configs
cfg.batch_size = 1
cfg.learning_rate = 1e-4
cfg.epochs = 150
cfg.save_point = 50

# loss weights
cfg.cls_weight = 0.5
cfg.recon_weight = 0.3
