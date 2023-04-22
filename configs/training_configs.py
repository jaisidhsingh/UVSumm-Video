from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# training configs
cfg.batch_size = 1
cfg.learning_rate = 1e-4
cfg.epochs = 20
cfg.save_point = 5