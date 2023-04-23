from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# training configs
cfg.batch_size = 1
cfg.learning_rate = 1e-2
cfg.epochs = 100
cfg.save_point = 5