from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# global configs
cfg.device = "cuda"
cfg.ckpt_dir = "../checkpoints"
cfg.plot_dir = "../plots"
cfg.dataset_dir = "../datasets"
cfg.runs_dir = "../runs"
cfg.runs_tracker = "../runs/tracker.json"
cfg.runs_stats_save_dir = "../runs/runs_stats_saves"