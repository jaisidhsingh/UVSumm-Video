from types import SimpleNamespace

cfg = SimpleNamespace(**{})
cfg.loading = {
    "tvsumm": {
    	"data_file": "../datasets/tvsumm/test_preprocessing.pt"
    },
    "summe": {
        "data_file": "../datasets/summe/test_preprocessing.pt"
    }
}
