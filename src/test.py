import torch
import torch.nn as nn
from configs.model_configs import cfg
from models.scoring_transformer import ScoringTransformer

def test():
	
	x = torch.randn((4, 320, 128)).cuda()
	model_config = cfg.trans_cfg
	model = ScoringTransformer(model_config)
	model.cuda()

	with torch.no_grad():
		scores, features = model(x)

	print(scores.shape, features.shape)



