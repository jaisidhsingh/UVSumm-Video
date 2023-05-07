import os
import torch
import torch.nn as nn
from configs.model_configs import cfg
from models.scoring_transformer import ScoringTransformer
import mat73
import scipy.io


def test():
	
	x = torch.randn((1, 320, 512)).cuda()
	model_config = cfg.trans_cfg
	model = ScoringTransformer(model_config)
	model.cuda()

	with torch.no_grad():
		scores, features = model(x)

	print(scores.shape, features.shape)

def test_summe():
	p = "../datasets/summe/GT/"
	for f in os.listdir(p):
		d = scipy.io.loadmat(os.path.join(p, f))
		print(d['nFrames'].item())

def test_ed():
	f = torch.randn((1, 320, 512))

	l1 = torch.ones((1, 160, 512))
	l2 = torch.zeros((1, 160, 512))
	m = torch.cat([l1, l2], dim=1)
	t = f * m
	
	encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
	decoder = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
	x = encoder(f)
	y = decoder(t, x)
	
	print(y.shape)

def test2():
	x = torch.randn((1, 320, 512)).cuda()
	model = ScoringTransformer(cfg.trans_cfg)
	model.cuda()
	y = model(x)
	print(y.shape)
