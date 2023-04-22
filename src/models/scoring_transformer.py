import torch
import torch.nn as nn
from configs.model_configs import cfg


class ScoringTransformer(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.embedding_dim = cfg.embedding_dim

		self.encoder_blocks1 = []
		self.encoder_blocks2 = []

		for i in range(cfg.num_blocks // 2):
			block = nn.TransformerEncoderLayer(
				d_model=cfg.embedding_dim,
				nhead=cfg.num_heads,
				batch_first=True
			)			
			self.encoder_blocks1.append(block)
		self.encoder_blocks1 = nn.Sequential(*self.encoder_blocks1)

		for i in range(cfg.num_blocks // 2):
			block = nn.TransformerEncoderLayer(
				d_model=cfg.embedding_dim,
				nhead=cfg.num_heads,
				batch_first=True
			)			
			self.encoder_blocks2.append(block)
		self.encoder_blocks2 = nn.Sequential(*self.encoder_blocks2)		
		self.activation = nn.Sigmoid()
		self.fc_out = nn.Linear(cfg.embedding_dim, 1)

	def forward(self, x):
		encoder_outputs = []
		for i in range(self.cfg.num_blocks // 2):
			x = self.encoder_blocks1[i](x)

			if i != (self.cfg.num_blocks // 2) - 1:
				encoder_outputs.append(x)
				
		encoder_outputs.reverse()

		for i in range(self.cfg.num_blocks // 2):
			if i == 0:
				x = self.encoder_blocks2[i](x)
			else:
				x += encoder_outputs[i-1]
				x = self.encoder_blocks2[i](x)
		
		scores = self.fc_out(x)
		return self.activation(scores).squeeze(2)



