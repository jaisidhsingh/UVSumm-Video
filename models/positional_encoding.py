import torch
import torch.nn as nn
import math


class LearnablePositionalEncoding(nn.Module):
	def __init__(self, dims, seq_len):
		super().__init__()
		self.dims = dims
		self.seq_len = seq_len
		self.positional_encoding = nn.Parameter(
				torch.randn((self.seq_len, self.dims))
		)

	def forward(self, x):
		x = x + self.positional_encoding.to(x.device)
		return x


class CosinePositionalEncoding(nn.Module):
	def __init__(self, dims, seq_len):
		super().__init__()
		self.seq_len = seq_len
		self.dims = dims

		pe = torch.zeros(self.seq_len, self.dims)
		position = torch.arange(0, self.seq_len).unsqueeze(1)
		div_term = torch.exp((torch.arange(0, self.dims, 2, dtype=torch.float) *
							-(math.log(10000.0) / self.dims)))
		pe[:, 0::2] = torch.sin(position.float() * div_term)
		pe[:, 1::2] = torch.cos(position.float() * div_term)
		self.pe = pe

	def forward(self, x):
		x = x + self.pe.to(x.device)
		return x