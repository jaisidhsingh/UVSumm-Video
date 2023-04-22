import torch
import torch.nn as nn
import clip
from copy import deepcopy

class FeatureExtractor():
	def __init__(self):
		model, preprocess = clip.load("ViT-B/32", device="cuda")
		self.vision_model = deepcopy(model)
		self.vision_model.cuda()
		self.transforms = preprocess

		for param in self.vision_model.parameters():
			param.requires_grad = False
		
		self.vision_model.eval()

	def extract_features(self, image):
		with torch.no_grad():
			img_tensor = self.transforms(image).unsqueeze(0).cuda()
			features = self.vision_model.encode_image(img_tensor)
		return features