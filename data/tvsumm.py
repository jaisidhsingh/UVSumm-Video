import torch
from torch.utils.data import Dataset


class TvSummDataset(Dataset):
	def __init__(self, dataset_file, split):
		self.data = torch.load(dataset_file)
		self.video_features = []
		self.labels = []
		self.lengths = []
		self.change_points = []
		self.n_frame_per_seg = []
		self.picks = []
		self.user_summary = []
		self.video_names = []
		
		for k, v in self.data.items():
			self.video_names.append(str(k))
			self.video_features.append(v['feature'])
			self.labels.append(v['label'])
			self.lengths.append(v['length'])
			self.change_points.append(v['change_points'])
			self.n_frame_per_seg.append(v['n_frame_per_seg'])
			self.picks.append(v['picks'])
			self.user_summary.append(v['user_summary'])

		self.split = split
		if split == 'train':
			self.split_slice = int(0.8*len(self.video_names))
		else:
			self.split_slice = int(-1*(0.2 * len(self.video_names)))
			
	def __len__(self):
		if self.split == 'train':
			return len(self.video_names[:self.split_slice])
		if self.split == 'test':
			return len(self.video_names[self.split_slice:])

	def __getitem__(self, idx):
		video_name = self.video_names[idx]
		features = self.video_features[idx]
		labels = torch.tensor(self.labels[idx])
		length = int(self.lengths[idx])
		change_points = self.change_points[idx]
		nfps = self.n_frame_per_seg[idx]
		picks = self.picks[idx]
		user_summary = self.user_summary[idx]

		data2return = {
			"video_name": video_name,
			"features": features,
			"labels": labels,
			"length": length,
			"change_points": change_points,
			"nfps": nfps,
			"picks": picks,
			"user_summary": user_summary
		}
		return data2return

