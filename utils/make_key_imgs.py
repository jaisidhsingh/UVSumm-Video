from configs.global_configs import cfg as global_config
from configs.data_configs import cfg as data_config

from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
import argparse
import warnings
warnings.simplefilter("ignore")

def make_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--run-id",
		type=str,
		required=True
	)
	parser.add_argument(
		"--dataset-name",
		type=str,
		required=True
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		required=True
	)
	args = parser.parse_args()
	return args


def save_selected_frames(video_path, selected_array, output_dir):
	video = cv2.VideoCapture(video_path.as_uri())

	i = 0
	success, frame = video.read()
	while success:
		if i in selected_array:
			img = Image.fromarray(frame)
			img.save(os.path.join(
				output_dir, f"frame_{i}.png"
			))
		i += 1
		if i > max(selected_array):
			break
		else:
			success, frame = video.read()
	
	tqdm.write(f"Saved {len(output_dir)} keyframes into {output_dir}")
	tqdm.write(" ")

def main(args):
	run_id = args.run_id
	dataset_name = args.dataset_name

	eval_results = torch.load(
		os.path.join(global_config.ckpt_dir, f"eval_results_id_{run_id}.pt")
	)[-1]
	video_dir = data_config.video_dirs[dataset_name]	

	for k in tqdm(list(eval_results.keys())[:-1]):
		selected = eval_results[k]['pred_selected']
		video_name = k[6:] + ".mp4"
		video_path = os.path.join(video_dir, video_name)
		video_path = Path(video_path).resolve()

		output_dir = os.path.join(args.output_dir, video_name[:-4])
		os.makedirs(output_dir, exist_ok=True)
		
		save_selected_frames(video_path, selected, output_dir)


args = make_args()
main(args)

