from configs.global_configs import cfg as global_config
from configs.model_configs import cfg as model_config
from configs.training_configs import cfg as training_config
from configs.data_configs import cfg as data_config

from models.scoring_transformer import ScoringTransformer

from data import *

from utils.schedulers import *
from utils.misc import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import argparse
import random
import string
import json
import warnings
warnings.simplefilter('ignore')


# setup command line arguments
def make_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset-name",
		type=str,
		default="tvsumm"
	)
	parser.add_argument(
		"--device",
		type=str,
		default=global_config.device
	)
	args = parser.parse_args()
	return args

def train(args):
	# initialize unique run-id to this run
	run_id = ''.join(random.choices(string.ascii_letters, k=7))
	with open(global_config.runs_tracker) as f:
		runs_data = json.load(f)
	my_run_data = {'run_id': run_id}
	
	# load in the dataset to train
	train_dataset = VideoSummarizationDatasets[args.dataset_name](
		dataset_file=data_config.loading[args.dataset_name]['data_file'], 
		split='train'
	)
	train_loader = DataLoader(
		train_dataset, 
		batch_size=training_config.batch_size, 
		shuffle=True,
	)

	test_dataset = VideoSummarizationDatasets[args.dataset_name](
		dataset_file=data_config.loading[args.dataset_name]['data_file'], 
		split='test'
	)
	test_loader = DataLoader(
		test_dataset, 
		batch_size=training_config.batch_size, 
		shuffle=True,
	)
	
	# load in the model to score the frames
	scoring_model = ScoringTransformer(model_config.trans_cfg)
	scoring_model.to(args.device)

	criterion = nn.BCELoss()
	optimizer = optim.AdamW(scoring_model.parameters(), lr=training_config.learning_rate)
	scheduler = WarmupCosineSchedule(optimizer, warmup_steps=20, t_total=40)

	stats = {
		'precision': [],
		'recall': [],
		'f1_score': [],
		'test_loss': [],
		'train_loss': []
	}
	# start training loop
	for epoch in range(training_config.epochs):
		train_loss = train_scoring_model(
			args, epoch,
			training_config, global_config,
			scoring_model, 
			train_loader, 
			optimizer, 
			criterion, 
			scheduler
		)
		eval_results, test_loss = evaluate(
			args,
			scoring_model, 
			test_loader, 
			criterion, 
		)	

		test_precision = round(eval_results['evaluation_results'][0], 4)
		test_recall = round(eval_results['evaluation_results'][1], 4)
		test_f1_score = round(eval_results['evaluation_results'][2], 4)

		print(f"Epoch: {epoch+1}, Training Loss: {train_loss}")
		print(f"---------- Test Loss: {test_loss}")
		print(f"---------- Test Precision: {test_precision}")
		print(f"---------- Test Recall: {test_recall}")
		print(f"---------- Test F1 Score: {test_f1_score}")
		print2spaces()

		stats['precision'].append(test_precision)
		stats['recall'].append(test_recall)
		stats['f1_score'].append(test_f1_score)
		stats['test_loss'].append(test_loss)
		stats['train_loss'].append(train_loss)

	run_stats_save_path = f"stats_id_{run_id}.pt"
	run_stats_save_path = os.path.join(global_config.runs_stats_save_dir, run_stats_save_path)
	torch.save(stats, run_stats_save_path)

	model_save_path = f"trained_scoring_model_id_{run_id}.pt"
	model_save_path = os.path.join(global_config.ckpt_dir, model_save_path)
	torch.save(scoring_model.state_dict(), model_save_path)

	my_run_data['run_stats_save_path'] = run_stats_save_path
	my_run_data['run_model_save_path'] = model_save_path

	runs_data.append(my_run_data)
	with open(global_config.runs_tracker, "w") as f:
		json.dump(runs_data, f)


# run the training process
if __name__ == "__main__":
	args = make_args()
	train(args)