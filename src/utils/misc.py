import torch
import os
from tqdm import tqdm
from .eval import *
import warnings
warnings.simplefilter('ignore')


def train_scoring_model(
		args, epoch,
		training_config, global_config,
		scoring_model, 
		train_loader, 
		optimizer, 
		criterion, 
		scheduler):
	scoring_model.train()
	running_loss = 0
	batch_counter = 0
	# start with batches
	for data in tqdm(train_loader):
		features = data['features'].float().squeeze(2).to(args.device)
		labels = data['labels'].float().to(args.device)
		# zero out the gradients accumulated
		optimizer.zero_grad()
		# forward pass
		scores = scoring_model(features)
		loss = criterion(scores, labels)
		running_loss += loss.item()
		# backward pass
		loss.backward()
		optimizer.step()
		scheduler.step()
		batch_counter += 1


	if (epoch + 1) % training_config.save_point == 0:
		ckpt_save_path = os.path.join(
			global_config.ckpt_dir, f"{args.dataset_name}_scoring_model_epoch_{epoch+1}.pt"
		)
		data2save = {
			'model': scoring_model.state_dict(),
			'optimizer': optimizer.state_dict()
		}
		torch.save(data2save, ckpt_save_path)
		print("Checkpoint saved")

	return running_loss

def evaluate(args, scoring_model, test_loader, criterion):
	scoring_model.eval()
	eval_arr = []
	data2return = {}

	running_loss = 0
	batch_counter = 0
	for data in tqdm(test_loader):
		features = data['features'].float().squeeze(2).to(args.device)
		labels = data['labels'].float().to(args.device)

		with torch.no_grad():
			scores = scoring_model(features)
			loss = criterion(scores, labels)
			running_loss += loss.item()
		
		pred_score = torch.softmax(scores, dim=1) # softmax across frames
		pred_score, pred_selected, pred_summary = select_keyshots(data, pred_score[0])
		true_summary_arr = data['user_summary'][0]
		pred_summary = torch.tensor(pred_summary)

		eval_res = [eval_metrics(pred_summary.numpy(), true_summary.numpy()) for true_summary in true_summary_arr]
		eval_res = np.mean(eval_res, axis=0).tolist()

		eval_arr.append(eval_res)

		video_name = data['video_name'][0]
		data2return[video_name] = {
			'evaluation': eval_res,
			'pred_score': pred_score,
			'pred_selected': pred_selected,
			'pred_summary': pred_summary,
		}
		batch_counter += 1

	data2return["evaluation_results"] = np.stack(eval_arr).mean(axis=0)
	return data2return, running_loss

def print2spaces():
	print(" ")
	print(" ")

def print2lines():
	print("-------------------------------------------------------------------------------------")
	print("-------------------------------------------------------------------------------------")