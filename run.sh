EPOCHS=200
BATCH_SIZE=1
LEARNING_RATE=0.0001
DATASET_NAME="summe"
DEVICE="cuda"


python3 train.py \
	--dataset-name=$DATASET_NAME \
	--epochs=$EPOCHS \
	--batch-size=$BATCH_SIZE \
	--learning-rate=$LEARNING_RATE \
	--device=$DEVICE



