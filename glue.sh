export GLUE_DIR=./glue_data/
export TASK_NAME=SST-2

python run_glue.py \
  --model_type albert \
  --model_name_or_path albert-xxlarge-v2 \
  --task_name $TASK_NAME \
  --do_train \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME/