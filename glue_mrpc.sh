GPU="0"
EPOCH=5
LR=2e-5
BSZ=16
SEED=1234
TASK_NAME=mrpc  
# train file, with instance difficulty label
TRAIN_FILE=data_difficulty/${TASK_NAME}.train.dif_5seeds-2level.json
CONF_MARGIN=0.5
DAR_WEIGHT=0.5
LAYER=2
#PATH_TO_SMALL_BERT
SMALL_MODEL=uncased_L-2_H-768_A-12  #-uncased
BIG_MODEL=bert-base-uncased
CASCADE_MODEL="${SMALL_MODEL};${BIG_MODEL}"
OUTPUT_DIR=saved_models/${TASK_NAME}_${SEED}_${LAYER}L-12L

# Training
CUDA_VISIBLE_DEVICES=$GPU python run_cascade.py --train_file $TRAIN_FILE \
  --cascade_model_name_or_path $CASCADE_MODEL  --fp16 --cascade_model_layers "2;12" \
  --confidence_margin $CONF_MARGIN --dar_weight $DAR_WEIGHT \
  --task_name $TASK_NAME  --overwrite_output_dir \
  --do_eval  --do_train --evaluation_strategy steps --metric_for_best_model "accuracy"  \
  --save_steps 200 --logging_steps 200 --save_total_limit 2 \
  --max_seq_length 128 \
  --per_device_train_batch_size ${BSZ}  \
  --learning_rate $LR \
  --num_train_epochs $EPOCH --seed $SEED \
  --output_dir "$OUTPUT_DIR"


# Evaluation, take confidence threshold as 0.5 for example
for TH in 0.6 0.7 0.8 0.9
do
CUDA_VISIBLE_DEVICES=$GPU python run_cascade.py --train_file $TRAIN_FILE \
  --saved_model_path "$OUTPUT_DIR" \
  --cascade_model_name_or_path $CASCADE_MODEL  --fp16 --cascade_model_layers "2;12" \
  --confidence_threshold $TH --task_name $TASK_NAME \
  --do_eval   --infer_mode "cascade"\
  --max_seq_length 128 \
  --per_device_eval_batch_size ${BSZ}  \
  --output_dir "$OUTPUT_DIR"
done 
