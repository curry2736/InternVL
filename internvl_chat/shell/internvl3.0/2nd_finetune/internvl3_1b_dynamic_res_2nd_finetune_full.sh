set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# Generate a timestamp e.g., 2023-10-27_15-30-00
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

MODEL_TYPE_DIR='work_dirs/internvl_chat_v3/internvl3_1b_dynamic_res_2nd_finetune_full'
OUTPUT_DIR="${MODEL_TYPE_DIR}/${TIMESTAMP}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

LOG_DIR="${OUTPUT_DIR}/training_logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi


LOG_FILE="${LOG_DIR}/training_log_${TIMESTAMP}.txt"

ORIGINAL_MODEL_PATH="/home/jovyan/data/arisrei_ws/vlm_deep_scan/vlm_deep_scan/third_party_models/InternVL/pretrained/InternVL3-1B"

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path ${ORIGINAL_MODEL_PATH} \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/home/jovyan/data/arisrei_ws/vlm_deep_scan/ip_data/internvl3_finetuning_datasets/gpt_he.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 50 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --per_device_eval_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "epoch" \
  --do_eval True \
  --load_best_model_at_end True \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --load_best_model_at_end True \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${LOG_FILE}"

  python /home/jovyan/data/arisrei_ws/vlm_deep_scan/vlm_deep_scan/scripts/copy_py_files.py ${ORIGINAL_MODEL_PATH} ${OUTPUT_DIR}