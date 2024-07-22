DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

torchrun $DISTRIBUTED_ARGS src/train_bash.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn \
    --model_name_or_path $MODEL_PATH \
    --dataset your_dataset \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16