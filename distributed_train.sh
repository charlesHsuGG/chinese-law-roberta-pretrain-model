#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr 127.0.0.2 \
    --master_port 29501 \
    run_lm_finetuning.py \
    --do_train --train_data_file train.txt \
    --per_gpu_train_batch_size 32 \
    --do_eval --eval_data_file test.txt \
    --per_gpu_eval_batch_size 32 \
    --output_dir output/2020080601 \
    --max_train_lines 50000000 \
    --max_val_lines 10000 \
    --block_size 512 \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --logging_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --model_name_or_path model/roberta/pytorch_model.bin \
    --config_name model/roberta/bert_config.json \
    --tokenizer_name model/roberta/vocab.txt \
    --evaluate_during_training \
    --do_lower_case \
    --fp16 \
    --mlm