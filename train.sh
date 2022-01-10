#!/bin/bash
python run_lm_finetuning.py --do_train --train_data_file train.txt \
    --per_gpu_train_batch_size 32 \
    --do_eval --eval_data_file test.txt \
    --per_gpu_eval_batch_size 32 \
    --output_dir output \
    --max_train_lines 100000 \
    --max_val_lines 80000 \
    --block_size 512 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 500 \
    --save_steps 1 \
    --save_total_limit 2 \
    --model_name_or_path model/roberta/pytorch_model.bin \
    --config_name model/roberta/bert_config.json \
    --tokenizer_name model/roberta/vocab.txt \
    --evaluate_during_training \
    --do_lower_case \
    --mlm