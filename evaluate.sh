#!/bin/bash
python run_lm_evaluate.py \
    --eval_data_file test.txt \
    --per_gpu_eval_batch_size 32 \
    --eval_output_dir eval_results \
    --max_val_lines 100 \
    --block_size 512 \
    --model_name_or_path output/checkpoint-582500 \
    --do_lower_case \
    --mlm