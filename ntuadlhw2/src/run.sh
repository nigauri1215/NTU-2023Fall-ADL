accelerate launch test.py \
  --model_name_or_path temp_07 \
  --output_dir ${2} \
  --test_file ${1} \
  --text_column maintext \
  --max_source_length 256 \
  --max_target_length 64 \
  --num_beams 15 \