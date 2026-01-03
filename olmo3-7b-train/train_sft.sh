cd ../

python src/scripts/train/sft/Olmo-3-7B-SFT.py launch \
  run_name \
  /path/to/sft/data \
  /path/to/checkpoint \
  ai2/jupiter-cirrascale-2 \
  --seq_len=4096 \
  --num_nodes=2
