cd ../

torchrun --nproc-per-node=8 src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py \
  --save-folder=../output/train_longtext \
  --name=olmo3-7b-long-context
