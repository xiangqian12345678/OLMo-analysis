cd ../

torchrun --nproc-per-node=8 src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-2.py \
  --save-folder=../output/pretrain2 \
  --name=olmo3-7b-pretrain-stage2
