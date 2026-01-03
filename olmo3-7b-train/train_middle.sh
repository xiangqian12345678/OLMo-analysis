cd ../

torchrun --nproc-per-node=8 src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py \
  --save-folder=../output/middle \
  --name=olmo3-7b-midtrain
