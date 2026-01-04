cd ../

# 单GPU训练,使用 train_single 子命令自动禁用 FSDP/DP
# 测试配置：使用 v3_small_ppl_validation 数据集（约几百万 tokens）
# 生产配置：将 dataset_config 中的 DataMix 改回 OLMo_mix_0625_official
#           将 hard_stop 从 100 改回 95_000
python src/scripts/official/OLMo3-190m/OLMo-3-190m-pretrain.py train_single \
  --save-folder=../output/190m/pretrain \
  --name=olmo3-190m-pretrain-stage1


