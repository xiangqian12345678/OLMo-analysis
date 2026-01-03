# 训练顺序

## 阶段 1: 预训练

### 步骤 1.1: 预训练第一阶段（0 - 5.93 万亿 tokens）

```bash
    sh train_pretrain1.sh
```

### 步骤 1.2: 预训练第二阶段（从 checkpoint 继续，扩展至 7 万亿 tokens）

```bash
    sh train_pretrain2.sh
``` 

## 阶段 2: 中期训练（1000 亿 tokens）

```bash
    sh train_middle.sh
```

## 阶段 3: 长上下文训练（500 亿 tokens，65,536 上下文长度）

```bash
    sh train_longtext.sh
```

## 阶段 4: SFT 训练（监督微调）

```bash
    sh train_sft.sh
```

