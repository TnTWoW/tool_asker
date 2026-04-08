# VERL SFT + RL Experiment Pipeline

这套 pipeline 直接复用当前仓库里已经准备好的 `gap_samples`、`sft_train/dev/test`，再按 VERL 的数据接口导出成 parquet。

## 1. 先准备 VERL 数据

```bash
python scripts/09_build_verl_datasets.py
```

生成结果：

- `data/verl/sft/train.parquet`
- `data/verl/sft/val.parquet`
- `data/verl/sft/test.parquet`
- `data/verl/rl/train.parquet`
- `data/verl/rl/val.parquet`

其中：

- SFT 数据使用 `prompt` / `response` 两列，对应 `verl.trainer.fsdp_sft_trainer`
- RL 数据使用 `prompt` / `reward_model` / `extra_info`，对应 `verl.trainer.main_ppo`
- RL reward 通过 [src/verl_reward.py](/D:/pythonCode/tool_asker/src/verl_reward.py) 计算，复用了你项目里 “decision + slot + hallucination” 的思路

## 2. 跑 SFT

参考 VERL 官方 quickstart 里提到的 SFT 入口 `verl.trainer.fsdp_sft_trainer`，这里给了一个最小脚本：

```bash
bash scripts/run_verl_sft.sh 4 checkpoints/verl_sft Qwen/Qwen2.5-1.5B-Instruct
```

默认设置：

- LoRA SFT
- `prompt_key=prompt`
- `response_key=response`
- `max_length=2048`
- `total_epochs=2`

常用覆盖方式：

```bash
MICRO_BATCH_SIZE=2 TOTAL_EPOCHS=3 LORA_RANK=16 bash scripts/run_verl_sft.sh 2 checkpoints/verl_sft
```

## 3. 把 SFT checkpoint merge 成 HF 模型

如果你想把 SFT 结果继续作为 RL 初始模型，最稳妥的做法是先 merge：

```bash
bash scripts/run_verl_merge.sh \
  checkpoints/verl_sft/tool-asker-sft/itge-sft-qwen2.5-1.5b/global_step_XXX/actor \
  checkpoints/verl_sft_merged
```

这一步对应 VERL quickstart 里给的 `verl.model_merger merge` 用法。

## 4. 跑 RL

RL 阶段使用 quickstart 里的 `verl.trainer.main_ppo`，只是把 reward 换成了当前任务的自定义打分：

```bash
bash scripts/run_verl_rl.sh 1 checkpoints/verl_rl checkpoints/verl_sft_merged
```

默认脚本现在按当前 VERL 版本的 async rollout 要求走 `vllm`：

- `ROLLOUT_NAME=vllm`
- `total_training_steps=200`
- 自定义 reward：格式解析 + decision correctness + gap type correctness + slot F1 + hallucination penalty

单卡最小命令：

```bash
bash scripts/run_verl_rl.sh 1 checkpoints/verl_rl checkpoints/verl_sft_merged
```

如果显存比较紧，可以先收紧 rollout 占用：

```bash
ROLLOUT_TP_SIZE=1 ROLLOUT_GPU_MEMORY_UTILIZATION=0.3 bash scripts/run_verl_rl.sh 1 checkpoints/verl_rl checkpoints/verl_sft_merged
```

## 5. 推荐实验顺序

1. 先用 `python scripts/09_build_verl_datasets.py` 导出 parquet
2. 用 `bash scripts/run_verl_sft.sh ...` 跑一个 0.5B 或 1.5B 的小模型做 sanity check
3. merge SFT actor checkpoint
4. 用 `bash scripts/run_verl_rl.sh ...` 跑 100 到 200 steps 验证 reward 是否正常
5. 再逐步调大 batch、response length、训练步数

## 6. 当前这套实现里的 reward 目标

模型输出仍然沿用你现有 SFT target 的结构：

```text
[DECISION]
ask

[GAP_TYPE]
missing_required_parameter

[MISSING_SLOTS]
reservation_id

[QUESTION]
Please provide the reservation ID.
```

`compute_score` 会奖励：

- `DECISION` 是否正确
- `GAP_TYPE` 是否正确
- `MISSING_SLOTS` 的 F1
- `QUESTION` 在需要提问时是否非空

也会惩罚：

- 幻觉 slot
- 不必要的提问

## 7. 需要你按机器资源调的参数

最先调这几个：

- `MICRO_BATCH_SIZE`
- `TRAIN_BATCH_SIZE`
- `ACTOR_MICRO_BATCH_SIZE`
- `CRITIC_MICRO_BATCH_SIZE`
- `MAX_PROMPT_LENGTH`
- `MAX_RESPONSE_LENGTH`
- `ROLLOUT_NAME`

如果显存比较紧，优先：

- 把模型换成 `Qwen/Qwen2.5-0.5B-Instruct`
- 降低 `ROLLOUT_GPU_MEMORY_UTILIZATION`
- 降低 `MICRO_BATCH_SIZE`
- 降低 `MAX_RESPONSE_LENGTH`
