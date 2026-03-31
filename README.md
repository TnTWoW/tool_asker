

你的主线可以先固定成一句话：

> 基于 APIGen 构造“工具/skill 缺口”数据，训练一个能判断缺口、抽取缺失项、并主动向用户发问的模型；先用 SFT 建立基础行为，再用 RL 优化“该不该问、问什么、问多少”。APIGen 本身适合作为母体，因为它提供了可执行 API、分层验证，以及 60k 级函数调用数据；ToolSandbox 后续适合做“信息不足”与多轮交互评测。

------

## 0. 先把项目目标缩到一个最小版本

**先不要做完整 skill acquisition。先做 v0：missing-parameter / missing-endpoint / missing-auth 三类缺口。**

原因很简单：

- 这三类最容易从 APIGen 样本自动扰动出来。[arXiv](https://arxiv.org/abs/2406.18518?utm_source=chatgpt.com)
- 标注最清楚，容易做 rule-based reward。
- 能直接形成一版论文中最关键的闭环：
   **完整工具调用样本 → 缺口注入 → 模型发问 → 用户补充 → 重新执行**

你可以把 v0 任务名定为：

**Interactive Tool Gap Elicitation (ITGE-v0)**

------

## 1. 本周内必须完成的目录结构

先把 repo 起好。建议直接建成下面这样：

```
project/
  README.md
  requirements.txt

  data/
    raw/
      apigen/
    processed/
      sft_train.jsonl
      sft_dev.jsonl
      sft_test.jsonl
      rl_env_cases.jsonl

  scripts/
    01_download_apigen.py
    02_normalize_apigen.py
    03_inject_gaps.py
    04_build_sft_data.py
    05_build_eval_data.py
    06_rule_based_user_simulator.py
    07_eval_sft.py
    08_eval_rl.py

  src/
    data_schema.py
    gap_injection.py
    prompt_templates.py
    model_heads.py
    train_sft.py
    train_rl.py
    reward.py
    env.py
    metrics.py

  configs/
    sft.yaml
    rl.yaml
    gap_types.yaml
```

**验收标准：**

- 目录建立完成
- `requirements.txt` 可装
- 每个脚本都能空跑
- README 先写 15 行，说明任务、输入、输出、训练顺序

------

## 2. 数据阶段：先只做 APIGen 母体清洗

APIGen 提供了 3,673 个可执行 API、21 个类别，以及 60,000 条函数调用数据，很适合做你这个任务的“正常样本母体”。[arXiv+1](https://arxiv.org/abs/2406.18518?utm_source=chatgpt.com)

### 2.1 你本周要做的具体事

执行顺序：

1. 下载 APIGen / xLAM-function-calling-60k
2. 统一解析成一个内部 schema
3. 只保留单工具、单轮、参数字段清晰的样本
4. 先抽一个小子集，例如 **5k 条** 作为 v0

### 2.2 统一后的内部数据格式

先统一成这种 JSON：

```
JSON








{
  "id": "apigen_000001",
  "user_query": "Find the weather in Boston tomorrow.",
  "tool_name": "get_weather",
  "tool_description": "Get weather by city and date.",
  "arguments": {
    "city": "Boston",
    "date": "tomorrow"
  },
  "required_params": ["city", "date"],
  "optional_params": [],
  "api_base_url": "https://api.example.com",
  "auth_type": "api_key",
  "ground_truth_call": {
    "tool_name": "get_weather",
    "arguments": {
      "city": "Boston",
      "date": "tomorrow"
    }
  }
}
```

### 2.3 筛选规则

先只保留满足下面条件的样本：

- 有清楚的 `required_params`
- 工具描述不是空
- 参数值不是复杂嵌套对象
- 最好是单函数调用
- 不依赖长上下文历史

**验收标准：**

- 得到 `processed/apigen_base_5k.jsonl`
- 统计出每条样本 required params 数量分布
- 目测抽查 50 条，确认 schema 干净

------

## 3. 缺口注入：先只做 4 类扰动

这里是你项目真正的起点。先别贪多，先实现 4 类：

### 3.1 Gap Type A：missing_required_parameter

做法：从 `arguments` 中删除 1–2 个必须参数。

例子：

- 原始：`{"city": "Boston", "date": "tomorrow"}`
- 扰动后：`{"city": "Boston"}`

期望模型行为：

- 判断有 gap
- 缺的是 required parameter
- 发问：“请提供日期”

------

### 3.2 Gap Type B：missing_api_endpoint

做法：把 `api_base_url` 置空或删掉。

期望模型行为：

- 判断缺 endpoint
- 发问：“请提供该接口的 base URL 或 API 文档地址”

------

### 3.3 Gap Type C：missing_auth

做法：删掉 `auth_type` 或将认证字段打散成 unknown。

期望模型行为：

- 发问“该接口使用什么鉴权方式？需要 API key、Bearer token 还是其他认证？”

------

### 3.4 Gap Type D：ambiguous_tool_choice

做法：给一条 query 配两个近似工具描述，但故意让工具名不暴露。

例子：

- `search_orders(order_id)`
- `search_shipments(tracking_id)`

用户 query：

- “帮我查一下 9482 的状态”

期望模型行为：

- 不直接瞎选
- 发问：“这是订单号还是物流追踪号？”

------

## 4. 缺口注入脚本必须产出的字段

`03_inject_gaps.py` 对每条正常样本生成一个缺口样本，输出：

```
JSON








{
  "id": "gap_000001",
  "base_sample_id": "apigen_000001",
  "user_query": "Find the weather in Boston tomorrow.",
  "available_tool_spec": {
    "tool_name": "get_weather",
    "tool_description": "Get weather by city and date.",
    "required_params": ["city", "date"],
    "known_arguments": {
      "city": "Boston"
    },
    "api_base_url": null,
    "auth_type": "api_key"
  },
  "gap_type": "missing_required_parameter",
  "gold_missing_slots": ["date"],
  "gold_decision": "ask",
  "gold_question": "Please provide the date for the weather query.",
  "gold_after_user_reply": {
    "date": "tomorrow"
  },
  "gold_final_call": {
    "tool_name": "get_weather",
    "arguments": {
      "city": "Boston",
      "date": "tomorrow"
    }
  }
}
```

**验收标准：**

- 每种 gap 至少 1,000 条
- 总计先做 4,000–8,000 条
- 每种扰动写成单独函数，便于 ablation

------

## 5. 先不要真的做“三头模型”，先做“软三头数据格式”

你现在最重要的不是改模型，而是把监督信号做出来。

### 5.1 SFT target 先统一成三段式文本

训练样本 target 先写成：

```
[DECISION]
ask

[GAP_TYPE]
missing_required_parameter

[MISSING_SLOTS]
date

[QUESTION]
Please provide the date for the weather query.
```

这就是“软三头”。先让普通 causal LM 学会这个输出结构。

### 5.2 为什么先这样做

因为这样你可以先完成：

- 数据管道
- baseline SFT
- 初步评测

而不需要一上来改 Transformer 结构。

**验收标准：**

- `04_build_sft_data.py` 能输出 `instruction -> target`
- 至少有 train/dev/test 三份
- 每份 gap type 分布平衡

------

## 6. 第一版 baseline：先训一个普通 SFT 模型

### 6.1 你最小可行的 baseline

先做两个 baseline：

**Baseline 1：直接问答 SFT**
 输入：

- user query
- tool spec
- known args

输出：

- 直接生成 question / 或 act

**Baseline 2：软三头 SFT**
 输出固定为：

- `[DECISION]`
- `[GAP_TYPE]`
- `[MISSING_SLOTS]`
- `[QUESTION]`

后者通常会更稳，因为结构更清楚。

### 6.2 训练建议

- 先用 7B 以下模型，或者更小模型快速试通
- LoRA 即可
- 不要一开始上 MoE
- batch 尽量大，保证格式稳定

**验收标准：**

- 模型能稳定输出完整字段标签
- 在 dev 上不崩格式
- 能在 100 条人工检查中大致做到“会问，不瞎编”

------

## 7. 第一版评测：只做 5 个指标

先别做太复杂。v0 只做下面五个。

### 7.1 Decision Accuracy

是否正确输出 `ask / infer / act`

### 7.2 Gap Type Accuracy

是否正确识别缺口类型

### 7.3 Missing Slot F1

预测缺失字段集合 vs gold 缺失字段集合

### 7.4 Question Sufficiency Rate

问句是否覆盖所有 gold 缺失字段
 规则就行：如果 gold 缺 `date, auth_type`，问句里都出现则算 sufficient

### 7.5 Hallucination Rate

在应该 ask 的样本上，模型是否擅自编造：

- endpoint
- auth
- parameter default

**验收标准：**

- `07_eval_sft.py` 一次跑完输出 JSON 指标
- 能按 gap type 分 bucket 报告

------

## 8. RL 不要一开始全量上，先做 decision+slot 的轻量 RL

这里最容易翻车。建议你把 RL 分成两步。

### 8.1 RL v1：只优化 `decision` 和 `missing_slots`

也就是先不优化自然语言问句措辞，只优化：

- 该不该问
- 该问哪些字段

把问句先模板化：

```
Please provide the following information: {slot_list}.
```

这样 reward 可以很干净。

### 8.2 RL 环境怎么做

你不需要真实用户。先做一个 **rule-based user simulator**：

- 如果模型问到了 gold missing slot，就返回该字段值
- 如果没问到，就不补
- 如果问了冗余字段，可以忽略或轻罚
- 如果模型直接 act 但参数不够，则执行失败

### 8.3 最小 reward

先用这个版本：

R=1.0⋅success+0.5⋅decision_correct+0.5⋅slot_F1−0.2⋅num_questions−1.0⋅hallucinationR = 1.0 \cdot \text{success} + 0.5 \cdot \text{decision\_correct} + 0.5 \cdot \text{slot\_F1} - 0.2 \cdot \text{num\_questions} - 1.0 \cdot \text{hallucination}R=1.0⋅success+0.5⋅decision_correct+0.5⋅slot_F1−0.2⋅num_questions−1.0⋅hallucination

其中：

- `success`: 用户回复后能否恢复成正确 final call
- `decision_correct`: ask / act 是否选对
- `slot_F1`: 选中缺失字段集合质量
- `num_questions`: 多轮就罚
- `hallucination`: 乱补参数重罚

### 8.4 训练策略

先用 ranking / group-based 方法更省事，再考虑 PPO。因为你的关键问题是“多个候选决策谁更好”，这类问题很适合相对排序优化。

**验收标准：**

- `06_rule_based_user_simulator.py` 可运行
- `train_rl.py` 能基于 SFT checkpoint 继续训练
- RL 后 `decision accuracy` 或 `slot F1` 至少有一项明显提升

------

## 9. 真正的“三头结构”放到第二阶段

当软三头 SFT + 轻量 RL 跑顺以后，再上硬三头。

### 9.1 真三头最小定义

共享 backbone，三个 head：

- **Head 1**：decision / gap type 分类头
- **Head 2**：missing slot multi-label 头
- **Head 3**：question generation LM head

### 9.2 第二阶段再上的理由

因为如果你现在一上来就改模型结构，你会同时卡在：

- 数据
- 训练
- 指标
- RL
- 网络结构

这会拖慢起盘速度。

------

## 10. 第一轮消融设计，先做 4 组就够

论文初版最小消融如下：

### A. No-gap-type supervision

去掉 `[GAP_TYPE]` 监督
 看是否影响 `decision` 和 `slot F1`

### B. No-slot supervision

只让模型生成 question
 看是否更容易“会说不会问准”

### C. No-RL

只有 SFT
 看 RL 是否提升 ask/act 决策质量

### D. No-ambiguity data

去掉 ambiguous_tool_choice
 看模型是否仍会乱选工具

**任务定义**
 给定用户请求和不完备工具描述，模型需判断是否存在工具缺口，并主动询问最小充分信息。

**v0 支持的缺口类型**

- missing_required_parameter
- missing_api_endpoint
- missing_auth
- ambiguous_tool_choice

**训练流程**
 APIGen → gap injection → soft three-head SFT → RL refinement

**评测指标**
 Decision Accuracy / Gap Type Accuracy / Missing Slot F1 / Question Sufficiency / Hallucination Rate

**当前状态**
 “v0 focuses on parameter-level and interface-level tool gaps; out-of-tool skill acquisition is left for future work.”

第一，不要一开始就做“缺工具本体 + 缺接口 + 缺 schema + 多轮 acquisition + 真三头 + PPO 全套”。范围太大。
 第二，不要先纠结模型结构。**先把数据和评测闭环跑通。**

你的最小闭环应该是：

**APIGen 正常样本**
 → **自动注入 gap**
 → **软三头 SFT**
 → **rule-based user reply**
 → **恢复 final call**
 → **评测 success / decision / slot / hallucination**

这条线一旦跑通，你的项目就已经站住了。APIGen适合做母体数据，ToolSandbox后续适合补多轮“信息不足”场景；若你后面想扩展到真实多工具发现，再接 MCP-Atlas 会更自然。