# 训练计划：laya-multilingual 双语（en+zh）canon spec 判断

## 1. 目标

一个模型同时服务 canon spec 的英文与中文判断（`classify` / `compare` /
`extract` / `rerank` / `find`）。底座：**laya-multilingual**（mmBERT-base，
100+ 语言，en/zh 均在预训练内）。

## 2. 可行性与成功率（评估结论）

| 目标 | 概率 | 依据 |
|---|---|---|
| 机械成功（收敛、双语提升） | ~0.85 | 多语言联合微调是标准做法；仓库内 recipe 已验证（typed-decisions 2000 行：0.362→0.766） |
| 追平两个专精 checkpoint | ~0.5–0.6 | 中文大概率追平（基座中文强）；英文有天花板——基座英文 MASSIVE 0.657 vs 英文 checkpoint 0.783 |
| 元判断题（evidence_sufficient 等） | ~0.2–0.3 | 两个 checkpoint 都是掷硬币；须引入元判断训练数据才能改善 |

决策：已选定 laya-multilingual 路线（一个模型干两种事，省 3-checkpoint 常驻
~3.9GB 内存与路由复杂度，接受 ~10 个百分点英文精度 tradeoff）。

## 3. 数据

### 格式（rows JSONL；RLCD 需要概率分布 gold）

见 [`data/README.md`](../data/README.md)。要点：`gold` 是每题选项上的概率
分布（硬标签用 one-hot 近似）；`state_hash` 是分组键；`lang` 用于平衡审计。

### 来源

1. **蒸馏（主力）**：用现有 typed-decisions（英文）/ multilingual（中文）
   checkpoint 对历史状态跑 predict，取输出分布作 gold（强模型标注）
2. **真实记录**：canon 历史 review 记录、中文工单/分诊日志
3. **公开集（辅助，非主场景）**：MASSIVE zh-CN、CLUE（TNEWS/ChnSentiCorp）

### 规模与平衡

- 几千条高质量即可（先例：2000 行带来 0.362→0.766）
- **en/zh 各半**（防单侧灾难性遗忘）；按 `lang` 字段审计平衡

## 4. 训练/校验/测试划分（纪律）

- 三分 **80/10/10**；训练前一次性完成；seed 固定（42）；划分结果入
  `splits/manifest.json`（可复现）
- **分组划分**：同一 state（diff/工单/会话）的所有 question 行必须整体落入
  同一集合——`make_splits.py` 按 `state_hash` 分组。否则同一文档的近重复
  跨 train/val 泄漏，验证集虚高
- **validation 只用于**：超参选择、早停、温度校准；**绝不参与训练**
  （拟合过度的经典来源）
- **test 只在最终验收用一次**；任何基于 test 的调整都污染它，调整后必须
  重新划分
- 增强（选项顺序翻转）只作用于 train
- 归一化/统计量在划分之后计算（防信息泄漏）

## 5. 配方

1. **SFT 决策头**：laya 的 `train_cli.py`（`--text-column` / `--label-column`
   / `--question-id` / `--instructions`；`--freeze-encoder` 先冻编码器）——
   本机 CPU 小时级可完成
2. **RLCD 校准**：`train.py` 消费 rows JSONL；参考 layazh 超参：
   head-lr 3e-4、encoder-lr 2e-5、rl_weight 8、kl_weight 1、group 4、
   σ 0.5、τ 1.5；解冻顶部 8–12 层 encoder + 头
3. **温度校准**：在 **validation** 上 refit（multilingual 中文出厂 ECE 0.219
   → 目标 <0.15；注意 noul:2 过自信可能超出出厂 `TEMP_MAX=5.0` 钳制，
   需放宽）
4. **验收**（在 **test** 上报，一次性）：accuracy、ECE、option-order
   flip rate <10%


### 5.1 实际命令（已验证的接口）

数据在本仓库，训练在 laya 仓库跑（驱动 `research/scripts/finetune_single_device.py`
单设备复现 2xT4 训练循环：预处理 → RLCD → 每题型拟合一个校准温度 → 存可加载
checkpoint）：

```bash
cd ~/projects/laya
HF_HUB_OFFLINE=1 .venv/bin/python research/scripts/finetune_single_device.py \
    --data ~/projects/laya-canon-judgments/data/splits/train.jsonl \
    --model-dir ~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/a9597e0e438d6971d2eb6742d7d8610cefd9f262/multilingual \
    --output-dir ~/projects/laya-canon-judgments/results/finetune-v1 \
    --device cuda --epochs 4 --seed 42
```

驱动内部在训练前从训练集切出校准片（≤400 条或 10%，固定 seed，不参与训练），
与 §4 的划分纪律一致。验收用 `laya-evals` 跑 test 划分（行内 `expected` 即
laya-evals 的读法）：

```bash
.venv/bin/laya-evals --data ~/projects/laya-canon-judgments/data/splits/test.jsonl \
    --model-dir ~/projects/laya-canon-judgments/results/finetune-v1
```

验收通过后才注册进 laya-serve（`LAYA_EXTRA_MODELS`），路由表加语言→新模型映射，
gate 的 `JEV_MODEL=auto` 渐进切到新 checkpoint；不过关则留在实验区，gate 继续
走现有三 checkpoint。
## 6. Kaggle 免费 GPU 使用法

- **额度**：30 GPU-h/周（2xT4 或 P100）；TPU v3-8 亦有；CPU 无周配额
- **会话上限**：GPU/CPU 12h、TPU 9h
- **前置**：账号手机验证解锁 GPU/TPU；无需信用卡；周配额每周重置，随需求波动
- **工作流**：
  1. kaggle.com → 新建 Notebook → Settings → Accelerator 选 GPU（T4x2）
  2. 数据：上传到 Kaggle Datasets（私有数据集免费），或挂 Google Drive
     （notebook 内一键挂载）
  3. CLI：Account → Create New API Token 下载 `kaggle.json`；
     `pip install kaggle` 后 `kaggle datasets upload`
  4. 环境：laya 仓库 `pip install -e .`；checkpoint 从 HF 下载
     （Kaggle 直连 HF）
- **分工**：本机 CPU 做数据准备 + SFT head-only；Kaggle GPU 做 encoder
  解冻训练 + RLCD（30h/周足够 2–3 轮完整实验）

## 7. 与现有架构的关系

训练完成前，gate 继续用 3-checkpoint + 路由（review_chain 的
`JEV_MODEL=auto`）。新 checkpoint 验收通过后，作为 `LAYA_EXTRA_MODELS`
注册进 laya-serve，路由表加 语言→新模型 映射，渐进替换。
