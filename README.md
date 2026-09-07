# CogAlpha — 实现

基于 `paper.md`（《Cognitive Alpha Mining via LLM-Driven Code-Based Evolution》）
和 `prompts/` 提示词模板的完整可运行实现，使用 Python `openai` SDK 与 `pydantic`。

## 架构

```
src/
├── run.py                  # 命令行入口（--demo / --self-test / 真实运行）
├── requirements.txt
└── cogalpha/
    ├── models.py           # 所有 dataclass 与 pydantic 模型（LLM/进化配置、Factor、质检输出、SearchResult）
    ├── llm_client.py       # OpenAI 兼容客户端封装（chat + JSON 结构化输出）
    ├── prompt_loader.py    # 从 prompts/ 解析并组装提示词
    ├── data_loader.py      # 载入 OHLCV 面板、列描述
    ├── executor.py         # 静态检查（AST 禁嵌套循环）+ 编译执行因子
    ├── evaluator.py        # IC / RankIC / ICIR / RankICIR / MI
    ├── quality.py          # 多智能体质量检查器（质量/修复/评判/逻辑优化）
    ├── generation.py       # 七级生成智能体（21 个）
    ├── evolution.py        # 思维进化（变异 / 交叉）
    ├── feedback.py         # 自适应生成反馈（有效/无效因子摘要）
    ├── selection.py        # 池管理 + 合格/精英分类
    ├── pipeline.py         # 生成→质检→执行→评估→分类 单因子流水线
    └── search.py           # 主进化搜索编排器（CogAlpha 类）
```

## 运行

```bash
cd src
pip install -r requirements.txt

# 离线自检（静态检查 / 解析器）
python run.py --self-test

# 离线端到端演示（合成数据，无需 API key）
python run.py --demo

# 真实运行（需 API key）
export OPENAI_API_KEY=sk-...
python run.py --data data.parquet --horizon 10 --output ./out --model gpt-4o-mini
```

若不使用官方端点而是走网关/代理，可设置 `OPENAI_BASE_URL`（地址）与
`OPENAI_CHAT_MODEL`（模型名），两者同样适用 openai 客户端；项目自身的
`COGALPHA_BASE_URL` / `COGALPHA_MODEL` 优先于这两个变量。

## 关键设计

- **提示词组装**：`prompt_loader.PromptLibrary` 从 `prompts/` 目录解析 21 个七级
  智能体（intro + guidance）与共享模块（requirements / libraries / output_format /
  system_message / 反馈分析等），按顶层 README 的 8 步顺序组装。
- **LLM 调用**：生成/进化智能体从 `{0.7,…,1.2}` 均匀抽样温度；质量检查器固定 0.8；
  评判/摘要等返回 JSON 并强制转换为 `pydantic` 模型（`complete_json`）。
- **质量检查器**：静态 AST 检查（嵌套循环 / `while True` / 语法 / 返回列名）为确定层，
  LLM 质量、评判、逻辑优化、修复为可开关层（`cfg.use_llm`）。
- **执行**：因子函数按 ticker 分组应用（`apply_factor`），返回 `(date,ticker)` 索引的 Series；
  NaN 比例超 30% 的因子丢弃。
- **评估**：前向收益标签（默认 10 日 open-to-open），逐日横截面相关平均得
  IC/RankIC，`mean/std` 得 ICIR/RankICIR，MI 用互信息（`sklearn` 优先，回退手写）。
- **进化循环**：初始池 → 父池 →（生成/变异/交叉）子池 → 质检 → 执行 → 评估 →
  每 `inject_every` 代把新 alpha 注入父池、刷新自适应反馈，精英跨代携带。

## 离线模式

`cfg.use_llm=False` 时仅运行确定性的静态检查 + 执行 + 评估，不产生任何网络调用，
用于在没有 API key 的环境下验证流水线。

## 免责声明

本实现仅用于学术研究，不构成金融建议。使用真实数据前请自行验证数据与因子。
