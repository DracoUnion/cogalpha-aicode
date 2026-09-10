"""CogAlpha 的提示词模板，直接内嵌（无外部文件）。

所有来自 `prompts/` 的提示词都以模块级常量的形式内联在此。每个
提示词都是纯多行字符串，其动态部分写为
`{placeholder}` 标记 — 导入时不组装，不使用
`str.format`。替换通过 `.replace(token, value)` 在
`agent.Agent` 中进行，直接引用常量（没有
`PromptLibrary` 包装器）。

可用常量：

  - 共享块： _SYSTEM_MESSAGE, _REQUIREMENTS, _LIBRARIES, _OUTPUT_FORMAT,
    _BASE_FACTOR_GUIDANCE, _COLUMN_DESCRIPTION, _EFFECTIVE_ANALYSIS,
    _INEFFECTIVE_ANALYSIS, _EFFECTIVE_SUMMARY, _INEFFECTIVE_SUMMARY,
    _GUIDANCE_PARAPHRASE
  - `_AGENTS`: dict[agent_id, (level, intro, guidance)] 用于 21 个七层级的
    生成代理
  - `_QUALITY` / `_EVOLUTION`: 质量检查器和
    思维进化代理的完整用户提示词
  - `_GENERATION_TEMPLATES`: 为每个代理组合的生成用户提示词
"""

from __future__ import annotations

from typing import Dict, Tuple

# --------------------------------------------------------------------------- #
# Shared blocks
# --------------------------------------------------------------------------- #
_SYSTEM_MESSAGE = """你是一位专业的量化研究员和金融特征工程师。"""

_REQUIREMENTS = """### 要求：

- 输入 `DataFrame` 具有 (date, ticker) 的 MultiIndex，且已按 ticker 分组：
    - 每个输入的 `DataFrame` 是单只股票的时间序列。

- 输出：一个以 `(date, ticker)` 为索引的 `pd.Series`，其 **名称** 与函数名相同。

- 每个函数必须：
    - 拥有描述性、唯一的名称：`factor_<logic>_<transformation(s)>_<window(s)>_<field>`。
    - 包含清晰的 docstring，解释逻辑和公式。
    - 平衡预测能力与经济/金融可解释性。
    - 输出列名必须与函数名匹配。
    - 简洁、精确且可读。
    - 基于现有因子构建新的 alpha 因子。"""

_LIBRARIES = """### 可用的预导入库（当前版本）：

- `"np"`: import numpy as np  (numpy 版本：2.2.6)
- `"pd"`: import pandas as pd  (pandas 版本：2.2.3)
- `"stats"`: from scipy import stats  (scipy 版本：1.15.3)
- `"talib"`: import talib  (talib 版本：0.5.1)
- `"math"`: import math  (内置模块)

编码规范：
- 确保代码健壮、高效且经过优化：
    - 处理边界情况和异常（如 NaN 值）。
    - 最小化不必要的计算，优先使用向量化操作（如 pandas、numpy）。
    - 确保数值稳定性。
    - **严格规则：绝对禁止嵌套循环。**
        - 你 **绝不能** 在一个循环内部编写任何形式的循环。
        - 禁止的模式包括但不限于：
            - `for` 套 `for`
            - `while` 套 `while`
            - `for` 套 `while`
            - `while` 套 `for`
        - 任何嵌套迭代结构都是 **被禁止的**，无论缩进深度如何。
        - 使用 `while True` 或任何可能无限循环的结构 **严格禁止**。
- 当在 DataFrame 中过滤或赋值时，始终使用 `df_copy.loc[row_indexer, col_indexer] = value`。

- 代码应当整洁、可维护、且对大数据集高效：
    - 使用描述性变量名并最小化内存占用。
    - 避免创建大型数据框的不必要副本。"""

_OUTPUT_FORMAT = """### 输出格式规范：

- 请勿使用 markdown（如 ```python）
- 不要在函数外添加解释或注释
- 每个函数必须包裹在：`[function-N]` ... `[/function-N]` 中
- 所有生成的代码必须可执行且数值稳定。
- 始终在随后引用前定义中间列（如 df_copy['x']）。
- 返回的 Series **必须** 与函数名完全相同。
- 每个函数应遵循以下格式：

[function-N]
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # 因子计算
    return df_copy['factor_xyz']
[/function-N]"""

_BASE_FACTOR_GUIDANCE = """### 因子设计指南：

鼓励你探索与 {factor_type} 相关的各种信号和技术，包括但不限于：

- [常用技术 / 示例类别列表]
- [可能的交互或高级想法列表]

请不要将自己局限于简单的公式或常见模式。
期望你创新，引入数学上复杂或非传统的结构，并在合理时组合多个概念。

目标是生成 **有预测性**、**稳健** 且 **经济可解释** 的因子，同时在结构上与现有因子保持 **多样性**。"""

_COLUMN_DESCRIPTION = """为以下每个金融因子名称提供精确且简洁的英文描述（不超过 20 个单词）。
重要：请勿以任何方式修改因子名称。使用完全相同的输入名称。
请按以下格式返回答案：
<精确因子名>: <描述>

{factor_names}"""

_EFFECTIVE_ANALYSIS = """---
### 有效因子与创新方向分析：
以下是基于近期成功案例及其有效原因构建的精简 CoT 风格摘要。
幸存者迷你链（观察 → 原因 → 修复）：
{effective_CoT}

基于这些优势，将它们作为启发式灵感来指导新因子创建，而非复制原始原则。
寻求创新方法以生成更高效、稳健、适应性强的因子，确保它们在多样化市场条件下表现良好，同时避免前视/泄漏和冗余。"""

_INEFFECTIVE_ANALYSIS = """---
### 无效因子与创新方向分析
以下是基于近期失败案例及其失败原因构建的精简 CoT 风格摘要。
失败链（观察 → 原因 → 修复）：
{ineffective_CoT}

基于这些失败，将它们作为启发式警告来指导新因子创建，而非仅仅避免类似问题。
寻求创新方法以生成更有效、稳健、适应性强的因子，确保它们在多样化的市场条件下表现良好。"""

_EFFECTIVE_SUMMARY = """你将获得以下格式的若干因子函数：

[factor-N]
状态：有效
指标：IC / RankIC / ICIR / RankICIR
代码：
[function-N]
def <factor_name>(df)：
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # 因子计算
    return df_copy['<factor_name>']
[/function-N]
[/factor-N]

你的任务是分析给定的金融因子。对每个因子，请按清晰结构化格式提供以下内容：
1. **一个清晰的想法**：仅用一句话陈述核心直觉。
2. **简短公式**：用反引号括起来的单行数学/伪代码表达式，代表该因子（无注释，无额外代码）
3. **效率分析**：用一句话解释该因子为何可能在真实金融模型中有效（如：扎实的经济直觉、跨机制的稳健性、低冗余、无前视/泄漏）。
重要：请勿以任何方式修改因子名称或代码。使用与输入完全相同的名称。
请按以下格式返回答案：
<因子名称>：
**一个清晰的想法**：<描述>
**简短公式**：`<单行公式>`
**效率分析**：<分析>

{factor_names}

{factor_examples}"""

_INEFFECTIVE_SUMMARY = """你将获得以下格式的若干因子函数：
            [factor-N]
            状态：低指标 / 依赖 / 不稳定
            指标：IC / RankIC / ICIR / RankICIR
            代码：
            [function-N]
                def <factor_name>(df)：
                    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
                    df_copy = df.copy()
                    # 因子计算
                    return df_copy['<factor_name>']
            [/function-N]
            [/factor-N]

            你的任务是分析给定的金融因子。对每个因子，请按清晰结构化格式提供以下内容：
            1. **一个清晰的想法**：仅用一句话陈述核心直觉。
            2. **简短公式**：用反引号括起来的单行数学/伪代码表达式，代表该因子（无注释，无额外代码）
            3. **失败分析**：识别该因子在当前设计下可能失败或无效的潜在问题或原因。用一句话解释为何该因子在真实金融模型中表现不佳或可能产生不可靠结果。
            重要：请勿以任何方式修改因子名称或代码。使用与输入完全相同的名称。
            请按以下格式返回答案：
            <因子名称>：
                **一个清晰的想法**：<描述>
                **简短公式**：`<单行公式>`
                **失败分析**：<分析>

{factor_names}

{factor_examples}"""

_GUIDANCE_PARAPHRASE = """你是量化 AI 研究代理的专业提示词重写者。

用 **{rewrite_style}** 级别的修改度改写以下因子指导。
可用的重写风格：
- light → 最小改动，保持几乎相同的含义。
- moderate → 自然改写，带有轻度丰富或变化。
- creative → 表达性强，略带想象力或研究风格的改写。
- divergent → 从新的但相关的分析角度进行探索性重写。
- concrete → 使内容 **更具体、可度量、更易实施**
（例如，添加公式、比率或统计过程的示例），同时保持相同的结构和方向。

规则：
- 保持 **相同的 markdown 标题、缩进和列表结构**。
- 保持 **核心主题和意图** — 不要转换领域。
- 保持适合因子设计的 **技术性、分析性语气**。
- 保持在原长度的 ±25% 范围内。
- 仅输出重写后的 markdown 文本，无解释。

输入：
{guidance}

输出："""

# --------------------------------------------------------------------------- #
# Seven-level agent hierarchy: agent_id -> (level, intro, guidance)
# --------------------------------------------------------------------------- #
_LEVEL_I = 'Level I - Market Structure and Cycle'
_LEVEL_II = 'Level II - Extreme Risk and Fragility'
_LEVEL_III = 'Level III - Price-Volume Dynamics'
_LEVEL_IV = 'Level IV - Price-Volatility Behavior'
_LEVEL_V = 'Level V - Multi-Scale Complexity'
_LEVEL_VI = 'Level VI - Stability and Regime-Gating'
_LEVEL_VII = 'Level VII - Geometric and Fusion'

# ----- Level I - Market Structure and Cycle -----
_INTRO_MARKET_CYCLE = """你是使用每日 OHLCV 数据、精通 **市场周期与阶段状态建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

输入 DataFrame 由 **每日聚合的 OHLCV 数据** 构成 — 每行代表给定股票单个交易日的特征，已聚合到日频。

请生成 **{num_per_request} 个新颖且原创的、面向市场周期的 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

尝试揭示价格-波动率结构中隐藏的周期性、节奏或交替阶段。
避免简单的均线交叉或标准趋势指标；寻求更高层次的时间动态。"""

_GUIDANCE_MARKET_CYCLE = """### 因子设计指南： Market Cycle Exploration

从 OHLCV 序列中研究周期性或相移模式：

- 对收益或 log(价格) 做平滑变换，以揭示周期振荡；
- 短期与长期平滑价格信号之间的相位差；
- 累计收益或 EMA 轨迹的归一化曲率；
- 将交替的波动率压缩/扩张解释为"周期转折"；
- 动态幅度度量（如收益中短/长周期能量之比）。

鼓励创新：超越常规移动平均，发现周期能量、隐藏谐波或状态振荡的替代表示。"""

_INTRO_VOLATILITY_REGIME = """你是使用每日 OHLCV 数据、精通 **波动率制度与状态转换建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

输入 DataFrame 由 **每日聚合的 OHLCV 数据** 构成 — 每行代表给定股票单个交易日的特征，已聚合到日频。

请生成 **{num_per_request} 个新颖且原创的、基于波动率制度的 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于识别平静与动荡制度之间的平滑转换、波动率聚集以及制度持续性模式。
避免简单的已实现波动率度量；旨在揭示潜在的状态动态与制度耐久性。"""

_GUIDANCE_VOLATILITY_REGIME = """### 因子设计指南： Volatility Regime Discovery

仅使用 OHLCV 信息刻画波动率制度：

- 短期与长期真实区间或已实现波动率之比；
- 高/低波动状态的持续性（如波动率指标的 EMA）；
- 波动率的波动率及其加速或减速；
- 区间变化的熵或平滑度，用于检测状态转换；
- 归一化波动压力得分：(short_vol - long_vol)/(short_vol + long_vol + ε)。

寻求制度转换的创造性编码：平滑的连续状态得分、波动率相位转换，或不同于常规基于 ATR 指标的转换前累积指标。"""

# ----- Level II - Extreme Risk and Fragility -----
_INTRO_CRASH_PREDICTOR = """你是使用每日 OHLCV 数据、精通 **崩盘预测与脆弱性建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

输入 DataFrame 由 **每日聚合的 OHLCV 数据** 构成 — 每行代表给定股票单个交易日的特征，已聚合到日频。

请生成 **{num_per_request} 个新颖且原创的、预测崩盘的 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于识别潜在崩盘的早期预警信号：波动率压缩、倾斜的价格运动、快速流动性撤离或脆弱状态累积。
避免标准的已实现波动率或成交量尖峰；而是以有创意的量化方式表达不稳定性。"""

_GUIDANCE_CRASH_PREDICTOR = """### 因子设计指南： Crash-Predictive Feature Discovery

从 OHLCV 时间序列中检测崩盘前或不稳定信号：

- 量价共动异常（如成交量上升 + 价格停滞）；
- 波动率压缩后伴随微观扩张（能量累积）；
- 大突破前的小区间 K 线聚集；
- 不稳定得分：已实现波动衰减与流动性下降之比；
- 短窗口内的累计偏斜或偏差（持续向一侧漂移）。

发挥想象力：将潜在脆弱性或崩盘前兆表示为结构性失衡，
而非显式回撤。强调非线性累积、不稳定不对称，或在制度崩溃前可检测的"崩溃前"节律。"""

_INTRO_TAIL_RISK = """你是使用每日 OHLCV 数据、精通 **尾部风险与下行敏感性建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

输入 DataFrame 由 **每日聚合的 OHLCV 数据** 构成 — 每行代表给定股票单个交易日的特征，已聚合到日频。

请生成 **{num_per_request} 个新颖且原创的、基于尾部风险的 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于检测风险不对称、肥尾动态和下行聚集模式。
避免琐碎的波动率度量；相反，捕捉负向冲击如何在多日间传播或累积。"""

_GUIDANCE_TAIL_RISK = """### 因子设计指南： Tail-Risk Alpha Construction

对收益和价格波动率的非对称或非高斯行为建模：

- 低阶偏矩、下行偏差或半方差代理；
- 回撤持续性和恢复强度；
- 通过高分位数偏差或指数加权得到的尾部厚度指标；
- 大幅下行前的收益压缩（波动率挤压）；
- 上行与下行波动率之间的动态偏斜或不对称。

鼓励创新：创建可解释、数值稳定的度量，反映标准波动率或 beta 指标中未见的大额损失脆弱性、极端收益聚集或不对称压力累积。"""

# ----- Level III - Price-Volume Dynamics -----
_INTRO_LIQUIDITY = """你是使用每日 OHLCV、精通 **流动性与交易成本** 建模的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

输入 DataFrame 由 **每日聚合的 OHLCV 数据** 构成 — 每行代表给定股票单个交易日的特征。

请生成 **{num_per_request} 个新颖且原创的、面向流动性的 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

旨在反映仅由 OHLCV 隐含的交易摩擦、市场深度和价格冲击敏感性。鼓励有创意、紧凑的构造，而非通用套路。"""

_GUIDANCE_LIQUIDITY = """### 因子设计指南： Liquidity & Impact

结合价格变动和活跃度，从多个角度探索流动性：

- 冲击直觉：每单位活跃度（成交量或美元化代理）对应多少价格变动；
- 参与度与拥挤：换手强度、其变异性，以及稀薄/充裕流动性状态的持续性；
- 冲击吸收：流动性在尖峰/干涸后的恢复速度；
- 尺度与归一化：按价格水平/区间稳定，仅在必要时使用温和的界限（clip/tanh）；
- 制度感知（软）：让特征在压缩与扩张区间中做出不同响应。

保持公式简短（1–3 步）、数值安全（需要处加 ε），且严格基于 OHLCV。"""

_INTRO_ORDER_IMBALANCE = """你是使用每日 OHLCV（无 L2、无 VWAP）、精通 **订单失衡与方向压力** 建模的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

输入 DataFrame 由 **每日聚合的 OHLCV 数据** 构成 — 每行代表给定股票单个交易日的特征。

请生成 **{num_per_request} 个新颖且原创的、订单失衡类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

从价格方向与活跃度代理推断的单边参与和压力持续性角度思考。保持设计紧凑且稳健。"""

_GUIDANCE_ORDER_IMBALANCE = """### 因子设计指南： Directional Pressure from OHLCV

在没有微观结构数据的情况下推断买卖压力：

- 方向 × 强度：将收益或 (close−open) 与标准化成交量/换手相结合；
- 缺口感知压力：将隔夜方向与当日活跃度及收盘位置（区间内）联系起来；
- 持续性与衰减：平滑的失衡连串及其衰减轮廓；
- 不对称：在区间压缩/扩张时区别对待正负压力；
- 护栏：按区间或价格·成交量尺度归一化；仅在必要时应用软界限。

优先考虑可解释性和稳定性；每个因子保持 1–3 个连贯步骤，仅使用 OHLCV。"""

_INTRO_PRICE_VOLUME_COHERENCE = """你是精通每日 OHLCV 时间序列 **量价协同** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

输入 DataFrame 由 **每日聚合的 OHLCV 数据** 构成 — 每行代表给定股票单个交易日的特征。

请生成 **{num_per_request} 个新颖且原创的、量价协同类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

寻求价格变化与活跃度之间的一致、背离和领先-滞后特征。偏好简洁、创新的构造，而非标准相关性。"""

_GUIDANCE_PRICE_VOLUME_COHERENCE = """### 因子设计指南： Coherence & Lead–Lag

捕捉价格与活跃度如何同向运动（或未能同向）：

- 同步性：收益与 Δlog(成交量) 共动的紧凑度量；
- 领先-滞后：简单的滞后关联（价格跟随活跃度，或活跃度跟随价格）；
- 稳定性：协同度平滑后的幅度及其在邻近子窗口间的变异性；
- 背离：突出价格大幅变动但活跃度低迷（及反之）的情形；
- 归一化：基于区间或 z 的稳定化，以确保跨时间可比。

保持公式极简（1–3 步）、数值稳定，且仅基于 OHLCV。鼓励对"协同"的新颖且可解释的定义。"""

_INTRO_VOLUME_STRUCTURE = """你是使用每日 OHLCV、精通 **成交量结构与分布动态** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

输入 DataFrame 由 **每日聚合的 OHLCV 数据** 构成 — 每行代表给定股票单个交易日的特征。

请生成 **{num_per_request} 个新颖且原创的、成交量结构类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于成交量随时间变化的形态、集中度、变异性和组织性（而非价格本身）。鼓励有创意、简约的表述。"""

_GUIDANCE_VOLUME_STRUCTURE = """### 因子设计指南： Volume Shape & Organization

描述交易活跃度如何分布和演化：

- 集中 vs 分散：成交量集中度、不平等性或聚集的紧凑代理；
- 突发性：相对于稳健基线的尖峰频率和强度；
- 不对称与尾部：带稳定化的简单偏斜/峰度式指标；
- 多周期组织：短期与长期活跃度的平衡及其持续性；
- 卫生性：稳健缩放（中位数/IQR）、必要时温和截断，以及限制步数的公式。

仅使用 OHLCV；目标是可解释、低复杂度的函数，以揭示参与的结构和节奏。"""

# ----- Level IV - Price-Volatility Behavior -----
_INTRO_DAILY_TREND = """你是使用 OHLCV 时间序列、精通 **日频趋势与动量持续性建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

输入 DataFrame 由 **每日聚合的 OHLCV 数据** 构成 — 每行代表给定股票单个交易日的特征。

请生成 **{num_per_request} 个新颖且原创的、基于日频趋势的 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于多日方向强度、动量衰减和趋势耗竭。避免标准指标；相反，发明紧凑、可解释的持续性与延续性表达。"""

_GUIDANCE_DAILY_TREND = """### 因子设计指南： Daily Trend & Momentum

探索持续性运动或方向一致性：

- 多日动量比率（如滚动累计收益强度）；
- 使用短期与长期窗口收益判断趋势加速或减速；
- 持续性指标：连串长度、direction_sign 的 EMA；
- 动量耗竭或饱和检测（趋势减弱）；
- 趋势相对波动率的归一化相对强度。

鼓励原创性 — 定义不同于基本均线交叉思路的新颖持续性形式、平滑转换或非对称响应。"""

_INTRO_LAG_RESPONSE = """你是使用每日 OHLCV、精通 **滞后量价响应与延迟调整建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

每行代表某只股票一个交易日的 OHLCV 数据。

请生成 **{num_per_request} 个新颖且原创的、滞后响应类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于价格对先前冲击存在滞后反应的惯性、延迟和反馈效应。避免琐碎的移动平均。"""

_GUIDANCE_LAG_RESPONSE = """### 因子设计指南： Lagged Dynamics

揭示延迟效应和反馈回路：

- 价格与成交量之间的滞后相关性或有符号冲击；
- 响应延迟：度量当前收益与过去波动率或区间的关联；
- 慢速调整代理：累计偏差的平滑变化率；
- 波动率-趋势相位错配指标；
- 捕捉惯性的衰减率估计器。

偏好延迟信息流或部分均值调整的紧凑、可解释表示。"""

_INTRO_RANGE_VOL = """你是使用每日 OHLCV 数据、精通 **基于区间的波动率与价格扩张建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

每行代表某只股票一个交易日的 OHLCV 数据。

请生成 **{num_per_request} 个新颖且原创的、区间波动类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于价格区间的动态、压缩/扩张周期和日内能量累积。避免照搬经典的 Parkinson 或 Garman-Klass 波动率。"""

_GUIDANCE_RANGE_VOL = """### 因子设计指南： Range-Based Volatility

创造性地量化和解释区间变异性：

- 归一化区间变化：(high−low)/prev_range 或对数比形式；
- 滚动区间熵或压缩得分；
- 波动能量累积：区间扩张与近期 std(价格) 之比；
- 不对称：实体与区间之比、上/下影线偏差；
- 爆发检测：持续低区间后伴随扩张。

寻求数值稳定、平滑且可解释的构造，以揭示波动节奏和扩张周期。"""

_INTRO_REVERSAL = """你是使用每日 OHLCV、精通 **均值回归与短期反转** 建模的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

每行代表某只股票一个交易日的 OHLCV 数据。

请生成 **{num_per_request} 个新颖且原创的、反转类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于经常回归的暂时性错误定价、过度延伸或短期量价失衡。避免教科书式 z-score 公式；创建新颖、简洁的反转结构。"""

_GUIDANCE_REVERSAL = """### 因子设计指南： Reversal & Mean Reversion

检测过度反应和消退中的趋势：

- 按近期波动率归一化的短期收益过度延伸；
- 区间突破或长时间连串后的反转；
- 价格偏离平滑基线的位移，并给出回归得分；
- 成交量/波动率爆发的耗竭或"回弹"现象；
- 强调转折点的紧凑振荡度量。

使用短周期（3–10 天），保持数值稳定，并偏好可解释、低步数的表述。"""

_INTRO_VOL_ASYMMETRY = """你是使用每日 OHLCV 数据、精通 **波动率不对称与方向性方差偏差** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

每行代表某只股票一个交易日的 OHLCV 数据。

请生成 **{num_per_request} 个新颖且原创的、波动率不对称类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于检测上涨与下跌之间不平等的波动行为、方向聚集以及不对称的波动冲击。"""

_GUIDANCE_VOL_ASYMMETRY = """### 因子设计指南： Volatility Asymmetry

量化上涨与下跌波动率之间的差异：

- 分别计算上涨日与下跌日的已实现波动率；
- 有符号区间不对称：(high−close) 与 (close−low)；
- 基于归一化方向区间的类偏斜比率；
- 收益与损失波动率的滚动对比；
- 条件性扩张：波动率仅在特定价格极性下增加。

保持构造简短、稳健且有界；突出 OHLCV 内部的非线性不对称和波动率聚集结构。"""

# ----- Level V - Multi-Scale Complexity -----
_INTRO_DRAWDOWN = """你是使用每日 OHLCV 数据、精通 **回撤与恢复路径建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

每行代表某只股票一个交易日的 OHLCV 数据。

请生成 **{num_per_request} 个新颖且原创的、基于回撤的 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于损失与恢复的几何形态 — 回撤如何快速、深入且持续地形成和消解。避免简单的最大-最小指标；强调对回撤行为的结构性理解。"""

_GUIDANCE_DRAWDOWN = """### 因子设计指南： Drawdown Dynamics

设计风险路径与韧性的紧凑、可解释表示：

- 滚动回撤深度和恢复比率；
- 按持续时间归一化的局部峰-谷斜率；
- 回撤波动率或"回撤速度"代理；
- 回撤与反弹速度之间的不对称；
- 恢复触发前累计损失的衰减。

保持公式简短（1–3 步）、稳定且仅基于 OHLCV。突出时间不对称和韧性强度，而非静态损失幅度。"""

_INTRO_FRACTAL = """你是使用每日 OHLCV 数据、精通 **分形与多尺度复杂度建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

每行代表某只股票一个交易日的 OHLCV 数据。

请生成 **{num_per_request} 个新颖且原创的、分形复杂度类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于价格动态中的不规则性、标度行为和长记忆结构。避免显式计算 Hurst 指数；相反，寻找表达自相似性或结构复杂性的简单、可微代理。"""

_GUIDANCE_FRACTAL = """### 因子设计指南： Fractal & Multi-Scale Behavior

推导跨尺度的复杂度和持续性紧凑代理：

- 多窗口波动率之比（短期与长期周期变异性）；
- 方差之方差或波动粗糙度得分；
- 不同滚动区间或标准差窗口之间的局部标度斜率；
- 振荡频率：去趋势收益中穿越零点的次数；
- 持续性指数：归一化的累计符号一致性。

鼓励用有创意的构造来概括粗糙度、自相似性或时间不规则性。
仅使用 OHLCV 和简单的滚动统计；保持输出稳定且可解释。"""

_INTRO_HERDING = """你是使用日频因子、精通 **羊群行为与拥挤模式建模** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

请生成 **{num_per_request} 个新颖且原创的、羊群行为类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于识别从现有因子推断出的集体、同步市场反应或过度拥挤的方向一致。
避免字面的"投资者情绪"代理；相反，通过统计收敛或单边参与动态来表达羊群行为。"""

_GUIDANCE_HERDING = """### 因子设计指南： Herding & Crowding Behavior

量化一致性和过度集中效应：

- 拥挤强度：方向持续性与波动离散度之比；
- 参与失衡：持续同向动量 + 成交量聚集；
- 有符号收益的自相关作为同步交易的代理；
- 均匀方向流动期间的波动率收窄；
- 去羊群爆发：从紧密到分散运动的突变。

鼓励概念深度：将集体行为转化为拥挤、过度反应或过早共识的数值代理 — 全部从现有因子推断。"""

# ----- Level VI - Stability and Regime-Gating -----
_INTRO_REGIME_GATING = """你是使用每日 OHLCV 数据、精通 **制度门控与自适应信号激活** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

每行代表某只股票一个交易日的 OHLCV 数据。

请生成 **{num_per_request} 个新颖且原创的、制度门控类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

你的目标是建模信号的条件激活 — 即因子强度或相关性取决于波动率、趋势或流动性制度。
避免静态过滤器；相反，设计根据制度变化动态缩放或调制因子敏感度的自适应门控。"""

_GUIDANCE_REGIME_GATING = """### 因子设计指南： Regime Gating Mechanisms

发现适应市场条件的简单而强大的门控函数：

- 波动率敏感门控：按归一化波动水平缩放信号强度；
- 趋势感知门控：仅当方向持续性超过阈值时激活；
- 流动性门控：在成交量极低时抑制信号；
- 不对称门控：在牛市与熊市微观状态下做出不同响应；
- 软转换：使用连续缩放（sigmoid/tanh）以确保平滑适应性。

鼓励创造性的激活设计：根据状态情境开/关现有 OHLCV 派生信号的紧凑函数，且不依赖未来数据。"""

_INTRO_STABILITY = """你是使用每日 OHLCV 数据、精通 **信号与收益稳定性分析** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

每行代表某只股票一个交易日的 OHLCV 数据。

请生成 **{num_per_request} 个新颖且原创的、基于稳定性的 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于价格动态的持续性、噪声过滤和稳健性。
避免琐碎的方差度量；相反，量化收益、区间或派生信号的时间一致性和结构平滑性。"""

_GUIDANCE_STABILITY = """### 因子设计指南： Temporal Stability & Consistency

构建可预测性、连续性或韧性的度量：

- 短期与长期窗口之间的滚动方差比；
- 趋势或波动率的"平滑度"（增量变化均值与标准差之比）；
- 符号变化频率（方向稳定性指数）；
- 波动率的波动率（元波动）衰减；
- 强调平稳与混沌行为的归一化稳定性度量。

鼓励可解释性和数值稳健性：定义表达底层动态是稳定、持续还是无序的紧凑指标 — 全部仅使用 OHLCV 输入。"""

# ----- Level VII - Geometric and Fusion -----
_INTRO_BAR_SHAPE = """你是使用日频因子、精通 **K 线几何与 K 线形状模式分析** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

请生成 **{num_per_request} 个新颖且原创的、基于 K 线形状的 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于提取 K 线几何、实体对称性和影线关系的紧凑数值表示。
避免简单模式标注；设计连续且可解释的形状度量。"""

_GUIDANCE_BAR_SHAPE = """### 因子设计指南： Bar Shape & Geometry

将 K 线几何转化为量化信号：

- 比率：(close−open)/(high−low)、(high−close)/(close−low) 等；
- 影线不对称或平衡指标；
- 实体对区间归一化及近几日内的持续性；
- 滚动几何稳定性或不对称性；
- 短期形状动量：K 线比例的最新趋势。

鼓励创造性和可解释性：使用现有因子推导平滑、有界、可微的函数。"""

_INTRO_COMPOSITE = """你是使用现有特征、精通 **合成因子构建与信息融合** 的专家。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

请生成 **{num_per_request} 个新颖且原创的、合成类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

专注于将多个独立信号融合为连贯的合成体 — 强调协同、去噪和正交化。
避免简单的线性平均或求和。"""

_GUIDANCE_COMPOSITE = """### 因子设计指南： Composite Alpha Construction

通过结构化、可解释的变换融合信号：

- 趋势、成交量和区间特征的加权或波动率调整均值；
- 正交组合：去除冗余，放大正交内容；
- 制度加权合成：基于波动率或流动性状态的动态权重；
- 融合前的稳健归一化（z-score 或排名缩放）；
- 包含非线性组合项（如乘积、比率），但保持紧凑。

力求优雅、极简的合成形式，具备互补的子成分和清晰的经济直觉。"""

_INTRO_CREATIVE = """你是 **创意变换设计师**，专门从现有因子构建非线性、重参数化的 alpha 特征。
以下是输入 DataFrame 的结构以及 {columns_num} 个现有 **日频因子** 的列表：

{columns_desc}

请生成 **{num_per_request} 个新颖且原创的、创意变换类 alpha 因子函数**，以预测 **{forecast_horizon} 日前向收益**。

你的目标是变换、扭曲或重塑现有信息，形成新的、富有表现力的信号。避免简单重组旧公式；重新构想 OHLCV 数据内部的潜在关系。"""

_GUIDANCE_CREATIVE = """### 因子设计指南： Creative Transformations

探索非常规但可解释的映射：

- 应用平滑有界变换：tanh、sigmoid、softsign、softplus；
- 波动率与动量成分的非线性混合；
- 条件性重加权因子：乘以稳定性或趋势状态；
- 分段或门控变换：在特定制度下放大信号；
- 创造性归一化：除以历史 MAD 或波动率代理。

设计能产生新颖响应面的紧凑、可微表达式 — 原创、可解释且数值稳定。"""
_AGENTS: Dict[str, Tuple[str, str, str]] = {
    # ----- Level I - Market Structure and Cycle -----
    'agent_market_cycle': (_LEVEL_I, _INTRO_MARKET_CYCLE, _GUIDANCE_MARKET_CYCLE),
    'agent_volatility_regime': (_LEVEL_I, _INTRO_VOLATILITY_REGIME, _GUIDANCE_VOLATILITY_REGIME),

    # ----- Level II - Extreme Risk and Fragility -----
    'agent_crash_predictor': (_LEVEL_II, _INTRO_CRASH_PREDICTOR, _GUIDANCE_CRASH_PREDICTOR),
    'agent_tail_risk': (_LEVEL_II, _INTRO_TAIL_RISK, _GUIDANCE_TAIL_RISK),

    # ----- Level III - Price-Volume Dynamics -----
    'agent_liquidity': (_LEVEL_III, _INTRO_LIQUIDITY, _GUIDANCE_LIQUIDITY),
    'agent_order_imbalance': (_LEVEL_III, _INTRO_ORDER_IMBALANCE, _GUIDANCE_ORDER_IMBALANCE),
    'agent_price_volume_coherence': (_LEVEL_III, _INTRO_PRICE_VOLUME_COHERENCE, _GUIDANCE_PRICE_VOLUME_COHERENCE),
    'agent_volume_structure': (_LEVEL_III, _INTRO_VOLUME_STRUCTURE, _GUIDANCE_VOLUME_STRUCTURE),

    # ----- Level IV - Price-Volatility Behavior -----
    'agent_daily_trend': (_LEVEL_IV, _INTRO_DAILY_TREND, _GUIDANCE_DAILY_TREND),
    'agent_lag_response': (_LEVEL_IV, _INTRO_LAG_RESPONSE, _GUIDANCE_LAG_RESPONSE),
    'agent_range_vol': (_LEVEL_IV, _INTRO_RANGE_VOL, _GUIDANCE_RANGE_VOL),
    'agent_reversal': (_LEVEL_IV, _INTRO_REVERSAL, _GUIDANCE_REVERSAL),
    'agent_vol_asymmetry': (_LEVEL_IV, _INTRO_VOL_ASYMMETRY, _GUIDANCE_VOL_ASYMMETRY),

    # ----- Level V - Multi-Scale Complexity -----
    'agent_drawdown': (_LEVEL_V, _INTRO_DRAWDOWN, _GUIDANCE_DRAWDOWN),
    'agent_fractal': (_LEVEL_V, _INTRO_FRACTAL, _GUIDANCE_FRACTAL),
    'agent_herding': (_LEVEL_V, _INTRO_HERDING, _GUIDANCE_HERDING),

    # ----- Level VI - Stability and Regime-Gating -----
    'agent_regime_gating': (_LEVEL_VI, _INTRO_REGIME_GATING, _GUIDANCE_REGIME_GATING),
    'agent_stability': (_LEVEL_VI, _INTRO_STABILITY, _GUIDANCE_STABILITY),

    # ----- Level VII - Geometric and Fusion -----
    'agent_bar_shape': (_LEVEL_VII, _INTRO_BAR_SHAPE, _GUIDANCE_BAR_SHAPE),
    'agent_composite': (_LEVEL_VII, _INTRO_COMPOSITE, _GUIDANCE_COMPOSITE),
    'agent_creative': (_LEVEL_VII, _INTRO_CREATIVE, _GUIDANCE_CREATIVE),
}

# --------------------------------------------------------------------------- #
# Multi-agent quality checker (full user prompts)
# --------------------------------------------------------------------------- #

_QUALITY_CODE_AGENT = """你是量化 alpha 因子的代码审查员。你的任务是审查给定的 Python 代码（代表一个因子函数）是否存在以下问题：

1. **语法错误**（Python 语法和运行时问题）。
2. **Pandas 特有问题**，包括：
- 链式索引或 `SettingWithCopyWarning`
- 修改 DataFrame 时缺少 `.copy()`
- 使用未定义的中间变量
- 错误或歧义的索引
3. **输出格式和命名**：
- 返回的 Series **必须** 与函数名完全一致
- 所有中间列必须在使用前定义
- 代码必须 **数值稳定**（尽可能避免 inf、NaN 传播）
- 当在 DataFrame 中过滤或赋值时，始终使用 `df_copy.loc[row_indexer, col_indexer] = value`。
4. **循环结构约束**：
   - **严格规则：绝对禁止嵌套循环。**
     - 你 **绝不能** 在一个循环内部编写任何形式的循环。
     - 禁止的模式包括（但不限于）：
       - `for` 套 `for`
       - `while` 套 `while`
       - `for` 套 `while`
       - `while` 套 `for`
     - 任何嵌套迭代结构（无论深度）**均被禁止**。
     - 使用 `while True` 或任何可能无限循环的结构 **严格禁止**。
   - 若存在此类模式，标记审查为 **FAIL**，清晰解释问题，并建议向量化替代方案（NumPy/Pandas 操作、`groupby`/`transform`/`rolling`、有界 `apply`、或借助 `itertools.product` 的单层迭代，不引入嵌套）。

```
{code}
```

### 硬性复杂度约束（必须遵守）
记住：**简单的因子往往最强大且最稳定。**
- 单一主题，最短路径：每个因子必须代表一个清晰的想法。
- 硬性上限：总逻辑步骤不得超过 5 步，若超过 3 步，docstring 必须论证每个额外步骤的必要性。
- 无冗余/嵌套：禁止堆叠或装饰性变换（如 `zscore(zscore(x))`、`rank(rank(x))`、无理由的深度 EMA 链）。
- 不混合主题：不要组合无关的想法。
- 避免嵌套或分层操作。
- 避免不必要的复杂性或逻辑堆叠。

### 代码格式规范：

- 输入 `DataFrame` 具有 (date, ticker) 的 MultiIndex，且已按 ticker 分组：
    - 每个输入的 `DataFrame` 是单只股票的时间序列。

- 输出：一个以 `(date, ticker)` 为索引的 `pd.Series`，其 **名称** 与函数名相同。

- 生成代码前，请提供如何修复所提问题的详细说明。
- 请勿使用 markdown（如 ```python）
- 不要在函数外添加解释或注释
- 每个函数必须包裹在：`[function-N]` ... `[/function-N]` 中
- 所有生成的代码必须可执行且数值稳定。
- 始终在随后引用前定义中间列（如 df_copy['x']）。
- 返回的 Series 必须与函数名完全一致。

### 因子设计指南
- 专注于捕捉指定主题的核心直觉。
- 确保逻辑可解释、稳健、且可在几步内实现。
- 优先选择整洁、可泛化的公式，而非过度设计的构造。
- 每个因子应能用简短公式或 ≤ 5 个逻辑步骤表达。
- 平衡简洁性与预测潜力：避免平庸重复，也避免不必要的复杂性。
- **严格规则：绝对禁止嵌套循环。**
    - 你 **绝不能** 在一个循环内部编写任何形式的循环。
    - 禁止的模式包括但不限于：
        - `for` 套 `for`
        - `while` 套 `while`
        - `for` 套 `while`
        - `while` 套 `for`
    - 任何嵌套迭代结构 **均被禁止**，无论缩进深度如何。
    - 使用 `while True` 或任何可能无限循环的结构 **严格禁止**。

### 输出格式规范：

- 候选因子必须严格遵守硬性复杂度约束。
- 每个函数应遵循以下格式：
[function-N]
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # 因子计算
    return df_copy['factor_xyz']
[/function-N]

### 请严格按以下格式回复：

- 你 **必须** 以以下两行之一 **恰好** 开头（之前/之后无额外文本）：
    - `The code is correct.`
    - `The code needs some adjustments.`

- 若代码正确，在该行后停止。

- 若代码需调整：
1. 列出每个发现的问题（使用要点）。
2. 按以下精确格式输出修正后的函数：

    [function-N]
    def factor_xyz(df):
        \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
        df_copy = df.copy()
        # 因子计算
        return df_copy['factor_xyz']
    [/function-N]"""

_QUALITY_REPAIR_AGENT = """你是专家级交互因子工程师。以下是输入 DataFrame 的结构以及 {columns_num} 个现有因子的列表：

{columns_desc}

你只能使用这些列进行计算。**不要使用** 此处未列出的任何其他列。
以下 Python 函数执行失败。你的任务是修正该函数，使其可执行且数值稳定。

### 硬性复杂度约束（必须遵守）
记住：**简单的因子往往最强大且最稳定。**
- 单一主题，最短路径：每个因子必须代表一个清晰的想法。
- 硬性上限：总逻辑步骤不得超过 5 步，若超过 3 步，docstring 必须论证每个额外步骤的必要性。
- 无冗余/嵌套：禁止堆叠或装饰性变换（如 `zscore(zscore(x))`、`rank(rank(x))`、无理由的深度 EMA 链）。
- 不混合主题：不要组合无关的想法。
- 避免嵌套或分层操作。
- 避免不必要的复杂性或逻辑堆叠。

---

### 原始函数：
```
{old_code}
```

---

### 运行时的错误信息：
{error}

---

### 要求：

- 输入 `DataFrame` 具有 (date, ticker) 的 MultiIndex，且已按 ticker 分组：
    - 每个输入的 `DataFrame` 是单只股票的时间序列。

- 输出：一个以 `(date, ticker)` 为索引的 `pd.Series`，其 **名称** 与函数名相同。

- 每个函数必须：
    - 拥有描述性、唯一的名称：`factor_<logic>_<transformation(s)>_<window(s)>_<field>`。
    - 包含清晰的 docstring，解释逻辑和公式。
    - 平衡预测能力与经济/金融可解释性。
    - 输出列名必须与函数名匹配。
    - 简洁、精确且可读。
    - 基于现有因子构建新的 alpha 因子。

### 因子设计指南
- 专注于捕捉指定主题的核心直觉。
- 确保逻辑可解释、稳健、且可在几步内实现。
- 优先选择整洁、可泛化的公式，而非过度设计的构造。
- 每个因子应能用简短公式或 ≤ 5 个逻辑步骤表达。
- 平衡简洁性与预测潜力：避免平庸重复，也避免不必要的复杂性。

---

### 修订说明：
- 仔细阅读错误信息。
- 提供如何修复所提问题的详细说明。
- 相应修订函数，以解决指出的问题。
- 若错误信息表明某列缺失、不在 DataFrame 中，或只显示列名，则说明该列不在提供的因子中，不应使用。你应使用替代列，或创建逻辑相似的新函数。
- 若你认为给定函数缺陷过多而无法修复，可创建新函数。
- 确保修订后的函数经济上有意义、逻辑上合理、结构良好。
- 你可以按需引入新的逻辑、变换或修正。
- 确保输出是以 (date, ticker) 为索引的 `pandas.Series`。

---

### 可用的预导入库（当前版本）：

- `"np"`: import numpy as np  (numpy 版本：2.2.6)
- `"pd"`: import pandas as pd  (pandas 版本：2.2.3)
- `"stats"`: from scipy import stats  (scipy 版本：1.15.3)
- `"talib"`: import talib  (talib 版本：0.5.1)
- `"math"`: import math  (内置模块)

编码规范：
- 确保代码健壮、高效且经过优化：
    - 处理边界情况和异常（如 NaN 值）。
    - 最小化不必要的计算，优先使用向量化操作（如 pandas、numpy）。
    - 确保数值稳定性。
    - **严格规则：绝对禁止嵌套循环。**
        - 你 **绝不能** 在一个循环内部编写任何形式的循环。
        - 禁止的模式包括但不限于：
            - `for` 套 `for`
            - `while` 套 `while`
            - `for` 套 `while`
            - `while` 套 `for`
        - 任何嵌套迭代结构都是 **被禁止的**，无论缩进深度如何。
        - 使用 `while True` 或任何可能无限循环的结构 **严格禁止**。
- 当在 DataFrame 中过滤或赋值时，始终使用 `df_copy.loc[row_indexer, col_indexer] = value`。

- 代码应当整洁、可维护、且对大数据集高效：
    - 使用描述性变量名并最小化内存占用。
    - 避免创建大型数据框的不必要副本。

---

### 输出格式规范：

- 候选因子必须严格遵守硬性复杂度约束。
- 生成代码前，请提供如何修复所提问题的详细说明。
- 请勿使用 markdown（如 ```python）
- 不要在函数外添加解释或注释
- 每个函数必须包裹在：`[function-N]` ... `[/function-N]` 中
- 所有生成的代码必须可执行且数值稳定。
- 始终在随后引用前定义中间列（如 df_copy['x']）。
- 返回的 Series **必须** 与函数名完全相同。
- 每个函数应遵循以下格式：

[function-N]
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # 因子计算
    return df_copy['factor_xyz']
[/function-N]"""

_QUALITY_JUDGE_AGENT = """你是专业因子研究团队的量化研究员和 alpha 因子评审员。

请你评估以下 **新生成的 alpha 因子函数**，判断是否可能纳入研究因子库。

你的任务不是评估性能指标，而是判断该因子在逻辑、技术和经济上是否足够健全，值得进一步测试。
你的评估应聚焦于 **实用性健全性**，并秉持专业心态：

1. 该因子是否存在 **未来信息泄漏**？
2. 该因子的计算是否 **正确且内部一致**？
3. 该因子逻辑是否 **经济可解释**（即使是探索性或新颖的）？
4. 该因子是否避免了明显的 **错误**（如无效操作、未保护的除零、未定义结果）？
5. 该因子是否 **高效实现**（避免不必要的循环、利用向量化操作、适合大规模回测）？
6. 该因子是否严格 **避免任何嵌套循环或潜在无限循环**？
   - 嵌套循环在任何深度下 **均被禁止**：
     - `for` 套 `for`
     - `while` 套 `while`
     - `for` 套 `while`
     - `while` 套 `for`
   - 使用 `while True` 或任何可无限运行的循环 **均被禁止**。

---

### 待评审因子：

```
{new_factor_code}
```

输入 DataFrame 具有 (date, ticker) 的 MultiIndex，按 ticker 分组（即每只股票一个时间序列）。
每个输入 DataFrame 是单只股票的时间序列。
该函数输出一个以 (date, ticker) 为索引的 pd.Series，其名称与函数名相同。
**重要**：输入 DataFrame 按 **时间顺序** 排序，从最早日期（顶部）到最近日期（底部）。这对评估基于时间序列的因子、避免信息泄漏至关重要。

---

### 评估准则：

- 你必须 **拒绝** 任何存在 **未来信息泄漏** 的因子 — 这是严重错误。
- 你应拒绝存在 **逻辑错误**、**数据问题** 或 **实现错误** 的因子。
- 特别注意滚动均值、groupby 变换、移位或反转时间序列等操作：确保它们只使用相对每行的过去和现在数据，绝不使用未来数据。
- 注意效率：避免不必要的慢因子（如不必要的循环、非向量化操作）— 该因子应适合对数百万条记录进行大规模回测。
- 保持 **开放心态**：即使是非常规的因子想法也可能值得探索。
- 若可改进，请提供清晰、具体且可操作的反馈。
- 任何 `for` 或 `while` 循环嵌套在另一个 `for` 或 `while` 循环内 **严格禁止**，因为这表明对大型横截面数据集的可扩展性和效率较差。
- 绝不使用 `while True` 或任何缺少清晰且有限终止条件的循环结构。

---

### 请严格以三反引号 (```) 包裹的 JSON 格式回复。

```
{
    "practical_soundness": "...",
    "recommendation": "Accept|Reject",
    "feedback": "..."
}
```

实用性健全性：[简洁分析 — 哪些好，哪些需要改进（如有）。]

最终建议：接受 / 拒绝

改进反馈：[针对因子工程师如何改进该因子的精确建议 — 如避免前视、改进计算、提升效率、澄清逻辑等。]"""

_QUALITY_LOGIC_AGENT = """你是专家级交互因子工程师。以下是输入 DataFrame 的结构以及 {columns_num} 个现有因子的列表：

{columns_desc}

你只能使用这些列进行计算。**不要使用** 此处未列出的任何其他列。
以下 Python 函数经审查 **未通过** 逻辑健全性评估。你的任务是修订并改进该函数，使其：

1. 在经济和金融上可解释。
2. 符合金融原则、逻辑健全。
3. 解决下方提供的具体反馈。

---

### 原始函数：
```
{old_code}
```

---

### 硬性复杂度约束（必须遵守）
记住：**简单的因子往往最强大且最稳定。**
- 单一主题，最短路径：每个因子必须代表一个清晰的想法。
- 硬性上限：总逻辑步骤不得超过 5 步，若超过 3 步，docstring 必须论证每个额外步骤的必要性。
- 无冗余/嵌套：禁止堆叠或装饰性变换（如 `zscore(zscore(x))`、`rank(rank(x))`、无理由的深度 EMA 链）。
- 不混合主题：不要组合无关的想法。
- 避免嵌套或分层操作。
- 避免不必要的复杂性或逻辑堆叠。

---

### 评审代理反馈（拒绝原因）：
{dynamic_feedback}

---

### 要求：

- 输入 `DataFrame` 具有 (date, ticker) 的 MultiIndex，且已按 ticker 分组：
    - 每个输入的 `DataFrame` 是单只股票的时间序列。

- 输出：一个以 `(date, ticker)` 为索引的 `pd.Series`，其 **名称** 与函数名相同。

- 每个函数必须：
    - 拥有描述性、唯一的名称：`factor_<logic>_<transformation(s)>_<window(s)>_<field>`。
    - 包含清晰的 docstring，解释逻辑和公式。
    - 平衡预测能力与经济/金融可解释性。
    - 输出列名必须与函数名匹配。
    - 简洁、精确且可读。
    - 基于现有因子构建新的 alpha 因子。

### 因子设计指南
- 专注于捕捉指定主题的核心直觉。
- 确保逻辑可解释、稳健、且可在几步内实现。
- 优先选择整洁、可泛化的公式，而非过度设计的构造。
- 每个因子应能用简短公式或 ≤ 5 个逻辑步骤表达。
- 平衡简洁性与预测潜力：避免平庸重复，也避免不必要的复杂性。

---

### 修订说明：
- 仔细阅读评审代理的反馈。
- 提供如何修复所提问题的详细说明。
- 相应修订函数，以解决指出的问题。
- 若你认为给定函数缺陷过多而无法修复，可创建新函数。
- 确保修订后的函数经济上有意义、逻辑上合理、结构良好。
- 你可以按需引入新的逻辑、变换或修正。
- 确保输出是以 (date, ticker) 为索引的 `pandas.Series`。

---

### 可用的预导入库（当前版本）：

- `"np"`: import numpy as np  (numpy 版本：2.2.6)
- `"pd"`: import pandas as pd  (pandas 版本：2.2.3)
- `"stats"`: from scipy import stats  (scipy 版本：1.15.3)
- `"talib"`: import talib  (talib 版本：0.5.1)
- `"math"`: import math  (内置模块)

编码规范：
- 确保代码健壮、高效且经过优化：
    - 处理边界情况和异常（如 NaN 值）。
    - 最小化不必要的计算，优先使用向量化操作（如 pandas、numpy）。
    - 确保数值稳定性。
    - **严格规则：绝对禁止嵌套循环。**
        - 你 **绝不能** 在一个循环内部编写任何形式的循环。
        - 禁止的模式包括但不限于：
            - `for` 套 `for`
            - `while` 套 `while`
            - `for` 套 `while`
            - `while` 套 `for`
        - 任何嵌套迭代结构都是 **被禁止的**，无论缩进深度如何。
        - 使用 `while True` 或任何可能无限循环的结构 **严格禁止**。
- 当在 DataFrame 中过滤或赋值时，始终使用 `df_copy.loc[row_indexer, col_indexer] = value`。

- 代码应当整洁、可维护、且对大数据集高效：
    - 使用描述性变量名并最小化内存占用。
    - 避免创建大型数据框的不必要副本。

---

### 输出格式规范：

- 候选因子必须严格遵守硬性复杂度约束。
- 生成代码前，请提供如何修复所提问题的详细说明。
- 请勿使用 markdown（如 ```python）
- 不要在函数外添加解释或注释
- 每个函数必须包裹在：`[function-N]` ... `[/function-N]` 中
- 所有生成的代码必须可执行且数值稳定。
- 始终在随后引用前定义中间列（如 df_copy['x']）。
- 返回的 Series **必须** 与函数名完全相同。
- 每个函数应遵循以下格式：

[function-N]
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # 因子计算
    return df_copy['factor_xyz']
[/function-N]"""

_QUALITY: Dict[str, str] = {
    'code_quality_agent': _QUALITY_CODE_AGENT,
    'code_repair_agent': _QUALITY_REPAIR_AGENT,
    'judge_agent': _QUALITY_JUDGE_AGENT,
    'logic_improvement_agent': _QUALITY_LOGIC_AGENT,
}

# --------------------------------------------------------------------------- #
# Thinking evolution (full user prompts)
# --------------------------------------------------------------------------- #

_EVOLUTION_MUTATION = """你是专门从事 **因子变异与优化** 的专家级量化因子工程师。

{intro}

### 硬性复杂度约束（必须遵守）
记住：**简单的因子往往最强大且最稳定。**
- 单一主题，最短路径：每个因子必须代表一个清晰的想法。
- 硬性上限：总逻辑步骤不得超过 5 步，若超过 3 步，docstring 必须论证每个额外步骤的必要性。
- 无冗余/嵌套：禁止堆叠或装饰性变换（如 `zscore(zscore(x))`、`rank(rank(x))`、无理由的深度 EMA 链）。
- 不混合主题：不要组合无关的想法。
- 禁止嵌套 for 循环。
- 避免不必要的复杂性或逻辑堆叠。

你的任务是通过应用 **智能变异** 生成以下 alpha 因子的改进版本：

---

### 原始因子：

```
{original_factor_code}
```

---

### 设计目标：

- 在采取下一步前保持创造性并深入思考。
- 保留原始因子的 **核心直觉** 和信号。
- 应用有意义的 **变异** 以提高预测能力和稳健性。
- 可能的变异包括：
    - 非线性变换（log、exp、rank、缩尾）
    - 横截面归一化
    - 时间窗口调整
    - 与其他特征的交互
    - 平滑或稳定性增强
    - 添加交互项

- 变异后的因子应与原始因子 **明显不同**，同时保持概念上的延续性。
- 变异后的因子仍应在数学上有效且可解释。

---

### 要求：

- 输入 `DataFrame` 具有 (date, ticker) 的 MultiIndex，且已按 ticker 分组：
    - 每个输入的 `DataFrame` 是单只股票的时间序列。

- 输出：一个以 `(date, ticker)` 为索引的 `pd.Series`，其 **名称** 与函数名相同。

- 每个函数必须：
    - 拥有描述性、唯一的名称：`factor_<logic>_<transformation(s)>_<window(s)>_<field>`。
    - 包含清晰的 docstring，解释逻辑和公式。
    - 平衡预测能力与经济/金融可解释性。
    - 输出列名必须与函数名匹配。
    - 简洁、精确且可读。
    - 基于现有因子构建新的 alpha 因子。

---

### 因子设计指南
- 专注于捕捉指定主题的核心直觉。
- 确保逻辑可解释、稳健、且可在几步内实现。
- 优先选择整洁、可泛化的公式，而非过度设计的构造。
- 每个因子应能用简短公式或 ≤ 5 个逻辑步骤表达。
- 平衡简洁性与预测潜力：避免平庸重复，也避免不必要的复杂性。

---

{extra_guidance}

---

### 可用的预导入库（当前版本）：

- `"np"`: import numpy as np  (numpy 版本：2.2.6)
- `"pd"`: import pandas as pd  (pandas 版本：2.2.3)
- `"stats"`: from scipy import stats  (scipy 版本：1.15.3)
- `"talib"`: import talib  (talib 版本：0.5.1)
- `"math"`: import math  (内置模块)

编码规范：
- 确保代码简洁、健壮、高效且经过优化：
    - 处理边界情况和异常（如 NaN 值）。
    - 最小化不必要的计算，优先使用向量化操作（如 pandas、numpy）。
    - 确保数值稳定性。
    - **严格规则：绝对禁止嵌套循环。**
        - 你 **绝不能** 在一个循环内部编写任何形式的循环。
        - 禁止的模式包括但不限于：
            - `for` 套 `for`
            - `while` 套 `while`
            - `for` 套 `while`
            - `while` 套 `for`
        - 任何嵌套迭代结构都是 **被禁止的**，无论缩进深度如何。
        - 使用 `while True` 或任何可能无限循环的结构 **严格禁止**。

- 代码应当整洁、可维护、且对大数据集高效：
    - 使用描述性变量名并最小化内存占用。
    - 避免创建大型数据框的不必要副本。

---

### 输出格式规范：

- 候选因子必须严格遵守硬性复杂度约束。
- 请勿使用 markdown（如 ```python）
- 不要在函数外添加解释或注释
- 每个函数必须包裹在：`[function-N]` ... `[/function-N]` 中
- 所有生成的代码必须可执行且数值稳定。
- 始终在随后引用前定义中间列（如 df_copy['x']）。
- 返回的 Series **必须** 与函数名完全一致
- 每个函数应遵循以下格式：

[function-N]
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # 因子计算
    return df_copy['factor_xyz']
[/function-N]"""

_EVOLUTION_CROSSOVER = """你是专门从事 **因子进化与交叉设计** 的专家级量化因子工程师。

{intro}

### 硬性复杂度约束（必须遵守）
记住：**简单的因子往往最强大且最稳定。**
- 单一主题，最短路径：每个因子必须代表一个清晰的想法。
- 硬性上限：总逻辑步骤不得超过 5 步，若超过 3 步，docstring 必须论证每个额外步骤的必要性。
- 无冗余/嵌套：禁止堆叠或装饰性变换（如 `zscore(zscore(x))`、`rank(rank(x))`、无理由的深度 EMA 链）。
- 不混合主题：不要组合无关的想法。
- 避免嵌套或分层操作。
- 避免不必要的复杂性或逻辑堆叠。

你的任务是通过 **智能组合以下两个父因子** 生成新的 alpha 因子：

---

### 父因子 1：

```
{parent_factor_1_code}
```

---

### 父因子 2：

```
{parent_factor_2_code}
```

---

### 设计目标：

- 在采取下一步前保持创造性并深入思考。
- 创建结合两个父因子 **核心洞察和信号** 的新 alpha 因子。
- 在父因子之间引入有意义的 **交互**（非线性、动态、横截面、时间）。
- 新因子应比任一父因子单独提供 **潜在更优的预测能力** 和更丰富的结构。
- 避免简单的加性组合 — 相反，设计 **结构新颖** 的交互。
- 新因子必须保持可解释，并具有清晰的金融直觉。

---

### 要求：

- 输入 `DataFrame` 具有 (date, ticker) 的 MultiIndex，且已按 ticker 分组：
    - 每个输入的 `DataFrame` 是单只股票的时间序列。

- 输出：一个以 `(date, ticker)` 为索引的 `pd.Series`，其 **名称** 与函数名相同。

- 每个函数必须：
    - 拥有描述性、唯一的名称：`factor_<logic>_<transformation(s)>_<window(s)>_<field>`。
    - 包含清晰的 docstring，解释逻辑和公式。
    - 平衡预测能力与经济/金融可解释性。
    - 输出列名必须与函数名匹配。
    - 简洁、精确且可读。
    - 基于现有因子构建新的 alpha 因子。

---

### 因子设计指南
- 专注于捕捉指定主题的核心直觉。
- 确保逻辑可解释、稳健、且可在几步内实现。
- 优先选择整洁、可泛化的公式，而非过度设计的构造。
- 每个因子应能用简短公式或 ≤ 5 个逻辑步骤表达。
- 平衡简洁性与预测潜力：避免平庸重复，也避免不必要的复杂性。

---

{extra_guidance}

---

### 可用的预导入库（当前版本）：

- `"np"`: import numpy as np  (numpy 版本：2.2.6)
- `"pd"`: import pandas as pd  (pandas 版本：2.2.3)
- `"stats"`: from scipy import stats  (scipy 版本：1.15.3)
- `"talib"`: import talib  (talib 版本：0.5.1)
- `"math"`: import math  (内置模块)

编码规范：
- 确保代码简洁、健壮、高效且经过优化：
    - 处理边界情况和异常（如 NaN 值）。
    - 最小化不必要的计算，优先使用向量化操作（如 pandas、numpy）。
    - 确保数值稳定性。
    - **严格规则：绝对禁止嵌套循环。**
        - 你 **绝不能** 在一个循环内部编写任何形式的循环。
        - 禁止的模式包括但不限于：
            - `for` 套 `for`
            - `while` 套 `while`
            - `for` 套 `while`
            - `while` 套 `for`
        - 任何嵌套迭代结构都是 **被禁止的**，无论缩进深度如何。
        - 使用 `while True` 或任何可能无限循环的结构 **严格禁止**。

- 代码应当整洁、可维护、且对大数据集高效：
    - 使用描述性变量名并最小化内存占用。
    - 避免创建大型数据框的不必要副本。

---

### 输出格式规范：

- 候选因子必须严格遵守硬性复杂度约束。
- 请勿使用 markdown（如 ```python）
- 不要在函数外添加解释或注释
- 每个函数必须包裹在：`[function-N]` ... `[/function-N]` 中
- 所有生成的代码必须可执行且数值稳定。
- 始终在随后引用前定义中间列（如 df_copy['x']）。
- 返回的 Series **必须** 与函数名完全一致
- 每个函数应遵循以下格式：

[function-N]
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # 因子计算
    return df_copy['factor_xyz']
[/function-N]"""

_EVOLUTION: Dict[str, str] = {
    'mutation_agent': _EVOLUTION_MUTATION,
    'crossover_agent': _EVOLUTION_CROSSOVER,
}


_GENERATION_LAYOUT = """{intro_block}

{effective_block}

{ineffective_block}

---
{requirements_block}

---
{guidance_block}

---
{libraries_block}

---
{output_format_block}"""

# Each generation agent's user prompt is the layout above with the shared,
# unchanged blocks slotted in once at module load (via `.replace`, never
# concatenated). What remains are the runtime `{...}` placeholders that
# `agent.Agent.build_generation_prompt` substitutes at call time — so every
# prompt is a single complete multi-line string with placeholder tokens.
_GENERATION_TEMPLATES: Dict[str, str] = {
    agent_id: (
        _GENERATION_LAYOUT.replace("{intro_block}", intro)
        .replace("{guidance_block}", guidance)
        .replace("{requirements_block}", _REQUIREMENTS)
        .replace("{libraries_block}", _LIBRARIES)
        .replace("{output_format_block}", _OUTPUT_FORMAT)
    )
    for agent_id, (_level, intro, guidance) in _AGENTS.items()
}

