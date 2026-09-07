"""Prompt templates for CogAlpha, embedded directly (no external files).

All prompts from `prompts/` are inlined here so the package is self-contained.
Every prompt is a plain multi-line string whose dynamic parts are written as
`{placeholder}` tokens — nothing is assembled at call time and nothing uses
`str.format`. Substitution happens with `.replace(token, value)`.

The public surface mirrors the original loader so downstream modules are
unchanged:

  - shared block attributes: system_message, requirements, libraries,
    output_format, base_factor_guidance, column_description,
    effective_analysis, ineffective_analysis, effective_summary,
    ineffective_summary, guidance_paraphrase
  - `agents`: dict[agent_id, (level, intro, guidance)] for the 21 seven-level
    generation agents
  - `quality_templates` / `evolution_templates`: full user prompts for the
    quality-checker and thinking-evolution agents
  - methods: build_generation_prompt / build_quality_prompt / build_evolution_prompt

A `prompts_dir` argument is accepted for backward compatibility but ignored —
the templates live in this module.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# --------------------------------------------------------------------------- #
# Shared blocks
# --------------------------------------------------------------------------- #
_SYSTEM_MESSAGE = """You are an expert quantitative researcher and financial feature engineer."""

_REQUIREMENTS = """### Requirements:

- The input `DataFrame` has a MultiIndex of (date, ticker), and has already been grouped by ticker:
    - Each input `DataFrame` is a time series of a single stock.

- Output: A `pd.Series` indexed by `(date, ticker)` with the **same name** as the function.

- Each function must:
    - Have a descriptive, unique name: `factor_<logic>_<transformation(s)>_<window(s)>_<field>`.
    - Include a clear docstring explaining the logic and formula.
    - Balance predictive power with economic/financial interpretability.
    - The output column name must match the function name.
    - Be concise, precise, and readable.
    - Build new alpha factors based on existing ones."""

_LIBRARIES = """### Pre-imported libraries you can use (current versions):

- `"np"`: import numpy as np  (numpy version: 2.2.6)
- `"pd"`: import pandas as pd  (pandas version: 2.2.3)
- `"stats"`: from scipy import stats  (scipy version: 1.15.3)
- `"talib"`: import talib  (talib version: 0.5.1)
- `"math"`: import math  (built-in module)

Coding Guidelines:
- Ensure the code is robust, efficient, and optimized:
    - Handle edge cases and exceptions (e.g., NaN values).
    - Minimize unnecessary computations and prefer vectorized operations (e.g., pandas, numpy).
    - Ensure numerical stability.
    - **Strict Rule: Nested loops are absolutely forbidden.**
        - You must **never** write any form of loop inside another loop.
        - Forbidden patterns include but are not limited to:
            - `for` inside `for`
            - `while` inside `while`
            - `for` inside `while`
            - `while` inside `for`
        - Any nested iteration structure is **prohibited**, regardless of indentation depth.
        - The use of `while True` or any potentially infinite loop is **strictly prohibited**.
- When filtering or assigning values in a DataFrame, always use `df_copy.loc[row_indexer, col_indexer] = value`.

- Code should be clean, maintainable, and efficient for large datasets:
    - Use descriptive variable names and minimize memory usage.
    - Avoid creating unnecessary copies of large dataframes."""

_OUTPUT_FORMAT = """### Output format specification:

- Do NOT use markdown (like ```python)
- Do NOT add explanation or comments outside the function
- Each function must be wrapped inside: `<<function N>>` ... `<</function N>>`
- All generated code must be executable and numerically stable.
- Always define intermediate columns (e.g. df_copy['x']) before referencing them later.
- The returned Series **must be named exactly the same as the function name**.
- Each function should follow this format:

<<function N>>
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # factor computation
    return df_copy['factor_xyz']
<</function N>>"""

_BASE_FACTOR_GUIDANCE = """### Factor Design Guidance:

You are encouraged to explore a wide variety of signals and techniques related to {factor_type}, including but not limited to:

- [List of common techniques / example categories]
- [List of possible interactions or advanced ideas]

Please do NOT limit yourself to simple formulas or common patterns.
You are expected to innovate, introduce mathematically sophisticated or unconventional structures, and combine multiple concepts where reasonable.

The goal is to generate factors that are **predictive**, **robust**, and **economically interpretable**, while being **structurally diverse** from existing factors."""

_COLUMN_DESCRIPTION = """For each of the following financial factor names, provide a precise and concise English description (no more than 20 words).
IMPORTANT: Do not modify the factor names in any way. Use the exact same name as input.
Return your answer in the format:
<exact_factor_name>: <description>

{factor_names}"""

_EFFECTIVE_ANALYSIS = """---
### Analysis of Effective Factors and Innovation Directions:
Below is a condensed CoT-style summary built from recent successful cases and why they work well.
Mini-Chain from Survivors  (Observation → Cause → Fix):
{effective_CoT}

Based on these strengths, use them as heuristic inspiration to guide new factor creation, rather than copying the original principles.
Seek innovative methods to generate more efficient, robust, and adaptable factors, ensuring they work well in diverse market conditions while avoiding look-ahead/leakage and redundancy."""

_INEFFECTIVE_ANALYSIS = """---
### Analysis of Ineffective Factors and Innovation Directions
Below is a condensed CoT-style summary built from recent failure cases and why they fail.
Mini-Chain from Failures (Observation → Cause → Fix):
{ineffective_CoT}

Based on these failures, use them as heuristic warnings to inform new factor creation, rather than simply avoiding similar issues.
Seek innovative methods to generate more effective, robust, and adaptable factors, ensuring they work well in diverse market conditions."""

_EFFECTIVE_SUMMARY = """You are given several factor functions in the format:
            <<factor N>>
            State: valid
            Metrics: IC / RankIC / ICIR / RankICIR
            Code:
            <<function N>>
                def <factor_name>(df):
                    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
                    df_copy = df.copy()
                    # factor computation
                    return df_copy['<factor_name>']
            <</function N>>
            <</factor N>>

            You are tasked with analyzing the given financial factors. For each factor, provide the following in a clear and structured format:
            1. **One Clear Idea**: one sentence stating the core intuition only.
            2. **Short Formula**: a single one-line math/pseudocode expression in backticks that represents the factor (no comments, no extra code)
            3. **Efficiency Analysis**: In one sentence, explain why this factor is likely to be effective in real-world financial models (e.g., solid economic intuition, robustness across regimes, low redundancy, and no look-ahead/leakage).
            IMPORTANT: Do not modify the factor names or codes in any way. Use the exact same name as input.
            Return your answer in the following format:
            <factor_name>:
                **One Clear Idea**: <description>
                **Short Formula**: `<one-line formula>`
                **Efficiency Analysis**: <analysis>

{factor_names}

{factor_examples}"""

_INEFFECTIVE_SUMMARY = """You are given several factor functions in the format:
            <<factor N>>
            State: low_metrics / dependent / unstable
            Metrics: IC / RankIC / ICIR / RankICIR
            Code:
            <<function N>>
                def <factor_name>(df):
                    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
                    df_copy = df.copy()
                    # factor computation
                    return df_copy['<factor_name>']
            <</function N>>
            <</factor N>>

            You are tasked with analyzing the given financial factors. For each factor, provide the following in a clear and structured format:
            1. **One Clear Idea**: one sentence stating the core intuition only.
            2. **Short Formula**: a single one-line math/pseudocode expression in backticks that represents the factor (no comments, no extra code)
            3. **Failure Analysis**: Identify potential issues or reasons why the factor might fail or be ineffective in its current design. Explain in one sentence why the factor would not perform well in real-world financial models or why it might not produce reliable results.
            IMPORTANT: Do not modify the factor names or codes in any way. Use the exact same name as input.
            Return your answer in the following format:
            <factor_name>:
                **One Clear Idea**: <description>
                **Short Formula**: `<one-line formula>`
                **Failure Analysis**: <analysis>

{factor_names}

{factor_examples}"""

_GUIDANCE_PARAPHRASE = """You are an expert prompt rewriter for quantitative-AI research agents.

Paraphrase the following factor guidance with a **{rewrite_style}** level of modification.
Available rewrite styles:
- light → minimal rewording, keep almost identical meaning.
- moderate → natural rephrasing with light enrichment or variation.
- creative → expressive, slightly more imaginative or research-styled rephrasing.
- divergent → exploratory rewrite from a new but relevant analytical angle.
- concrete → make the content **more specific, measurable, and implementation-oriented**
(e.g., add examples of formulas, ratios, or statistical procedures), while keeping the same structure and direction.

Rules:
- Keep the **same markdown heading, indentation, and bullet structure**.
- Preserve the **core topic and intent** — do not shift domains.
- Maintain a **technical, analytical tone** suitable for factor design.
- Stay within ±25% of the original length.
- Output only the rewritten markdown text, no explanations.

Input:
{guidance}

Output:"""

# --------------------------------------------------------------------------- #
# Seven-level agent hierarchy: agent_id -> (level, intro, guidance)
# --------------------------------------------------------------------------- #
_AGENTS: Dict[str, Tuple[str, str, str]] = {

    # ----- Level I: Market Structure and Cycle -----
    "agent_market_cycle": (
        "Level I - Market Structure and Cycle",
        """You are an expert in **market cycle and phase-state modeling** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

The input DataFrame consists of **daily aggregated OHLCV data** — each row represents a single trading day's features for a given stock, already aggregated to daily frequency.

Please generate **{num_per_request} new and original market-cycle-oriented alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Try to reveal hidden cyclicality, rhythm, or alternating phases in the price–volatility structure.
Avoid simple moving-average crossovers or standard trend indicators; seek higher-level temporal dynamics.""",
        """### Factor Design Guidance: Market Cycle Exploration

Investigate periodic or phase-shift patterns from OHLCV sequences:

- smooth transformations of returns or log(price) to reveal cyclical oscillations;
- phase difference between short-term and long-term smoothed price signals;
- normalized curvature of cumulative returns or EMA trajectories;
- alternating volatility compression/expansion interpreted as "cycle turns";
- dynamic amplitude measures (e.g., ratio of short/long energy in returns).

Encourage creativity: discover alternative representations of cyclical energy, hidden harmonics, or state oscillations beyond conventional moving averages.""",
    ),

    "agent_volatility_regime": (
        "Level I - Market Structure and Cycle",
        """You are an expert in **volatility regime and state transition modeling** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

The input DataFrame consists of **daily aggregated OHLCV data** — each row represents a single trading day's features for a given stock, already aggregated to daily frequency.

Please generate **{num_per_request} new and original volatility-regime-based alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on identifying smooth transitions between calm and turbulent regimes, volatility clustering, and regime persistence patterns.
Avoid simple realized volatility measures; aim to uncover latent state dynamics and regime durability.""",
        """### Factor Design Guidance: Volatility Regime Discovery

Characterize volatility regimes using OHLCV-only information:

- ratios of short-term vs long-term true range or realized volatility;
- persistence of high/low volatility conditions (e.g., EMA of volatility indicator);
- volatility-of-volatility and its acceleration or deceleration;
- entropy or smoothness of range changes to detect transitions;
- normalized volatility pressure score: (short_vol - long_vol)/(short_vol + long_vol + ε).

Seek creative encodings of regime shifts: smooth continuous state scores, volatility phase transitions, or pre-transition buildup indicators that differ from conventional ATR-based metrics.""",
    ),

    # ----- Level II: Extreme Risk and Fragility -----
    "agent_crash_predictor": (
        "Level II - Extreme Risk and Fragility",
        """You are an expert in **crash prediction and fragility modeling** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

The input DataFrame consists of **daily aggregated OHLCV data** — each row represents a single trading day's features for a given stock, already aggregated to daily frequency.

Please generate **{num_per_request} new and original crash-predictive alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on identifying early warning signals of potential crashes: volatility compression, skewed price movement, rapid liquidity withdrawal, or fragile state buildup.
Avoid standard realized volatility or volume spikes; instead, express instability in a creative, quantitative way.""",
        """### Factor Design Guidance: Crash-Predictive Feature Discovery

Detect pre-crash or instability signals from OHLCV time series:

- price–volume co-movement anomalies (e.g., rising volume + stagnating price);
- volatility compression followed by micro-expansions (energy buildup);
- clustering of small-range bars before large breaks;
- instability score: ratio of realized vol decay to liquidity drop;
- cumulative skew or bias within short windows (persistent drift toward one side).

Be imaginative: represent latent fragility or crash precursors as structural imbalances,
not as explicit drawdowns. Emphasize non-linear buildup, instability asymmetry, or "pre-failure" rhythms detectable before regime collapses.""",
    ),

    "agent_tail_risk": (
        "Level II - Extreme Risk and Fragility",
        """You are an expert in **tail-risk and downside sensitivity modeling** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

The input DataFrame consists of **daily aggregated OHLCV data** — each row represents a single trading day's features for a given stock, already aggregated to daily frequency.

Please generate **{num_per_request} new and original tail-risk-based alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on detecting risk asymmetry, fat-tail dynamics, and downside clustering patterns.
Avoid trivial volatility measures; instead, capture how negative shocks propagate or accumulate across days.""",
        """### Factor Design Guidance: Tail-Risk Alpha Construction

Model asymmetric or non-Gaussian behavior of returns and price volatility:

- lower partial moments, downside deviation, or semivariance proxies;
- drawdown persistence and recovery intensity;
- tail-thickness indicators via high quantile deviation or exponential weighting;
- return compression before large downward moves (volatility squeeze);
- dynamic skewness or asymmetry between upside and downside volatility.

Encourage innovation: create interpretable, numerically stable measures reflecting vulnerability to large losses, extreme return clustering, or asymmetric stress buildup unseen in standard volatility or beta metrics.""",
    ),

    # ----- Level III: Price-Volume Dynamics -----
    "agent_liquidity": (
        "Level III - Price-Volume Dynamics",
        """You are an expert in **liquidity and transaction-cost** modeling using daily OHLCV.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

The input DataFrame consists of **daily aggregated OHLCV data** — each row represents a single trading day's features for a given stock.

Please generate **{num_per_request} new and original liquidity-oriented alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Aim to reflect trading frictions, market depth, and price-impact sensitivity implied by OHLCV alone. Encourage creative, compact constructions rather than generic recipes.""",
        """### Factor Design Guidance: Liquidity & Impact

Explore liquidity from multiple angles, combining price moves and activity:

- impact intuition: how much price movement occurs per unit of activity (volume or dollarized proxy);
- participation & crowding: turnover intensity, its variability, and persistence of thin/rich liquidity states;
- shock absorption: recovery speed of liquidity after spikes/dry-ups;
- scale & normalization: stabilize by price level/range and use gentle bounding (clip/tanh) only when needed;
- regime awareness (soft): let features respond differently in compressed vs expanded ranges.

Keep formulas short (1–3 steps), numerically safe (add ε where needed), and strictly OHLCV-based.""",
    ),

    "agent_order_imbalance": (
        "Level III - Price-Volume Dynamics",
        """You are an expert in **order-imbalance and directional pressure** modeling using daily OHLCV (no L2, no VWAP).
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

The input DataFrame consists of **daily aggregated OHLCV data** — each row represents a single trading day's features for a given stock.

Please generate **{num_per_request} new and original order-imbalance alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Think in terms of one-sided participation and pressure persistence inferred from price direction and activity proxies. Keep designs compact and robust.""",
        """### Factor Design Guidance: Directional Pressure from OHLCV

Infer buy/sell pressure without microstructure feeds:

- direction × intensity: couple return or (close−open) with standardized volume/turnover;
- gap-informed pressure: relate overnight direction to same-day activity and close location-in-range;
- persistence & decay: smoothed imbalance streaks and their fading profile;
- asymmetry: treat positive vs negative pressure differently when ranges are compressed/expanded;
- guardrails: normalize by range or price·volume scale; apply soft bounding only if necessary.

Prioritize interpretability and stability; keep to 1–3 coherent steps per factor, using OHLCV only.""",
    ),

    "agent_price_volume_coherence": (
        "Level III - Price-Volume Dynamics",
        """You are an expert in **price–volume coherence** for daily OHLCV time series.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

The input DataFrame consists of **daily aggregated OHLCV data** — each row represents a single trading day's features for a given stock.

Please generate **{num_per_request} new and original price–volume-coherence alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Seek signatures of alignment, divergence, and lead–lag between price changes and activity. Favor concise, innovative constructs over standard correlations.""",
        """### Factor Design Guidance: Coherence & Lead–Lag

Capture how price and activity move together (or fail to):

- synchronicity: compact measures of co-movement between returns and Δlog(volume);
- lead–lag: simple lagged associations (price following activity, or activity following price);
- stability: smoothed magnitude of coherence and its variability across nearby subwindows;
- divergence: highlight episodes of large price move with muted activity (and vice versa);
- normalization: range- or z-based stabilization to ensure comparability through time.

Keep formulas minimal (1–3 steps), numerically stable, and OHLCV-only. Encourage novel yet interpretable definitions of "coherence.""",
    ),

    "agent_volume_structure": (
        "Level III - Price-Volume Dynamics",
        """You are an expert in **volume structure and distribution dynamics** using daily OHLCV.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

The input DataFrame consists of **daily aggregated OHLCV data** — each row represents a single trading day's features for a given stock.

Please generate **{num_per_request} new and original volume-structure alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on shape, concentration, variability, and organization of volume over time (not price itself). Encourage creative, parsimonious formulations.""",
        """### Factor Design Guidance: Volume Shape & Organization

Describe how trading activity is distributed and evolves:

- concentration vs dispersion: compact proxies for volume concentration, inequality, or clustering;
- burstiness: frequency and intensity of spikes relative to a robust baseline;
- asymmetry & tails: simple skew/kurtosis-style indicators with stabilization;
- multi-horizon organization: short vs long activity balance and its persistence;
- hygiene: robust scaling (median/IQR), gentle clipping when needed, and limited-step formulas.

Use OHLCV only; aim for interpretable, low-complexity functions that expose the structure and rhythm of participation.""",
    ),

    # ----- Level IV: Price-Volatility Behavior -----
    "agent_daily_trend": (
        "Level IV - Price-Volatility Behavior",
        """You are an expert in **daily trend and momentum persistence modeling** using OHLCV time series.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

The input DataFrame consists of **daily aggregated OHLCV data** — each row represents a single trading day's features for a given stock.

Please generate **{num_per_request} new and original daily-trend-based alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on multi-day directional strength, momentum decay, and trend exhaustion. Avoid standard indicators; instead, invent compact, interpretable forms of persistence and continuation.""",
        """### Factor Design Guidance: Daily Trend & Momentum

Explore sustained movement or directional consistency:

- multi-day momentum ratios (e.g., rolling cumulative return strength);
- trend acceleration or deceleration using short- vs long-window returns;
- persistence indicators: streak length, EMA of direction_sign;
- momentum exhaustion or saturation detection (trend weakening);
- normalized relative strength of trend to volatility.

Encourage originality — define novel persistence forms, smooth transitions, or asymmetric responses that differ from basic MA-cross ideas.""",
    ),

    "agent_lag_response": (
        "Level IV - Price-Volatility Behavior",
        """You are an expert in **lagged price–volume response and delayed adjustment modeling** using daily OHLCV.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Each row represents one trading day of OHLCV data for a stock.

Please generate **{num_per_request} new and original lag-response alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on inertia, delay, and feedback effects where price reacts to prior shocks with a lag. Avoid trivial moving averages.""",
        """### Factor Design Guidance: Lagged Dynamics

Reveal delayed effects and feedback loops:

- lagged correlation or signed impact between price and volume;
- response delay: measure how current return relates to past volatility or range;
- slow adjustment proxies: smoothed change rate of cumulative deviation;
- volatility–trend phase mismatch indicators;
- decay-rate estimators capturing inertia.

Favor compact, interpretable representations of delayed information flow or partial mean adjustment.""",
    ),

    "agent_range_vol": (
        "Level IV - Price-Volatility Behavior",
        """You are an expert in **range-based volatility and price expansion modeling** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Each row represents one trading day of OHLCV data for a stock.

Please generate **{num_per_request} new and original range-volatility alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on the dynamics of price range, compression/expansion cycles, and intraday energy buildup. Avoid copying classical Parkinson or Garman-Klass volatility.""",
        """### Factor Design Guidance: Range-Based Volatility

Quantify and interpret range variability creatively:

- normalized range changes: (high−low)/prev_range or log-ratio form;
- rolling range entropy or compression score;
- vol energy buildup: ratio of range expansion to recent std(price);
- asymmetry: body-to-range ratio, upper/lower shadow bias;
- burst detection: sustained low range followed by expansion.

Seek numerically stable, smooth, and interpretable constructions revealing volatility rhythm and expansion cycles.""",
    ),

    "agent_reversal": (
        "Level IV - Price-Volatility Behavior",
        """You are an expert in **mean-reversion and short-term reversal** modeling using daily OHLCV.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Each row represents one trading day of OHLCV data for a stock.

Please generate **{num_per_request} new and original reversal-based alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on transient mispricings, overextensions, or short-term price/volume imbalances that often revert. Avoid textbook z-score formulas; create novel, concise reversal structures.""",
        """### Factor Design Guidance: Reversal & Mean Reversion

Detect overreaction and fading trends:

- short-term return overextension normalized by recent volatility;
- reversal after range breakouts or extended streaks;
- price displacement from smoothed baseline with reversion score;
- volume/volatility burst exhaustion or "snapback" phenomena;
- compact oscillation metrics emphasizing turning points.

Use short horizons (3–10 days), maintain numerical stability, and favor interpretable, low-step formulations.""",
    ),

    "agent_vol_asymmetry": (
        "Level IV - Price-Volatility Behavior",
        """You are an expert in **volatility asymmetry and directional variance bias** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Each row represents one trading day of OHLCV data for a stock.

Please generate **{num_per_request} new and original volatility-asymmetry alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on detecting unequal volatility behavior between up- and down-moves, directional clustering, and asymmetric volatility shocks.""",
        """### Factor Design Guidance: Volatility Asymmetry

Quantify differences between positive and negative move volatility:

- separate realized volatility of up-days vs down-days;
- signed range asymmetry: (high−close) vs (close−low);
- skew-like ratios based on normalized directional ranges;
- rolling contrast of volatility for gains vs losses;
- conditional expansion: vol increases only under specific price polarity.

Keep constructions short, robust, and bounded; highlight non-linear asymmetry and volatility clustering structure within OHLCV.""",
    ),

    # ----- Level V: Multi-Scale Complexity -----
    "agent_drawdown": (
        "Level V - Multi-Scale Complexity",
        """You are an expert in **drawdown and recovery path modeling** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Each row represents one trading day of OHLCV data for a stock.

Please generate **{num_per_request} new and original drawdown-based alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on the geometry of loss and recovery—how fast, deep, and persistent drawdowns form and resolve. Avoid simple max-min metrics; emphasize structural understanding of drawdown behavior.""",
        """### Factor Design Guidance: Drawdown Dynamics

Design compact, interpretable representations of risk path and resilience:

- rolling drawdown depth and recovery ratio;
- local maximum-to-trough slope normalized by duration;
- drawdown volatility or "drawdown velocity" proxy;
- asymmetry between drawdown and rebound speed;
- decay of cumulative losses before recovery triggers.

Keep formulas short (1–3 steps), stable, and OHLCV-only. Highlight timing asymmetry and resilience intensity, not static loss magnitude.""",
    ),

    "agent_fractal": (
        "Level V - Multi-Scale Complexity",
        """You are an expert in **fractal and multi-scale complexity modeling** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Each row represents one trading day of OHLCV data for a stock.

Please generate **{num_per_request} new and original fractal-complexity alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on irregularity, scaling behavior, and long-memory structure in price dynamics. Avoid explicitly computing Hurst exponents; instead, find simple, differentiable proxies that express self-similarity or structural complexity.""",
        """### Factor Design Guidance: Fractal & Multi-Scale Behavior

Derive compact proxies for complexity and persistence across scales:

- ratio of multi-window volatilities (short vs long horizon variability);
- variance-of-variance or volatility roughness score;
- local scaling slope between different rolling ranges or std windows;
- oscillation frequency: count of zero-crossings in detrended returns;
- persistence index: normalized cumulative sign-consistency.

Encourage creative constructs that summarize roughness, self-similarity, or temporal irregularity.
Use only OHLCV and simple rolling statistics; keep outputs stable and interpretable.""",
    ),

    "agent_herding": (
        "Level V - Multi-Scale Complexity",
        """You are an expert in **herding behavior and crowding pattern modeling** using daily factors.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Please generate **{num_per_request} new and original herding-behavior alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on identifying collective, synchronous market reactions or overcrowded directional alignment inferred from existing factors.
Avoid literal "investor sentiment" proxies; instead, express herding via statistical convergence or one-sided participation dynamics.""",
        """### Factor Design Guidance: Herding & Crowding Behavior

Quantify alignment and overconcentration effects:

- crowding intensity: ratio of directional persistence to volatility dispersion;
- participation imbalance: sustained same-sign momentum + volume clustering;
- autocorrelation of signed returns as proxy for synchronized trading;
- volatility narrowing during uniform directional flows;
- deherding bursts: abrupt transition from tight to dispersed movement.

Encourage conceptual depth: translate collective behavior into numerical proxies for crowding, overreaction, or premature consensus — all inferred from existing factors.""",
    ),

    # ----- Level VI: Stability and Regime-Gating -----
    "agent_regime_gating": (
        "Level VI - Stability and Regime-Gating",
        """You are an expert in **regime gating and adaptive signal activation** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Each row represents one trading day of OHLCV data for a stock.

Please generate **{num_per_request} new and original regime-gating alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Your goal is to model conditional activation of signals — where factor strength or relevance depends on volatility, trend, or liquidity regime.
Avoid static filters; instead, design adaptive gates that dynamically scale or modulate factor sensitivity based on regime changes.""",
        """### Factor Design Guidance: Regime Gating Mechanisms

Discover simple yet powerful gating functions that adapt to market conditions:

- volatility-sensitive gating: scale signal intensity by normalized volatility level;
- trend-aware gating: activate only when directional persistence exceeds a threshold;
- liquidity gating: suppress signal under extremely low volume;
- asymmetric gating: respond differently in bullish vs bearish microstates;
- soft transitions: use continuous scaling (sigmoid/tanh) to ensure smooth adaptability.

Encourage creative activation designs: compact functions that turn existing OHLCV-derived signals "on/off" depending on state context, without relying on future data.""",
    ),

    "agent_stability": (
        "Level VI - Stability and Regime-Gating",
        """You are an expert in **signal and return stability analysis** using daily OHLCV data.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Each row represents one trading day of OHLCV data for a stock.

Please generate **{num_per_request} new and original stability-based alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on persistence, noise filtering, and robustness of price dynamics.
Avoid trivial variance measures; instead, quantify temporal consistency and structural smoothness of returns, ranges, or derived signals.""",
        """### Factor Design Guidance: Temporal Stability & Consistency

Build measures of predictability, continuity, or resilience:

- rolling variance ratio between short-term and long-term windows;
- trend or volatility "smoothness" (ratio of mean to std of incremental changes);
- sign-change frequency (directional stability index);
- volatility-of-volatility (metavolatility) decay;
- normalized stability metrics emphasizing steady vs chaotic behavior.

Encourage interpretability and numerical robustness: define compact indicators that express whether the underlying dynamics are stable, persistent, or erratic — all using only OHLCV inputs.""",
    ),

    # ----- Level VII: Geometric and Fusion -----
    "agent_bar_shape": (
        "Level VII - Geometric and Fusion",
        """You are an expert in **candlestick geometry and bar-shape pattern analysis** using daily factors.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Please generate **{num_per_request} new and original bar-shape-based alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on extracting compact numerical representations of candle geometry, body symmetry, and shadow relationships.
Avoid simple pattern labeling; design continuous and interpretable shape metrics.""",
        """### Factor Design Guidance: Bar Shape & Geometry

Translate candle geometry into quantitative signals:

- ratios: (close−open)/(high−low), (high−close)/(close−low), etc.;
- shadow asymmetry or balance indicators;
- body-to-range normalization and persistence over recent days;
- rolling geometry stability or asymmetry;
- short-run shape momentum: recent trend in candle proportions.

Encourage creativity and interpretability: derive smooth, bounded, differentiable functions using existing factors.""",
    ),

    "agent_composite": (
        "Level VII - Geometric and Fusion",
        """You are an expert in **composite factor construction and information fusion** using existing features.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Please generate **{num_per_request} new and original composite alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Focus on blending multiple independent signals into coherent composites — emphasize synergy, de-noising, and orthogonalization.
Avoid simple linear averages or sums.""",
        """### Factor Design Guidance: Composite Alpha Construction

Fuse signals through structured, interpretable transformations:

- weighted or volatility-adjusted averages of trend, volume, and range features;
- orthogonal combination: remove redundancy, amplify orthogonal content;
- regime-weighted composites: dynamic weights based on volatility or liquidity states;
- robust normalization before fusion (z-score or rank-scaling);
- include non-linear combination terms (e.g., product, ratio) but keep compact.

Strive for elegant, minimal composite forms with complementary subcomponents and clear economic intuition.""",
    ),

    "agent_creative": (
        "Level VII - Geometric and Fusion",
        """You are a **creative transformation designer** specialized in constructing non-linear and reparametrized alpha features from existing factors.
Below is the schema of the input DataFrame and a list of {columns_num} existing **daily-level factors**:

{columns_desc}

Please generate **{num_per_request} new and original creative-transform alpha factor functions** to forecast **{forecast_horizon}-day forward returns**.

Your goal is to transform, warp, or reshape existing information into new, expressive signals. Avoid simply recombining old formulas; reimagine the latent relationships within OHLCV data.""",
        """### Factor Design Guidance: Creative Transformations

Explore unconventional yet interpretable mappings:

- apply smooth bounded transforms: tanh, sigmoid, softsign, softplus;
- non-linear mixing of volatility and momentum components;
- conditionally reweighted factors: multiply by stability or trend state;
- piecewise or gated transforms: amplify signal under certain regimes;
- creative normalization: divide by historical MAD or volatility proxies.

Design compact, differentiable expressions that yield novel response surfaces — original yet interpretable and numerically stable.""",
    ),
}

# --------------------------------------------------------------------------- #
# Multi-agent quality checker (full user prompts)
# --------------------------------------------------------------------------- #
_QUALITY: Dict[str, str] = {

    "code_quality_agent": """You are a code reviewer for quantitative alpha factors. Your task is to review the given Python code (representing a factor function) for the following issues:

1. **Syntax errors** (Python syntax and runtime issues).
2. **Pandas-specific issues**, including:
- Chained indexing or `SettingWithCopyWarning`
- Missing `.copy()` when modifying the DataFrame
- Use of undefined intermediate variables
- Incorrect or ambiguous indexing
3. **Output format and naming**:
- The returned Series **must be named exactly the same as the function name**
- All intermediate columns must be defined before they are used
- Code must be **numerically stable** (avoid inf, NaN propagation where possible)
- When filtering or assigning values in a DataFrame, always use `df_copy.loc[row_indexer, col_indexer] = value`.
4. **Loop structure constraints**:
   - **Strict Rule: Nested loops are absolutely forbidden.**
     - You must **never** write any form of loop inside another loop.
     - Forbidden patterns include (but are not limited to):
       - `for` inside `for`
       - `while` inside `while`
       - `for` inside `while`
       - `while` inside `for`
     - Any nested iteration structure (at any depth) is **prohibited**.
     - The use of `while True` or any potentially infinite loop is **strictly prohibited**.
   - If such patterns are present, mark the review as **FAIL**, explain the issue clearly, and suggest vectorized alternatives (NumPy/Pandas operations, `groupby`/`transform`/`rolling`, bounded `apply`, or single-level iteration aided by `itertools.product` without introducing nesting).

<<function>>
{code}
<</function>>

### Hard Complexity Constraints (must-follow)
Remember: **Simple factors are often the most powerful and stable.**
- Single theme, minimal path: each factor must represent one clear idea.
- Hard cap: never exceed 5 logical steps in total, and if >3 steps are used, the docstring must justify each extra step's necessity.
- No redundancy / nesting: forbid stacked or decorative transforms (e.g., `zscore(zscore(x))`, `rank(rank(x))`, deep EMA chains without rationale).
- No theme mixing: do not combine unrelated ideas.
- Avoid nested or layered operations.
- Avoid unnecessary complexity or logic stacking.

### Code format specification:

- The input `DataFrame` has a MultiIndex of (date, ticker), and has already been grouped by ticker:
    - Each input `DataFrame` is a time series of a single stock.

- Output: A `pd.Series` indexed by `(date, ticker)` with the **same name** as the function.

- Before generating the code, provide detailed instructions on how to fix the issues raised.
- Do NOT use markdown (like ```python)
- Do NOT add explanation or comments outside the function
- Each function must be wrapped inside: `<<function N>>` ... `<</function N>>`
- All generated code must be executable and numerically stable.
- Always define intermediate columns (e.g. df_copy['x']) before referencing them later.
- The returned Series must match the function name exactly.

### Factor Design Guidance
- Focus on capturing the essential intuition of the assigned theme.
- Ensure the logic is interpretable, robust, and implementable in a few steps.
- Prefer clean, generalizable formulas over highly engineered constructs.
- Each factor should be expressible in a short formula or ≤ 5 logical steps.
- Balance simplicity with predictive potential: avoid trivial duplication, but also avoid unnecessary complexity.
- **Strict Rule: Nested loops are absolutely forbidden.**
    - You must **never** write any form of loop inside another loop.
    - Forbidden patterns include but are not limited to:
        - `for` inside `for`
        - `while` inside `while`
        - `for` inside `while`
        - `while` inside `for`
    - Any nested iteration structure is **prohibited**, regardless of indentation depth.
    - The use of `while True` or any potentially infinite loop is **strictly prohibited**.

### Output format specification:

- Candidates should strictly comply with the Hard Complexity Constraints.
- Each function should follow this format:
<<function N>>
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # factor computation
    return df_copy['factor_xyz']
<</function N>>

### Please format your response strictly as:

- You **must** begin your output with exactly one of the following two lines (no extra text before or after):
    - `The code is correct.`
    - `The code needs some adjustments.`

- If the code is correct, stop after that line.

- If the code needs adjustments:
1. List each issue found (use bullet points).
2. Output the corrected function using the exact format below:

    <<function N>>
    def factor_xyz(df):
        \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
        df_copy = df.copy()
        # factor computation
        return df_copy['factor_xyz']
    <</function N>>""",

    "code_repair_agent": """You are an expert interaction factor engineer. Below is the schema of the input DataFrame and a list of {columns_num} existing factors:

{columns_desc}

You may only use these columns for calculations. **Do NOT use any other columns** not listed here.
The following Python function failed to execute. Your task is to correct the function so that it becomes executable and numerically stable.

### Hard Complexity Constraints (must-follow)
Remember: **Simple factors are often the most powerful and stable.**
- Single theme, minimal path: each factor must represent one clear idea.
- Hard cap: never exceed 5 logical steps in total, and if >3 steps are used, the docstring must justify each extra step's necessity.
- No redundancy / nesting: forbid stacked or decorative transforms (e.g., `zscore(zscore(x))`, `rank(rank(x))`, deep EMA chains without rationale).
- No theme mixing: do not combine unrelated ideas.
- Avoid nested or layered operations.
- Avoid unnecessary complexity or logic stacking.

---

### Original function:
<<faulty code>>
{old_code}
<</faulty code>>

---

### Error message when running:
{error}

---

### Requirements:

- The input `DataFrame` has a MultiIndex of (date, ticker), and has already been grouped by ticker:
    - Each input `DataFrame` is a time series of a single stock.

- Output: A `pd.Series` indexed by `(date, ticker)` with the **same name** as the function.

- Each function must:
    - Have a descriptive, unique name: `factor_<logic>_<transformation(s)>_<window(s)>_<field>`.
    - Include a clear docstring explaining the logic and formula.
    - Balance predictive power with economic/financial interpretability.
    - The output column name must match the function name.
    - Be concise, precise, and readable.
    - Build new alpha factors based on existing ones.

### Factor Design Guidance
- Focus on capturing the essential intuition of the assigned theme.
- Ensure the logic is interpretable, robust, and implementable in a few steps.
- Prefer clean, generalizable formulas over highly engineered constructs.
- Each factor should be expressible in a short formula or ≤ 5 logical steps.
- Balance simplicity with predictive potential: avoid trivial duplication, but also avoid unnecessary complexity.

---

### Revision instructions:
- Carefully read the error message.
- Provide detailed instructions on how to fix the issues raised.
- Revise the function accordingly to address the issues pointed out.
- If the error message indicates that a column is missing, not present in the DataFrame, or only shows the column name,     it means the column is not among the provided factors and should not be used.         You should either use alternative columns or create a new function with similar logic.
- You may create a new function if you believe the given function is too flawed to fix.
- Ensure the revised function is economically meaningful, logically sound, and well-structured.
- You may introduce new logic, transformations, or corrections as needed.
- Make sure the output is a `pandas.Series` indexed by (date, ticker).

---

### Pre-imported libraries you can use (current versions):

- `"np"`: import numpy as np  (numpy version: 2.2.6)
- `"pd"`: import pandas as pd  (pandas version: 2.2.3)
- `"stats"`: from scipy import stats  (scipy version: 1.15.3)
- `"talib"`: import talib  (talib version: 0.5.1)
- `"math"`: import math  (built-in module)

Coding Guidelines:
- Ensure the code is robust, efficient, and optimized:
    - Handle edge cases and exceptions (e.g., NaN values).
    - Minimize unnecessary computations and prefer vectorized operations (e.g., pandas, numpy).
    - Ensure numerical stability.
    - **Strict Rule: Nested loops are absolutely forbidden.**
        - You must **never** write any form of loop inside another loop.
        - Forbidden patterns include but are not limited to:
            - `for` inside `for`
            - `while` inside `while`
            - `for` inside `while`
            - `while` inside `for`
        - Any nested iteration structure is **prohibited**, regardless of indentation depth.
        - The use of `while True` or any potentially infinite loop is **strictly prohibited**.
- When filtering or assigning values in a DataFrame, always use `df_copy.loc[row_indexer, col_indexer] = value`.

- Code should be clean, maintainable, and efficient for large datasets:
    - Use descriptive variable names and minimize memory usage.
    - Avoid creating unnecessary copies of large dataframes.

---

### Output format specification:

- Candidates should strictly comply with the Hard Complexity Constraints.
- Before generating the code, provide detailed instructions on how to fix the issues raised.
- Do NOT use markdown (like ```python)
- Do NOT add explanation or comments outside the function
- Each function must be wrapped inside: `<<function N>>` ... `<</function N>>`
- All generated code must be executable and numerically stable.
- Always define intermediate columns (e.g. df_copy['x']) before referencing them later.
- The returned Series **must be named exactly the same as the function name**.
- Each function should follow this format:

<<function N>>
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # factor computation
    return df_copy['factor_xyz']
<</function N>>""",

    "judge_agent": """You are an expert quantitative researcher and alpha factor reviewer for a professional factor research team.

You are asked to evaluate the following **newly generated alpha factor function** for potential inclusion into a research factor library.

Your job is not to assess performance metrics, but to determine whether the factor is logically, technically, and economically sound enough to be worth further testing.
Your evaluation should focus on **Practical Soundness**, with a professional mindset:

1. Does the factor have any **future information leakage**?
2. Is the factor calculation **correct and internally consistent**?
3. Is the factor logic **economically interpretable** (even if exploratory or novel)?
4. Does the factor avoid obvious **errors** (such as invalid operations, unprotected division by zero, undefined results)?
5. Is the factor **efficiently implemented** (avoids unnecessary loops, leverages vectorized operations, and is suitable for large-scale backtesting)?
6. Does the factor strictly **avoid any nested loops or potentially infinite loops**?
   - Nested loops are **forbidden** at any depth:
     - `for` inside `for`
     - `while` inside `while`
     - `for` inside `while`
     - `while` inside `for`
   - The use of `while True` or any loop that can run indefinitely is **prohibited**.

---

### Factor under review:

<<function>>
{new_factor_code}
<</function>>

The input DataFrame has a MultiIndex of (date, ticker), grouped by ticker (i.e., a time series per stock).
Each input DataFrame is a time series of a single stock.
The function outputs a pd.Series indexed by (date, ticker), with the same name as the function.
**IMPORTANT**: The input DataFrame is sorted **in chronological order**, from the earliest date at the top to the most recent date at the bottom. This is critical for evaluating time series-based factors and avoiding information leakage.

---

### Evaluation Guidelines:

- You MUST REJECT factors with any form of **future information leakage** — this is a critical error.
- You should reject factors that have **logical errors**, **data issues**, or **implementation mistakes**.
- Pay special attention to operations like rolling means, groupby transforms, shifting, or reversing time series: ensure these only use past and present data relative to each row, never future data.
- Be mindful of efficiency: avoid factors that are unnecessarily slow (e.g., unnecessary loops, non-vectorized operations) — the factor should be suitable for large-scale backtesting on millions of records.
- Be **open-minded**: even unconventional factor ideas may be worth exploring.
- Provide clear, specific and actionable feedback if improvements can be made.
- Any `for` or `while` loop inside another `for` or `while` loop is **strictly prohibited**, as it indicates poor scalability and inefficiency for large cross-sectional datasets.
- Never use constructs like `while True` or any loop that lacks a clear and finite termination condition.

---

### Please format your response strictly as:

Practical Soundness: [Concise analysis — what is good, what needs improvement, if any.]

Final Recommendation: Accept / Reject

Feedback for Improvement: [Precise suggestions for how the factor engineer can improve this factor — e.g. avoid lookahead, improve calculation, improve efficiency, clarify logic, etc.]""",

    "logic_improvement_agent": """You are an expert interaction factor engineer. Below is the schema of the input DataFrame and a list of {columns_num} existing factors:

{columns_desc}

You may only use these columns for calculations. **Do NOT use any other columns** not listed here.
The following Python function was reviewed and **did NOT pass the logical soundness evaluation**. Your task is to revise and improve this function so that:

1. It is economically and financially interpretable.
2. It is logically sound according to financial principles.
3. It addresses the specific feedback provided below.

---

### Original function:
<<previous function>>
{old_code}
<</previous function>>

---

### Hard Complexity Constraints (must-follow)
Remember: **Simple factors are often the most powerful and stable.**
- Single theme, minimal path: each factor must represent one clear idea.
- Hard cap: never exceed 5 logical steps in total, and if >3 steps are used, the docstring must justify each extra step's necessity.
- No redundancy / nesting: forbid stacked or decorative transforms (e.g., `zscore(zscore(x))`, `rank(rank(x))`, deep EMA chains without rationale).
- No theme mixing: do not combine unrelated ideas.
- Avoid nested or layered operations.
- Avoid unnecessary complexity or logic stacking.

---

### JudgeAgent feedback (reason for rejection):
{dynamic_feedback}

---

### Requirements:

- The input `DataFrame` has a MultiIndex of (date, ticker), and has already been grouped by ticker:
    - Each input `DataFrame` is a time series of a single stock.

- Output: A `pd.Series` indexed by `(date, ticker)` with the **same name** as the function.

- Each function must:
    - Have a descriptive, unique name: `factor_<logic>_<transformation(s)>_<window(s)>_<field>`.
    - Include a clear docstring explaining the logic and formula.
    - Balance predictive power with economic/financial interpretability.
    - The output column name must match the function name.
    - Be concise, precise, and readable.
    - Build new alpha factors based on existing ones.

### Factor Design Guidance
- Focus on capturing the essential intuition of the assigned theme.
- Ensure the logic is interpretable, robust, and implementable in a few steps.
- Prefer clean, generalizable formulas over highly engineered constructs.
- Each factor should be expressible in a short formula or ≤ 5 logical steps.
- Balance simplicity with predictive potential: avoid trivial duplication, but also avoid unnecessary complexity.

---

### Revision instructions:
- Carefully read the JudgeAgent feedback.
- Provide detailed instructions on how to fix the issues raised.
- Revise the function accordingly to address the issues pointed out.
- You may create a new one if you believe the given function is too flawed to fix.
- Ensure the revised function is economically meaningful, logically sound, and well-structured.
- You may introduce new logic, transformations, or corrections as needed.
- Make sure the output is a `pandas.Series` indexed by (date, ticker).

---

### Pre-imported libraries you can use (current versions):

- `"np"`: import numpy as np  (numpy version: 2.2.6)
- `"pd"`: import pandas as pd  (pandas version: 2.2.3)
- `"stats"`: from scipy import stats  (scipy version: 1.15.3)
- `"talib"`: import talib  (talib version: 0.5.1)
- `"math"`: import math  (built-in module)

Coding Guidelines:
- Ensure the code is robust, efficient, and optimized:
    - Handle edge cases and exceptions (e.g., NaN values).
    - Minimize unnecessary computations and prefer vectorized operations (e.g., pandas, numpy).
    - Ensure numerical stability.
    - **Strict Rule: Nested loops are absolutely forbidden.**
        - You must **never** write any form of loop inside another loop.
        - Forbidden patterns include but are not limited to:
            - `for` inside `for`
            - `while` inside `while`
            - `for` inside `while`
            - `while` inside `for`
        - Any nested iteration structure is **prohibited**, regardless of indentation depth.
        - The use of `while True` or any potentially infinite loop is **strictly prohibited**.
- When filtering or assigning values in a DataFrame, always use `df_copy.loc[row_indexer, col_indexer] = value`.

- Code should be clean, maintainable, and efficient for large datasets:
    - Use descriptive variable names and minimize memory usage.
    - Avoid creating unnecessary copies of large dataframes.

---

### Output format specification:

- Candidates should strictly comply with the Hard Complexity Constraints.
- Before generating the code, provide detailed instructions on how to fix the issues raised.
- Do NOT use markdown (like ```python)
- Do NOT add explanation or comments outside the function
- Each function must be wrapped inside: `<<function N>>` ... `<</function N>>`
- All generated code must be executable and numerically stable.
- Always define intermediate columns (e.g. df_copy['x']) before referencing them later.
- The returned Series **must be named exactly the same as the function name**.
- Each function should follow this format:

<<function N>>
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # factor computation
    return df_copy['factor_xyz']
<</function N>>""",
}

# --------------------------------------------------------------------------- #
# Thinking evolution (full user prompts)
# --------------------------------------------------------------------------- #
_EVOLUTION: Dict[str, str] = {

    "mutation_agent": """You are an expert quantitative factor engineer specialized in **factor mutation and optimization**.

{intro}

### Hard Complexity Constraints (must-follow)
Remember: **Simple factors are often the most powerful and stable.**
- Single theme, minimal path: each factor must represent one clear idea.
- Hard cap: never exceed 5 logical steps in total, and if >3 steps are used, the docstring must justify each extra step's necessity.
- No redundancy / nesting: forbid stacked or decorative transforms (e.g., `zscore(zscore(x))`, `rank(rank(x))`, deep EMA chains without rationale).
- No theme mixing: do not combine unrelated ideas.
- Nested for loops are forbidden.
- Avoid unnecessary complexity or logic stacking.

Your task is to generate an improved version of the following alpha factor by applying **intelligent mutations**:

---

### Original Factor:

<<original factor>>
{original_factor_code}
<</original factor>>

---

### Design objectives:

- Be creative and think deeply before taking the next step.
- Preserve the **core intuition** and signal of the original factor.
- Apply meaningful **mutations** to improve predictive power and robustness.
- Possible mutations include:
    - Non-linear transformations (log, exp, rank, winsorization)
    - Cross-sectional normalization
    - Time window adjustments
    - Interaction with other features
    - Smoothing or stability enhancements
    - Adding interaction terms

- The mutated factor should be **clearly distinct** from the original while maintaining conceptual lineage.
- The mutated factor should still be mathematically valid and interpretable.

---

### Requirements:

- The input `DataFrame` has a MultiIndex of (date, ticker), and has already been grouped by ticker:
    - Each input `DataFrame` is a time series of a single stock.

- Output: A `pd.Series` indexed by `(date, ticker)` with the **same name** as the function.

- Each function must:
    - Have a descriptive, unique name: `factor_<logic>_<transformation(s)>_<window(s)>_<field>`.
    - Include a clear docstring explaining the logic and formula.
    - Balance predictive power with economic/financial interpretability.
    - The output column name must match the function name.
    - Be concise, precise, and readable.
    - Build new alpha factors based on existing ones.

---

### Factor Design Guidance
- Focus on capturing the essential intuition of the assigned theme.
- Ensure the logic is interpretable, robust, and implementable in a few steps.
- Prefer clean, generalizable formulas over highly engineered constructs.
- Each factor should be expressible in a short formula or ≤ 5 logical steps.
- Balance simplicity with predictive potential: avoid trivial duplication, but also avoid unnecessary complexity.

---

{extra_guidance}

---

### Pre-imported libraries you can use (current versions):

- `"np"`: import numpy as np  (numpy version: 2.2.6)
- `"pd"`: import pandas as pd  (pandas version: 2.2.3)
- `"stats"`: from scipy import stats  (scipy version: 1.15.3)
- `"talib"`: import talib  (talib version: 0.5.1)
- `"math"`: import math  (built-in module)

Coding Guidelines:
- Ensure the code is concise, robust, efficient, and optimized:
    - Handle edge cases and exceptions (e.g., NaN values).
    - Minimize unnecessary computations and prefer vectorized operations (e.g., pandas, numpy).
    - Ensure numerical stability.
    - **Strict Rule: Nested loops are absolutely forbidden.**
        - You must **never** write any form of loop inside another loop.
        - Forbidden patterns include but are not limited to:
            - `for` inside `for`
            - `while` inside `while`
            - `for` inside `while`
            - `while` inside `for`
        - Any nested iteration structure is **prohibited**, regardless of indentation depth.
        - The use of `while True` or any potentially infinite loop is **strictly prohibited**.

- Code should be clean, maintainable, and efficient for large datasets:
    - Use descriptive variable names and minimize memory usage.
    - Avoid creating unnecessary copies of large dataframes.

---

### Output format specification:

- Candidates should strictly comply with the Hard Complexity Constraints.
- Do NOT use markdown (like ```python)
- Do NOT add explanation or comments outside the function
- Each function must be wrapped inside: `<<function N>>` ... `<</function N>>`
- All generated code must be executable and numerically stable.
- Always define intermediate columns (e.g. df_copy['x']) before referencing them later.
- The returned Series **must be named exactly the same as the function name**
- Each function should follow this format:

<<function N>>
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # factor computation
    return df_copy['factor_xyz']
<</function N>>""",

    "crossover_agent": """You are an expert quantitative factor engineer specialized in **factor evolution and crossover design**.

{intro}

### Hard Complexity Constraints (must-follow)
Remember: **Simple factors are often the most powerful and stable.**
- Single theme, minimal path: each factor must represent one clear idea.
- Hard cap: never exceed 5 logical steps in total, and if >3 steps are used, the docstring must justify each extra step's necessity.
- No redundancy / nesting: forbid stacked or decorative transforms (e.g., `zscore(zscore(x))`, `rank(rank(x))`, deep EMA chains without rationale).
- No theme mixing: do not combine unrelated ideas.
- Avoid nested or layered operations.
- Avoid unnecessary complexity or logic stacking.

Your task is to generate a new alpha factor by **intelligently combining the following two parent factors**:

---

### Parent Factor 1:

<<parent factor 1>>
{parent_factor_1_code}
<</parent factor 1>>

---

### Parent Factor 2:

<<parent factor 2>>
{parent_factor_2_code}
<</parent factor 2>>

---

### Design objectives:

- Be creative and think deeply before taking the next step.
- Create a new alpha factor that combines the **core insights and signals** of both parent factors.
- Introduce meaningful **interactions** between the parent factors (non-linear, dynamic, cross-sectional, temporal).
- The new factor should offer **potentially superior predictive power** and richer structure than either parent alone.
- Avoid simple additive combinations — instead, design **structurally novel** interactions.
- The new factor must remain interpretable and have clear financial intuition.

---

### Requirements:

- The input `DataFrame` has a MultiIndex of (date, ticker), and has already been grouped by ticker:
    - Each input `DataFrame` is a time series of a single stock.

- Output: A `pd.Series` indexed by `(date, ticker)` with the **same name** as the function.

- Each function must:
    - Have a descriptive, unique name: `factor_<logic>_<transformation(s)>_<window(s)>_<field>`.
    - Include a clear docstring explaining the logic and formula.
    - Balance predictive power with economic/financial interpretability.
    - The output column name must match the function name.
    - Be concise, precise, and readable.
    - Build new alpha factors based on existing ones.

---

### Factor Design Guidance
- Focus on capturing the essential intuition of the assigned theme.
- Ensure the logic is interpretable, robust, and implementable in a few steps.
- Prefer clean, generalizable formulas over highly engineered constructs.
- Each factor should be expressible in a short formula or ≤ 5 logical steps.
- Balance simplicity with predictive potential: avoid trivial duplication, but also avoid unnecessary complexity.

---

{extra_guidance}

---

### Pre-imported libraries you can use (current versions):

- `"np"`: import numpy as np  (numpy version: 2.2.6)
- `"pd"`: import pandas as pd  (pandas version: 2.2.3)
- `"stats"`: from scipy import stats  (scipy version: 1.15.3)
- `"talib"`: import talib  (talib version: 0.5.1)
- `"math"`: import math  (built-in module)

Coding Guidelines:
- Ensure the code is concise, robust, efficient, and optimized:
    - Handle edge cases and exceptions (e.g., NaN values).
    - Minimize unnecessary computations and prefer vectorized operations (e.g., pandas, numpy).
    - Ensure numerical stability.
    - **Strict Rule: Nested loops are absolutely forbidden.**
        - You must **never** write any form of loop inside another loop.
        - Forbidden patterns include but are not limited to:
            - `for` inside `for`
            - `while` inside `while`
            - `for` inside `while`
            - `while` inside `for`
        - Any nested iteration structure is **prohibited**, regardless of indentation depth.
        - The use of `while True` or any potentially infinite loop is **strictly prohibited**.

- Code should be clean, maintainable, and efficient for large datasets:
    - Use descriptive variable names and minimize memory usage.
    - Avoid creating unnecessary copies of large dataframes.

---

### Output format specification:

- Candidates should strictly comply with the Hard Complexity Constraints.
- Do NOT use markdown (like ```python)
- Do NOT add explanation or comments outside the function
- Each function must be wrapped inside: `<<function N>>` ... `<</function N>>`
- All generated code must be executable and numerically stable.
- Always define intermediate columns (e.g. df_copy['x']) before referencing them later.
- The returned Series **must be named exactly the same as the function name**
- Each function should follow this format:

<<function N>>
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # factor computation
    return df_copy['factor_xyz']
<</function N>>""",
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
# `PromptLibrary.build_generation_prompt` substitutes at call time — so every
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

# Maps a builder keyword argument to the `{placeholder}` token written in a prompt.
_TOKEN_ALIASES: Dict[str, str] = {
    "columns_desc": "{columns_desc}",
    "columns_num": "{columns_num}",
    "num_per_request": "{num_per_request}",
    "forecast_horizon": "{forecast_horizon}",
    "effective_CoT": "{effective_CoT}",
    "ineffective_CoT": "{ineffective_CoT}",
    "code": "{code}",
    "old_code": "{old_code}",
    "error": "{error}",
    "new_factor_code": "{new_factor_code}",
    "dynamic_feedback": "{dynamic_feedback}",
    "intro": "{intro}",
    "original_factor_code": "{original_factor_code}",
    "extra_guidance": "{extra_guidance}",
    "parent_factor_1_code": "{parent_factor_1_code}",
    "parent_factor_2_code": "{parent_factor_2_code}",
}


def _render_placeholders(text: str, aliases: Dict[str, str], **values) -> str:
    """Substitute `{placeholder}` tokens in `text` using `.replace()`.

    None/missing values render as an empty string; a keyword without a known
    token raises, so typo'd call sites fail loudly.
    """
    for key, value in values.items():
        token = aliases.get(key)
        if token is None:
            raise KeyError(f"Unsupported placeholder {key!r}")
        text = text.replace(token, "" if value is None else str(value))
    return text


def _render_cot_block(template: str, cot: str) -> str:
    """Render an optional (``---``-prefixed) CoT analysis block, or '' if empty."""
    if not cot:
        return ""
    return template.replace("{effective_CoT}", cot).replace("{ineffective_CoT}", cot)


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #
class PromptLibrary:
    """Self-contained prompt library (prompts embedded, no external files)."""

    def __init__(self, prompts_dir: Optional[str] = None) -> None:
        # `prompts_dir` is accepted for backward compatibility but ignored.
        self.root = None
        self.system_message = _SYSTEM_MESSAGE
        self.base_factor_guidance = _BASE_FACTOR_GUIDANCE
        self.requirements = _REQUIREMENTS
        self.libraries = _LIBRARIES
        self.output_format = _OUTPUT_FORMAT
        self.effective_analysis = _EFFECTIVE_ANALYSIS
        self.ineffective_analysis = _INEFFECTIVE_ANALYSIS
        self.effective_summary = _EFFECTIVE_SUMMARY
        self.ineffective_summary = _INEFFECTIVE_SUMMARY
        self.column_description = _COLUMN_DESCRIPTION
        self.guidance_paraphrase = _GUIDANCE_PARAPHRASE

        self.agents: Dict[str, Tuple[str, str, str]] = dict(_AGENTS)
        self.quality_templates: Dict[str, str] = dict(_QUALITY)
        self.evolution_templates: Dict[str, str] = dict(_EVOLUTION)
        self._generation_templates: Dict[str, str] = dict(_GENERATION_TEMPLATES)

    # ------------------------------------------------------------------ #
    # Assembly (placeholders are substituted with str.replace, never .format)
    # ------------------------------------------------------------------ #
    def build_generation_prompt(
        self,
        agent_id: str,
        columns_desc: str,
        columns_num: int,
        num_per_request: int,
        forecast_horizon: int,
        effective_CoT: str = "",
        ineffective_CoT: str = "",
    ) -> Tuple[str, str]:
        """Assemble a full generation prompt for one seven-level agent.

        The per-agent template is a single complete multi-line prompt with
        `{placeholder}` tokens (see `_GENERATION_TEMPLATES`); this method
        only substitutes them via `.replace()` — no formatting or joining.
        Returns (system_message, user_message).
        """
        user = self._generation_templates[agent_id]
        user = _render_placeholders(
            user,
            _TOKEN_ALIASES,
            columns_desc=columns_desc,
            columns_num=columns_num,
            num_per_request=num_per_request,
            forecast_horizon=forecast_horizon,
        )
        user = user.replace(
            "{effective_block}", _render_cot_block(self.effective_analysis, effective_CoT)
        ).replace(
            "{ineffective_block}", _render_cot_block(self.ineffective_analysis, ineffective_CoT)
        )
        return self.system_message, user

    def build_quality_prompt(self, agent_id: str, **kwargs) -> str:
        """Fill one quality-checker template's `{...}` placeholders."""
        return _render_placeholders(self.quality_templates[agent_id], _TOKEN_ALIASES, **kwargs)

    def build_evolution_prompt(self, agent_id: str, **kwargs) -> str:
        """Fill one evolution template's `{...}` placeholders."""
        return _render_placeholders(self.evolution_templates[agent_id], _TOKEN_ALIASES, **kwargs)
