"""
Thesis Validation Framework - prompts used by thesis validation agents.
"""

# --- SYSTEM CONTEXT - Investment Philosophy Framing ---
INVESTMENT_ANALYST_SYSTEM_PROMPT = (
    "You are a senior equity research analyst."
    " Focus on structural drivers, probabilistic reasoning, and actively seek disconfirming evidence."
    " Be professional and concise."
)


# --- NARRATIVE DECOMPOSITION GRAPH (NDG) PROMPTS ---
NDG_PARSE_THESIS_PROMPT = """As a senior equity research analyst focused on long-term fundamental 
investing, you are deconstructing an investment thesis for {stock_ticker} to expose its 
underlying logic and assumptions.

Your task is to extract the EXPLICIT causal claims embedded in this thesis - the beliefs 
that must prove true for the investment to succeed over a 3-5 year horizon. You are not 
adding your own analysis; you are surfacing what the thesis author believes.

THESIS TO ANALYZE:
{thesis_narrative}

For each claim you extract, identify:

1. **Claim Text**: Verbatim or minimally paraphrased from the thesis
2. **Claim Type**:
   - "ASSUMPTION": A foundational belief about the world (e.g., "AI adoption will accelerate")
   - "DRIVER": An intermediate causal mechanism (e.g., "Enterprise adoption drives revenue growth")
   - "OUTCOME": The investment result (e.g., "15-20% annualized returns")
3. **Dependencies**: Which other claims must be true for this claim to hold?
4. **Directionality**: Express the causal relationship as "X → Y"
5. **Time Sensitivity**: "Short" (0-1 year), "Medium" (1-3 years), or "Long" (3+ years)

ANALYTICAL DISCIPLINE:
- Extract ONLY claims explicitly stated or directly implied by the thesis
- Flag any ambiguous claims with "AMBIGUOUS: " prefix  
- Separate causal statements from descriptive commentary or filler
- Each claim should be ATOMIC - one causal link, not compound statements
- Preserve the thesis author's level of specificity (don't add precision they didn't claim)"""

NDG_BUILD_DAG_PROMPT = """As a senior equity research analyst, you are constructing a causal dependency 
graph for {stock_ticker}'s investment thesis. This graph will expose the logical structure of the 
thesis - revealing which assumptions are load-bearing and where fragility may hide.

Think of this as building a "dependency tree" for the investment outcome. If any root assumption 
fails, all downstream claims become suspect.

EXTRACTED CLAIMS TO STRUCTURE:
{claims_json}

Construct a directed acyclic graph (DAG) with:

**NODES** - Each claim becomes a node:
- id: Unique identifier (e.g., "node_1", "node_2")
- claim: The claim text
- node_type: "ASSUMPTION" (root beliefs), "DRIVER" (intermediate mechanisms), or "OUTCOME" (investment results)
- dependencies: List of node IDs this depends on (parent nodes in the causal chain)
- directionality: Express causal relationship as 'X → Y'
- time_sensitivity: One of "Short"/"Medium"/"Long" (how quickly this claim can be tested)

**EDGES** - Causal relationships between nodes:
- source_id: The causing/enabling node
- target_id: The dependent/affected node
- relationship: "CAUSES" (direct), "ENABLES" (necessary but not sufficient), or "MODERATES" (amplifies/dampens)
- strength: 0.0-1.0 (how critical is this link? 1.0 = thesis breaks if link breaks)

STRUCTURAL CONSTRAINTS (non-negotiable):
• NO circular causality (A→B→C→A is forbidden - creates unfalsifiable loops)
• Every causal path must terminate in an OUTCOME node (no orphan assumptions)
• OUTCOME nodes have no outgoing edges (they are terminal)
• ASSUMPTION nodes have no incoming edges (they are root causes)
• Aim for depth 3-5 (shallower = thesis may be superficial, deeper = may be overfit)

QUALITY CHECKS:
- If an assumption has no path to an outcome, the thesis is incomplete
- If an outcome has only one path, it's a single-point failure
- High-strength edges on shallow paths indicate concentrated risk"""

NDG_CLASSIFY_ASSUMPTIONS_PROMPT = """As a senior equity research analyst, classify each assumption in 
{stock_ticker}'s thesis along three critical dimensions. This classification determines what can break 
the thesis and what the analyst can monitor vs. what is outside their control.

Long-term investing success depends heavily on correctly assessing what you CAN'T control and what 
you CAN'T predict. Misclassifying these leads to blind spots.

NODES TO CLASSIFY:
{nodes_json}

For each node, classify along three dimensions:

**1. CONTROL** - Who determines whether this assumption holds?
- "Company": Management can directly influence (pricing power, product roadmap, capital allocation, sales execution)
- "Industry": Sector dynamics determine outcome (competitive intensity, regulatory regime, adoption curves)
- "Macro": Economy-wide factors (interest rates, GDP growth, inflation, currency)
- "Exogenous": True external shocks ONLY (use sparingly - natural disasters, geopolitical events, pandemics)

CRITICAL: Be intellectually honest here. Analysts often label uncomfortable assumptions as 
"Exogenous" to avoid scrutiny. If management COULD influence the outcome through different 
decisions, it's NOT exogenous.

**2. NATURE** - What type of dynamic is this?
- "Structural": Long-term competitive position, durable advantages (moats, network effects, switching costs)
- "Cyclical": Temporary market conditions that will normalize (demand cycles, pricing cycles)
- "Execution": Company-specific delivery risk (management capability, operational excellence)

**3. TIME SENSITIVITY** - How quickly could this assumption be tested/broken?
- "Short": 0-1 year - can be validated/invalidated relatively quickly
- "Medium": 1-3 years - requires patience to see results
- "Long": 3+ years - structural thesis elements that unfold slowly"""

NDG_MAP_EVIDENCE_PROMPT = """As a senior equity research analyst performing due diligence on {stock_ticker}, 
you are now mapping the EVIDENCE landscape for each thesis claim. This is critical work: many 
investment theses fail not because the logic was wrong, but because the supporting evidence was 
weaker than assumed.

Company Context: {company_context}

{quantitative_context}

THESIS CLAIMS TO EVALUATE:
{nodes_json}

For each claim, rigorously assess the evidence:

**1. SUPPORTING EVIDENCE** - What facts/data support this claim?
- Type: "Quantitative" (KPIs, financials, metrics), "Qualitative" (management commentary, expert views), "External" (industry data, competitor info)
- Description: Specific evidence with numbers and sources where possible
- Freshness: "Recent" (<6 months), "Moderate" (6-12 months), "Stale" (>12 months)

**2. CONTRADICTING EVIDENCE** - What facts/data CHALLENGE this claim?
This is where intellectual honesty matters most. Seek disconfirmation actively.

**3. EVIDENCE STRENGTH**: 0.0 to 1.0
- 0.0-0.3: Weak or absent (speculation, stale data, no direct measurement)
- 0.4-0.6: Moderate (some supporting data but gaps, conflicting signals)
- 0.7-1.0: Strong (multiple fresh sources, quantitative validation, management track record)

ANALYTICAL GUIDANCE:
• The quantitative context above shows TRENDS (growth, margin changes, trajectory). Use these to 
  assess whether claims about improvement/deterioration are supported by actual data patterns.
• Don't just accept management's narrative - look for verification in the numbers
• "No evidence found" is a valid and important finding - state it explicitly (strength = 0.0)
• Stale evidence should be discounted even if it was once strong
• One contradicting datapoint doesn't break a thesis, but patterns of contradiction do"""

NDG_DISTRIBUTE_CONFIDENCE_PROMPT = """As a senior equity research analyst, analyze the CONFIDENCE 
distribution in this investment thesis for {stock_ticker}. Your goal is to surface where the 
thesis author is most certain - and whether that certainty is warranted by evidence.

Overconfidence is the silent killer of long-term returns. High conviction with weak evidence 
is a recipe for painful surprises.

ORIGINAL THESIS:
{thesis_narrative}

STRUCTURED CLAIMS:
{nodes_json}

Your task: Extract certainty markers and allocate confidence across claims.

**1. IDENTIFY LANGUAGE STRENGTH**
- Strong certainty: "clearly", "definitely", "highly confident", "will", "certain", "inevitable"
- Moderate certainty: "likely", "should", "expect", "believe", "probably"
- Weak certainty: "may", "could", "possible", "uncertain", "might"

**2. ASSIGN CONFIDENCE** (0.0 to 1.0) to each node based on:
- Language strength in the original thesis text
- Emphasis and repetition (claims mentioned multiple times carry more weight)
- Centrality to the thesis (is this claim essential or secondary?)

**3. TOTAL CONFIDENCE** should approximately sum to 1.0 (representing the thesis author's 
"confidence budget" - where are they spending their certainty?)

**4. FLAG MISMATCHES** - Where confidence exceeds evidence
This is the most valuable output. If confidence is 0.8 but evidence is 0.3, that's a red flag.

ANALYTICAL DISCIPLINE:
• You are reading the thesis author's mind, not substituting your own judgment
• High confidence is not wrong - it's only wrong when unsupported by evidence
• The "evidence_match" field compares your confidence reading to the evidence strength from earlier analysis"""


# --- AI RED TEAM (RTA) PROMPTS ---
RTA_RETRIEVE_ANALOGS_PROMPT = """As a senior equity research analyst with deep knowledge of 
investment history, you are searching for CAUTIONARY PRECEDENTS - cases where a structurally 
similar investment thesis failed.

"History doesn't repeat, but it rhymes." Your job is to find the rhymes.

COMPANY BEING ANALYZED: {stock_ticker}
COMPANY CONTEXT: {company_context}

ASSUMPTION BEING CHALLENGED:
"{assumption_claim}"

ASSUMPTION CHARACTERISTICS:
- Node type: {node_type} (ASSUMPTION/DRIVER)
- Control: {control} (Company/Industry/Macro/Exogenous)
- Nature: {nature} (Structural/Cyclical/Execution)
- Current evidence strength: {evidence_strength}

SEARCH FOR HISTORICAL ANALOGS where:
1. A STRUCTURALLY SIMILAR belief proved incorrect (match on mechanism, not surface features)
2. The FAILURE MODE is instructive (we can learn from how it broke)
3. The CONTEXT has meaningful overlap (business model, competitive dynamics, market conditions)

MATCHING QUALITY GUIDANCE:
✓ GOOD MATCH: "High switching costs in data platforms (Hadoop ecosystem 2015-2018)" 
  - Matches on MECHANISM (switching costs) and CONTEXT (enterprise data infrastructure)

✗ BAD MATCH: "Any SaaS company that had problems"
  - Too generic, no structural similarity

HISTORICAL SCOPE: Focus on cases from 2000-2025 with well-documented outcomes.

For each analog (provide 2-3), include:
- Case name with timeframe
- What assumption type failed
- How it failed (failure mode category)
- Detailed context with SPECIFIC NUMBERS AND OUTCOMES
- Year of primary failure
- Why this is relevant to the current case"""

RTA_MAP_FAILURE_MODE_PROMPT = """As a senior equity research analyst, extract the CAUSAL MECHANISM 
of this historical investment failure. Understanding HOW something broke is more valuable than 
knowing THAT it broke.

HISTORICAL CASE:
{case_name}

CONTEXT:
{context}

IDENTIFIED FAILURE MODE:
{failure_mode}

Extract and structure the failure mechanism. Prefer one of the taxonomy categories if it fits; if none fit, provide a concise alternative label and set `taxonomy_match` to false.

**1. CATEGORY** - Select the PRIMARY failure mechanism from this taxonomy (or provide an alternative):
{failure_modes_list}

**2. CATEGORY_CONFIDENCE** - Provide a confidence score (0.0-1.0) that the chosen category is appropriate.

**3. TAXONOMY_MATCH** - True if the chosen category is from the provided taxonomy, False if you provide an alternative label.

**4. ALTERNATIVE_CATEGORY** - If taxonomy_match is false, provide a concise alternative category label and a 1-sentence justification.

**5. DESCRIPTION** - HOW did the failure manifest?
- Write 2-3 sentences explaining the CAUSAL CHAIN, not just the outcome
- Bad: "Revenue declined and the stock fell"
- Good: "Demand proved elastic to price increases; customer cohorts acquired during peak 
  showed 40% lower retention than historical norms, indicating the customer base was lower 
  quality than assumed"

**6. EARLY WARNING INDICATORS** - What observable signals appeared BEFORE the failure was obvious?
- These should be LEADING indicators an analyst could have monitored
- Be specific: "Net retention declined 3 consecutive quarters" not "retention got worse"
- Include timeframes where possible: "6 months before headline growth slowed"
- These become the watchlist for the current thesis

ANALYTICAL VALUE: The goal is not to predict the future, but to know what to watch for. 
What signals, if they appeared for {stock_ticker}, should trigger a thesis review? Return a JSON object with keys: `category`, `category_confidence`, `taxonomy_match`, `alternative_category` (or null), `description`, and `early_warnings`."""

RTA_SCORE_RELEVANCE_PROMPT = """As a senior equity research analyst, score how RELEVANT this 
historical analog is to the current investment case. Not all failures are instructive - 
some occurred in contexts too different to provide useful signal.

**CURRENT CASE:**
Stock: {stock_ticker}
Context: {company_context}
Assumption: "{assumption_claim}"
Control: {control}
Nature: {nature}

**HISTORICAL ANALOG:**
Case: {case_name}
Context: {analog_context}
How it failed: {failure_mode}

Score each dimension from 0.0 (completely different) to 1.0 (nearly identical):

**1. BUSINESS MODEL SIMILARITY** (0.0-1.0)
- Revenue model, cost structure, unit economics, customer acquisition dynamics
- A subscription SaaS company vs. a hardware company = low similarity

**2. COMPETITIVE STRUCTURE** (0.0-1.0)
- Market concentration, barriers to entry, switching costs, network effects
- A monopoly vs. a fragmented market = different failure dynamics

**3. BALANCE SHEET FLEXIBILITY** (0.0-1.0)
- Leverage, cash position, burn rate, access to capital
- A company that can weather storms vs. one that can't = different risk profiles

**4. REGULATORY ENVIRONMENT** (0.0-1.0)
- Regulatory burden, political risk, compliance costs
- Heavily regulated vs. unregulated = different constraint sets

**5. CYCLE POSITION** (0.0-1.0)
- Economic cycle, industry maturity, technology adoption curve
- Early hypergrowth vs. mature optimization = different failure modes

For each score, provide 1-sentence reasoning explaining the comparison.

**OVERALL RELEVANCE**: Weight and synthesize into an overall score (0.0-1.0).
- Below 0.4: Analog may not be relevant enough to learn from
- 0.4-0.7: Instructive despite context differences
- Above 0.7: Strong structural match, high learning value"""

RTA_SYNTHESIZE_CHALLENGE_PROMPT = """As a senior equity research analyst, write a professional 
challenge to this investment assumption. Your goal is to help the thesis author think more 
rigorously - not to attack or dismiss their thesis.

The best challenges are CONSTRUCTIVE: they identify specific risks and what to monitor, 
rather than just saying "this could go wrong."

**ASSUMPTION BEING CHALLENGED:**
"{assumption_claim}"
(Evidence strength: {evidence_strength}, Thesis author's confidence: {confidence})

**HISTORICAL PRECEDENT:**
{case_name}
{analog_context}

**FAILURE MECHANISM:**
{failure_description}

**RELEVANCE TO CURRENT CASE:**
Overall score: {relevance_score}
{relevance_justification}

**EARLY WARNING INDICATORS:**
{early_warnings}

Write a 2-3 paragraph challenge that:

**PARAGRAPH 1: State the assumption and the risk**
- Clearly articulate what the thesis assumes
- Explain why this assumption carries risk (reference the historical precedent)

**PARAGRAPH 2: Why it matters for this specific case**
- Connect the historical failure mode to the current situation
- Acknowledge differences but explain why the risk is still relevant
- Be specific about what conditions would cause this risk to materialize

**PARAGRAPH 3: What to monitor**
- List specific, observable indicators to watch
- These should be LEADING indicators (warn before it's too late)
- Connect to the early warnings from the historical case

**TONE REQUIREMENTS:**
✓ Neutral and professional - not adversarial
✓ Falsifiable - the analyst can provide counter-evidence
✓ Constructive - helps the analyst think better, doesn't just criticize
✗ Avoid: "This is wrong" / "The analyst should have known" / Editorial judgment

Example opening: "The thesis assumes [X]. Historical precedent from [case] suggests that..."
NOT: "The analyst has made a critical error in assuming..."

Return plain text only (no JSON)."""


# --- COUNTERFACTUAL RESEARCH ENGINE (CRE) PROMPTS ---
CRE_BOUND_ASSUMPTIONS_PROMPT = """As a senior equity research analyst with deep experience in 
{stock_ticker}'s sector, you are setting EMPIRICALLY-GROUNDED bounds for each assumption.

This is the most critical step in stress testing. Fantasy bounds (revenue can grow 500%!) produce 
worthless scenario analysis. Your bounds must be anchored in HISTORICAL REALITY - what has 
actually happened to companies like {stock_ticker} under various conditions.

COMPANY: {stock_ticker}
CONTEXT: {company_context}

{quantitative_context}

THESIS METRICS TO BOUND:
{metrics_json}

QUALITATIVE CLAIMS:
{claims_json}

For EACH metric, provide EMPIRICALLY-GROUNDED bounds:

**1. LOWER BOUND**: What's the worst sustained outcome for {stock_ticker} or structurally 
similar peers? NOT a one-quarter blip, but a scenario that persists.
- Cite SPECIFIC historical analogs: company name, time period, actual numbers
- Example: "During the 2022 SaaS correction, {stock_ticker} saw growth drop from 68% to 29%"

**2. UPPER BOUND**: What's the best sustained outcome achievable? Not a one-off spike.
- Again cite SPECIFIC historical analogs
- Be realistic about mean reversion - hypergrowth doesn't last forever

**3. JUSTIFICATION**: Real historical data with dates and figures. No hand-waving.

GUIDANCE ON USING QUANTITATIVE CONTEXT:
The data above shows {stock_ticker}'s ACTUAL recent performance (growth, margins, trends). 
Use this as a sanity check - if the thesis assumes 25% revenue growth but trailing growth 
is 8%, that's a material gap requiring explanation.

HARD CONSTRAINTS (violations indicate unrealistic thesis):
- Revenue growth: -30% to +100% (no company sustains higher)
- Gross margin: 0% to 85% (cost floor exists)
- Operating margin: -30% to 50% (rare to exceed)
- Net retention: 70% to 150% (below 70% = dying, above 150% = implausible)
- WACC: 5% to 20% (market-driven reality) — use as bounding range only; do NOT substitute a single default if WACC is not inferable
- Terminal multiple: 5x to 25x (empirical range for mature companies) — use as bounding range only; do NOT substitute a single default if terminal multiple is not inferable"""

CRE_GENERATE_SCENARIOS_PROMPT = """As a senior equity research analyst, you are constructing 
PLAUSIBLE stress scenarios for {stock_ticker}'s thesis. These scenarios will test whether 
the investment survives realistic adversity.

Good stress testing asks: "What specific, historically-grounded adversity could this thesis face?"
Bad stress testing asks: "What if everything went wrong?" (too vague to be useful)

SUGGESTED SCENARIO TEMPLATES (use as inspiration, not constraints):
{scenario_templates_json}

PRIORITY CLAIMS (optional): {priority_claims_json}
DOWNSIDE BIAS: {downside_bias}

You may create scenarios that fit these templates OR generate thesis-specific scenarios 
that don't fit these categories. For example, if the thesis involves currency exposure, 
supply chain dependencies, management quality, technology disruption, or other factors 
not covered by the templates - create scenarios that stress-test those specific risks.

The goal is to stress-test THIS THESIS, not to fill a checklist.

COMPANY: {stock_ticker}

{quantitative_context}

THESIS CLAIMS (what must be true):
{claims_json}

BASE CASE METRICS (what the thesis assumes):
{metrics_json}

BOUNDED RANGES (scenarios MUST stay within these):
{bounds_json}

Generate 4-6 INTERNALLY CONSISTENT scenarios:

**SCENARIO DESIGN PRINCIPLES:**
1. Each scenario should have a clear CAUSAL NARRATIVE - not just "things get worse"
2. Adjustments must be DIRECTIONALLY CONSISTENT (demand slowdown → lower growth, not higher)
3. Each adjustment must cite a SPECIFIC HISTORICAL ANALOG (company, timeframe, what happened)
4. Include at least ONE UPSIDE scenario (intellectual honesty requires symmetry)
5. Plausibility weights should sum to approximately 1.0 across all scenarios

**ADJUSTMENT FORMAT (CRITICAL):**
For multiplicative changes (percentage of base case):
- Use "_factor" suffix: "revenue_growth_factor": 0.7 means growth is 70% of base case
  Example: if base growth = 25%, factor of 0.7 → stressed growth = 17.5%

For additive changes (percentage point shifts):
- Use plain metric name: "gross_margin": -0.05 means margin drops 5 percentage points
  Example: if base margin = 70%, delta of -0.05 → stressed margin = 65%

**QUALITY CHECKS:**
- Does each scenario tell a coherent story?
- Are the adjustments proportional to the historical analog cited?
- Would an investment committee find this scenario credible?
- Is there a plausible path from "today" to "this scenario"?"""


# --- Financial Translation Agent (FTA) PROMPTS ---
CRE_GENERATE_REASONING_PROMPT = """Analyze why this counterfactual scenario leads to the given valuation outcome for {stock_ticker}.

SCENARIO: {scenario_name}
{scenario_description}

{base_summary}

{stressed_summary}

FACTOR CONTRIBUTIONS (JSON): {factor_contributions_json}
METRIC->FACTOR MAPPING (JSON): {metric_map_json}

VALUATION IMPACT: {valuation_change}
OUTCOME TIER: {outcome_tier}

Return a JSON object with keys: `explanation` (2-3 sentence causal chain), `key_drivers` (array of 2-3 most important drivers), `historical_precedent` (a concise historical example or analog), `related_factors` (list of factor names referenced in reasoning), and `factor_explanation` (2-3 sentence explanation tying factor movements to valuation). Use the structured schema provided for this prompt."""

METRIC_TO_FACTOR_CLASSIFICATION_PROMPT = """You are a domain expert mapping thesis metrics to canonical value factors.

Given a metric name and (optional) value context, return a JSON object that lists the factor(s) this metric most influences and a coefficient for each factor in the range -1.0..1.0 indicating direction and relative strength. Also include a `confidence` (0.0-1.0) and a short `explanation`.

Metric: {metric}
Metric value (optional): {metric_value}
Company/context: {company_context}

Return a JSON object matching the provided schema and do not add extra keys."""

CRE_SUMMARY_PROMPT = """As a senior equity research analyst presenting stress test findings, 
summarize the key contradictions revealed by the scenario analysis for {stock_ticker}.

This summary is for the investment committee. Focus on what matters most: where does the 
thesis break, and what did the analyst potentially miss?

ORIGINAL THESIS CLAIMS:
{claims_json}

STRESS TEST RESULTS:
{results_json}

DEFAULTS/APPLIED INFERENCES (JSON): {defaults_json}
AGGREGATED FACTOR SCORES (JSON): {factor_scores_json}
METRIC->FACTOR MAPPING (JSON): {metric_map_json}

Return a JSON object matching the provided summary schema with the following keys:
- `vulnerable_claims`: list of {claim, failure_count, example_scenarios}
- `contradicting_evidence`: list of strings
- `blind_spots`: list of strings
- `asymmetry`: {direction: "DOWNSIDE"|"UPSIDE"|"BALANCED", description: string}
- `bullets`: 3-6 concise bullet points (each 1-2 sentences)

Additionally, include an explicit diagnostic section in the JSON for `defaults_applied` (list), `top_factors` (array of {factor, score}), and `metric_factor_mapping` (object) so the committee knows what was inferred or defaulted.

Use the schema carefully and ensure all fields are filled with concise, factual statements."""


# --- IDEA HALF-LIFE ESTIMATOR (IHLE) PROMPTS ---
IHLE_ADJUST_REGIME_PROMPT = """As a senior equity research analyst with macro awareness, assess 
how the current macroeconomic regime affects this thesis for {stock_ticker}.

Some theses are regime-dependent - they work in certain macro conditions but fail in others.
Understanding this dependency is critical for estimating thesis durability.

THESIS REGIME DEPENDENCIES IDENTIFIED:
{regime_tags}

CURRENT MACRO CONTEXT: {current_macro_context}

EVALUATE THREE DIMENSIONS:

**1. REGIME STATE** - Current macro stability
- "Stable": Clear regime with low uncertainty (steady rates, predictable policy)
- "Transitioning": Active regime change (rate hiking/cutting cycle, policy shifts)
- "Unstable": High uncertainty, multiple possible paths

**2. ALIGNMENT** (0.0 to 1.0) - How well does the thesis fit current regime?
- 1.0: Thesis is tailored for exactly this environment
- 0.5: Thesis works in most environments
- 0.0: Thesis requires opposite conditions

Example: A "growth at any price" thesis in a "higher for longer rates" regime = low alignment

**3. ADJUSTMENT FACTOR** (0.5x to 1.5x) - How should half-life be adjusted?
- Transitioning regimes ACCELERATE decay: 0.7-0.9x (thesis may become stale faster)
- Stable aligned regimes EXTEND half-life: 1.1-1.5x (thesis has longer runway)
- Unstable regimes SIGNIFICANTLY accelerate: 0.5-0.8x (high uncertainty shortens all time horizons)

REASONING: Explain the logic connecting regime state → alignment → adjustment.
- Be specific about which macro factors matter for this thesis
- Consider second-order effects (e.g., regime change might hurt near-term but help long-term)"""


