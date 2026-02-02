# AI Research System - Model Architecture

## Overview

This system implements a systematic investment research pipeline that validates investment theses through adversarial testing, counterfactual analysis, and quantitative decomposition. It combines traditional quantitative analysis with AI-powered thesis validation to provide comprehensive investment recommendations grounded in probabilistic reasoning.

The core philosophy recognizes that investment theses are fragile narratives built on chains of assumptions. Traditional research suffers from confirmation bias, seeking evidence that supports the thesis while ignoring contradictory signals. This system inverts that approach by systematically attacking the thesis, searching for disconfirming evidence, and quantifying failure probability before recommending a position.

## Two-Stage Architecture

### Stage 1: Stock Analytics

Stage 1 establishes an objective quantitative baseline using market data, financials, technical indicators, and macro sensitivity. It provides empirical metrics that anchor the subsequent narrative analysis while identifying red flags that warrant deeper investigation.

The stage includes technical calculators (price momentum, trend strength, support/resistance), financial calculators (growth decomposition, margin analysis, ROIC trends), peer intelligence (relative performance, correlation analysis), and macro sensitivity measures (interest rate exposure, cycle positioning). 

Output is a structured ThesisQuantitativeContext that feeds into Stage 2 as supplementary context rather than authoritative truth.

### Stage 2: Thesis Validation (Adversarial Analysis)
**Purpose**: Stress-test the investment narrative through systematic decomposition and challenge generation

This is where AI fundamentally transforms the investment process. Traditional research is narrative-driven and confirmation-biased. Stage 2 inverts this by:

1. **Decomposing the thesis into falsifiable claims** (Narrative Decomposition Graph)
2. **Generating adversarial challenges** for each claim (AI Red Team)
3. **Creating counterfactual scenarios** that stress-test assumptions (Counterfactual Research)
4. **Translating scenarios to valuations** to quantify downside risk (Financial Translation)
5. **Estimating the thesis hal

Stage 2 stress-tests the investment narrative through systematic decomposition and adversarial analysis. The process decomposes the thesis into falsifiable claims, generates specific challenges for each claim, creates counterfactual scenarios, translates those scenarios to valuations, estimates the thesis half-life for monitoring cadence, and aggregates results into actionable recommendations.

Output is a FinalThesisReport containing survival rate, fragility score, dominant failure modes, and calibrated confidence intervals.

---

## Agent Functions and Investment Logicy vs. macro), nature (structural vs. cyclical), and time sensitivity
- Map sNarrative Decomposition Graph (NDG)

Traditional analyst narratives bundle assumptions together in ways that are hard to stress-test because the causal logic remains implicit. The NDG agent extracts explicit causal claims from natural language theses and builds a directed acyclic graph showing dependencies between assumptions.

Each node is classified by control (company vs. macro), nature (structural vs. cyclical), and time sensitivity. The agent maps supporting and contradicting evidence to each node, then computes fragility by measuring the probability that a single assumption failure cascades through the graph.

This converts fuzzy narratives into testable structures and identifies single-point failures where breaking one assumption collapses the entire thesis
- Assigns severity scores (HIGH/MEDIUM/LOW) based on evidence strength
- Focuses on the most vulnerable nodes (high fragility, weak evidence)

**Value Add**: Forces confrontation with specific failure modes that human analysts might rationalize away. Provides historical parallels that ground abstract risks in concrete outcomes.

### 3. Counterfactual Scenario Generation (The "What If" Engine)
I Red Team

Traditional risk analysis produces generic concerns that lack specificity and historical grounding. The Red Team agent generates specific, evidence-based challenges for each node in the NDG, using historical analogs to ground challenges in real-world precedent.

Challenges are assigned severity scores based on evidence strength, with focus on the most vulnerable nodes. This forces confrontation with specific failure modes that analysts might rationalize away and p
### 4. Financial Translation (Narrative → Numbers)

**Traditional Approach**: DCF models are built separately from the narrative. Assumptions are input manually, leading to disconnect between story and valuation.

**AI Enhancement**: The Financial Translation (FT) agent:
- Takes each counterfactual scenario and translates it to financial outcomes (revenue CAGR, margin trajectory, terminal growth, WACC)
- Computes implied valuations and upside/downside potentials
- Identifies "impairedResearch Engine (CRE)

Traditional scenario analysis suffers from anchoring bias, with base/bull/bear cases clustered too closely together. The CRE extracts canonical assumptions from the NDG and assigns empirical bounds using historical analogs to ensure scenarios reflect realistic ranges.

The agent generates diverse scenarios including extreme variants and cases focused on specific Red Team challenges. When challenges are severe, it creates asymmetric distributions with more downside scenarios. This produces genuinely diverse scenario sets that reflect tail risks rather than clustered variations on the
- Considers causal path dependencies (if upstream assumptions decay fast, downstream nodes are vulnerable)
- Accounts for macro regime sensitivity (rate changes, cycle shifts can invalidate assumptions rapidly)
- Outputs recommended monitoring cadence (daily/weekly/monthly/quarterly)

**Value Add**: Provides data-driven guidance on how actively to monitor a position. High-frequency trading setups get daily monitoring; structural theses get quarterly checks.

### 6. Validity Evaluation & Aggregation (The Verdict)


DCF models are typically built separately from narratives, creating disconnects between story and valuation. The Financial Translation agent takes each counterfactual scenario and translates it to financial outcomes including revenue CAGR, margin trajectory, terminal growth, and WACC adjustments.

The agent computes implied valuations and upside/downside potentials, identifies impaired scenarios where valuation drops significantly, and aggregates scenario results with plausibility weights. This ece by explicitly penalizing fragile theses and severe challenges.

---

## Architecture & Components

### Directory Structure

```Idea Half-Life Estimator (IHLE)

Traditional approaches lack systematic methods for determining how quickly a thesis can decay. The IHLE analyzes assumption decay rates for each node, considers causal path dependencies where upstream failures propagate downstream, and accounts for macro regime sensitivity.

The agent outputs a recommended monitori

Traditional conviction levels are subjective without systematic aggregation of evidence. The Validity Evaluator aggregates scenario results including weighted upside and downside, incorporates the NDG fragility score, and considers Red Team challenge severity.

The agent classifies thesis status as VALID, QUESTIONABLE, or INVALID, and computes calibrated confidence as survival rate multiplied by one minus fragility score. This produces a probabilistic verdict that reflects all evidence while preventing overconfidence through explicit penalties for
│   ├── stock_analytics_orchestrator.py # Stage 1 coordinator
│   └── thesis_validation_orchestrator.py # Stage 2 coordinator
│
├── services/             # Data providers and external integrations
│   ├── stock_data_provider.py          # yfinance wrapper for price/financial data
│   ├── index_provider.py               # Benchmark data (S&P 500, sector indices)
│   ├── peer_discovery_provider.py      # Sector/industry peer identification
│   ├── fred_provider.py                # FRED economic data (rates, GDP)
│   ├── research_data_provider.py       # Aggregate data for thesis context
│   ├── anthropic_helper.py             # Claude API client wrapper
│  Directory Structure

The system is organized into calculators for quantitative analysis, thesis agents for validation logic, orchestration for pipeline coordination, services for data providers, prompts for LLM interactions, and core types for data structures.

Key directories include:
- calculators: Technical and financial calculators for Stage 1 analysis
- thesis_agents: Seven agents implementing the validation pipeline
- orchestration: Pipeline coordinators for both stages
- services: Data providers for market data, SEC filings, and research
- prompts: LLM prompts and structured output schemas
- core: Type definitions and configuration
- tests: 129 tests covering both stages
**Why This Matters**: Exposes hidden assumptions. A thesis like "revenue growth will accelerate" actually contains multiple sub-assumptions (TAM expansion + share gains + retention improvement). The NDG makes these explicit and shows which ones are most fragile.

#### Step 2: AI Red Team with Memory
**Input**: NDG output + company context  
**Process**:
1. For each vulnerable node (high fragility, low evidence), generate specific challenges
2. Use historical analogs to ground challenges ("Zoom also had 40% growth in 2020 but decelerated to 10% by 2022 as pull-forward demand normalized")
3. Assign severity (HIGH/MEDIUM/LOW) based on evidence strength and relevance
4. Focus on single-point failures and outcome nodes

**Output**: `RedTeamOutput` with challenges, severity distribution, coverage metrics

**Why This Matters**: Forces engagement with disconfirming evidence. Prevents "this time is different" thinking by providing concrete historical parallels.

#### Step 3: Counterfactual Research Engine (CRE)
**Input**: NDG + Red Team output + company context  
**Process**:
1. Extract canonical assumptions from NDG (e.g., "revenue_growth: 40%")
2. Assign empirical bounds using historical analogs (min=20%, max=60%)
3. Generate base/bull/bear scenarios
4. Augment with Red Team-focused scenarios (stress-test specific challenges)
5. Generate extreme variants (push assumptions to empirical bounds)
6. Apply defaults to any missing metrics (e.g., if margin not specified, use industry median)

**Output**: `CREGenerationResult` with `CREScenarioSet` containing 8-15 scenarios

**Why This Matters**: Creates genuinely diverse scenarios. Historical bounds prevent anchoring. Red Team augmentation ensures the worst-case scenarios aren't ignored.

#### Step 4: Financial Translation (FT)
**Input**: CRE scenario set  
**Process**:
1. For each scenario, translate stressed assumptions to financial outcomes:
   - Revenue CAGR (5-year)
   - Margin trajectory (expansion or compression)
   - Terminal growth rate
   - WACC adjustment (risk premium changes)
2. Compute implied DCF valuation
3. Calculate upside/downside potential vs. current price
4. Identify "impaired scenarios" (>30% downside)
5. Aggregate results with plausibility weights

**Output**: `FTOutput` with valuation results, weighted upside/downside, tail loss

**Why This Matters**: Connects narrative stress-testing to valuation impact. Quantifies the cost of being wrong. Tail loss (probability-weighted downside in worst scenarios) is often the most important metric.

#### Step 5: Thesis Validity Evaluation
**Input**: NDG + Red Team + FT outputs  
**Process**:
1. Aggregate scenario results (weighted upside, downside, tail loss)
2. Incorporate fragility score from NDG
3. Factor in Red Team challenge severity (high-severity challenges penalize survival rate)
4. Classify thesis status:
   - **VALID**: Positive weighted upside, low fragility, manageable challenges
   - **QUESTIONABLE**: Mixed signals, moderate fragility or severe challenges
   - **INVALID**: Negative weighted upside, high fragility, or overwhelming challenges
5. Build contradiction list (specific reasons the thesis might fail)
6. Pipeline Execution Flow

The pipeline executes sequentially with each agent building on prior outputs. Agents support independent re-execution with adjustable parameters, allowing iteration on specific components without rerunning the entire pipeline.

```python
orchestrator = ThesisValidationOrchestrator(stock_ticker='SNOW')

report = orchestrator.run(
    thesis_narrative="Snowflake will grow 20%+ due to data consolidation trends...",
    company_context="Snowflake is a cloud data platform...",
    quantitative_context=stage1_output  # Optional
**Output**: `IHLEOutput` with half_life_days, monitoring_cadence

**Why This Matters**: High-frequency setups (earnings beats, product cycles) need active monitoring. Structural theses (demographic trends, secular growth) can be checked quarterly. This prevents both overtrading and negligent monitoring.

#### Step 7: Aggregation Diagnostics
**Input**: All prior stage outputs  
**Process**:
1. Compute overall score (0-100) combining fragility, upside, challenge severity
2. Aggregate key metrics (survival rate, tail loss, half-life)
3. Generate executive summary

**Output**: `AggregationDiagnosticsOutput` with overall_score, key_metrics

**Final Report**: `FinalThesisReport` combining all outputs with:
- `calibrated_confidence` = survival_rate × (1 - fragility_score)
- `suggested_position_size_factor` (0.0-1.0 multiplier for portfolio weight)
- `dominant_failure_modes` (top 3-5 risks)
- `half_life_months` (monitoring cadence guidance)

**S Step 2: AI Red Team

The Red Team agent generates specific challenges for each vulnerable node, using historical analogs to ground challenges in real-world precedent. For example, it might note that Zoom had similar growth in 2020 but decelerated significantly as pull-forward demand normalized.

Challenges are assigned severity ratings based on evidence strength and relevance, with focus on single-point failures and outcome nodes. Output includes challenges, severity distribution, and coverage metrics.

This forces engagement with disconfirming evidence and prevents wishful

## Existing Infrastructure Supporting Future Enhancements

The current V1 implementation includes several features that enable future monitoring, tracking, and learning capabilities:

### Step 3: Counterfactual Research Engine

The CRE extracts canonical assumptions from the NDG and assigns empirical bounds using historical analogs. It generates base, bull, and bear scenarios, then augments with Red Team-focused scenarios that stress-test specific challenges and extreme variants that push assumptions to empirical bounds.

The agent applies defaults to missing metrics using industry medians. Output is a scenario set containing 8-15 scenarios with varying plausibility weights.

This creates genuinely diverse scenarios where historical bounds prevent anchoring and Red Team augmentation ensures worst-case scenarios receive proper attention
- **V2 Value**: Provides concrete benchmarks to compare against actual results in monitoring stage

### 4. Half-Life Guidance
- **Implementation**: IHLE computes monitoring cadence (daily/weekly/monthly/quarterly)
- **Usage**: Outputs `monitoring_cadence` enum with explicit recommendation
- **V2 Value**: Determines how often to run disconfirming evidence checks

### 5. NDG as Monitoring Template
- **Implementation**: Narrative Decomposition Graph explicitly maps thesis structure
- **Usage**: Each node represents a testable assumption with evidence requirements
- **V2 Value**: Provides explicit list of "what to monitor" rather than generic metrics

### 6. Dominant Failure Modes Identification
- **Implementation**: Validity Evaluator identifies top 3-5 risks from Red Team challenges
- **Usage**: `dominant_failure_modes` list in `FinalThesisReport`
- **V2 Value**: Enables targeted monitoring (did these specific risks materialize?)

### 7. Agent-Specific Parameters
- **Implementation**: Each agent exposes configuration constants (thresholds, limits, formulas)
- **Usage**: `MAX_CHALLENGES`, `HIGH_SEVERITY_THRESHOLD`, decay rate constants, etc.
- **V2 Value**: Enables A/B testing and prompt evolution based on calibration data

### 8. Test Suite for Reliability
- **Implementation**: 129 tests validate agent logic and edge cases
- **Usage**: Ensures consistent behavior across different thesis types
- **V2 Value**: Provides confidence baseline for predictions (agents are systematically tested)

These features make V2 implementation more feasible—the infrastructure for persistence, monitoring, and iteration is already present.

---

## Usage Patterns

### 1. Full Pipeline Execution
```python
from model.orchestration.thesis_validation_orchestrator import ThesisValidationOrchestrator

# Define thesis and context
thesis = """
Snowflake will grow 20%+ annually as enterprises consolidate data workloads
onto its platform. Management execution + platform stickiness will sustain
abo Step 6: Idea Half-Life Estimator

The IHLE analyzes assumption decay rates to determine how quickly evidence can change, considers causal path decay where upstream failures propagate downstream, and accounts for regime sensitivity where macro shifts can invalidate theses rapidly.

The agent computes a half-life estimate in days until the thesis needs re-evaluation and determines monitoring cadence ranging from daily to quarterly. Output includes half-life days and monitoring cadence.

This ensures high-frequency setups receive active monitoring while structural theses receive less frequent checks, preventing
    company_context=company_context,
    quantitative_context=None  # Optional: add Stage 1 output
)

# Interpret results
print(f"Status: {report.detailed_components['validity'].thesis_status}")
pri Step 7: Aggregation Diagnostics

The final agent computes an overall score combining fragility, upside, and challenge severity, aggregates key metrics including survival rate, tail loss, and half-life, and generates an executive summary.

The FinalThesisReport combines all outputs with calibrated confidence (survival rate multiplied by one minus fragility score), suggested position size factor (0.0-1.0 multiplier for portfolio weight), dominant failure modes (top 3-5 risks), and half-life months for monitoring guidance.

Each stage supports parameter customization and independent re-execution. Analysts can adjust thresholds, limits, and formulas for individual agents without rerunning the entire pipeline, though downstream stages should be re-executed if upstream inputs change materially
### 3. Incremental Agent Testing
```python
# Test individual agents during development
from model.thesis_agents.narrative_decomposition import NarrativeDecompositionGraph
from model.thesis_agents.red_team import AIRedTeamWithMemory

ndg = NarrativeDecompositionGraph('SNOW')
ndg_output = ndg.run(thesis_narrative=thesis, company_context=company_context)

print(f"Nodes: {len(ndg_output.nodes)}")
print(f"Fragility: {ndg_output.fragility_metrics.fragility_score:.2f}")


- **Max Tokens**: 2000-4000 per call (varies by agent complexity)

### Prompt Engineering Principles
1. **Explicit Role Definition**: Each agent has a clear role (e.g., "You are an adversarial analyst attacking an investment thesis")
2. **Historical Grounding**: Prompts require historical analogs to prevent hallucination
3. **Structured Outputs**: JSON schemas enforce consistent response formats
4. **Chain-of-Thought**: Agents are prompted to explain reasoning before conclusions
5. **Calibration**: Confidence scores must be justified with evidence

### Agent-Specific Configuration

| Agent | Temperature | Max Tokens | Key Schema Fields |
|-------|------------|------------|-------------------|
| NDG | 0.0 | 3000 | claims[], metrics{}, dependencies[] |
| Red Team | 0.2 | 3000 | challenges[], severity, evidence_strength |
| CRE | 0.3 | 4000 | scenarios[], stressed_assumptions{}, plausibility_weight |
| FT | 0.0 | 2500 | revenue_cagr, margin_expansion, implied_price |
| IHLE | 0.1 | 2000 | half_life_days, confidence_score, reasoning |
| Validity | 0.0 | 2000 | thesis_status, survival_rate, contradictions[] |

---

## Testing & Validation

The model includes comprehensive test coverage:

- **Stock Analytics Tests**: 13 files, 75 tests covering all calculators and data providers
- **Thesis Validation Tests**: 9 files, 61 tests covering all agents and orchestration

Run tests:
```bash
# All tests
pytest model/tests/ -v

# Stock analytics only
pytest model/tests/stock_analytics/ -v

# Thesis validation only
pytest model/tests/thesis_validation/ -v

# With coverage
pytest model/tests/ --cov=model --cov-report=html
```

---

## Extension Points

### Adding Custom Calculators
Extend `CExamples

###odel.calculators.calculator_base import CalculatorBase

class CustomCalculator(CalculatorBase):
    def calculate_custom_metric(self, price_history: pd.DataFrame) -> float:
        # Your calculation logic
        return result
```

### Adding Custom Thesis Agents
Extend `LLMHelperMixin` in `model/thesis_agents/llm_helper.py`:
```python
from model.thesis_agents.llm_helper import LLMHelperMixin

class CustomAgent(LLMHelperMixin):
    def run(self, input_data) -> OutputType:
        prompt = self.format_prompt(CUSTOM_PROMPT, **kwargs)
        result = self._call_llm_structured(prompt, CUSTOM_SCHEMA)
        return self._process_output(result)
```

### Modifying the Pipeline
Edit orchestrators in `model/orchestration/`:
- Add stages to `ThesisValidationOrchestrator.run()`
- Inject custom agents via constructor
- Modify output aggregation in `AggregationDiagnostics`

---

## Performance Considerations

### LLM API Costs
- Full thesis validation: ~7 LLM calls (1 per agent)
- Avge tokens per run: 20,000-30,000 (input + output)
- Estimated cost: $0.50-1.00 per thesis validation (Claude pricing)

### Optimization Strategies
1. **Cache NDG outputs**: Reuse decomposition if thesis hasn't changed
2. **Parallel execution**: Red Team + CRE can run concurrently (not implemented yet)
3. **Conditional stages**: Skip IHLE or Aggregation for quick validation
4. **Batch processing**: Validate multiple theses in parallel

### Scaling Considerations
- Store results in database (not implemented - currently in-memory)
- Rate limit LLM calls to avoid API throttling
- Consider local LLM for cost-sensitive applications (with quality tradeoff)

---



---

## V2 Architecture

The current implementation uses policy parameters based on explicit priors and domain knowledge rather than empirically-calibrated parameters learned from historical outcomes. V2 will replace these policy parameters with learned parameters once sufficient outcome data becomes available.

### V2 Enhancements

V2 will focus on three key areas:

**1. Outcome Tracking and Calibration**

The system will record thesis predictions and actual outcomes to enable calibration analysis. A database will store predicted survival rates, fragility scores, and dominant failure modes alongside realized stock performance and thesis invalidation events. This data will support calibration metrics including Brier scores and calibration curves showing predicted versus actual success rates.

**2. Learning Loop**

Historical outcomes will inform prompt refinement and parameter adjustment. The system will identify systematic biases such as overestimating tech stock survival rates or underweighting regulatory risks. Prompt evolution will address these biases through A/B testing on historical data before deployment. Agent weights and thresholds will adjust based on which agents prove most predictive.

**3. Empirical Parameter Calibration**

Policy parameters will transition to learned parameters once sufficient outcome data exists. Financial Translation will learn factor weights from return attribution analysis rather than using assumed coefficients. Red Team severity thresholds will calibrate to actual failure rates. Half-Life estimation will use sector-specific baselines derived from measured thesis persistence. Validity Evaluator thresholds will ground in realized portfolio drawdown tolerance.

### Implementation Timeline

Phase 1 (Months 1-6) establishes outcome tracking infrastructure with a database recording predictions alongside actual stock performance and thesis invalidation events.

Phase 2 (Months 6-12) adds disconfirming evidence monitoring that scans news, earnings releases, and SEC filings for contradictions to thesis assumptions, generating alerts when evidence degrades.

Phase 3 (Months 12-24) implements the calibration and learning loop, analyzing which predictions proved accurate, identifying systematic biases, and refining prompts based on outcome patterns.

Phase 4 (Months 24+) deploys empirically-calibrated parameters replacing policy priors with learned weights from historical data.

### Data Requirements

V2 calibration requires 50-100 historical thesis outcomes with labeled failures. This enables learning optimal factor weights from return attribution, calibrating survival thresholds from predicted versus actual success rates, and validating Red Team severity against materialized risks.

The current implementation makes no false claims of predictive power while documenting all policy decisions explicitly. V2 will add empirical grounding once sufficient outcome data exists.


