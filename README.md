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
5. **Estimating the thesis half-life** for monitoring cadence (Idea Half-Life Estimator)
6. **Aggregating results** into actionable recommendations

Output is a FinalThesisReport containing survival rate, fragility score, dominant failure modes, and calibrated confidence intervals.

---

## Agent Functions and Investment Logic

### 1. Narrative Decomposition Graph (NDG)

Traditional analyst narratives bundle assumptions together in ways that are hard to stress-test because the causal logic remains implicit. The NDG agent extracts explicit causal claims from natural language theses and builds a directed acyclic graph showing dependencies between assumptions.

Each node is classified by control (company vs. macro), nature (structural vs. cyclical), and time sensitivity. The agent maps supporting and contradicting evidence to each node, then computes fragility by measuring the probability that a single assumption failure cascades through the graph.

**Value Add**: Forces confrontation with specific failure modes that human analysts might rationalize away. Provides historical parallels that ground abstract risks in concrete outcomes.

### 2. AI Red Team

The Red Team agent generates specific challenges for each vulnerable node, using historical analogs to ground challenges in real-world precedent. For example, it might note that Zoom had similar growth in 2020 but decelerated significantly as pull-forward demand normalized.

Challenges are assigned severity ratings based on evidence strength and relevance, with focus on single-point failures and outcome nodes. Output includes challenges, severity distribution, and coverage metrics.

**Value Add**: Forces engagement with disconfirming evidence and prevents wishful thinking.

### 3. Counterfactual Research Engine (CRE)

The CRE extracts canonical assumptions from the NDG and assigns empirical bounds using historical analogs. It generates base, bull, and bear scenarios, then augments with Red Team-focused scenarios that stress-test specific challenges and extreme variants that push assumptions to empirical bounds.

The agent applies defaults to missing metrics using industry medians. Output is a scenario set containing 8-15 scenarios with varying plausibility weights.

**Value Add**: Creates genuinely diverse scenarios where historical bounds prevent anchoring and Red Team augmentation ensures worst-case scenarios receive proper attention.

### 4. Financial Translation (FT)

The Financial Translation agent takes each scenario and translates stressed assumptions to financial outcomes including revenue CAGR, margin trajectory, terminal growth, and WACC adjustments. It computes implied DCF valuations and upside/downside potentials.

The agent identifies impaired scenarios (>30% downside) and aggregates results with plausibility weights. Output includes valuation results, weighted upside/downside, and tail loss.

**Value Add**: Connects narrative stress-testing to valuation impact. Quantifies the cost of being wrong.

### 5. Thesis Validity Evaluation

The Validity Evaluator aggregates scenario results including weighted upside and downside, incorporates the NDG fragility score, and considers Red Team challenge severity. It classifies thesis status as VALID, QUESTIONABLE, or INVALID.

The agent computes calibrated confidence as survival rate multiplied by one minus fragility score. Output includes thesis status, survival rate, and specific contradictions.

**Value Add**: Provides a probabilistic verdict that reflects all evidence while preventing overconfidence through explicit penalties for fragile theses.

### 6. Idea Half-Life Estimator (IHLE)

The IHLE analyzes assumption decay rates to determine how quickly evidence can change. It considers causal path decay where upstream failures propagate downstream, and accounts for regime sensitivity where macro shifts can invalidate theses rapidly.

The agent computes a half-life estimate in days until the thesis needs re-evaluation and determines monitoring cadence ranging from daily to quarterly.

**Value Add**: Ensures high-frequency setups receive active monitoring while structural theses receive less frequent checks, preventing both overtrading and negligent monitoring.

### 7. Aggregation Diagnostics

The final agent computes an overall score combining fragility, upside, and challenge severity, aggregates key metrics including survival rate, tail loss, and half-life, and generates an executive summary.

The FinalThesisReport combines all outputs with calibrated confidence, suggested position size factor (0.0-1.0 multiplier for portfolio weight), dominant failure modes (top 3-5 risks), and half-life months for monitoring guidance.

---

## Architecture & Components

### Directory Structure

```
model/
├── calculators/          # Quantitative analysis
│   ├── calculator_base.py
│   ├── technical_calculator.py
│   ├── advanced_technical_calculator.py
│   ├── financial_calculator.py
│   ├── peer_intelligence_calculator.py
│   ├── macro_sensitivity_calculator.py
│   └── volume_positioning_calculator.py
│
├── thesis_agents/        # Validation pipeline agents
│   ├── narrative_decomposition_agent.py
│   ├── red_team_agent.py
│   ├── counterfactual_research_agent.py
│   ├── financial_translation_agent.py
│   ├── conviction_calibration_agent.py
│   ├── half_life_estimator_agent.py
│   ├── disconfirming_evidence_agent.py
│   └── llm_helpers.py
│
├── orchestration/        # Pipeline coordination
│   ├── stock_analytics_orchestrator.py     # Stage 1 coordinator
│   └── thesis_validation_orchestrator.py   # Stage 2 coordinator
│
├── services/             # Data providers and external integrations
│   ├── stock_data_provider.py
│   ├── index_provider.py
│   ├── peer_discovery_provider.py
│   ├── fred_provider.py
│   ├── research_data_provider.py
│   ├── anthropic_helper.py
│   ├── perplexity_research_provider.py
│   ├── sec_filing_manager.py
│   └── ten_k_processor.py
│
├── prompts/              # LLM prompts and schemas
│   ├── system_prompts.py
│   ├── thesis_validation_prompts.py
│   ├── perplexity_research_prompts.py
│   ├── output_schemas.py
│   └── utils.py
│
├── core/                 # Type definitions
│   ├── analytics_types.py
│   ├── thesis_types.py
│   ├── config.py
│   └── logging_config.py
│
└── tests/                # Comprehensive test suite
    ├── unit_tests/       # 129 tests
    └── examples/         # Integration examples
```

### Key Data Types

- **ThesisQuantitativeContext**: Stage 1 output with technical, financial, peer, and macro metrics
- **NDGOutput**: Narrative graph with nodes, dependencies, and fragility scores
- **RedTeamOutput**: Challenges with severity ratings and evidence
- **CREGenerationResult**: Scenario set with 8-15 plausible scenarios
- **FTOutput**: Valuations with upside/downside and tail loss
- **ThesisValidityOutput**: Status classification and survival rate
- **IHLEOutput**: Half-life estimates and monitoring cadence
- **FinalThesisReport**: Complete verdict with all metrics

---

## Usage Patterns

### 1. Full Pipeline Execution
```python
from model.orchestration.thesis_validation_orchestrator import ThesisValidationOrchestrator

thesis = """
Snowflake will grow 20%+ annually as enterprises consolidate data workloads
onto its platform. Management execution + platform stickiness will sustain
above-market growth.
"""

orchestrator = ThesisValidationOrchestrator(stock_ticker='SNOW')
report = orchestrator.run(
    thesis_narrative=thesis,
    company_context="Snowflake is a cloud data platform...",
    quantitative_context=None  # Optional: add Stage 1 output
)

print(f"Status: {report.detailed_components['validity'].thesis_status}")
print(f"Confidence: {report.calibrated_confidence:.2%}")
print(f"Half-Life: {report.half_life_months} months")
```

### 2. Stage 1 Analytics Only
```python
from model.orchestration.stock_analytics_orchestrator import StockAnalyticsOrchestrator

orchestrator = StockAnalyticsOrchestrator(stock_ticker='SNOW')
context = orchestrator.all_metrics()

print(f"Technical: {context.technical_metrics}")
print(f"Financial: {context.financial_metrics}")
print(f"Peer Position: {context.peer_metrics}")
print(f"Macro Sensitivity: {context.macro_sensitivity}")
```

### 3. Incremental Agent Testing
```python
from model.thesis_agents.narrative_decomposition_agent import NarrativeDecompositionAgent

agent = NarrativeDecompositionAgent('SNOW')
ndg_output = agent.run(
    thesis_narrative=thesis,
    company_context=company_context
)

print(f"Nodes: {len(ndg_output.nodes)}")
print(f"Fragility: {ndg_output.fragility_score:.2f}")
```

---

## Testing & Validation

The model includes comprehensive test coverage:

- **Stock Analytics Tests**: 75 tests covering all calculators and data providers
- **Thesis Validation Tests**: 57 tests covering all agents and orchestration

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

## Prompt Engineering Principles

1. **Explicit Role Definition**: Each agent has a clear role (e.g., "You are an adversarial analyst attacking an investment thesis")
2. **Historical Grounding**: Prompts require historical analogs to prevent hallucination
3. **Structured Outputs**: JSON schemas enforce consistent response formats
4. **Chain-of-Thought**: Agents are prompted to explain reasoning before conclusions
5. **Calibration**: Confidence scores must be justified with evidence

---

## Performance Considerations

### LLM API Costs
- Full thesis validation: ~7 LLM calls (1 per agent)
- Average tokens per run: 20,000-30,000 (input + output)
- Estimated cost: $0.50-1.00 per thesis validation (Claude pricing)

### Optimization Strategies
1. **Cache NDG outputs**: Reuse decomposition if thesis hasn't changed
2. **Parallel execution**: Red Team + CRE can run concurrently
3. **Conditional stages**: Skip IHLE or Aggregation for quick validation
4. **Batch processing**: Validate multiple theses in parallel

---

## Extension Points

### Adding Custom Calculators
```python
from model.calculators.calculator_base import CalculatorBase

class CustomCalculator(CalculatorBase):
    def calculate_custom_metric(self, price_history: pd.DataFrame) -> float:
        # Your calculation logic
        return result
```

### Adding Custom Thesis Agents
```python
from model.thesis_agents.llm_helpers import LLMHelperMixin

class CustomAgent(LLMHelperMixin):
    def run(self, input_data) -> OutputType:
        prompt = self.format_prompt(CUSTOM_PROMPT, **kwargs)
        result = self._call_llm_structured(prompt, CUSTOM_SCHEMA)
        return self._process_output(result)
```

---

## Future Enhancements (V2)

V2 will add outcome tracking, learning loops, and empirically-calibrated parameters. The current implementation uses policy parameters based on explicit priors and domain knowledge. Once sufficient historical thesis outcomes become available, V2 will transition to learned parameters grounded in actual prediction performance.

Key V2 areas:
- Outcome tracking database (predicted vs. actual results)
- Disconfirming evidence monitoring from news, earnings, SEC filings
- Calibration analysis with Brier scores and calibration curves
- Empirical parameter learning from return attribution
- A/B testing of prompts on historical data
