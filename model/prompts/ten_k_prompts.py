"""
10-K Extraction Prompts for SEC Filing Analysis.

This prompt is designed to extract credibility signals and strategic insights
from SEC filings, focusing on what management is saying (and NOT saying) rather
than financial numbers. The analysis mimics professional investor diligence.
"""

# --- MAIN 10-K EXTRACTION PROMPT ---
EXTRACTION_PROMPT = """EXTRACT CREDIBILITY SIGNALS AND STRATEGIC INSIGHTS FROM THIS 10-K FILING.

You are analyzing this filing as a professional investor would: looking for what management 
is REALLY saying beneath the boilerplate, assessing management credibility, and identifying 
strategic priorities through narrative analysis (not financial metrics).

CRITICAL INSTRUCTIONS:
- You MUST respond with ONLY a JSON object, nothing else
- No explanatory text, no markdown, no preamble, no code blocks
- Response must start with {{ and end with }}
- ALL values must be grounded in actual text from the filing
- Quote directly when possible to support your analysis
- Identify what is being AVOIDED or MINIMIZED as much as what is stated
- Focus on narrative quality, consistency, and management candor

ANALYSIS FRAMEWORK (Think like an investor reading the filing):

1. **Management Tone & Candor**: 
   - Are they being straightforward or heavily hedging risk?
   - How much do they acknowledge headwinds vs. focusing only on opportunities?
   - Do they discuss failures/setbacks or always pivot to positives?

2. **Strategic Narrative & Consistency**:
   - What are the core strategic pillars? Are they stable or shifting year-to-year?
   - Does the narrative feel coherent or are there contradictions?
   - How do they explain past guidance misses?

3. **Risk Disclosure Quality**:
   - Are material risks described specifically or vaguely?
   - Do they acknowledge emerging threats or minimize them?
   - How credible are their risk mitigation strategies?

4. **Attribution Patterns**:
   - When things go well: do they credit execution/strategy or external factors?
   - When things go poorly: do they take responsibility or blame externals?
   - This asymmetry reveals management alignment with shareholders

5. **Forward Guidance**:
   - Is forward guidance specific (with numbers) or vague qualitative statements?
   - Do they acknowledge contingencies and uncertainties?
   - How does current guidance compare to past promised targets?

6. **Capital Allocation & Priorities**:
   - What does CapEx policy reveal about growth confidence?
   - What do M&A activities and amounts say about strategy?
   - How much capital returned to shareholders vs. reinvested?

7. **Operational Transparency**:
   - Do they break out performance by segment/geography clearly?
   - How much detail on customer concentration, supplier dependencies?
   - Do they discuss competitive positioning vulnerabilities?

REQUIRED JSON OUTPUT SCHEMA:
{
  "management_tone": {
    "candor_assessment": "high|moderate|low",
    "hedging_intensity": "high|moderate|low",
    "acknowledgment_of_challenges": "transparent|selective|minimal",
    "optimism_bias": "evident|balanced|cautious",
    "key_quotes": [
      "direct quote showing tone",
      "another revealing quote"
    ]
  },
  "strategic_narrative": {
    "core_strategy_statement": "one sentence capturing the core strategic narrative",
    "strategic_pillars": ["pillar 1", "pillar 2", "pillar 3"],
    "narrative_consistency": "stable and consistent|moderately evolving|notably shifting",
    "consistency_notes": "brief explanation if narrative is shifting (e.g., new focus on cloud, de-emphasizing legacy products)",
    "management_agility": "rigid and committed|balanced and adaptable|reactive to changes",
    "evidence": "quote or reference supporting consistency assessment"
  },
  "risk_disclosure": {
    "material_risks_identified": [
      "specific risk with brief context",
      "another identified risk"
    ],
    "emerging_risks_acknowledged": [
      "risk management sees on horizon"
    ],
    "risk_minimization_signals": [
      "risk discussed but appears downplayed"
    ],
    "disclosure_quality": "detailed and specific|reasonably detailed|vague and generic",
    "mitigation_credibility": "high (specific plans)|moderate (general approach)|low (reassurance without substance)",
    "most_concerning_gap": "risk that should be disclosed but appears minimized"
  },
  "management_credibility": {
    "success_attribution": {
      "pattern": "mostly internal execution|balanced internal-external|mostly external factors",
      "example": "quote or reference showing how they attribute wins"
    },
    "failure_attribution": {
      "pattern": "takes responsibility|balanced|deflects externally",
      "example": "quote or reference showing how they explain setbacks"
    },
    "attribution_asymmetry": "high (credit wins, blame externals)|moderate|low (balanced)",
    "candor_indicators": {
      "past_guidance_misses_addressed": true,
      "past_guidance_statement": "how they addressed/explained prior misses",
      "competitive_losses_acknowledged": true,
      "competitive_loss_statement": "example of acknowledging competitive setback"
    },
    "overall_credibility_score": "1-10 with reasoning"
  },
  "forward_guidance": {
    "type": "quantitative (specific numbers)|qualitative (general direction)|mixed",
    "specificity_examples": [
      "example of specific guidance (e.g., revenue growth 10-12% in FY2024)",
      "example of vague guidance (e.g., expect to continue strong growth)"
    ],
    "conservatism": "conservative (targets likely to be beat)|realistic (reasonable targets)|aggressive (aggressive targets)",
    "contingency_acknowledgment": "explicitly discusses uncertainties|acknowledges major risks|limited uncertainty discussion",
    "guidance_track_record_note": "any commentary on whether company hit prior guidance"
  },
  "capital_allocation": {
    "capex_philosophy": "growth-oriented description|maintenance-focused|opportunistic",
    "capex_trends": "increasing investment in X|maintaining baseline spending|reducing/deferring capex",
    "m_and_a_strategy": "active M&A pursuing growth|selective M&A|minimal M&A activity",
    "capital_return_approach": "dividend focused|buyback focused|balanced|minimal returns",
    "allocation_coherence": "allocation matches stated strategy|misalignment between stated strategy and capital decisions"
  },
  "operational_transparency": {
    "segment_disclosure_quality": "detailed breakout by segment|some disclosure|limited transparency",
    "geographic_exposure_clarity": "clear country/region breakdown|general regions mentioned|minimal geographic detail",
    "customer_concentration_disclosed": true,
    "customer_concentration_note": "top customer is X% of revenue" or "no major customer represents >10%",
    "competitive_positioning": "discusses competitive strengths and vulnerabilities|mostly positive positioning|vague on competition",
    "key_operational_metrics": [
      "important non-financial metric disclosed (e.g., units sold, customer count, retention rate)"
    ]
  },
  "red_flags": [
    "red flag 1 - be specific",
    "red flag 2 - something that raises credibility concerns"
  ],
  "green_flags": [
    "positive sign 1 - management transparency or strategic clarity",
    "positive sign 2 - example of credible risk acknowledgment"
  ],
  "investor_summary": {
    "management_credibility_assessment": "high credibility|moderate credibility|low credibility",
    "key_strengths": "1-2 sentence summary of what management is executing well",
    "key_concerns": "1-2 sentence summary of main investor concerns from analysis",
    "narrative_quality": "coherent and detailed|reasonably clear|unclear or inconsistent"
  }
}

GUIDANCE ON ANALYSIS:
- Look for PATTERNS in how management discusses performance, not isolated quotes
- Compare Risk Factors section (Required Disclosure) with MD&A (Chosen Disclosure) - gaps reveal priorities
- Track strategic focus: are they investing in stated priorities or elsewhere?
- Assess narrative evolution: does this year's story contradict last year's?
- Identify what's missing: what material topics are absent or minimally discussed?
- Real credibility comes from candid acknowledgment of problems + specific mitigation plans
- Vague forward guidance + aggressive tone = higher execution risk
- Management credibility = transparency about past execution + realistic forward outlook
"""
