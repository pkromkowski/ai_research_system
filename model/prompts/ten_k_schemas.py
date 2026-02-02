"""JSON output schemas for 10-K extraction."""

TEN_K_EXTRACTION_SCHEMA = {
    "name": "extract_10k_signals",
    "description": "Extract credibility signals and strategic insights from a 10-K filing",
    "input_schema": {
        "type": "object",
        "properties": {
            "management_tone": {
                "type": "object",
                "properties": {
                    "candor_assessment": {
                        "type": "string",
                        "enum": ["high", "moderate", "low"],
                        "description": "Overall assessment of management candor"
                    },
                    "hedging_intensity": {
                        "type": "string",
                        "enum": ["high", "moderate", "low"],
                        "description": "Level of hedging language used"
                    },
                    "acknowledgment_of_challenges": {
                        "type": "string",
                        "enum": ["transparent", "selective", "minimal"],
                        "description": "How openly management discusses challenges"
                    },
                    "optimism_bias": {
                        "type": "string",
                        "enum": ["evident", "balanced", "cautious"],
                        "description": "Degree of optimism bias in narrative"
                    },
                    "key_quotes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Direct quotes showing tone"
                    }
                },
                "required": ["candor_assessment", "hedging_intensity", "acknowledgment_of_challenges", "optimism_bias", "key_quotes"]
            },
            "strategic_narrative": {
                "type": "object",
                "properties": {
                    "core_strategy_statement": {
                        "type": "string",
                        "description": "One sentence capturing core strategic narrative"
                    },
                    "strategic_pillars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key strategic pillars"
                    },
                    "narrative_consistency": {
                        "type": "string",
                        "enum": ["stable and consistent", "moderately evolving", "notably shifting"],
                        "description": "Consistency of strategic narrative"
                    },
                    "consistency_notes": {
                        "type": "string",
                        "description": "Explanation if narrative is shifting"
                    },
                    "management_agility": {
                        "type": "string",
                        "enum": ["rigid and committed", "balanced and adaptable", "reactive to changes"],
                        "description": "Management adaptability assessment"
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Quote or reference supporting consistency assessment"
                    }
                },
                "required": ["core_strategy_statement", "strategic_pillars", "narrative_consistency", "consistency_notes", "management_agility", "evidence"]
            },
            "risk_disclosure": {
                "type": "object",
                "properties": {
                    "material_risks_identified": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific material risks identified"
                    },
                    "emerging_risks_acknowledged": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Emerging risks management acknowledges"
                    },
                    "risk_minimization_signals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Risks that appear downplayed"
                    },
                    "disclosure_quality": {
                        "type": "string",
                        "enum": ["detailed and specific", "reasonably detailed", "vague and generic"],
                        "description": "Quality of risk disclosure"
                    },
                    "mitigation_credibility": {
                        "type": "string",
                        "enum": ["high (specific plans)", "moderate (general approach)", "low (reassurance without substance)"],
                        "description": "Credibility of risk mitigation strategies"
                    },
                    "most_concerning_gap": {
                        "type": "string",
                        "description": "Most concerning risk disclosure gap"
                    }
                },
                "required": ["material_risks_identified", "emerging_risks_acknowledged", "risk_minimization_signals", "disclosure_quality", "mitigation_credibility", "most_concerning_gap"]
            },
            "management_credibility": {
                "type": "object",
                "properties": {
                    "success_attribution": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "enum": ["mostly internal execution", "balanced internal-external", "mostly external factors"]
                            },
                            "example": {"type": "string"}
                        },
                        "required": ["pattern", "example"]
                    },
                    "failure_attribution": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "enum": ["takes responsibility", "balanced", "deflects externally"]
                            },
                            "example": {"type": "string"}
                        },
                        "required": ["pattern", "example"]
                    },
                    "attribution_asymmetry": {
                        "type": "string",
                        "enum": ["high (credit wins, blame externals)", "moderate", "low (balanced)"],
                        "description": "Asymmetry in attribution patterns"
                    },
                    "candor_indicators": {
                        "type": "object",
                        "properties": {
                            "past_guidance_misses_addressed": {"type": "boolean"},
                            "past_guidance_statement": {"type": "string"},
                            "competitive_losses_acknowledged": {"type": "boolean"},
                            "competitive_loss_statement": {"type": "string"}
                        },
                        "required": ["past_guidance_misses_addressed", "past_guidance_statement", "competitive_losses_acknowledged", "competitive_loss_statement"]
                    },
                    "overall_credibility_score": {
                        "type": "string",
                        "description": "1-10 score with reasoning"
                    }
                },
                "required": ["success_attribution", "failure_attribution", "attribution_asymmetry", "candor_indicators", "overall_credibility_score"]
            },
            "forward_guidance": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["quantitative (specific numbers)", "qualitative (general direction)", "mixed"],
                        "description": "Type of guidance provided"
                    },
                    "specificity_examples": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Examples of guidance specificity"
                    },
                    "conservatism": {
                        "type": "string",
                        "enum": ["conservative (targets likely to be beat)", "realistic (reasonable targets)", "aggressive (aggressive targets)"],
                        "description": "Conservatism of guidance"
                    },
                    "contingency_acknowledgment": {
                        "type": "string",
                        "enum": ["explicitly discusses uncertainties", "acknowledges major risks", "limited uncertainty discussion"],
                        "description": "How guidance acknowledges uncertainties"
                    },
                    "guidance_track_record_note": {
                        "type": "string",
                        "description": "Commentary on prior guidance accuracy"
                    }
                },
                "required": ["type", "specificity_examples", "conservatism", "contingency_acknowledgment", "guidance_track_record_note"]
            },
            "capital_allocation": {
                "type": "object",
                "properties": {
                    "capex_philosophy": {
                        "type": "string",
                        "enum": ["growth-oriented description", "maintenance-focused", "opportunistic"],
                        "description": "CapEx philosophy"
                    },
                    "capex_trends": {
                        "type": "string",
                        "description": "Description of CapEx trends"
                    },
                    "m_and_a_strategy": {
                        "type": "string",
                        "enum": ["active M&A pursuing growth", "selective M&A", "minimal M&A activity"],
                        "description": "M&A strategy"
                    },
                    "capital_return_approach": {
                        "type": "string",
                        "enum": ["dividend focused", "buyback focused", "balanced", "minimal returns"],
                        "description": "Approach to returning capital"
                    },
                    "allocation_coherence": {
                        "type": "string",
                        "enum": ["allocation matches stated strategy", "misalignment between stated strategy and capital decisions"],
                        "description": "Coherence between strategy and allocation"
                    }
                },
                "required": ["capex_philosophy", "capex_trends", "m_and_a_strategy", "capital_return_approach", "allocation_coherence"]
            },
            "operational_transparency": {
                "type": "object",
                "properties": {
                    "segment_disclosure_quality": {
                        "type": "string",
                        "enum": ["detailed breakout by segment", "some disclosure", "limited transparency"],
                        "description": "Quality of segment disclosure"
                    },
                    "geographic_exposure_clarity": {
                        "type": "string",
                        "enum": ["clear country/region breakdown", "general regions mentioned", "minimal geographic detail"],
                        "description": "Clarity of geographic exposure"
                    },
                    "customer_concentration_disclosed": {"type": "boolean"},
                    "customer_concentration_note": {"type": "string"},
                    "competitive_positioning": {
                        "type": "string",
                        "enum": ["discusses competitive strengths and vulnerabilities", "mostly positive positioning", "vague on competition"],
                        "description": "Quality of competitive positioning discussion"
                    },
                    "key_operational_metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Important non-financial metrics disclosed"
                    }
                },
                "required": ["segment_disclosure_quality", "geographic_exposure_clarity", "customer_concentration_disclosed", "customer_concentration_note", "competitive_positioning", "key_operational_metrics"]
            },
            "red_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Credibility concerns or red flags"
            },
            "green_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Positive transparency or strategic clarity signals"
            },
            "investor_summary": {
                "type": "object",
                "properties": {
                    "management_credibility_assessment": {
                        "type": "string",
                        "enum": ["high credibility", "moderate credibility", "low credibility"],
                        "description": "Overall credibility assessment"
                    },
                    "key_strengths": {
                        "type": "string",
                        "description": "1-2 sentence summary of key strengths"
                    },
                    "key_concerns": {
                        "type": "string",
                        "description": "1-2 sentence summary of main concerns"
                    },
                    "narrative_quality": {
                        "type": "string",
                        "enum": ["coherent and detailed", "reasonably clear", "unclear or inconsistent"],
                        "description": "Quality of strategic narrative"
                    }
                },
                "required": ["management_credibility_assessment", "key_strengths", "key_concerns", "narrative_quality"]
            }
        },
        "required": ["management_tone", "strategic_narrative", "risk_disclosure", "management_credibility", "forward_guidance", "capital_allocation", "operational_transparency", "red_flags", "green_flags", "investor_summary"]
    }
}
