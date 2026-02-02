"""Thesis Validation Structured Output Schemas."""

# --- NARRATIVE DECOMPOSITION GRAPH (NDG) SCHEMAS ---
NDG_PARSE_THESIS_SCHEMA = {
    "name": "parse_thesis_output",
    "description": "Extract structured causal claims and any quantifiable metrics from an investment thesis narrative",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "description": "List of extracted causal claims from the thesis",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "The claim text (verbatim or minimally paraphrased)"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["ASSUMPTION", "DRIVER", "OUTCOME"],
                            "description": "ASSUMPTION=belief about the world, DRIVER=intermediate KPI/behavior, OUTCOME=earnings/valuation/return"
                        },
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Names of other claims this depends on"
                        },
                        "directionality": {
                            "type": "string",
                            "description": "Causal relationship expressed as 'X → Y'"
                        },
                        "time_sensitivity": {
                            "type": "string",
                            "enum": ["Short", "Medium", "Long"],
                            "description": "Short=0-1yr, Medium=1-3yr, Long=3+yr"
                        },
                        "is_ambiguous": {
                            "type": "boolean",
                            "description": "True if claim is unclear or needs clarification"
                        }
                    },
                    "required": ["claim", "type", "dependencies", "directionality", "time_sensitivity"]
                }
            },
            "metrics": {
                "type": "object",
                "description": "Any quantifiable metrics implied by the thesis (optional). Keys may vary by thesis.",
                "additionalProperties": {"type": ["number", "array", "object", "null", "string"]}
            }
        },
        "required": ["claims"]
    }
}

NDG_BUILD_DAG_SCHEMA = {
    "name": "build_dag_output",
    "description": "Construct a directed acyclic graph from thesis claims",
    "input_schema": {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "description": "Graph nodes representing claims",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique node identifier (e.g., node_1)"},
                        "claim": {"type": "string", "description": "The claim text"},
                        "node_type": {
                            "type": "string",
                            "enum": ["ASSUMPTION", "DRIVER", "OUTCOME"]
                        },
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of node IDs this depends on (parent nodes)"
                        },
                        "directionality": {
                            "type": "string",
                            "description": "Causal relationship expressed as 'X → Y'"
                        },
                        "time_sensitivity": {
                            "type": "string",
                            "enum": ["Short", "Medium", "Long"],
                            "description": "Short=0-1yr, Medium=1-3yr, Long=3+yr"
                        }
                    },
                    "required": ["id", "claim", "node_type", "dependencies", "directionality", "time_sensitivity"]
                }
            },
            "edges": {
                "type": "array",
                "description": "Causal relationships between nodes",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string", "description": "Causing node ID"},
                        "target_id": {"type": "string", "description": "Affected node ID"},
                        "relationship": {
                            "type": "string",
                            "enum": ["CAUSES", "ENABLES", "MODERATES"]
                        },
                        "strength": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Strength of causal link (0.0-1.0)"
                        }
                    },
                    "required": ["source_id", "target_id", "relationship", "strength"]
                }
            }
        },
        "required": ["nodes", "edges"]
    }
}

NDG_CLASSIFY_ASSUMPTIONS_SCHEMA = {
    "name": "classify_assumptions_output",
    "description": "Classify each assumption along control, nature, and time dimensions",
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Node ID"},
                        "control": {
                            "type": "string",
                            "enum": ["Company", "Industry", "Macro", "Exogenous"],
                            "description": "Who controls this assumption"
                        },
                        "nature": {
                            "type": "string",
                            "enum": ["Structural", "Cyclical", "Execution"],
                            "description": "Type of belief"
                        },
                        "time_sensitivity": {
                            "type": "string",
                            "enum": ["Short", "Medium", "Long"]
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation for classification"
                        }
                    },
                    "required": ["id", "control", "nature", "time_sensitivity", "reasoning"]
                }
            },
            "ambiguous_nodes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Node IDs with unclear classifications and why"
            }
        },
        "required": ["classifications", "ambiguous_nodes"]
    }
}

NDG_MAP_EVIDENCE_SCHEMA = {
    "name": "map_evidence_output",
    "description": "Map supporting and contradicting evidence to each thesis claim",
    "input_schema": {
        "type": "object",
        "properties": {
            "evidence_map": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Node ID"},
                        "supporting_evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["Quantitative", "Qualitative", "External"]
                                    },
                                    "description": {"type": "string"},
                                    "freshness": {
                                        "type": "string",
                                        "enum": ["Recent", "Moderate", "Stale"]
                                    }
                                },
                                "required": ["type", "description", "freshness"]
                            }
                        },
                        "contradicting_evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Facts/data that challenge this claim"
                        },
                        "evidence_strength": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "0.0-0.3=weak, 0.4-0.6=moderate, 0.7-1.0=strong"
                        }
                    },
                    "required": ["id", "supporting_evidence", "contradicting_evidence", "evidence_strength"]
                }
            }
        },
        "required": ["evidence_map"]
    }
}

NDG_DISTRIBUTE_CONFIDENCE_SCHEMA = {
    "name": "distribute_confidence_output",
    "description": "Distribute thesis confidence across claims based on language and emphasis",
    "input_schema": {
        "type": "object",
        "properties": {
            "confidence_distribution": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "basis": {
                            "type": "string",
                            "description": "What language/emphasis drives this confidence"
                        },
                        "evidence_match": {
                            "type": "string",
                            "enum": ["HIGH", "MODERATE", "MISMATCH"],
                            "description": "How well confidence matches available evidence"
                        }
                    },
                    "required": ["id", "confidence", "basis", "evidence_match"]
                }
            },
            "total_confidence": {
                "type": "number",
                "description": "Sum of all confidence values (should be ~1.0)"
            },
            "high_confidence_low_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Warnings about nodes with confidence-evidence mismatch"
            }
        },
        "required": ["confidence_distribution", "total_confidence", "high_confidence_low_evidence"]
    }
}

# --- AI RED TEAM (RTA) SCHEMAS ---
RTA_RETRIEVE_ANALOGS_SCHEMA = {
    "name": "retrieve_analogs_output",
    "description": "Find historical cases where similar assumptions failed",
    "input_schema": {
        "type": "object",
        "properties": {
            "analogs": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "case_name": {
                            "type": "string",
                            "description": "Company/situation name with year range"
                        },
                        "assumption_type": {
                            "type": "string",
                            "description": "What belief failed"
                        },
                        "failure_mode": {
                            "type": "string",
                            "enum": ["Demand Elasticity Misjudged", "Switching Costs Overestimated", "Margin Durability Overestimated", "Competitive Response Underestimated", "Management Signal Misread"]
                        },
                        "context": {
                            "type": "string",
                            "description": "Detailed description with specific numbers and outcomes"
                        },
                        "year": {
                            "type": "integer",
                            "minimum": 2000,
                            "maximum": 2026
                        },
                        "relevance_reasoning": {
                            "type": "string",
                            "description": "Why this analog matters for the current case"
                        }
                    },
                    "required": ["case_name", "assumption_type", "failure_mode", "context", "year", "relevance_reasoning"]
                }
            }
        },
        "required": ["analogs"]
    }
}

RTA_MAP_FAILURE_MODE_SCHEMA = {
    "name": "map_failure_mode_output",
    "description": "Extract causal failure mechanism from historical case",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Primary failure category (use taxonomy when it fits, or supply an alternative)"
            },
            "category_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Model's confidence that the selected category is appropriate (0.0-1.0)"
            },
            "taxonomy_match": {
                "type": "boolean",
                "description": "True if the failure fits one of the provided taxonomy categories, false if an alternative label is provided"
            },
            "alternative_category": {
                "type": ["string", "null"],
                "description": "If taxonomy_match is false, provide a concise alternative label"
            },
            "description": {
                "type": "string",
                "description": "HOW the failure manifested (causal, not descriptive)"
            },
            "early_warnings": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 8,
                "description": "Observable indicators that appeared BEFORE failure was obvious"
            },
            "is_downside_transferable": {
                "type": "boolean",
                "description": "True if downside risk failure (margin compression, demand shock, competitive pressure); False if upside scenario failure (turnaround, innovation, transformation)"
            }
        },
        "required": ["category", "description", "early_warnings", "taxonomy_match", "category_confidence", "is_downside_transferable"]
    }
}

RTA_SCORE_RELEVANCE_SCHEMA = {
    "name": "score_relevance_output",
    "description": "Score relevance of historical analog to current case",
    "input_schema": {
        "type": "object",
        "properties": {
            "business_model_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "business_model_reasoning": {"type": "string"},
            "competitive_structure": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "competitive_structure_reasoning": {"type": "string"},
            "balance_sheet_flexibility": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "balance_sheet_flexibility_reasoning": {"type": "string"},
            "regulatory_environment": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "regulatory_environment_reasoning": {"type": "string"},
            "cycle_position": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "cycle_position_reasoning": {"type": "string"},
            "overall_relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "overall_reasoning": {"type": "string"}
        },
        "required": [
            "business_model_similarity", "business_model_reasoning",
            "competitive_structure", "competitive_structure_reasoning",
            "balance_sheet_flexibility", "balance_sheet_flexibility_reasoning",
            "regulatory_environment", "regulatory_environment_reasoning",
            "cycle_position", "cycle_position_reasoning",
            "overall_relevance", "overall_reasoning"
        ]
    }
}

RTA_SYNTHESIZE_CHALLENGE_SCHEMA = {
    "name": "synthesize_challenge_output",
    "description": "Generate a concise adversarial challenge and watchlist for an assumption",
    "input_schema": {
        "type": "object",
        "properties": {
            "challenge_text": {
                "type": "string",
                "description": "Full 2-3 paragraph challenge text (neutral, constructive)"
            },
            "monitor_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific, observable indicators to monitor that would signal risk"
            },
            "short_summary": {
                "type": "string",
                "description": "1-2 sentence summary of the challenge (optional)"
            }
        },
        "required": ["challenge_text"]
    }
}

# --- COUNTERFACTUAL RESEARCH ENGINE (CRE) SCHEMAS ---
CRE_BOUND_ASSUMPTIONS_SCHEMA = {
    "name": "bound_assumptions_output",
    "description": "Set empirically-grounded bounds for each metric",
    "input_schema": {
        "type": "object",
        "properties": {
            "bounds": {
                "type": "object",
                "properties": {
                    "revenue_growth": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                            "justification": {"type": "string", "description": "Historical analogs with dates and data"}
                        },
                        "required": ["min", "max", "justification"]
                    },
                    "gross_margin": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                            "justification": {"type": "string"}
                        },
                        "required": ["min", "max", "justification"]
                    },
                    "operating_margin": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                            "justification": {"type": "string"}
                        },
                        "required": ["min", "max", "justification"]
                    },
                    "net_retention": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                            "justification": {"type": "string"}
                        },
                        "required": ["min", "max", "justification"]
                    },
                    "wacc": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                            "justification": {"type": "string"}
                        },
                        "required": ["min", "max", "justification"]
                    },
                    "terminal_multiple": {
                        "type": "object",
                        "properties": {
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                            "justification": {"type": "string"}
                        },
                        "required": ["min", "max", "justification"]
                    }
                },
                "required": ["revenue_growth", "gross_margin", "operating_margin", "net_retention", "wacc", "terminal_multiple"]
            }
        },
        "required": ["bounds"]
    }
}

CRE_GENERATE_SCENARIOS_SCHEMA = {
    "name": "generate_scenarios_output",
    "description": "Generate bounded stress-test scenarios for the thesis",
    "input_schema": {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "minItems": 4,
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Scenario template name"
                        },
                        "description": {
                            "type": "string",
                            "description": "What happens in this scenario"
                        },
                        "impact": {
                            "type": "string",
                            "description": "Brief summary of thesis impact"
                        },
                        "stressed_assumptions": {
                            "type": "object",
                            "description": "Metric adjustments (use _factor suffix for multiplicative, plain for additive)"
                        },
                        "justification": {
                            "type": "string",
                            "description": "Historical analog with specific dates and data"
                        },
                        "plausibility_weight": {
                            "type": "number",
                            "minimum": 0.05,
                            "maximum": 0.5,
                            "description": "Probability weight (0.05-0.5)"
                        }
                    },
                    "required": ["name", "description", "impact", "stressed_assumptions", "justification", "plausibility_weight"]
                }
            }
        },
        "required": ["scenarios"]
    }
}

# --- Financial Translation Agent (FTA) SCHEMAS ---
FT_GENERATE_REASONING_SCHEMA = {
    "name": "generate_reasoning_output",
    "description": "Explain why a stress scenario produces a specific outcome",
    "input_schema": {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "2-3 sentence explanation of key drivers and causal chain"
            },
            "key_drivers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The 2-3 factors most responsible for the outcome"
            },
            "historical_precedent": {
                "type": "string",
                "description": "Specific historical evidence supporting this analysis"
            },
            "related_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of factor names referenced in the reasoning"
            },
            "factor_explanation": {
                "type": "string",
                "description": "2-3 sentence explanation tying factor movements to valuation"
            }
        },
        "required": ["explanation", "key_drivers", "historical_precedent"]
    }
} 

FT_SUMMARY_SCHEMA = {
    "name": "cre_summary_output",
    "description": "Structured executive summary of CRE results",
    "input_schema": {
        "type": "object",
        "properties": {
            "vulnerable_claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "failure_count": {"type": "integer"},
                        "example_scenarios": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["claim", "failure_count"]
                }
            },
            "contradicting_evidence": {
                "type": "array",
                "items": {"type": "string"}
            },
            "blind_spots": {
                "type": "array",
                "items": {"type": "string"}
            },
            "asymmetry": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["DOWNSIDE", "UPSIDE", "BALANCED"]},
                    "description": {"type": "string"}
                },
                "required": ["direction", "description"]
            },
            "bullets": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6
            },
            "defaults_applied": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of metric defaults or inferences applied during CRE"
            },
            "top_factors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "factor": {"type": "string"},
                        "score": {"type": "number"}
                    },
                    "required": ["factor", "score"]
                }
            },
            "metric_factor_mapping": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "description": "metric -> {factor: coefficient}"
                }
            }
        },
        "required": ["vulnerable_claims", "blind_spots", "asymmetry", "bullets"]
    }
} 

METRICS_BATCH_CLASSIFICATION_SCHEMA = {
    "name": "metrics_batch_classification",
    "description": "Classify multiple metrics into canonical value factors in a single call",
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "factor_influences": {
                            "type": "object",
                            "description": "factor -> coefficient (-1.0..1.0)",
                            "additionalProperties": {"type": "number"}
                        },
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "explanation": {"type": "string"}
                    },
                    "required": ["metric", "factor_influences", "confidence"]
                }
            }
        },
        "required": ["classifications"]
    }
}

# --- IDEA HALF-LIFE ESTIMATOR (IHLE) SCHEMAS ---
IHLE_ADJUST_REGIME_SCHEMA = {
    "name": "adjust_regime_output",
    "description": "Assess macro regime compatibility for thesis. Returns adjustment factor, regime state, alignment (0-1), and required reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "regime_state": {"type": "string", "enum": ["Stable", "Transitioning", "Unstable"], "description": "Short label describing current regime state"},
            "alignment": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Numeric alignment/confidence of regime assessment (0.0-1.0)"},
            "adjustment_factor": {"type": "number", "description": "Multiplicative factor to apply to the half-life"},
            "reasoning": {"type": "string", "description": "Explanation of regime assessment"},
            "source_confidence": {"type": "number", "description": "Optional confidence score of the model (0.0-1.0)"}
        },
        "required": ["regime_state", "alignment", "adjustment_factor", "reasoning"]
    }
}