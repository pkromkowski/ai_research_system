"""Tests for AIRedTeamWithMemory agent."""
import pytest
from unittest.mock import Mock, patch

from model.thesis_agents.red_team import AIRedTeamWithMemory


class TestAIRedTeamWithMemory:
    """Test suite for AIRedTeamWithMemory."""
    
    def test_initialization(self):
        """Test agent initialization."""
        red_team = AIRedTeamWithMemory('AAPL')
        assert red_team.stock_ticker == 'AAPL'
        assert red_team.MAX_TOKENS_ANALOGS > 0
    
    def test_validate_ndg_valid(self, sample_ndg_output):
        """Test NDG validation with valid input."""
        red_team = AIRedTeamWithMemory('AAPL')
        result = red_team._validate_and_normalize_ndg(sample_ndg_output)
        
        assert result is not None
        assert hasattr(result, 'nodes')
        assert isinstance(result.nodes, list)
    
    def test_validate_ndg_missing_nodes(self):
        """Test NDG validation with missing nodes."""
        red_team = AIRedTeamWithMemory('AAPL')
        
        with pytest.raises(ValueError, match='must have a.*nodes.*list'):
            red_team._validate_and_normalize_ndg(Mock(spec=[]))
    
    def test_determine_severity_high(self):
        """Test severity determination - high."""
        red_team = AIRedTeamWithMemory('AAPL')
        # High severity requires relevance >= 0.7 AND evidence < 0.4
        severity = red_team._determine_severity(0.9, 0.3)
        
        assert severity == 'HIGH'
    
    def test_determine_severity_medium(self):
        """Test severity determination - medium."""
        red_team = AIRedTeamWithMemory('AAPL')
        # Medium requires relevance >= 0.5 OR evidence < 0.5
        severity = red_team._determine_severity(0.6, 0.6)
        
        assert severity == 'MEDIUM'
    
    def test_determine_severity_low(self):
        """Test severity determination - low."""
        red_team = AIRedTeamWithMemory('AAPL')
        # Low is everything else
        severity = red_team._determine_severity(0.3, 0.6)
        
        assert severity == 'LOW'
    
    @patch.object(AIRedTeamWithMemory, '_call_llm_structured')
    def test_run(self, mock_llm, sample_ndg_output):
        """Test full red team execution."""
        mock_llm.return_value = {
            'challenges': [
                {
                    'node_id': 'node_1',
                    'challenge_text': 'Test challenge',
                    'severity': 'HIGH',
                    'relevance_score': 0.8,
                    'evidence_strength': 0.7
                }
            ]
        }
        
        red_team = AIRedTeamWithMemory('AAPL')
        result = red_team.run(ndg=sample_ndg_output, company_context='Test company')
        
        assert result is not None
        assert hasattr(result, 'challenges')
        assert hasattr(result, 'high_severity_count')
    
    def test_empty_ndg(self):
        """Test with empty NDG."""
        red_team = AIRedTeamWithMemory('AAPL')
        empty_ndg = Mock(nodes=[], edges=[])
        
        # Empty NDG is handled gracefully - should return output with 0 challenges
        result = red_team.run(ndg=empty_ndg, company_context='Test')
        assert result is not None
        assert hasattr(result, 'challenges')
        assert len(result.challenges) == 0
