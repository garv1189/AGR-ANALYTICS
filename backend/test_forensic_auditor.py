"""
Test Suite for SEC Forensic Auditor
Run with: pytest test_forensic_auditor.py -v
"""
import pytest
from unittest.mock import Mock, patch
import json
from datetime import datetime

# Import modules to test
import sys
sys.path.append('..')

from config import settings
from sec_extractor import edgar_extractor, Company, Filing
from data_preprocessor import data_preprocessor
from agent_numerical import numerical_analyst, FinancialAnomaly
from agent_textual import textual_investigator, TextualFlag
from agent_chief import chief_auditor


# ============================================================================
# SEC Extractor Tests
# ============================================================================

class TestSECExtractor:
    """Test SEC EDGAR data extraction"""
    
    def test_company_creation(self):
        """Test Company dataclass"""
        company = Company(cik="0000320193", name="APPLE INC", ticker="AAPL")
        assert company.cik == "0000320193"
        assert company.name == "APPLE INC"
        assert company.ticker == "AAPL"
    
    @patch('sec_extractor.SECEdgarExtractor._make_request')
    def test_search_company(self, mock_request):
        """Test company search"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "0": {
                "cik_str": 320193,
                "ticker": "AAPL",
                "title": "Apple Inc."
            }
        }
        mock_request.return_value = mock_response
        
        companies = edgar_extractor.search_company_by_name("Apple")
        assert len(companies) > 0


# ============================================================================
# Data Preprocessor Tests
# ============================================================================

class TestDataPreprocessor:
    """Test data preprocessing"""
    
    def test_clean_html(self):
        """Test HTML cleaning"""
        html = "<html><body><p>Test paragraph</p><script>alert('test')</script></body></html>"
        cleaned = data_preprocessor.clean_html(html)
        assert "Test paragraph" in cleaned
        assert "<script>" not in cleaned
    
    def test_extract_key_metrics(self):
        """Test metric extraction"""
        import pandas as pd
        table = pd.DataFrame({
            'Metric': ['Total Revenue', 'Net Income', 'Total Assets'],
            '2023': [1000, 200, 5000],
            '2022': [900, 180, 4500]
        })
        
        metrics = data_preprocessor._extract_key_metrics([table])
        assert 'total_revenue' in metrics or 'total_assets' in metrics
    
    def test_calculate_ratios(self):
        """Test ratio calculation"""
        metrics = {
            'net_income': 200,
            'total_revenue': 1000,
            'total_assets': 5000
        }
        ratios = data_preprocessor._calculate_ratios(metrics)
        
        assert 'profit_margin' in ratios
        assert ratios['profit_margin'] == 0.2  # 200/1000
        assert 'return_on_assets' in ratios
        assert ratios['return_on_assets'] == 0.04  # 200/5000


# ============================================================================
# Numerical Analyst Tests
# ============================================================================

class TestNumericalAnalyst:
    """Test numerical analysis agent"""
    
    def test_validate_data_complete(self):
        """Test data validation with complete data"""
        numerical_data = {
            'key_metrics': {
                'total_revenue': 1000,
                'net_income': 200,
                'total_assets': 5000,
                'total_liabilities': 3000,
                'shareholders_equity': 2000
            }
        }
        
        validation = numerical_analyst.validate_data(numerical_data)
        assert validation['is_valid'] == True
        assert validation['completeness_score'] == 1.0
    
    def test_validate_data_incomplete(self):
        """Test data validation with incomplete data"""
        numerical_data = {
            'key_metrics': {
                'total_revenue': 1000
            }
        }
        
        validation = numerical_analyst.validate_data(numerical_data)
        assert validation['completeness_score'] < 1.0
        assert len(validation['warnings']) > 0
    
    def test_detect_business_rule_anomalies(self):
        """Test business rule anomaly detection"""
        metrics = {
            'net_income': -500,  # Negative
            'total_revenue': 1000,
            'shareholders_equity': -100  # Negative (insolvency)
        }
        ratios = {}
        
        anomalies = numerical_analyst._detect_business_rule_anomalies(metrics, ratios)
        assert len(anomalies) >= 2  # Should detect loss and negative equity
        
        # Check for negative equity anomaly
        equity_anomalies = [a for a in anomalies if a.metric_name == 'shareholders_equity']
        assert len(equity_anomalies) > 0
        assert equity_anomalies[0].severity == 'high'
    
    def test_heuristic_risk_score(self):
        """Test heuristic risk scoring"""
        # Low risk features
        low_risk_features = {
            'profit_margin': 0.15,
            'debt_to_equity': 0.5,
            'cash_ratio': 0.2,
            'return_on_assets': 0.1
        }
        low_score = numerical_analyst._heuristic_risk_score(low_risk_features)
        assert low_score < 0.3
        
        # High risk features
        high_risk_features = {
            'profit_margin': -0.1,
            'debt_to_equity': 5.0,
            'cash_ratio': 0.02,
            'return_on_assets': -0.05,
            'shareholders_equity': -1000
        }
        high_score = numerical_analyst._heuristic_risk_score(high_risk_features)
        assert high_score > 0.5


# ============================================================================
# Textual Investigator Tests
# ============================================================================

class TestTextualInvestigator:
    """Test textual analysis agent"""
    
    def test_count_syllables(self):
        """Test syllable counting"""
        assert textual_investigator._count_syllables("cat") == 1
        assert textual_investigator._count_syllables("running") == 2
        assert textual_investigator._count_syllables("beautiful") == 3
    
    def test_compute_complexity_metrics(self):
        """Test complexity computation"""
        simple_text = "The cat sat on the mat. It was a nice day."
        complex_text = "The multifaceted organizational restructuring necessitates comprehensive evaluation of extraordinarily complicated financial implications."
        
        simple_metrics = textual_investigator.compute_complexity_metrics(simple_text)
        complex_metrics = textual_investigator.compute_complexity_metrics(complex_text)
        
        assert simple_metrics['gunning_fog_index'] < complex_metrics['gunning_fog_index']
        assert simple_metrics['obfuscation_score'] < complex_metrics['obfuscation_score']
    
    def test_scan_risk_keywords(self):
        """Test keyword scanning"""
        text = """
        The company is facing litigation and regulatory action.
        Material weakness in internal controls was identified.
        There is uncertainty regarding going concern.
        """
        
        flags = textual_investigator.scan_keywords_phrases(text, "test_section")
        
        # Should detect multiple risk keywords
        assert len(flags) >= 3
        risk_flags = [f for f in flags if f.category == 'risk_indicator']
        assert len(risk_flags) >= 3
    
    def test_detect_euphemisms(self):
        """Test euphemism detection"""
        text = """
        We experienced challenging environment and headwinds in the market.
        There was softness in demand and we made strategic realignment.
        These are one-time charges and non-recurring items.
        """
        
        flags = textual_investigator.scan_keywords_phrases(text, "test_section")
        
        euphemism_flags = [f for f in flags if f.category == 'euphemism']
        assert len(euphemism_flags) >= 2
    
    def test_linguistic_risk_score(self):
        """Test linguistic risk scoring"""
        # Low risk scenario
        low_risk_sentiment = {'negative_score': 0.1, 'neutral_score': 0.7, 'positive_score': 0.2}
        low_risk_complexity = {'obfuscation_score': 0.2}
        low_risk_flags = []
        low_risk_patterns = []
        
        low_score = textual_investigator.calculate_linguistic_risk_score(
            low_risk_sentiment, low_risk_complexity, low_risk_flags, low_risk_patterns
        )
        assert low_score < 0.3
        
        # High risk scenario
        high_risk_sentiment = {'negative_score': 0.8, 'neutral_score': 0.1, 'positive_score': 0.1}
        high_risk_complexity = {'obfuscation_score': 0.9}
        high_risk_flags = [
            TextualFlag('legal_term', 'litigation', 'section1', 'high', 0.9, 'test', 'test')
            for _ in range(5)
        ]
        high_risk_patterns = [{'type': 'test'} for _ in range(10)]
        
        high_score = textual_investigator.calculate_linguistic_risk_score(
            high_risk_sentiment, high_risk_complexity, high_risk_flags, high_risk_patterns
        )
        assert high_score > 0.6


# ============================================================================
# Chief Auditor Tests
# ============================================================================

class TestChiefAuditor:
    """Test chief forensic auditor agent"""
    
    def test_rule_based_classification_low(self):
        """Test low risk classification"""
        risk_level = chief_auditor._rule_based_classification(
            numerical_score=0.2,
            textual_score=0.2,
            combined_score=0.2,
            anomaly_count=1,
            high_severity_flags=0,
            legal_terms=0
        )
        assert risk_level == 'L1_LOW'
    
    def test_rule_based_classification_high(self):
        """Test high risk classification"""
        risk_level = chief_auditor._rule_based_classification(
            numerical_score=0.85,
            textual_score=0.80,
            combined_score=0.82,
            anomaly_count=8,
            high_severity_flags=5,
            legal_terms=3
        )
        assert risk_level == 'L3_HIGH'
    
    def test_ml_based_classification(self):
        """Test ML-based risk classification"""
        low_level = chief_auditor._ml_based_classification(0.2)
        assert low_level == 'L1_LOW'
        
        medium_level = chief_auditor._ml_based_classification(0.5)
        assert medium_level == 'L2_MEDIUM'
        
        high_level = chief_auditor._ml_based_classification(0.9)
        assert high_level == 'L3_HIGH'
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        # Agreement + clear score
        high_confidence = chief_auditor._calculate_confidence(
            'L3_HIGH', 'L3_HIGH', 0.95, 5
        )
        assert high_confidence > 0.7
        
        # Disagreement + unclear score
        low_confidence = chief_auditor._calculate_confidence(
            'L1_LOW', 'L3_HIGH', 0.5, 1
        )
        assert low_confidence < 0.7


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_complete_analysis_pipeline(self):
        """Test complete analysis pipeline with mock data"""
        # Mock processed data
        processed_data = {
            'filing_id': 'test-123',
            'company_info': {
                'name': 'Test Company',
                'cik': '0000000001',
                'ticker': 'TEST'
            },
            'filing_info': {
                'filing_date': '2024-01-01',
                'report_date': '2023-12-31',
                'accession_number': 'test-accession'
            },
            'numerical_data': {
                'metadata': {'company_info': {}, 'filing_info': {}},
                'key_metrics': {
                    'total_revenue': 1000,
                    'net_income': 100,
                    'total_assets': 5000,
                    'total_liabilities': 3000,
                    'shareholders_equity': 2000
                },
                'ratios': {
                    'profit_margin': 0.1,
                    'debt_to_equity': 1.5
                }
            },
            'textual_data': {
                'metadata': {'company_info': {}, 'filing_info': {}},
                'full_text': 'Test company annual report. Business operations continue.',
                'sections': {
                    'item_1a': 'Risk factors include market volatility and competition.'
                }
            }
        }
        
        # Run numerical analysis
        numerical_result = numerical_analyst.analyze(processed_data['numerical_data'])
        assert numerical_result is not None
        assert numerical_result.forensic_score >= 0
        assert numerical_result.forensic_score <= 1
        
        # Run textual analysis
        textual_result = textual_investigator.analyze(processed_data['textual_data'])
        assert textual_result is not None
        assert textual_result.linguistic_score >= 0
        assert textual_result.linguistic_score <= 1
        
        # Synthesize report
        report = chief_auditor.synthesize_report(numerical_result, textual_result)
        assert report is not None
        assert report.risk_classification.risk_level in ['L1_LOW', 'L2_MEDIUM', 'L3_HIGH']
        assert len(report.actionable_recommendations) > 0


# ============================================================================
# Pytest Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
