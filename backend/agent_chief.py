"""
Chief Forensic Auditor Agent
Synthesizes numerical and textual analysis, performs causal inference, and generates final report
"""
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from agent_numerical import NumericalAnalysisResult, FinancialAnomaly
from agent_textual import TextualAnalysisResult, TextualFlag
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class CausalNarrative:
    """Represents a causal inference between numerical and textual findings"""
    numerical_finding: str
    textual_finding: str
    causal_link: str
    confidence_score: float
    severity: str
    evidence: List[str]


@dataclass
class RiskClassification:
    """Final risk classification"""
    risk_level: str  # 'L1_LOW', 'L2_MEDIUM', 'L3_HIGH'
    confidence_score: float
    primary_factors: List[str]
    contributing_factors: List[str]
    methodology: str  # 'rule_based', 'ml_fusion', 'hybrid'


@dataclass
class ForensicReport:
    """Complete forensic audit report"""
    report_id: str
    company_info: Dict
    filing_info: Dict
    generation_date: str
    
    # Risk Assessment
    risk_classification: RiskClassification
    
    # Evidence Synthesis
    numerical_evidence: Dict
    textual_evidence: Dict
    causal_narratives: List[CausalNarrative]
    
    # Pattern Identification
    identified_patterns: List[Dict]
    anomaly_clusters: List[Dict]
    
    # Final Report Sections
    executive_summary: Dict
    detailed_findings: Dict
    actionable_recommendations: List[str]
    
    # Metadata
    analysis_metadata: Dict


class ChiefForensicAuditor:
    """Chief auditor agent that synthesizes all analyses and generates final report"""
    
    def __init__(self):
        self.risk_rules = self._initialize_risk_rules()
    
    def _initialize_risk_rules(self) -> Dict:
        """Initialize rule-based risk classification rules"""
        return {
            'L3_HIGH': {
                'numerical_score_threshold': 0.75,
                'textual_score_threshold': 0.75,
                'combined_score_threshold': 0.70,
                'anomaly_count_threshold': 5,
                'high_severity_flags_threshold': 3,
                'legal_terms_threshold': 2
            },
            'L2_MEDIUM': {
                'numerical_score_threshold': 0.45,
                'textual_score_threshold': 0.45,
                'combined_score_threshold': 0.40,
                'anomaly_count_threshold': 3,
                'high_severity_flags_threshold': 1,
                'legal_terms_threshold': 1
            },
            'L1_LOW': {
                # Default - anything below L2 thresholds
            }
        }
    
    def find_causal_narratives(
        self,
        numerical_result: NumericalAnalysisResult,
        textual_result: TextualAnalysisResult
    ) -> List[CausalNarrative]:
        """Find causal links between numerical anomalies and textual disclosures"""
        narratives = []
        
        # Pattern 1: Financial anomaly + related disclosure in risk factors
        for anomaly in numerical_result.anomalies:
            # Check if there's textual evidence discussing this metric
            related_flags = [
                flag for flag in textual_result.red_flags
                if self._is_related(anomaly.metric_name, flag.text_snippet)
            ]
            
            if related_flags:
                narratives.append(CausalNarrative(
                    numerical_finding=f"{anomaly.metric_name}: {anomaly.explanation}",
                    textual_finding=f"{related_flags[0].category}: {related_flags[0].text_snippet}",
                    causal_link=f"Numerical anomaly in {anomaly.metric_name} is acknowledged in disclosure",
                    confidence_score=0.8,
                    severity=anomaly.severity,
                    evidence=[flag.sentence_context for flag in related_flags[:2]]
                ))
        
        # Pattern 2: Revenue/profit decline + evasive language
        if numerical_result.trend_analysis.get('trends', {}).get('total_revenue', {}).get('direction') == 'decreasing':
            evasive_flags = [
                flag for flag in textual_result.red_flags
                if flag.category in ['euphemism', 'evasive_language']
            ]
            
            if evasive_flags:
                narratives.append(CausalNarrative(
                    numerical_finding="Declining revenue trend",
                    textual_finding="Evasive/euphemistic language in disclosures",
                    causal_link="Company may be downplaying revenue decline through euphemistic language",
                    confidence_score=0.75,
                    severity='medium',
                    evidence=[flag.sentence_context for flag in evasive_flags[:2]]
                ))
        
        # Pattern 3: High debt + risk factor disclosures
        if 'debt_to_equity' in numerical_result.ratio_analysis:
            if numerical_result.ratio_analysis['debt_to_equity'] > 2.5:
                debt_risk_flags = [
                    flag for flag in textual_result.red_flags
                    if any(term in flag.text_snippet.lower() 
                          for term in ['debt', 'leverage', 'covenant', 'liquidity'])
                ]
                
                if debt_risk_flags:
                    narratives.append(CausalNarrative(
                        numerical_finding=f"High debt-to-equity ratio: {numerical_result.ratio_analysis['debt_to_equity']:.2f}",
                        textual_finding="Debt-related risk factors disclosed",
                        causal_link="High leverage is reflected in risk factor disclosures",
                        confidence_score=0.85,
                        severity='high',
                        evidence=[flag.sentence_context for flag in debt_risk_flags[:2]]
                    ))
        
        # Pattern 4: Negative sentiment + financial losses
        if numerical_result.forensic_score > 0.6 and textual_result.linguistic_score > 0.6:
            narratives.append(CausalNarrative(
                numerical_finding=f"High forensic risk score: {numerical_result.forensic_score:.2f}",
                textual_finding=f"High linguistic risk score: {textual_result.linguistic_score:.2f}",
                causal_link="Both numerical and textual indicators suggest elevated risk",
                confidence_score=0.9,
                severity='high',
                evidence=["Combined quantitative and qualitative risk indicators"]
            ))
        
        # Pattern 5: Specific legal terms + related financial anomaly
        legal_flags = [f for f in textual_result.red_flags if f.category == 'legal_term']
        if legal_flags and numerical_result.anomalies:
            narratives.append(CausalNarrative(
                numerical_finding="Financial anomalies detected",
                textual_finding=f"Legal/regulatory issues mentioned: {', '.join([f.text_snippet for f in legal_flags[:2]])}",
                causal_link="Legal issues may be driving financial irregularities",
                confidence_score=0.8,
                severity='high',
                evidence=[f.sentence_context for f in legal_flags[:2]]
            ))
        
        return narratives
    
    def _is_related(self, metric_name: str, text_snippet: str) -> bool:
        """Check if a metric name is related to a text snippet"""
        metric_keywords = metric_name.lower().replace('_', ' ').split()
        text_lower = text_snippet.lower()
        
        # Check for direct matches or related terms
        related_terms = {
            'revenue': ['sales', 'income', 'revenue'],
            'debt': ['debt', 'liability', 'leverage', 'borrowing'],
            'profit': ['profit', 'earnings', 'income', 'margin'],
            'asset': ['asset', 'property', 'investment'],
            'equity': ['equity', 'shareholder', 'stockholder']
        }
        
        for keyword in metric_keywords:
            if keyword in text_lower:
                return True
            
            for key, terms in related_terms.items():
                if keyword in terms and any(term in text_lower for term in terms):
                    return True
        
        return False
    
    def classify_risk(
        self,
        numerical_result: NumericalAnalysisResult,
        textual_result: TextualAnalysisResult,
        causal_narratives: List[CausalNarrative]
    ) -> RiskClassification:
        """Classify overall risk using rule-based + ML fusion"""
        # Calculate combined score
        combined_score = (numerical_result.forensic_score * 0.5 + 
                         textual_result.linguistic_score * 0.5)
        
        # Count various indicators
        high_severity_anomalies = len([a for a in numerical_result.anomalies if a.severity == 'high'])
        total_anomalies = len(numerical_result.anomalies)
        high_severity_flags = len([f for f in textual_result.red_flags if f.severity == 'high'])
        legal_terms = len([f for f in textual_result.red_flags if f.category == 'legal_term'])
        high_severity_narratives = len([n for n in causal_narratives if n.severity == 'high'])
        
        # Rule-based classification
        rule_based_level = self._rule_based_classification(
            numerical_result.forensic_score,
            textual_result.linguistic_score,
            combined_score,
            total_anomalies,
            high_severity_flags,
            legal_terms
        )
        
        # ML-based classification (using scores)
        ml_based_level = self._ml_based_classification(combined_score)
        
        # Fusion: Use highest risk level
        risk_levels_priority = ['L3_HIGH', 'L2_MEDIUM', 'L1_LOW']
        final_level = rule_based_level
        if risk_levels_priority.index(ml_based_level) < risk_levels_priority.index(rule_based_level):
            final_level = ml_based_level
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            rule_based_level,
            ml_based_level,
            combined_score,
            high_severity_narratives
        )
        
        # Identify primary factors
        primary_factors = []
        if numerical_result.forensic_score > 0.6:
            primary_factors.append(f"High numerical risk score ({numerical_result.forensic_score:.2f})")
        if textual_result.linguistic_score > 0.6:
            primary_factors.append(f"High linguistic risk score ({textual_result.linguistic_score:.2f})")
        if high_severity_anomalies > 0:
            primary_factors.append(f"{high_severity_anomalies} high-severity financial anomalies")
        if legal_terms > 0:
            primary_factors.append(f"{legal_terms} legal/regulatory terms detected")
        
        # Contributing factors
        contributing_factors = []
        if total_anomalies > 3:
            contributing_factors.append(f"Total {total_anomalies} anomalies detected")
        if high_severity_flags > 0:
            contributing_factors.append(f"{high_severity_flags} high-severity textual flags")
        if len(causal_narratives) > 2:
            contributing_factors.append(f"{len(causal_narratives)} causal links identified")
        
        return RiskClassification(
            risk_level=final_level,
            confidence_score=confidence,
            primary_factors=primary_factors,
            contributing_factors=contributing_factors,
            methodology='hybrid'
        )
    
    def _rule_based_classification(
        self,
        numerical_score: float,
        textual_score: float,
        combined_score: float,
        anomaly_count: int,
        high_severity_flags: int,
        legal_terms: int
    ) -> str:
        """Rule-based risk classification"""
        rules_l3 = self.risk_rules['L3_HIGH']
        rules_l2 = self.risk_rules['L2_MEDIUM']
        
        # Check L3 (HIGH) conditions
        l3_conditions_met = 0
        if numerical_score >= rules_l3['numerical_score_threshold']:
            l3_conditions_met += 1
        if textual_score >= rules_l3['textual_score_threshold']:
            l3_conditions_met += 1
        if combined_score >= rules_l3['combined_score_threshold']:
            l3_conditions_met += 1
        if anomaly_count >= rules_l3['anomaly_count_threshold']:
            l3_conditions_met += 1
        if high_severity_flags >= rules_l3['high_severity_flags_threshold']:
            l3_conditions_met += 1
        if legal_terms >= rules_l3['legal_terms_threshold']:
            l3_conditions_met += 1
        
        if l3_conditions_met >= 3:  # Need at least 3 conditions
            return 'L3_HIGH'
        
        # Check L2 (MEDIUM) conditions
        l2_conditions_met = 0
        if numerical_score >= rules_l2['numerical_score_threshold']:
            l2_conditions_met += 1
        if textual_score >= rules_l2['textual_score_threshold']:
            l2_conditions_met += 1
        if combined_score >= rules_l2['combined_score_threshold']:
            l2_conditions_met += 1
        if anomaly_count >= rules_l2['anomaly_count_threshold']:
            l2_conditions_met += 1
        if high_severity_flags >= rules_l2['high_severity_flags_threshold']:
            l2_conditions_met += 1
        
        if l2_conditions_met >= 2:  # Need at least 2 conditions
            return 'L2_MEDIUM'
        
        return 'L1_LOW'
    
    def _ml_based_classification(self, combined_score: float) -> str:
        """ML-based risk classification using score thresholds"""
        if combined_score >= settings.RISK_L3_THRESHOLD:
            return 'L3_HIGH'
        elif combined_score >= settings.RISK_L2_THRESHOLD:
            return 'L2_MEDIUM'
        else:
            return 'L1_LOW'
    
    def _calculate_confidence(
        self,
        rule_based: str,
        ml_based: str,
        combined_score: float,
        causal_narratives_count: int
    ) -> float:
        """Calculate confidence in risk classification"""
        base_confidence = 0.5
        
        # Agreement between methods
        if rule_based == ml_based:
            base_confidence += 0.2
        
        # Clear score separation from thresholds
        if combined_score > 0.8 or combined_score < 0.2:
            base_confidence += 0.15
        
        # Strong causal evidence
        if causal_narratives_count > 2:
            base_confidence += 0.15
        
        return min(base_confidence, 1.0)
    
    def identify_patterns(
        self,
        numerical_result: NumericalAnalysisResult,
        textual_result: TextualAnalysisResult,
        causal_narratives: List[CausalNarrative]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Identify cross-cutting patterns and anomaly clusters"""
        patterns = []
        clusters = []
        
        # Pattern 1: Consistent decline across metrics
        declining_trends = [
            metric for metric, trend in numerical_result.trend_analysis.get('trends', {}).items()
            if trend.get('direction') == 'decreasing'
        ]
        if len(declining_trends) >= 2:
            patterns.append({
                'type': 'multi_metric_decline',
                'description': f"Multiple declining metrics: {', '.join(declining_trends)}",
                'severity': 'high' if len(declining_trends) >= 3 else 'medium',
                'evidence': declining_trends
            })
        
        # Pattern 2: Clustered textual flags in specific sections
        flags_by_section = {}
        for flag in textual_result.red_flags:
            if flag.location not in flags_by_section:
                flags_by_section[flag.location] = []
            flags_by_section[flag.location].append(flag)
        
        for section, flags in flags_by_section.items():
            if len(flags) >= 5:
                patterns.append({
                    'type': 'concentrated_risk_disclosure',
                    'description': f"High concentration of risk indicators in {section}",
                    'severity': 'medium',
                    'evidence': [f.text_snippet for f in flags[:3]]
                })
        
        # Pattern 3: Correlation between debt and textual caution
        debt_related_narratives = [
            n for n in causal_narratives
            if 'debt' in n.numerical_finding.lower() or 'debt' in n.textual_finding.lower()
        ]
        if debt_related_narratives:
            patterns.append({
                'type': 'debt_disclosure_correlation',
                'description': "Debt-related numerical and textual indicators aligned",
                'severity': 'high' if len(debt_related_narratives) > 1 else 'medium',
                'evidence': [n.causal_link for n in debt_related_narratives]
            })
        
        # Cluster 1: Financial anomalies by type
        anomaly_types = {}
        for anomaly in numerical_result.anomalies:
            category = 'leverage' if 'debt' in anomaly.metric_name else \
                      'profitability' if any(term in anomaly.metric_name for term in ['income', 'profit', 'revenue']) else \
                      'liquidity' if 'cash' in anomaly.metric_name else 'other'
            
            if category not in anomaly_types:
                anomaly_types[category] = []
            anomaly_types[category].append(anomaly)
        
        for category, anomalies in anomaly_types.items():
            if len(anomalies) >= 2:
                clusters.append({
                    'cluster_type': f'{category}_anomalies',
                    'item_count': len(anomalies),
                    'items': [a.metric_name for a in anomalies],
                    'avg_severity': sum(1 if a.severity == 'high' else 0.5 if a.severity == 'medium' else 0.25 
                                      for a in anomalies) / len(anomalies)
                })
        
        return patterns, clusters
    
    def generate_actionable_recommendations(
        self,
        risk_classification: RiskClassification,
        numerical_result: NumericalAnalysisResult,
        textual_result: TextualAnalysisResult,
        causal_narratives: List[CausalNarrative],
        patterns: List[Dict]
    ) -> List[str]:
        """Generate prioritized, actionable recommendations"""
        recommendations = []
        
        # Priority 1: Immediate actions for high risk
        if risk_classification.risk_level == 'L3_HIGH':
            recommendations.append(
                "IMMEDIATE ACTION REQUIRED: Conduct comprehensive forensic audit with external auditors"
            )
            recommendations.append(
                "Engage legal counsel to review disclosed legal/regulatory issues"
            )
            recommendations.append(
                "Request management response to all identified high-severity anomalies"
            )
        
        # Priority 2: Specific numerical issues
        high_severity_anomalies = [a for a in numerical_result.anomalies if a.severity == 'high']
        for anomaly in high_severity_anomalies[:2]:
            recommendations.append(
                f"Investigate {anomaly.metric_name}: {anomaly.explanation}"
            )
        
        # Priority 3: Textual concerns
        legal_flags = [f for f in textual_result.red_flags if f.category == 'legal_term']
        if legal_flags:
            recommendations.append(
                f"Review legal matters: {', '.join(set([f.text_snippet for f in legal_flags[:3]]))}"
            )
        
        # Priority 4: Causal narratives
        high_conf_narratives = [n for n in causal_narratives if n.confidence_score > 0.8]
        for narrative in high_conf_narratives[:2]:
            recommendations.append(
                f"Examine causal link: {narrative.causal_link}"
            )
        
        # Priority 5: Pattern-based recommendations
        for pattern in patterns[:2]:
            if pattern['severity'] == 'high':
                recommendations.append(
                    f"Address pattern: {pattern['description']}"
                )
        
        # Priority 6: Data quality
        if numerical_result.validation_results['completeness_score'] < 0.7:
            recommendations.append(
                "Request additional financial disclosures to complete analysis"
            )
        
        # Priority 7: Follow-up
        if risk_classification.risk_level in ['L2_MEDIUM', 'L3_HIGH']:
            recommendations.append(
                "Schedule follow-up analysis for next quarterly filing"
            )
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def synthesize_report(
        self,
        numerical_result: NumericalAnalysisResult,
        textual_result: TextualAnalysisResult
    ) -> ForensicReport:
        """Main method: Synthesize all analyses into final forensic report"""
        logger.info("Synthesizing forensic audit report")
        
        # Step 1: Find causal narratives
        causal_narratives = self.find_causal_narratives(numerical_result, textual_result)
        
        # Step 2: Classify risk
        risk_classification = self.classify_risk(numerical_result, textual_result, causal_narratives)
        
        # Step 3: Identify patterns
        patterns, clusters = self.identify_patterns(numerical_result, textual_result, causal_narratives)
        
        # Step 4: Generate recommendations
        recommendations = self.generate_actionable_recommendations(
            risk_classification,
            numerical_result,
            textual_result,
            causal_narratives,
            patterns
        )
        
        # Step 5: Build executive summary
        executive_summary = {
            'risk_level': risk_classification.risk_level,
            'confidence_score': risk_classification.confidence_score,
            'numerical_risk_score': numerical_result.forensic_score,
            'textual_risk_score': textual_result.linguistic_score,
            'combined_risk_score': (numerical_result.forensic_score + textual_result.linguistic_score) / 2,
            'total_anomalies': len(numerical_result.anomalies),
            'high_severity_anomalies': len([a for a in numerical_result.anomalies if a.severity == 'high']),
            'total_textual_flags': len(textual_result.red_flags),
            'high_severity_flags': len([f for f in textual_result.red_flags if f.severity == 'high']),
            'causal_narratives_found': len(causal_narratives),
            'patterns_identified': len(patterns),
            'key_concerns': risk_classification.primary_factors[:3],
            'immediate_actions': recommendations[:3]
        }
        
        # Step 6: Detailed findings
        detailed_findings = {
            'numerical_analysis': {
                'forensic_score': numerical_result.forensic_score,
                'validation_status': numerical_result.validation_results,
                'key_anomalies': [asdict(a) for a in numerical_result.anomalies[:5]],
                'trend_summary': numerical_result.trend_analysis,
                'financial_ratios': numerical_result.ratio_analysis
            },
            'textual_analysis': {
                'linguistic_score': textual_result.linguistic_score,
                'sentiment_summary': textual_result.sentiment_analysis,
                'complexity_metrics': textual_result.complexity_metrics,
                'key_red_flags': [asdict(f) for f in textual_result.red_flags[:10]],
                'suspicious_patterns': textual_result.suspicious_patterns[:5]
            },
            'synthesis': {
                'causal_narratives': [asdict(n) for n in causal_narratives],
                'cross_cutting_patterns': patterns,
                'anomaly_clusters': clusters
            }
        }
        
        # Step 7: Create report
        report = ForensicReport(
            report_id=f"FR-{numerical_result.company_info.get('cik', 'UNKNOWN')}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            company_info=numerical_result.company_info,
            filing_info=numerical_result.filing_info,
            generation_date=datetime.now().isoformat(),
            risk_classification=risk_classification,
            numerical_evidence={
                'forensic_score': numerical_result.forensic_score,
                'anomalies': [asdict(a) for a in numerical_result.anomalies],
                'shap_values': numerical_result.shap_summary
            },
            textual_evidence={
                'linguistic_score': textual_result.linguistic_score,
                'red_flags': [asdict(f) for f in textual_result.red_flags],
                'evidence_snippets': textual_result.evidence_snippets
            },
            causal_narratives=causal_narratives,
            identified_patterns=patterns,
            anomaly_clusters=clusters,
            executive_summary=executive_summary,
            detailed_findings=detailed_findings,
            actionable_recommendations=recommendations,
            analysis_metadata={
                'agent_versions': {
                    'numerical_analyst': '1.0.0',
                    'textual_investigator': '1.0.0',
                    'chief_auditor': '1.0.0'
                },
                'models_used': {
                    'xgboost': settings.XGBOOST_N_ESTIMATORS,
                    'finbert': settings.FINBERT_MODEL,
                    'longformer': settings.LONGFORMER_MODEL
                },
                'analysis_duration': 'calculated_at_runtime'
            }
        )
        
        logger.info(f"Report generated: {report.report_id}")
        logger.info(f"Final risk level: {risk_classification.risk_level} (confidence: {risk_classification.confidence_score:.2f})")
        
        return report
    
    def export_report_json(self, report: ForensicReport, filepath: str):
        """Export report to JSON file"""
        report_dict = asdict(report)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Report exported to {filepath}")


# Singleton instance
chief_auditor = ChiefForensicAuditor()
