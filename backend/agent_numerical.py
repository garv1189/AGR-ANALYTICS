"""
Numerical Analyst Agent
Performs forensic analysis on numerical/financial data using XGBoost and SHAP
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import json

# ML Libraries
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import shap

# Statistical Libraries
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class FinancialAnomaly:
    """Represents a detected financial anomaly"""
    metric_name: str
    value: float
    expected_range: Tuple[float, float]
    anomaly_score: float
    severity: str  # 'low', 'medium', 'high'
    shap_values: Dict[str, float]
    explanation: str


@dataclass
class NumericalAnalysisResult:
    """Complete numerical analysis result"""
    company_info: Dict
    filing_info: Dict
    validation_results: Dict
    anomalies: List[FinancialAnomaly]
    trend_analysis: Dict
    ratio_analysis: Dict
    forensic_score: float
    risk_flags: List[str]
    shap_summary: Dict
    quantitative_report: Dict


class NumericalAnalystAgent:
    """Agent for numerical/financial forensic analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.xgb_model = None
        self.shap_explainer = None
        
    def validate_data(self, numerical_data: Dict) -> Dict:
        """Validate financial data for completeness and consistency"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'completeness_score': 0.0
        }
        
        # Check for required metrics
        required_metrics = [
            'total_revenue', 'net_income', 'total_assets', 
            'total_liabilities', 'shareholders_equity'
        ]
        
        key_metrics = numerical_data.get('key_metrics', {})
        missing_metrics = [m for m in required_metrics if m not in key_metrics]
        
        if missing_metrics:
            validation_results['warnings'].append(
                f"Missing metrics: {', '.join(missing_metrics)}"
            )
        
        # Calculate completeness
        completeness = (len(required_metrics) - len(missing_metrics)) / len(required_metrics)
        validation_results['completeness_score'] = completeness
        
        # Validate data consistency
        if 'total_assets' in key_metrics and 'total_liabilities' in key_metrics and 'shareholders_equity' in key_metrics:
            assets = key_metrics['total_assets']
            liabilities = key_metrics['total_liabilities']
            equity = key_metrics['shareholders_equity']
            
            # Basic accounting equation: Assets = Liabilities + Equity
            balance_diff = abs(assets - (liabilities + equity))
            tolerance = assets * 0.01  # 1% tolerance
            
            if balance_diff > tolerance:
                validation_results['errors'].append(
                    f"Balance sheet doesn't balance: Assets ({assets}) != Liabilities ({liabilities}) + Equity ({equity})"
                )
                validation_results['is_valid'] = False
        
        # Check for negative values where they shouldn't be
        for metric, value in key_metrics.items():
            if metric in ['total_assets', 'total_revenue', 'shareholders_equity']:
                if value < 0:
                    validation_results['errors'].append(
                        f"{metric} has negative value: {value}"
                    )
        
        # Check ratios for extreme values
        ratios = numerical_data.get('ratios', {})
        if 'debt_to_equity' in ratios and ratios['debt_to_equity'] > 5.0:
            validation_results['warnings'].append(
                f"Extremely high debt-to-equity ratio: {ratios['debt_to_equity']:.2f}"
            )
        
        return validation_results
    
    def detect_anomalies(self, numerical_data: Dict, historical_data: Optional[List[Dict]] = None) -> List[FinancialAnomaly]:
        """Detect anomalies in financial data using statistical methods and ML"""
        anomalies = []
        key_metrics = numerical_data.get('key_metrics', {})
        
        if not key_metrics:
            return anomalies
        
        # Method 1: Statistical outlier detection (Z-score)
        if historical_data and len(historical_data) > 3:
            anomalies.extend(self._detect_statistical_outliers(key_metrics, historical_data))
        
        # Method 2: Isolation Forest for multivariate anomaly detection
        if historical_data and len(historical_data) > 5:
            anomalies.extend(self._detect_isolation_forest_anomalies(key_metrics, historical_data))
        
        # Method 3: Business rule-based anomalies
        anomalies.extend(self._detect_business_rule_anomalies(key_metrics, numerical_data.get('ratios', {})))
        
        return anomalies
    
    def _detect_statistical_outliers(self, current_metrics: Dict, historical_data: List[Dict]) -> List[FinancialAnomaly]:
        """Detect outliers using Z-score method"""
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            # Get historical values
            historical_values = [
                h.get('key_metrics', {}).get(metric_name)
                for h in historical_data
                if h.get('key_metrics', {}).get(metric_name) is not None
            ]
            
            if len(historical_values) < 3:
                continue
            
            # Calculate Z-score
            mean = np.mean(historical_values)
            std = np.std(historical_values)
            
            if std == 0:
                continue
            
            z_score = abs((current_value - mean) / std)
            
            # Flag if Z-score > 2.5 (outlier)
            if z_score > 2.5:
                severity = 'high' if z_score > 3.5 else 'medium' if z_score > 3.0 else 'low'
                
                anomalies.append(FinancialAnomaly(
                    metric_name=metric_name,
                    value=current_value,
                    expected_range=(mean - 2*std, mean + 2*std),
                    anomaly_score=z_score,
                    severity=severity,
                    shap_values={},
                    explanation=f"{metric_name} deviates {z_score:.2f} standard deviations from historical mean"
                ))
        
        return anomalies
    
    def _detect_isolation_forest_anomalies(self, current_metrics: Dict, historical_data: List[Dict]) -> List[FinancialAnomaly]:
        """Detect anomalies using Isolation Forest"""
        anomalies = []
        
        # Prepare feature matrix
        common_metrics = set(current_metrics.keys())
        for h in historical_data:
            common_metrics &= set(h.get('key_metrics', {}).keys())
        
        if len(common_metrics) < 3:
            return anomalies
        
        common_metrics = list(common_metrics)
        
        # Build feature matrix
        X_historical = []
        for h in historical_data:
            row = [h['key_metrics'][m] for m in common_metrics]
            X_historical.append(row)
        
        X_current = np.array([[current_metrics[m] for m in common_metrics]])
        X_historical = np.array(X_historical)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X_historical)
        
        # Predict anomaly
        prediction = iso_forest.predict(X_current)
        anomaly_score = -iso_forest.score_samples(X_current)[0]
        
        if prediction[0] == -1:  # Anomaly detected
            severity = 'high' if anomaly_score > 0.6 else 'medium'
            
            # Identify which features contribute most
            feature_importance = {}
            for i, metric in enumerate(common_metrics):
                hist_mean = X_historical[:, i].mean()
                deviation = abs(X_current[0, i] - hist_mean) / (X_historical[:, i].std() + 1e-6)
                feature_importance[metric] = float(deviation)
            
            anomalies.append(FinancialAnomaly(
                metric_name='multivariate_anomaly',
                value=anomaly_score,
                expected_range=(0.0, 0.5),
                anomaly_score=anomaly_score,
                severity=severity,
                shap_values=feature_importance,
                explanation=f"Multivariate anomaly detected with score {anomaly_score:.3f}"
            ))
        
        return anomalies
    
    def _detect_business_rule_anomalies(self, metrics: Dict, ratios: Dict) -> List[FinancialAnomaly]:
        """Detect anomalies based on business rules"""
        anomalies = []
        
        # Rule 1: Negative net income with positive revenue (loss)
        if 'net_income' in metrics and 'total_revenue' in metrics:
            if metrics['net_income'] < 0 and metrics['total_revenue'] > 0:
                loss_magnitude = abs(metrics['net_income'] / metrics['total_revenue'])
                if loss_magnitude > 0.2:  # 20% loss
                    anomalies.append(FinancialAnomaly(
                        metric_name='net_income',
                        value=metrics['net_income'],
                        expected_range=(0, metrics['total_revenue'] * 0.2),
                        anomaly_score=loss_magnitude,
                        severity='high' if loss_magnitude > 0.5 else 'medium',
                        shap_values={},
                        explanation=f"Significant net loss: {loss_magnitude*100:.1f}% of revenue"
                    ))
        
        # Rule 2: Very high debt-to-equity ratio
        if 'debt_to_equity' in ratios and ratios['debt_to_equity'] > 3.0:
            anomalies.append(FinancialAnomaly(
                metric_name='debt_to_equity',
                value=ratios['debt_to_equity'],
                expected_range=(0.0, 2.0),
                anomaly_score=ratios['debt_to_equity'] / 3.0,
                severity='high' if ratios['debt_to_equity'] > 5.0 else 'medium',
                shap_values={},
                explanation=f"High financial leverage: D/E ratio of {ratios['debt_to_equity']:.2f}"
            ))
        
        # Rule 3: Negative equity (balance sheet insolvency)
        if 'shareholders_equity' in metrics and metrics['shareholders_equity'] < 0:
            anomalies.append(FinancialAnomaly(
                metric_name='shareholders_equity',
                value=metrics['shareholders_equity'],
                expected_range=(0, float('inf')),
                anomaly_score=1.0,
                severity='high',
                shap_values={},
                explanation="Negative shareholders' equity indicates balance sheet insolvency"
            ))
        
        return anomalies
    
    def perform_trend_analysis(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """Analyze trends in financial metrics over time"""
        trend_analysis = {
            'revenue_trend': None,
            'profit_trend': None,
            'asset_trend': None,
            'trends': {}
        }
        
        if not historical_data or len(historical_data) < 2:
            return trend_analysis
        
        # Analyze key metric trends
        metrics_to_analyze = ['total_revenue', 'net_income', 'total_assets', 'total_liabilities']
        
        for metric in metrics_to_analyze:
            values = []
            dates = []
            
            for h in historical_data:
                if metric in h.get('key_metrics', {}):
                    values.append(h['key_metrics'][metric])
                    dates.append(h.get('filing_info', {}).get('report_date', ''))
            
            # Add current
            if metric in current_data.get('key_metrics', {}):
                values.append(current_data['key_metrics'][metric])
                dates.append(current_data.get('filing_info', {}).get('report_date', ''))
            
            if len(values) >= 2:
                # Calculate trend
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                trend_direction = 'increasing' if slope > 0 else 'decreasing'
                trend_strength = abs(r_value)
                
                # Calculate year-over-year growth
                yoy_growth = []
                for i in range(1, len(values)):
                    if values[i-1] != 0:
                        growth = ((values[i] - values[i-1]) / abs(values[i-1])) * 100
                        yoy_growth.append(growth)
                
                trend_analysis['trends'][metric] = {
                    'direction': trend_direction,
                    'strength': float(trend_strength),
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'yoy_growth': yoy_growth,
                    'avg_growth': float(np.mean(yoy_growth)) if yoy_growth else 0,
                    'volatility': float(np.std(values)) if len(values) > 1 else 0
                }
        
        return trend_analysis
    
    def compute_forensic_model(self, numerical_data: Dict, historical_data: Optional[List[Dict]] = None) -> Tuple[float, Dict]:
        """Compute forensic risk score using XGBoost model"""
        # Extract features
        features = self._extract_features(numerical_data)
        
        if not features:
            return 0.5, {}
        
        # Create feature dataframe
        feature_df = pd.DataFrame([features])
        
        # Train simple XGBoost model (in production, use pre-trained model)
        if historical_data and len(historical_data) > 10:
            X_train, y_train = self._prepare_training_data(historical_data)
            
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=settings.XGBOOST_N_ESTIMATORS,
                max_depth=settings.XGBOOST_MAX_DEPTH,
                learning_rate=settings.XGBOOST_LEARNING_RATE,
                random_state=42
            )
            self.xgb_model.fit(X_train, y_train)
            
            # Predict risk
            risk_proba = self.xgb_model.predict_proba(feature_df)[0, 1]
            
            # Compute SHAP values
            self.shap_explainer = shap.TreeExplainer(self.xgb_model)
            shap_values = self.shap_explainer.shap_values(feature_df)
            
            # Create SHAP explanation
            shap_explanation = {}
            for i, feature_name in enumerate(feature_df.columns):
                shap_explanation[feature_name] = float(shap_values[0][i])
        else:
            # Use heuristic scoring
            risk_proba = self._heuristic_risk_score(features)
            shap_explanation = self._heuristic_feature_importance(features)
        
        return float(risk_proba), shap_explanation
    
    def _extract_features(self, numerical_data: Dict) -> Dict:
        """Extract features for ML model"""
        features = {}
        
        key_metrics = numerical_data.get('key_metrics', {})
        ratios = numerical_data.get('ratios', {})
        
        # Add metrics as features
        for metric, value in key_metrics.items():
            if value is not None and not np.isnan(value):
                features[metric] = value
        
        # Add ratios as features
        for ratio, value in ratios.items():
            if value is not None and not np.isnan(value):
                features[ratio] = value
        
        return features
    
    def _prepare_training_data(self, historical_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data for XGBoost"""
        X = []
        y = []
        
        for data in historical_data:
            features = self._extract_features(data.get('numerical_data', {}))
            if features:
                X.append(features)
                # Dummy labels (in production, use actual fraud/risk labels)
                y.append(0)
        
        return pd.DataFrame(X), np.array(y)
    
    def _heuristic_risk_score(self, features: Dict) -> float:
        """Calculate risk score using heuristics"""
        risk_score = 0.0
        weights = 0.0
        
        # Profitability risk
        if 'profit_margin' in features:
            if features['profit_margin'] < 0:
                risk_score += 0.3
            elif features['profit_margin'] < 0.05:
                risk_score += 0.15
            weights += 0.3
        
        # Leverage risk
        if 'debt_to_equity' in features:
            if features['debt_to_equity'] > 3.0:
                risk_score += 0.25
            elif features['debt_to_equity'] > 2.0:
                risk_score += 0.15
            weights += 0.25
        
        # Liquidity risk
        if 'cash_ratio' in features:
            if features['cash_ratio'] < 0.05:
                risk_score += 0.2
            elif features['cash_ratio'] < 0.1:
                risk_score += 0.1
            weights += 0.2
        
        # Asset quality
        if 'return_on_assets' in features:
            if features['return_on_assets'] < 0:
                risk_score += 0.15
            elif features['return_on_assets'] < 0.02:
                risk_score += 0.08
            weights += 0.15
        
        # Solvency
        if 'shareholders_equity' in features:
            if features['shareholders_equity'] < 0:
                risk_score += 0.1
            weights += 0.1
        
        return min(risk_score / (weights if weights > 0 else 1.0), 1.0)
    
    def _heuristic_feature_importance(self, features: Dict) -> Dict:
        """Calculate feature importance heuristically"""
        importance = {}
        
        for feature, value in features.items():
            if 'debt' in feature.lower():
                importance[feature] = 0.15
            elif 'equity' in feature.lower():
                importance[feature] = 0.12
            elif 'revenue' in feature.lower():
                importance[feature] = 0.1
            elif 'profit' in feature.lower() or 'margin' in feature.lower():
                importance[feature] = 0.13
            elif 'asset' in feature.lower():
                importance[feature] = 0.11
            else:
                importance[feature] = 0.05
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def generate_quantitative_report(
        self, 
        numerical_data: Dict,
        validation_results: Dict,
        anomalies: List[FinancialAnomaly],
        trend_analysis: Dict,
        forensic_score: float,
        shap_values: Dict
    ) -> Dict:
        """Generate comprehensive quantitative report"""
        # Determine risk flags
        risk_flags = []
        
        if not validation_results['is_valid']:
            risk_flags.append("Data validation failed")
        
        if forensic_score > 0.7:
            risk_flags.append("High forensic risk score")
        
        if len(anomalies) > 3:
            risk_flags.append("Multiple anomalies detected")
        
        high_severity_anomalies = [a for a in anomalies if a.severity == 'high']
        if high_severity_anomalies:
            risk_flags.append(f"{len(high_severity_anomalies)} high-severity anomalies")
        
        # Build report
        report = {
            'executive_summary': {
                'forensic_score': forensic_score,
                'risk_level': self._determine_risk_level(forensic_score),
                'total_anomalies': len(anomalies),
                'high_severity_anomalies': len(high_severity_anomalies),
                'data_quality_score': validation_results['completeness_score'],
                'primary_concerns': risk_flags[:3]
            },
            'validation_summary': validation_results,
            'anomaly_details': [asdict(a) for a in anomalies],
            'trend_analysis': trend_analysis,
            'feature_importance': shap_values,
            'key_metrics': numerical_data.get('key_metrics', {}),
            'financial_ratios': numerical_data.get('ratios', {}),
            'risk_flags': risk_flags,
            'recommendations': self._generate_recommendations(forensic_score, anomalies, validation_results)
        }
        
        return report
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score"""
        if score >= settings.RISK_L3_THRESHOLD:
            return "L3_HIGH"
        elif score >= settings.RISK_L2_THRESHOLD:
            return "L2_MEDIUM"
        else:
            return "L1_LOW"
    
    def _generate_recommendations(self, score: float, anomalies: List[FinancialAnomaly], validation: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if score > 0.7:
            recommendations.append("Immediate detailed audit recommended")
            recommendations.append("Review all financial statements for accuracy")
        
        if not validation['is_valid']:
            recommendations.append("Resolve data validation errors before proceeding")
        
        for anomaly in anomalies:
            if anomaly.severity == 'high':
                recommendations.append(f"Investigate {anomaly.metric_name}: {anomaly.explanation}")
        
        if validation['completeness_score'] < 0.7:
            recommendations.append("Obtain missing financial data for complete analysis")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def analyze(self, numerical_data: Dict, historical_data: Optional[List[Dict]] = None) -> NumericalAnalysisResult:
        """Main analysis method - orchestrates all numerical analysis"""
        logger.info("Starting numerical forensic analysis")
        
        # Step 1: Validate data
        validation_results = self.validate_data(numerical_data)
        
        # Step 2: Detect anomalies
        anomalies = self.detect_anomalies(numerical_data, historical_data)
        
        # Step 3: Trend analysis
        trend_analysis = self.perform_trend_analysis(numerical_data, historical_data or [])
        
        # Step 4: Compute forensic model
        forensic_score, shap_values = self.compute_forensic_model(numerical_data, historical_data)
        
        # Step 5: Generate report
        quantitative_report = self.generate_quantitative_report(
            numerical_data,
            validation_results,
            anomalies,
            trend_analysis,
            forensic_score,
            shap_values
        )
        
        # Compile results
        result = NumericalAnalysisResult(
            company_info=numerical_data.get('metadata', {}).get('company_info', {}),
            filing_info=numerical_data.get('metadata', {}).get('filing_info', {}),
            validation_results=validation_results,
            anomalies=anomalies,
            trend_analysis=trend_analysis,
            ratio_analysis=numerical_data.get('ratios', {}),
            forensic_score=forensic_score,
            risk_flags=quantitative_report['risk_flags'],
            shap_summary=shap_values,
            quantitative_report=quantitative_report
        )
        
        logger.info(f"Analysis complete. Forensic score: {forensic_score:.3f}")
        return result


# Singleton instance
numerical_analyst = NumericalAnalystAgent()
