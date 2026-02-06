"""
Textual Investigator Agent
Performs forensic analysis on textual data using Longformer and FinBERT
"""
import re
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import numpy as np
import torch

# NLP Libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    LongformerTokenizer,
    LongformerModel
)
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob

from config import settings

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


@dataclass
class TextualFlag:
    """Represents a detected textual red flag"""
    category: str  # 'risk_indicator', 'legal_term', 'euphemism', 'evasive_language'
    text_snippet: str
    location: str  # section name
    severity: str  # 'low', 'medium', 'high'
    confidence_score: float
    explanation: str
    sentence_context: str
    attention_scores: Optional[List[float]] = None


@dataclass
class TextualAnalysisResult:
    """Complete textual analysis result"""
    company_info: Dict
    filing_info: Dict
    sentiment_analysis: Dict
    complexity_metrics: Dict
    red_flags: List[TextualFlag]
    suspicious_patterns: List[Dict]
    evidence_snippets: List[str]
    linguistic_score: float
    textual_report: Dict


class TextualInvestigatorAgent:
    """Agent for textual/narrative forensic analysis"""
    
    # Risk indicator keywords and phrases
    RISK_KEYWORDS = [
        'litigation', 'lawsuit', 'investigation', 'regulatory action',
        'material weakness', 'restatement', 'contingent liability',
        'adverse', 'uncertainty', 'bankruptcy', 'default',
        'impairment', 'write-down', 'write-off', 'restructuring',
        'going concern', 'liquidity crisis', 'covenant violation'
    ]
    
    LEGAL_TERMS = [
        'sec investigation', 'department of justice', 'class action',
        'derivative lawsuit', 'securities fraud', 'insider trading',
        'accounting irregularities', 'financial fraud', 'misstatement'
    ]
    
    EUPHEMISMS = [
        'challenging environment', 'headwinds', 'market conditions',
        'softness', 'pressures', 'adjustments', 'one-time charge',
        'non-recurring', 'exceptional items', 'strategic realignment'
    ]
    
    EVASIVE_PHRASES = [
        'to the best of our knowledge', 'substantially', 'approximately',
        'estimates suggest', 'we believe', 'anticipated', 'expected',
        'management judgment', 'certain circumstances', 'may vary'
    ]
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models lazily
        self._finbert_model = None
        self._finbert_tokenizer = None
        self._longformer_model = None
        self._longformer_tokenizer = None
    
    @property
    def finbert_model(self):
        """Lazy load FinBERT model"""
        if self._finbert_model is None:
            logger.info("Loading FinBERT model...")
            self._finbert_tokenizer = AutoTokenizer.from_pretrained(
                settings.FINBERT_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self._finbert_model = AutoModelForSequenceClassification.from_pretrained(
                settings.FINBERT_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            ).to(self.device)
            self._finbert_model.eval()
        return self._finbert_model
    
    @property
    def finbert_tokenizer(self):
        """Lazy load FinBERT tokenizer"""
        if self._finbert_tokenizer is None:
            _ = self.finbert_model  # Trigger model load
        return self._finbert_tokenizer
    
    @property
    def longformer_model(self):
        """Lazy load Longformer model"""
        if self._longformer_model is None:
            logger.info("Loading Longformer model...")
            self._longformer_tokenizer = LongformerTokenizer.from_pretrained(
                settings.LONGFORMER_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            self._longformer_model = LongformerModel.from_pretrained(
                settings.LONGFORMER_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            ).to(self.device)
            self._longformer_model.eval()
        return self._longformer_model
    
    @property
    def longformer_tokenizer(self):
        """Lazy load Longformer tokenizer"""
        if self._longformer_tokenizer is None:
            _ = self.longformer_model  # Trigger model load
        return self._longformer_tokenizer
    
    def analyze_sentiment(self, text: str, section_name: str = '') -> Dict:
        """Analyze sentiment and tone using FinBERT"""
        sentences = sent_tokenize(text[:10000])  # Limit for performance
        
        sentiments = []
        for sentence in sentences[:50]:  # Analyze first 50 sentences
            if len(sentence.strip()) < 10:
                continue
            
            try:
                inputs = self.finbert_tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.finbert_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # FinBERT classes: negative, neutral, positive
                sentiment_scores = predictions[0].cpu().numpy()
                sentiments.append({
                    'sentence': sentence,
                    'negative': float(sentiment_scores[0]),
                    'neutral': float(sentiment_scores[1]),
                    'positive': float(sentiment_scores[2])
                })
            except Exception as e:
                logger.warning(f"Error analyzing sentence sentiment: {e}")
                continue
        
        # Aggregate sentiment
        if sentiments:
            avg_negative = np.mean([s['negative'] for s in sentiments])
            avg_neutral = np.mean([s['neutral'] for s in sentiments])
            avg_positive = np.mean([s['positive'] for s in sentiments])
            
            overall_sentiment = 'negative' if avg_negative > max(avg_neutral, avg_positive) else \
                              'positive' if avg_positive > avg_neutral else 'neutral'
        else:
            avg_negative = avg_neutral = avg_positive = 0.33
            overall_sentiment = 'neutral'
        
        return {
            'section': section_name,
            'overall_sentiment': overall_sentiment,
            'negative_score': float(avg_negative),
            'neutral_score': float(avg_neutral),
            'positive_score': float(avg_positive),
            'sentiment_shift': float(avg_negative - avg_positive),
            'analyzed_sentences': len(sentiments),
            'top_negative_sentences': sorted(sentiments, key=lambda x: x['negative'], reverse=True)[:3]
        }
    
    def compute_complexity_metrics(self, text: str) -> Dict:
        """Compute Gunning Fog Index and obfuscation metrics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        if not sentences or not words:
            return {
                'gunning_fog_index': 0,
                'avg_sentence_length': 0,
                'complex_word_percentage': 0,
                'obfuscation_score': 0
            }
        
        # Gunning Fog Index = 0.4 * [(words/sentences) + 100 * (complex_words/words)]
        avg_sentence_length = len(words) / len(sentences)
        
        # Count complex words (3+ syllables)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        complex_word_percentage = (complex_words / len(words)) * 100
        
        gunning_fog = 0.4 * (avg_sentence_length + complex_word_percentage)
        
        # Obfuscation score (higher = more obfuscated)
        # Based on: excessive length, complex words, passive voice
        obfuscation_score = min(gunning_fog / 20, 1.0)  # Normalize to 0-1
        
        # Check for excessive use of complex/evasive language
        evasive_count = sum(
            1 for phrase in self.EVASIVE_PHRASES 
            if phrase.lower() in text.lower()
        )
        
        if evasive_count > 10:
            obfuscation_score = min(obfuscation_score + 0.2, 1.0)
        
        return {
            'gunning_fog_index': float(gunning_fog),
            'avg_sentence_length': float(avg_sentence_length),
            'complex_word_percentage': float(complex_word_percentage),
            'obfuscation_score': float(obfuscation_score),
            'evasive_phrase_count': evasive_count,
            'readability_grade': self._fog_to_grade(gunning_fog)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least 1 syllable
        return max(1, syllable_count)
    
    def _fog_to_grade(self, fog_index: float) -> str:
        """Convert Fog index to readability grade"""
        if fog_index < 6:
            return "Elementary"
        elif fog_index < 10:
            return "Middle School"
        elif fog_index < 14:
            return "High School"
        elif fog_index < 18:
            return "College"
        else:
            return "Post-Graduate"
    
    def scan_keywords_phrases(self, text: str, section_name: str) -> List[TextualFlag]:
        """Scan for risk indicators, legal terms, and red flags"""
        flags = []
        sentences = sent_tokenize(text)
        
        # Scan for risk keywords
        for keyword in self.RISK_KEYWORDS:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            matches = pattern.finditer(text.lower())
            
            for match in matches:
                # Find the sentence containing this match
                char_pos = match.start()
                sentence = self._find_sentence_at_position(sentences, text, char_pos)
                
                if sentence:
                    flags.append(TextualFlag(
                        category='risk_indicator',
                        text_snippet=keyword,
                        location=section_name,
                        severity='medium',
                        confidence_score=0.7,
                        explanation=f"Risk indicator keyword detected: {keyword}",
                        sentence_context=sentence
                    ))
        
        # Scan for legal terms
        for term in self.LEGAL_TERMS:
            if term.lower() in text.lower():
                sentence = self._find_sentence_containing(sentences, term)
                if sentence:
                    flags.append(TextualFlag(
                        category='legal_term',
                        text_snippet=term,
                        location=section_name,
                        severity='high',
                        confidence_score=0.9,
                        explanation=f"Legal/regulatory term detected: {term}",
                        sentence_context=sentence
                    ))
        
        # Scan for euphemisms
        for euphemism in self.EUPHEMISMS:
            if euphemism.lower() in text.lower():
                sentence = self._find_sentence_containing(sentences, euphemism)
                if sentence:
                    flags.append(TextualFlag(
                        category='euphemism',
                        text_snippet=euphemism,
                        location=section_name,
                        severity='low',
                        confidence_score=0.6,
                        explanation=f"Potential euphemism for negative news: {euphemism}",
                        sentence_context=sentence
                    ))
        
        # Scan for evasive language
        evasive_count = 0
        for phrase in self.EVASIVE_PHRASES:
            count = text.lower().count(phrase.lower())
            evasive_count += count
        
        if evasive_count > 5:
            flags.append(TextualFlag(
                category='evasive_language',
                text_snippet=f"{evasive_count} evasive phrases",
                location=section_name,
                severity='medium',
                confidence_score=0.65,
                explanation=f"High frequency of evasive/uncertain language ({evasive_count} instances)",
                sentence_context=f"Examples: {', '.join(self.EVASIVE_PHRASES[:3])}"
            ))
        
        return flags
    
    def _find_sentence_at_position(self, sentences: List[str], full_text: str, position: int) -> str:
        """Find the sentence at a given character position"""
        current_pos = 0
        for sentence in sentences:
            sent_start = full_text.find(sentence, current_pos)
            sent_end = sent_start + len(sentence)
            
            if sent_start <= position < sent_end:
                return sentence
            
            current_pos = sent_end
        
        return ""
    
    def _find_sentence_containing(self, sentences: List[str], phrase: str) -> str:
        """Find first sentence containing a phrase"""
        for sentence in sentences:
            if phrase.lower() in sentence.lower():
                return sentence
        return ""
    
    def detect_context_manipulation(self, text: str, section_name: str) -> List[Dict]:
        """Detect euphemism and context manipulation"""
        patterns = []
        sentences = sent_tokenize(text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Pattern 1: Minimizing negative news
            if any(neg in sentence_lower for neg in ['loss', 'decline', 'decrease', 'impairment']):
                if any(eup in sentence_lower for eup in self.EUPHEMISMS):
                    patterns.append({
                        'type': 'minimization',
                        'sentence': sentence,
                        'location': section_name,
                        'explanation': 'Negative news followed by euphemistic language'
                    })
            
            # Pattern 2: Burying bad news
            if i > 0 and any(good in sentence_lower for good in ['growth', 'increase', 'strong', 'positive']):
                prev_sentence_lower = sentences[i-1].lower()
                if any(bad in prev_sentence_lower for bad in ['however', 'although', 'despite']):
                    patterns.append({
                        'type': 'burying_bad_news',
                        'sentence': f"{sentences[i-1]} {sentence}",
                        'location': section_name,
                        'explanation': 'Negative information followed immediately by positive framing'
                    })
            
            # Pattern 3: Excessive qualification
            qualifier_count = sum(1 for qual in ['may', 'might', 'could', 'possibly', 'perhaps'] 
                                if qual in sentence_lower.split())
            if qualifier_count >= 2:
                patterns.append({
                    'type': 'excessive_qualification',
                    'sentence': sentence,
                    'location': section_name,
                    'explanation': f'Excessive use of qualifying language ({qualifier_count} qualifiers)'
                })
        
        return patterns[:10]  # Return top 10 patterns
    
    def analyze_with_longformer(self, text: str) -> Dict:
        """Analyze long document context using Longformer"""
        try:
            # Longformer can handle up to 4096 tokens
            inputs = self.longformer_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.longformer_model(**inputs)
                
                # Get attention scores
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    # Average attention across all layers and heads
                    attention = torch.stack(outputs.attentions).mean(dim=(0, 1))
                    attention_scores = attention[0].cpu().numpy()
                else:
                    attention_scores = None
            
            # Extract embeddings
            last_hidden_state = outputs.last_hidden_state[0].cpu().numpy()
            
            return {
                'context_preserved': True,
                'embedding_shape': last_hidden_state.shape,
                'avg_embedding_norm': float(np.linalg.norm(last_hidden_state, axis=1).mean()),
                'has_attention_scores': attention_scores is not None
            }
            
        except Exception as e:
            logger.error(f"Error in Longformer analysis: {e}")
            return {
                'context_preserved': False,
                'error': str(e)
            }
    
    def calculate_linguistic_risk_score(
        self,
        sentiment_analysis: Dict,
        complexity_metrics: Dict,
        red_flags: List[TextualFlag],
        suspicious_patterns: List[Dict]
    ) -> float:
        """Calculate overall linguistic risk score"""
        score = 0.0
        
        # Factor 1: Negative sentiment (0-0.25)
        score += sentiment_analysis.get('negative_score', 0) * 0.25
        
        # Factor 2: Complexity/obfuscation (0-0.25)
        score += complexity_metrics.get('obfuscation_score', 0) * 0.25
        
        # Factor 3: Red flags (0-0.3)
        high_severity_flags = len([f for f in red_flags if f.severity == 'high'])
        medium_severity_flags = len([f for f in red_flags if f.severity == 'medium'])
        flag_score = min((high_severity_flags * 0.1 + medium_severity_flags * 0.05), 0.3)
        score += flag_score
        
        # Factor 4: Suspicious patterns (0-0.2)
        pattern_score = min(len(suspicious_patterns) * 0.02, 0.2)
        score += pattern_score
        
        return min(score, 1.0)
    
    def generate_textual_report(
        self,
        textual_data: Dict,
        sentiment_analysis: Dict,
        complexity_metrics: Dict,
        red_flags: List[TextualFlag],
        suspicious_patterns: List[Dict],
        linguistic_score: float
    ) -> Dict:
        """Generate comprehensive textual analysis report"""
        # Categorize flags
        flags_by_category = {}
        for flag in red_flags:
            if flag.category not in flags_by_category:
                flags_by_category[flag.category] = []
            flags_by_category[flag.category].append(flag)
        
        # Extract evidence snippets
        evidence_snippets = [
            flag.sentence_context
            for flag in red_flags
            if flag.severity in ['high', 'medium']
        ][:10]
        
        report = {
            'executive_summary': {
                'linguistic_risk_score': linguistic_score,
                'overall_sentiment': sentiment_analysis.get('overall_sentiment', 'neutral'),
                'sentiment_score': sentiment_analysis.get('negative_score', 0),
                'obfuscation_score': complexity_metrics.get('obfuscation_score', 0),
                'total_red_flags': len(red_flags),
                'high_severity_flags': len([f for f in red_flags if f.severity == 'high']),
                'readability_grade': complexity_metrics.get('readability_grade', 'Unknown')
            },
            'sentiment_details': sentiment_analysis,
            'complexity_analysis': complexity_metrics,
            'red_flags_by_category': {
                cat: [asdict(f) for f in flags]
                for cat, flags in flags_by_category.items()
            },
            'suspicious_patterns': suspicious_patterns,
            'evidence_snippets': evidence_snippets,
            'key_concerns': self._identify_key_concerns(red_flags, suspicious_patterns, complexity_metrics),
            'recommendations': self._generate_textual_recommendations(linguistic_score, red_flags, complexity_metrics)
        }
        
        return report
    
    def _identify_key_concerns(
        self,
        red_flags: List[TextualFlag],
        patterns: List[Dict],
        complexity: Dict
    ) -> List[str]:
        """Identify key textual concerns"""
        concerns = []
        
        # Legal/regulatory concerns
        legal_flags = [f for f in red_flags if f.category == 'legal_term']
        if legal_flags:
            concerns.append(f"Legal/regulatory issues mentioned ({len(legal_flags)} instances)")
        
        # High-risk indicators
        high_risk_flags = [f for f in red_flags if f.severity == 'high']
        if len(high_risk_flags) > 3:
            concerns.append(f"Multiple high-severity risk indicators ({len(high_risk_flags)} found)")
        
        # Obfuscation
        if complexity.get('obfuscation_score', 0) > 0.7:
            concerns.append("High document obfuscation detected")
        
        # Context manipulation
        if len(patterns) > 5:
            concerns.append(f"Multiple suspicious language patterns ({len(patterns)} detected)")
        
        return concerns[:5]
    
    def _generate_textual_recommendations(
        self,
        score: float,
        red_flags: List[TextualFlag],
        complexity: Dict
    ) -> List[str]:
        """Generate textual analysis recommendations"""
        recommendations = []
        
        if score > 0.7:
            recommendations.append("High linguistic risk - detailed manual review of all sections recommended")
        
        legal_flags = [f for f in red_flags if f.category == 'legal_term']
        if legal_flags:
            recommendations.append("Investigate legal/regulatory issues mentioned in filing")
        
        if complexity.get('obfuscation_score', 0) > 0.6:
            recommendations.append("Request clearer disclosure - document shows signs of intentional complexity")
        
        risk_flags = [f for f in red_flags if f.category == 'risk_indicator']
        if len(risk_flags) > 10:
            recommendations.append("Significant risk factors identified - assess materiality of each")
        
        return recommendations[:5]
    
    def analyze(self, textual_data: Dict) -> TextualAnalysisResult:
        """Main analysis method - orchestrates all textual analysis"""
        logger.info("Starting textual forensic analysis")
        
        sections = textual_data.get('sections', {})
        full_text = textual_data.get('full_text', '')
        
        # Analyze each section
        all_sentiment_analysis = {}
        all_red_flags = []
        all_patterns = []
        
        for section_name, section_text in sections.items():
            if not section_text or len(section_text.strip()) < 100:
                continue
            
            logger.info(f"Analyzing section: {section_name}")
            
            # Sentiment analysis
            sentiment = self.analyze_sentiment(section_text, section_name)
            all_sentiment_analysis[section_name] = sentiment
            
            # Keyword scanning
            flags = self.scan_keywords_phrases(section_text, section_name)
            all_red_flags.extend(flags)
            
            # Context manipulation
            patterns = self.detect_context_manipulation(section_text, section_name)
            all_patterns.extend(patterns)
        
        # Overall complexity metrics
        complexity_metrics = self.compute_complexity_metrics(full_text[:50000])
        
        # Longformer analysis on full document
        longformer_analysis = self.analyze_with_longformer(full_text[:4000])
        
        # Calculate linguistic risk score
        linguistic_score = self.calculate_linguistic_risk_score(
            all_sentiment_analysis.get('item_7', all_sentiment_analysis.get(list(all_sentiment_analysis.keys())[0], {})) if all_sentiment_analysis else {},
            complexity_metrics,
            all_red_flags,
            all_patterns
        )
        
        # Generate report
        textual_report = self.generate_textual_report(
            textual_data,
            all_sentiment_analysis,
            complexity_metrics,
            all_red_flags,
            all_patterns,
            linguistic_score
        )
        
        # Extract evidence snippets
        evidence_snippets = list(set([
            flag.sentence_context[:200] + "..."
            for flag in all_red_flags
            if flag.severity in ['high', 'medium']
        ]))[:10]
        
        result = TextualAnalysisResult(
            company_info=textual_data.get('metadata', {}).get('company_info', {}),
            filing_info=textual_data.get('metadata', {}).get('filing_info', {}),
            sentiment_analysis=all_sentiment_analysis,
            complexity_metrics=complexity_metrics,
            red_flags=all_red_flags,
            suspicious_patterns=all_patterns,
            evidence_snippets=evidence_snippets,
            linguistic_score=linguistic_score,
            textual_report=textual_report
        )
        
        logger.info(f"Textual analysis complete. Linguistic risk score: {linguistic_score:.3f}")
        return result


# Singleton instance
textual_investigator = TextualInvestigatorAgent()
