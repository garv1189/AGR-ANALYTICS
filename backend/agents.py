import os
import asyncio
from typing import List, Dict, Optional
from emergentintegrations.llm.chat import LlmChat, UserMessage
from models import AGRQuery, RetrievedChunk, ReasonerAnalysis, AGRResponse
from document_processor import DocumentProcessor
import logging
import json
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class RetrieverAgent:
    """Section-Aware AGR Retriever Agent"""
    
    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
    
    async def retrieve(self, query: AGRQuery) -> List[RetrievedChunk]:
        """Search the vector database and retrieve top-k relevant chunks"""
        try:
            # Search for relevant chunks
            search_results = self.document_processor.search_chunks(
                query=query.query,
                top_k=query.top_k,
                company_filter=query.company_filter,
                year_filter=query.year_filter,
                section_filter=query.section_filter
            )
            
            # Convert to RetrievedChunk objects
            retrieved_chunks = []
            for result in search_results:
                chunk = RetrievedChunk(
                    chunk_id=result['chunk_id'],
                    content=result['content'],
                    document_filename=result['document_filename'],
                    company_name=result['company_name'],
                    year=result['year'],
                    section_type=result.get('section_type'),
                    page_number=result.get('page_number'),
                    relevance_score=result['relevance_score']
                )
                retrieved_chunks.append(chunk)
            
            if not retrieved_chunks:
                logger.info("No relevant chunks found for query")
            else:
                logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error in retriever agent: {e}")
            return []


class ReasonerAgent:
    """Contextual Financial Analyzer Agent"""
    
    def __init__(self):
        self.llm_chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id="reasoner_agent",
            system_message="""You are a expert financial analyst specializing in Annual General Reports (AGR) analysis. Your role is to examine user queries and retrieved AGR content to provide contextual analysis.

Your tasks:
1. Analyze if the retrieved content is sufficient to answer the user's query
2. Identify the type of analysis required (comparison, summary, trend analysis, risk detection)
3. Suggest query reformulation if needed
4. Provide structured context for the response generator

Guidelines:
- Always be objective and data-driven
- Identify if cross-year comparisons are needed
- Detect potential red flags or risks
- Assess confidence based on data availability
- Be concise but thorough in analysis

Response format: Provide analysis as structured JSON with fields: analysis_type, requires_reformulation, suggested_reformulation, context_summary, confidence_level"""
        ).with_model("openai", "gpt-4o-mini")
    
    async def analyze(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> ReasonerAnalysis:
        """Analyze query context and retrieved chunks"""
        try:
            if not retrieved_chunks:
                return ReasonerAnalysis(
                    analysis_type="insufficient_data",
                    requires_reformulation=True,
                    suggested_reformulation=f"Try rephrasing your query with more specific terms or check if relevant documents are uploaded",
                    context_summary="No relevant chunks found in the database",
                    confidence_level="low"
                )
            
            # Prepare context for analysis
            chunks_summary = []
            for chunk in retrieved_chunks[:3]:  # Analyze top 3 chunks
                chunks_summary.append({
                    "content_preview": chunk.content[:200] + "...",
                    "company": chunk.company_name,
                    "year": chunk.year,
                    "section": chunk.section_type or "Unknown",
                    "relevance_score": chunk.relevance_score
                })
            
            analysis_prompt = f"""
            Analyze this AGR query and retrieved content:
            
            USER QUERY: {query}
            
            RETRIEVED CHUNKS:
            {json.dumps(chunks_summary, indent=2)}
            
            Provide analysis as JSON with these exact fields:
            - analysis_type: (comparison|summary|trend_analysis|risk_detection|specific_data|insufficient_data)
            - requires_reformulation: (true|false)
            - suggested_reformulation: (string or null)
            - context_summary: (string describing what the chunks contain)
            - confidence_level: (high|medium|low)
            """
            
            user_message = UserMessage(text=analysis_prompt)
            response = await self.llm_chat.send_message(user_message)
            
            # Parse JSON response
            try:
                analysis_data = json.loads(response)
                return ReasonerAnalysis(
                    analysis_type=analysis_data.get('analysis_type', 'summary'),
                    requires_reformulation=analysis_data.get('requires_reformulation', False),
                    suggested_reformulation=analysis_data.get('suggested_reformulation'),
                    context_summary=analysis_data.get('context_summary', 'Analysis of retrieved AGR content'),
                    confidence_level=analysis_data.get('confidence_level', 'medium')
                )
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return ReasonerAnalysis(
                    analysis_type="summary",
                    requires_reformulation=False,
                    context_summary=f"Analysis of {len(retrieved_chunks)} retrieved chunks from AGR documents",
                    confidence_level="medium"
                )
                
        except Exception as e:
            logger.error(f"Error in reasoner agent: {e}")
            return ReasonerAnalysis(
                analysis_type="error",
                requires_reformulation=True,
                suggested_reformulation="Please try rephrasing your query",
                context_summary="Error occurred during analysis",
                confidence_level="low"
            )


class ResponderAgent:
    """Structured Report Answer Generator Agent"""
    
    def __init__(self):
        self.llm_chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id="responder_agent",
            system_message="""You are an expert financial report writer specializing in Annual General Reports (AGR). Your role is to provide clear, structured answers using ONLY the provided AGR content.

CRITICAL RULES:
- NEVER hallucinate numbers, claims, or facts not in the provided content
- Always cite section/page numbers when available
- If data is not in the AGR content, explicitly state "This detail is not available in the Annual General Report"
- Focus on factual, data-driven responses

Response Formats:
📊 TABLES: For comparisons, growth rates, KPIs (use markdown table format)
📈 CHARTS: For trends (describe chart data, don't create actual charts)
📝 SUMMARIES: For executive overviews
🚨 RED FLAGS: For risks, litigations, negative trends

Always include:
1. Direct answer to the query
2. Supporting evidence from AGR content
3. Citations (section/page numbers when available)
4. Confidence assessment based on data availability
"""
        ).with_model("openai", "gpt-4o-mini")
    
    async def generate_response(self, query: str, retrieved_chunks: List[RetrievedChunk], 
                              reasoner_analysis: ReasonerAnalysis) -> Dict:
        """Generate structured response based on query and analysis"""
        try:
            if not retrieved_chunks:
                return {
                    "formatted_answer": "I don't have relevant information in the uploaded Annual General Reports to answer your question. Please ensure the relevant AGR documents are uploaded or try rephrasing your query with more specific terms.",
                    "confidence_score": 0.1,
                    "citations": [],
                    "response_format": "text"
                }
            
            # Prepare comprehensive context
            context_chunks = []
            citations = set()
            
            for chunk in retrieved_chunks:
                context_info = f"""
                Content: {chunk.content}
                Source: {chunk.document_filename} (Company: {chunk.company_name}, Year: {chunk.year})
                Section: {chunk.section_type or 'Unspecified'}
                Page: {chunk.page_number or 'N/A'}
                Relevance Score: {chunk.relevance_score:.3f}
                """
                context_chunks.append(context_info)
                
                # Build citation
                citation = f"{chunk.document_filename}"
                if chunk.section_type:
                    citation += f", {chunk.section_type} section"
                if chunk.page_number:
                    citation += f", page {chunk.page_number}"
                citations.add(citation)
            
            # Determine response format based on analysis type
            format_instruction = ""
            if reasoner_analysis.analysis_type == "comparison":
                format_instruction = "Present the answer as a comparison table using 📊 format."
            elif reasoner_analysis.analysis_type == "trend_analysis":
                format_instruction = "Present the answer as trend analysis using 📈 format."
            elif reasoner_analysis.analysis_type == "risk_detection":
                format_instruction = "Present the answer highlighting risks using 🚨 format."
            else:
                format_instruction = "Present the answer as a structured summary using 📝 format."
            
            response_prompt = f"""
            Based on the following AGR content, answer the user's query:
            
            USER QUERY: {query}
            
            ANALYSIS CONTEXT: {reasoner_analysis.context_summary}
            CONFIDENCE LEVEL: {reasoner_analysis.confidence_level}
            
            AGR CONTENT:
            {chr(10).join(context_chunks)}
            
            INSTRUCTIONS:
            {format_instruction}
            
            Requirements:
            1. Answer ONLY based on the provided AGR content
            2. Include specific numbers and facts from the content
            3. Cite sources with document names and page numbers
            4. If information is missing, state it clearly
            5. Provide a confidence assessment at the end
            
            Format your response professionally with clear headings and structure.
            """
            
            user_message = UserMessage(text=response_prompt)
            response = await self.llm_chat.send_message(user_message)
            
            # Calculate confidence score based on various factors
            confidence_score = self._calculate_confidence_score(
                retrieved_chunks, reasoner_analysis.confidence_level
            )
            
            # Determine response format
            response_format = self._determine_response_format(response, reasoner_analysis.analysis_type)
            
            return {
                "formatted_answer": response,
                "confidence_score": confidence_score,
                "citations": list(citations),
                "response_format": response_format
            }
            
        except Exception as e:
            logger.error(f"Error in responder agent: {e}")
            return {
                "formatted_answer": "I encountered an error while processing your request. Please try again.",
                "confidence_score": 0.1,
                "citations": [],
                "response_format": "text"
            }
    
    def _calculate_confidence_score(self, chunks: List[RetrievedChunk], confidence_level: str) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not chunks:
            return 0.1
        
        # Base score from confidence level
        base_scores = {"high": 0.8, "medium": 0.6, "low": 0.4}
        base_score = base_scores.get(confidence_level, 0.5)
        
        # Adjust based on relevance scores
        avg_relevance = sum(chunk.relevance_score for chunk in chunks) / len(chunks)
        relevance_boost = min(0.2, avg_relevance * 0.2)
        
        # Adjust based on number of chunks
        chunk_boost = min(0.1, len(chunks) * 0.02)
        
        return min(0.95, base_score + relevance_boost + chunk_boost)
    
    def _determine_response_format(self, response: str, analysis_type: str) -> str:
        """Determine response format based on content and analysis type"""
        response_lower = response.lower()
        
        if "📊" in response or "|" in response:  # Table indicators
            return "table"
        elif "📈" in response or "trend" in response_lower:
            return "chart"
        elif "🚨" in response or "risk" in response_lower:
            return "red_flag"
        elif analysis_type == "summary":
            return "summary"
        else:
            return "text"


class AGRPipeline:
    """Main AGR Pipeline orchestrating all agents"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.retriever = RetrieverAgent(self.document_processor)
        self.reasoner = ReasonerAgent()
        self.responder = ResponderAgent()
    
    async def process_query(self, query: AGRQuery) -> AGRResponse:
        """Process AGR query through the complete pipeline"""
        try:
            logger.info(f"Processing query: {query.query}")
            
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = await self.retriever.retrieve(query)
            
            # Step 2: Analyze context and determine approach
            reasoner_analysis = await self.reasoner.analyze(query.query, retrieved_chunks)
            
            # Step 3: Generate structured response
            response_data = await self.responder.generate_response(
                query.query, retrieved_chunks, reasoner_analysis
            )
            
            # Create final response
            agr_response = AGRResponse(
                query=query.query,
                retrieved_chunks=retrieved_chunks,
                reasoner_analysis=reasoner_analysis,
                formatted_answer=response_data["formatted_answer"],
                confidence_score=response_data["confidence_score"],
                citations=response_data["citations"],
                response_format=response_data["response_format"],
                session_id=query.session_id
            )
            
            logger.info(f"Generated response with confidence: {response_data['confidence_score']:.2f}")
            return agr_response
            
        except Exception as e:
            logger.error(f"Error in AGR pipeline: {e}")
            # Return error response
            return AGRResponse(
                query=query.query,
                retrieved_chunks=[],
                reasoner_analysis=ReasonerAnalysis(
                    analysis_type="error",
                    requires_reformulation=True,
                    context_summary="System error occurred",
                    confidence_level="low"
                ),
                formatted_answer="I encountered an error while processing your request. Please try again later.",
                confidence_score=0.1,
                citations=[],
                response_format="text",
                session_id=query.session_id
            )