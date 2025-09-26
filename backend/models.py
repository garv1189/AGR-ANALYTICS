from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# Document Models
class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    company_name: str
    year: int
    file_type: str  # pdf, docx
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    total_chunks: Optional[int] = None
    file_path: str

class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_index: int
    content: str
    section_type: Optional[str] = None  # Financials, Risks, ESG, MD&A
    page_number: Optional[int] = None
    embeddings_stored: bool = False

# AGR Query and Response Models
class AGRQuery(BaseModel):
    query: str
    company_filter: Optional[str] = None
    year_filter: Optional[List[int]] = None
    section_filter: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    document_filename: str
    company_name: str
    year: int
    section_type: Optional[str] = None
    page_number: Optional[int] = None
    relevance_score: float

class ReasonerAnalysis(BaseModel):
    analysis_type: str  # comparison, summary, trend_analysis, risk_detection
    requires_reformulation: bool
    suggested_reformulation: Optional[str] = None
    context_summary: str
    confidence_level: str  # high, medium, low

class AGRResponse(BaseModel):
    query: str
    retrieved_chunks: List[RetrievedChunk]
    reasoner_analysis: ReasonerAnalysis
    formatted_answer: str
    confidence_score: float
    citations: List[str]
    response_format: str  # text, table, chart, summary, red_flag
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Chat Session Models
class ChatSession(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message_type: str  # user, assistant
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agr_response: Optional[AGRResponse] = None