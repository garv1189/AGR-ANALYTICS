import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import PyPDF2
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from models import Document, DocumentChunk
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.vector_store_path = "/app/backend/vector_store"
        self.embeddings_path = "/app/backend/embeddings"
        
        # Create directories if they don't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.embeddings_path, exist_ok=True)
        
        # Initialize or load FAISS index
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = None
        self.chunk_metadata = {}
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        metadata_path = os.path.join(self.vector_store_path, "chunk_metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new one.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.chunk_metadata = {}
        logger.info("Created new FAISS index")

    def _save_index(self):
        """Save FAISS index and metadata"""
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        metadata_path = os.path.join(self.vector_store_path, "chunk_metadata.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)

    def extract_text_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF file with page numbers"""
        text_pages = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_pages.append((text, page_num))
            logger.info(f"Extracted text from {len(text_pages)} pages in PDF")
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise
        return text_pages

    def extract_text_from_docx(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_path)
            text_pages = []
            current_text = ""
            page_num = 1
            
            for paragraph in doc.paragraphs:
                current_text += paragraph.text + "\n"
                # Simple page estimation (every ~1000 chars = new page)
                if len(current_text) > 1000:
                    text_pages.append((current_text, page_num))
                    current_text = ""
                    page_num += 1
            
            if current_text.strip():
                text_pages.append((current_text, page_num))
                
            logger.info(f"Extracted text from DOCX with {len(text_pages)} sections")
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise
        return text_pages

    def detect_section_type(self, text: str) -> Optional[str]:
        """Detect AGR section type based on content"""
        text_lower = text.lower()
        
        # Financial section indicators
        financial_keywords = ['revenue', 'profit', 'earnings', 'financial statements', 'balance sheet', 
                            'income statement', 'cash flow', 'assets', 'liabilities', 'equity']
        
        # Risk section indicators
        risk_keywords = ['risk', 'risks', 'risk factors', 'uncertainties', 'challenges', 'threats']
        
        # ESG section indicators
        esg_keywords = ['environmental', 'sustainability', 'governance', 'social responsibility', 
                       'esg', 'carbon', 'emissions', 'diversity', 'inclusion']
        
        # MD&A section indicators
        mda_keywords = ['management discussion', 'md&a', 'analysis', 'outlook', 'forward-looking']
        
        keyword_counts = {
            'Financials': sum(1 for kw in financial_keywords if kw in text_lower),
            'Risks': sum(1 for kw in risk_keywords if kw in text_lower),
            'ESG': sum(1 for kw in esg_keywords if kw in text_lower),
            'MD&A': sum(1 for kw in mda_keywords if kw in text_lower)
        }
        
        # Return section with highest keyword count (if above threshold)
        max_section = max(keyword_counts.items(), key=lambda x: x[1])
        return max_section[0] if max_section[1] > 2 else None

    def chunk_text(self, text: str, page_number: int) -> List[Tuple[str, int]]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to end at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + self.chunk_size * 0.7:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, page_number))
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
                
        return chunks

    def process_document(self, document: Document, file_content: bytes) -> List[DocumentChunk]:
        """Process uploaded document and create chunks"""
        # Save file temporarily
        temp_path = f"/tmp/{document.filename}"
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        
        try:
            # Extract text based on file type
            if document.file_type == 'pdf':
                text_pages = self.extract_text_from_pdf(temp_path)
            elif document.file_type == 'docx':
                text_pages = self.extract_text_from_docx(temp_path)
            else:
                raise ValueError(f"Unsupported file type: {document.file_type}")
            
            # Create chunks
            chunks = []
            chunk_index = 0
            
            for text, page_number in text_pages:
                # Detect section type
                section_type = self.detect_section_type(text)
                
                # Create chunks from page text
                page_chunks = self.chunk_text(text, page_number)
                
                for chunk_text, page_num in page_chunks:
                    if len(chunk_text.strip()) < 50:  # Skip very short chunks
                        continue
                        
                    chunk = DocumentChunk(
                        document_id=document.id,
                        chunk_index=chunk_index,
                        content=chunk_text,
                        section_type=section_type,
                        page_number=page_num
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            logger.info(f"Created {len(chunks)} chunks for document {document.filename}")
            return chunks
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def add_chunks_to_index(self, chunks: List[DocumentChunk], document: Document):
        """Add document chunks to FAISS vector index"""
        if not chunks:
            return
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            vector_id = self.index.ntotal - len(chunks) + i
            self.chunk_metadata[vector_id] = {
                'chunk_id': chunk.id,
                'document_id': chunk.document_id,
                'document_filename': document.filename,
                'company_name': document.company_name,
                'year': document.year,
                'section_type': chunk.section_type,
                'page_number': chunk.page_number,
                'content': chunk.content
            }
        
        # Save updated index
        self._save_index()
        logger.info(f"Added {len(chunks)} chunks to vector index")

    def search_chunks(self, query: str, top_k: int = 5, 
                     company_filter: Optional[str] = None,
                     year_filter: Optional[List[int]] = None,
                     section_filter: Optional[List[str]] = None) -> List[Dict]:
        """Search for relevant chunks using vector similarity"""
        if self.index.ntotal == 0:
            logger.warning("No documents indexed yet")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Search in FAISS index (get more results for filtering)
        search_k = min(top_k * 3, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Filter and rank results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # No more results
                break
                
            metadata = self.chunk_metadata.get(idx, {})
            if not metadata:
                continue
            
            # Apply filters
            if company_filter and metadata.get('company_name', '').lower() != company_filter.lower():
                continue
            if year_filter and metadata.get('year') not in year_filter:
                continue
            if section_filter and metadata.get('section_type') not in section_filter:
                continue
            
            result = {
                'chunk_id': metadata['chunk_id'],
                'content': metadata['content'],
                'document_filename': metadata['document_filename'],
                'company_name': metadata['company_name'],
                'year': metadata['year'],
                'section_type': metadata.get('section_type'),
                'page_number': metadata.get('page_number'),
                'relevance_score': float(similarity)
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        logger.info(f"Found {len(results)} relevant chunks for query")
        return results