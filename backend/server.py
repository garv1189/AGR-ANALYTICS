from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime

# Import AGR models and agents
from models import (
    Document, DocumentChunk, AGRQuery, AGRResponse, 
    ChatSession, ChatMessage, StatusCheck, StatusCheckCreate
)
from document_processor import DocumentProcessor
from agents import AGRPipeline

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize AGR Pipeline
agr_pipeline = AGRPipeline()

# Create the main app without a prefix
app = FastAPI(title="AGR Agentic RAG API", description="Advanced agentic RAG pipeline for Annual General Reports")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "AGR Agentic RAG Pipeline API", "status": "active"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# AGR Document Management Endpoints
@api_router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    company_name: str = Form(...),
    year: int = Form(...)
):
    """Upload and process AGR document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        # Create document record
        file_type = 'pdf' if file.filename.lower().endswith('.pdf') else 'docx'
        document = Document(
            filename=file.filename,
            company_name=company_name,
            year=year,
            file_type=file_type,
            file_path=f"/app/backend/uploads/{file.filename}"
        )
        
        # Read file content
        file_content = await file.read()
        
        # Process document and create chunks
        chunks = agr_pipeline.document_processor.process_document(document, file_content)
        
        # Add chunks to vector index
        agr_pipeline.document_processor.add_chunks_to_index(chunks, document)
        
        # Save document and chunks to database
        document.processed = True
        document.total_chunks = len(chunks)
        
        await db.documents.insert_one(document.dict())
        
        # Save chunks
        chunk_dicts = [chunk.dict() for chunk in chunks]
        if chunk_dicts:
            await db.document_chunks.insert_many(chunk_dicts)
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": document.id,
            "total_chunks": len(chunks),
            "filename": document.filename
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/documents")
async def get_documents():
    """Get list of uploaded documents"""
    try:
        documents = await db.documents.find().to_list(1000)
        return [Document(**doc) for doc in documents]
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    try:
        # Delete document
        delete_result = await db.documents.delete_one({"id": document_id})
        if delete_result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete associated chunks
        await db.document_chunks.delete_many({"document_id": document_id})
        
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AGR Query Endpoints
@api_router.post("/query", response_model=AGRResponse)
async def query_agr(query_data: AGRQuery):
    """Process AGR query through the agentic pipeline"""
    try:
        response = await agr_pipeline.process_query(query_data)
        
        # Save query and response to database
        chat_message = ChatMessage(
            session_id=query_data.session_id,
            message_type="user",
            content=query_data.query
        )
        await db.chat_messages.insert_one(chat_message.dict())
        
        response_message = ChatMessage(
            session_id=query_data.session_id,
            message_type="assistant",
            content=response.formatted_answer,
            agr_response=response
        )
        await db.chat_messages.insert_one(response_message.dict())
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat Session Endpoints
@api_router.post("/sessions")
async def create_chat_session():
    """Create a new chat session"""
    try:
        session = ChatSession(session_id=str(uuid.uuid4()))
        await db.chat_sessions.insert_one(session.dict())
        return {"session_id": session.session_id}
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/sessions/{session_id}/messages")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(1000)
        
        return [ChatMessage(**msg) for msg in messages]
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/sessions")
async def get_chat_sessions():
    """Get all chat sessions"""
    try:
        sessions = await db.chat_sessions.find().sort("created_at", -1).to_list(100)
        return [ChatSession(**session) for session in sessions]
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Info Endpoints
@api_router.get("/system/info")
async def get_system_info():
    """Get system information"""
    try:
        doc_count = await db.documents.count_documents({})
        chunk_count = await db.document_chunks.count_documents({})
        session_count = await db.chat_sessions.count_documents({})
        
        return {
            "documents_uploaded": doc_count,
            "total_chunks": chunk_count,
            "vector_index_size": agr_pipeline.document_processor.index.ntotal if agr_pipeline.document_processor.index else 0,
            "active_sessions": session_count,
            "system_status": "operational"
        }
    except Exception as e:
        logger.error(f"Error fetching system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create uploads directory
uploads_dir = "/app/backend/uploads"
os.makedirs(uploads_dir, exist_ok=True)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
