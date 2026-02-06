"""
FastAPI Application - SEC Forensic Auditor API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging
from datetime import datetime
import asyncio

from config import settings
from sec_extractor import edgar_extractor
from data_preprocessor import data_preprocessor
from agent_numerical import numerical_analyst
from agent_textual import textual_investigator
from agent_chief import chief_auditor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI/ML-powered forensic auditing system for SEC 10-K filings"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class CompanySearchRequest(BaseModel):
    company_name: str = Field(..., description="Company name to search")


class CompanySearchResponse(BaseModel):
    companies: List[Dict]


class AnalysisRequest(BaseModel):
    cik: str = Field(..., description="Company CIK number")
    filing_count: int = Field(default=1, ge=1, le=5, description="Number of recent filings to analyze")
    include_historical: bool = Field(default=False, description="Include historical data for trend analysis")


class AnalysisStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[int] = None
    message: Optional[str] = None


class ForensicReportResponse(BaseModel):
    report_id: str
    company_name: str
    cik: str
    filing_date: str
    risk_level: str
    confidence_score: float
    numerical_risk_score: float
    textual_risk_score: float
    executive_summary: Dict
    detailed_findings: Optional[Dict] = None
    recommendations: List[str]


# ============================================================================
# In-Memory Task Storage (Use Redis/Celery in production)
# ============================================================================

analysis_tasks = {}


class AnalysisTask:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "pending"
        self.progress = 0
        self.message = ""
        self.result = None
        self.error = None
        self.created_at = datetime.now()


# ============================================================================
# Background Task Functions
# ============================================================================

async def run_forensic_analysis(task_id: str, cik: str, filing_count: int, include_historical: bool):
    """Background task to run complete forensic analysis"""
    task = analysis_tasks[task_id]
    
    try:
        # Step 1: Extract 10-K filings
        task.status = "extracting"
        task.progress = 10
        task.message = "Extracting 10-K filings from SEC EDGAR..."
        logger.info(f"Task {task_id}: Extracting filings for CIK {cik}")
        
        filings = edgar_extractor.get_10k_filings(cik, count=filing_count)
        
        if not filings:
            raise ValueError(f"No 10-K filings found for CIK {cik}")
        
        # Process first filing (most recent)
        target_filing = filings[0]
        
        # Step 2: Download document
        task.progress = 20
        task.message = "Downloading 10-K document..."
        logger.info(f"Task {task_id}: Downloading document {target_filing.accession_number}")
        
        content, metadata = edgar_extractor.download_10k_document(target_filing)
        sections = edgar_extractor.extract_sections(content)
        
        # Step 3: Preprocess data
        task.progress = 30
        task.message = "Preprocessing and structuring data..."
        logger.info(f"Task {task_id}: Preprocessing data")
        
        processed_data = data_preprocessor.process_10k_filing(content, metadata, sections)
        
        # Step 4: Numerical analysis
        task.status = "analyzing_numerical"
        task.progress = 45
        task.message = "Running numerical forensic analysis..."
        logger.info(f"Task {task_id}: Running numerical analysis")
        
        # Get historical data if requested
        historical_data = None
        if include_historical and len(filings) > 1:
            historical_data = []
            for hist_filing in filings[1:]:
                try:
                    hist_content, hist_metadata = edgar_extractor.download_10k_document(hist_filing)
                    hist_sections = edgar_extractor.extract_sections(hist_content)
                    hist_processed = data_preprocessor.process_10k_filing(hist_content, hist_metadata, hist_sections)
                    historical_data.append(hist_processed)
                except Exception as e:
                    logger.warning(f"Failed to process historical filing: {e}")
                    continue
        
        numerical_result = numerical_analyst.analyze(
            processed_data['numerical_data'],
            historical_data
        )
        
        # Step 5: Textual analysis
        task.status = "analyzing_textual"
        task.progress = 65
        task.message = "Running textual forensic analysis..."
        logger.info(f"Task {task_id}: Running textual analysis")
        
        textual_result = textual_investigator.analyze(processed_data['textual_data'])
        
        # Step 6: Synthesize report
        task.status = "synthesizing"
        task.progress = 85
        task.message = "Synthesizing final forensic report..."
        logger.info(f"Task {task_id}: Synthesizing report")
        
        forensic_report = chief_auditor.synthesize_report(numerical_result, textual_result)
        
        # Step 7: Complete
        task.status = "completed"
        task.progress = 100
        task.message = "Analysis complete"
        task.result = forensic_report
        
        logger.info(f"Task {task_id}: Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Task {task_id}: Error during analysis - {str(e)}", exc_info=True)
        task.status = "failed"
        task.error = str(e)
        task.message = f"Analysis failed: {str(e)}"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "endpoints": {
            "search_company": "/api/v1/search/company",
            "start_analysis": "/api/v1/analysis/start",
            "analysis_status": "/api/v1/analysis/status/{task_id}",
            "get_report": "/api/v1/reports/{report_id}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post(f"{settings.API_V1_PREFIX}/search/company", response_model=CompanySearchResponse)
async def search_company(request: CompanySearchRequest):
    """
    Search for a company by name to get CIK
    """
    try:
        companies = edgar_extractor.search_company_by_name(request.company_name)
        
        return CompanySearchResponse(
            companies=[
                {
                    "cik": c.cik,
                    "name": c.name,
                    "ticker": c.ticker
                }
                for c in companies
            ]
        )
    except Exception as e:
        logger.error(f"Error searching company: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.API_V1_PREFIX}/analysis/start", response_model=AnalysisStatus)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Start forensic analysis for a company's 10-K filings
    """
    try:
        # Generate task ID
        task_id = f"task-{request.cik}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create task
        task = AnalysisTask(task_id)
        analysis_tasks[task_id] = task
        
        # Start background analysis
        background_tasks.add_task(
            run_forensic_analysis,
            task_id,
            request.cik,
            request.filing_count,
            request.include_historical
        )
        
        logger.info(f"Started analysis task: {task_id}")
        
        return AnalysisStatus(
            task_id=task_id,
            status="pending",
            progress=0,
            message="Analysis task queued"
        )
        
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{settings.API_V1_PREFIX}/analysis/status/{{task_id}}", response_model=AnalysisStatus)
async def get_analysis_status(task_id: str):
    """
    Get the status of an analysis task
    """
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    
    return AnalysisStatus(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        message=task.message
    )


@app.get(f"{settings.API_V1_PREFIX}/analysis/result/{{task_id}}", response_model=ForensicReportResponse)
async def get_analysis_result(task_id: str):
    """
    Get the result of a completed analysis task
    """
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    
    if task.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Current status: {task.status}"
        )
    
    if not task.result:
        raise HTTPException(status_code=500, detail="No result available")
    
    report = task.result
    
    return ForensicReportResponse(
        report_id=report.report_id,
        company_name=report.company_info.get('name', 'Unknown'),
        cik=report.company_info.get('cik', ''),
        filing_date=report.filing_info.get('filing_date', ''),
        risk_level=report.risk_classification.risk_level,
        confidence_score=report.risk_classification.confidence_score,
        numerical_risk_score=report.executive_summary.get('numerical_risk_score', 0),
        textual_risk_score=report.executive_summary.get('textual_risk_score', 0),
        executive_summary=report.executive_summary,
        detailed_findings=report.detailed_findings,
        recommendations=report.actionable_recommendations
    )


@app.get(f"{settings.API_V1_PREFIX}/filings/{{cik}}")
async def get_company_filings(cik: str, count: int = 10):
    """
    Get 10-K filings for a company
    """
    try:
        filings = edgar_extractor.get_10k_filings(cik, count=min(count, 20))
        
        return {
            "cik": cik,
            "filings": [
                {
                    "accession_number": f.accession_number,
                    "filing_date": f.filing_date,
                    "report_date": f.report_date,
                    "form_type": f.form_type
                }
                for f in filings
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching filings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"{settings.API_V1_PREFIX}/reports")
async def list_reports(skip: int = 0, limit: int = 50):
    """
    List all forensic reports (from in-memory storage)
    """
    completed_tasks = [
        task for task in analysis_tasks.values()
        if task.status == "completed" and task.result
    ]
    
    reports = []
    for task in completed_tasks[skip:skip+limit]:
        report = task.result
        reports.append({
            "report_id": report.report_id,
            "company_name": report.company_info.get('name'),
            "cik": report.company_info.get('cik'),
            "risk_level": report.risk_classification.risk_level,
            "confidence_score": report.risk_classification.confidence_score,
            "generated_at": report.generation_date
        })
    
    return {
        "total": len(completed_tasks),
        "skip": skip,
        "limit": limit,
        "reports": reports
    }


@app.delete(f"{settings.API_V1_PREFIX}/tasks/{{task_id}}")
async def delete_task(task_id: str):
    """
    Delete a task from memory
    """
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del analysis_tasks[task_id]
    return {"message": "Task deleted successfully"}


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"API available at: http://{settings.HOST}:{settings.PORT}{settings.API_V1_PREFIX}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
