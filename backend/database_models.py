"""
Database models for SEC Forensic Auditor
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Company(Base):
    """Company information"""
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True, index=True)
    cik = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    ticker = Column(String(10), index=True)
    sic = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    filings = relationship("Filing", back_populates="company")
    reports = relationship("ForensicReportDB", back_populates="company")


class Filing(Base):
    """SEC Filing information"""
    __tablename__ = 'filings'
    
    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=False)
    accession_number = Column(String(20), unique=True, index=True, nullable=False)
    form_type = Column(String(10), nullable=False)
    filing_date = Column(DateTime, nullable=False)
    report_date = Column(DateTime)
    document_url = Column(Text)
    
    # Processing status
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    processed_at = Column(DateTime)
    
    # Stored data
    raw_content = Column(Text)
    processed_data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="filings")
    analyses = relationship("AnalysisResult", back_populates="filing")
    reports = relationship("ForensicReportDB", back_populates="filing")


class AnalysisResult(Base):
    """Analysis results storage"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True, index=True)
    filing_id = Column(Integer, ForeignKey('filings.id'), nullable=False)
    analysis_type = Column(String(20), nullable=False)  # numerical, textual, combined
    
    # Scores
    risk_score = Column(Float)
    confidence_score = Column(Float)
    
    # Results
    analysis_data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    filing = relationship("Filing", back_populates="analyses")


class ForensicReportDB(Base):
    """Forensic audit reports"""
    __tablename__ = 'forensic_reports'
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(50), unique=True, index=True, nullable=False)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=False)
    filing_id = Column(Integer, ForeignKey('filings.id'), nullable=False)
    
    # Risk classification
    risk_level = Column(String(10), nullable=False)  # L1_LOW, L2_MEDIUM, L3_HIGH
    confidence_score = Column(Float, nullable=False)
    
    # Scores
    numerical_risk_score = Column(Float)
    textual_risk_score = Column(Float)
    combined_risk_score = Column(Float)
    
    # Report data
    executive_summary = Column(JSON)
    detailed_findings = Column(JSON)
    recommendations = Column(JSON)
    full_report = Column(JSON)
    
    # Metadata
    generated_at = Column(DateTime, default=datetime.utcnow)
    generated_by = Column(String(50), default='system')
    
    # Relationships
    company = relationship("Company", back_populates="reports")
    filing = relationship("Filing", back_populates="reports")


class User(Base):
    """User accounts"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)


class AuditLog(Base):
    """Audit logging"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(Integer)
    details = Column(JSON)
    ip_address = Column(String(45))
    timestamp = Column(DateTime, default=datetime.utcnow)
