"""
Configuration management for SEC Forensic Auditor
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "SEC Forensic Auditor"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database - PostgreSQL
    POSTGRES_USER: str = "forensic_user"
    POSTGRES_PASSWORD: str = "forensic_pass"
    POSTGRES_DB: str = "forensic_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "forensic_documents"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # SEC EDGAR
    SEC_USER_AGENT: str = "Forensic Auditor forensicauditor@example.com"
    SEC_API_RATE_LIMIT: int = 10  # requests per second
    
    # ML Models
    MODEL_CACHE_DIR: str = "./models_cache"
    FINBERT_MODEL: str = "ProsusAI/finbert"
    LONGFORMER_MODEL: str = "allenai/longformer-base-4096"
    
    # LLM
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4000
    
    # XGBoost Parameters
    XGBOOST_N_ESTIMATORS: int = 100
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.1
    
    # SHAP
    SHAP_MAX_SAMPLES: int = 100
    
    # Risk Thresholds
    RISK_L1_THRESHOLD: float = 0.3  # Low risk
    RISK_L2_THRESHOLD: float = 0.6  # Medium risk
    RISK_L3_THRESHOLD: float = 0.85  # High risk
    
    # Authentication
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "forensic_auditor.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Create necessary directories
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
