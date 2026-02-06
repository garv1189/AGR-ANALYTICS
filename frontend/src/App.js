import React, { useState, useEffect } from 'react';
import { 
  Search, 
  TrendingUp, 
  AlertTriangle, 
  FileText, 
  BarChart3, 
  Brain,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  RefreshCw
} from 'lucide-react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

function App() {
  const [companyName, setCompanyName] = useState('');
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [analysisTask, setAnalysisTask] = useState(null);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Search for companies
  const searchCompanies = async () => {
    if (!companyName.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/search/company`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company_name: companyName })
      });
      
      if (!response.ok) throw new Error('Search failed');
      
      const data = await response.json();
      setCompanies(data.companies);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Start forensic analysis
  const startAnalysis = async (cik) => {
    setLoading(true);
    setError(null);
    setReport(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/analysis/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          cik,
          filing_count: 1,
          include_historical: true
        })
      });
      
      if (!response.ok) throw new Error('Failed to start analysis');
      
      const data = await response.json();
      setAnalysisTask(data);
      pollAnalysisStatus(data.task_id);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Poll analysis status
  const pollAnalysisStatus = async (taskId) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/analysis/status/${taskId}`);
        
        if (!response.ok) throw new Error('Failed to get status');
        
        const data = await response.json();
        setAnalysisTask(data);
        
        if (data.status === 'completed') {
          clearInterval(interval);
          fetchAnalysisResult(taskId);
        } else if (data.status === 'failed') {
          clearInterval(interval);
          setError(data.message || 'Analysis failed');
          setLoading(false);
        }
      } catch (err) {
        clearInterval(interval);
        setError(err.message);
        setLoading(false);
      }
    }, 2000);
  };

  // Fetch analysis result
  const fetchAnalysisResult = async (taskId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/analysis/result/${taskId}`);
      
      if (!response.ok) throw new Error('Failed to get result');
      
      const data = await response.json();
      setReport(data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Risk level styling
  const getRiskLevelColor = (level) => {
    switch(level) {
      case 'L3_HIGH': return 'text-red-600 bg-red-50 border-red-200';
      case 'L2_MEDIUM': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'L1_LOW': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getRiskLevelIcon = (level) => {
    switch(level) {
      case 'L3_HIGH': return <AlertTriangle className="w-6 h-6" />;
      case 'L2_MEDIUM': return <AlertTriangle className="w-6 h-6" />;
      case 'L1_LOW': return <CheckCircle className="w-6 h-6" />;
      default: return <Clock className="w-6 h-6" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Brain className="w-10 h-10 text-blue-500" />
              <div>
                <h1 className="text-3xl font-bold text-white">SEC Forensic Auditor</h1>
                <p className="text-sm text-slate-400">AI-Powered Financial Forensics</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search Section */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl shadow-2xl p-6 mb-8 border border-slate-700">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <Search className="w-5 h-5 mr-2 text-blue-500" />
            Search Company
          </h2>
          
          <div className="flex gap-4">
            <input
              type="text"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && searchCompanies()}
              placeholder="Enter company name (e.g., Apple, Tesla, Microsoft)"
              className="flex-1 px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={searchCompanies}
              disabled={loading || !companyName.trim()}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* Company Results */}
          {companies.length > 0 && (
            <div className="mt-6 space-y-3">
              <h3 className="text-sm font-medium text-slate-400 uppercase">Select a company:</h3>
              {companies.map((company) => (
                <div
                  key={company.cik}
                  className="bg-slate-900 border border-slate-700 rounded-lg p-4 cursor-pointer hover:border-blue-500 transition-colors"
                  onClick={() => {
                    setSelectedCompany(company);
                    startAnalysis(company.cik);
                  }}
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <h4 className="text-lg font-semibold text-white">{company.name}</h4>
                      <p className="text-sm text-slate-400">
                        CIK: {company.cik} {company.ticker && `| Ticker: ${company.ticker}`}
                      </p>
                    </div>
                    <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors">
                      Analyze
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 mb-8 flex items-start">
            <XCircle className="w-5 h-5 text-red-400 mr-3 mt-0.5 flex-shrink-0" />
            <div className="text-red-200">{error}</div>
          </div>
        )}

        {/* Analysis Progress */}
        {analysisTask && analysisTask.status !== 'completed' && (
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl shadow-2xl p-6 mb-8 border border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white flex items-center">
                <RefreshCw className="w-5 h-5 mr-2 text-blue-500 animate-spin" />
                Analysis in Progress
              </h2>
              <span className="text-sm text-slate-400">{analysisTask.progress}%</span>
            </div>
            
            <div className="w-full bg-slate-900 rounded-full h-3 mb-4">
              <div
                className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${analysisTask.progress}%` }}
              />
            </div>
            
            <p className="text-slate-300">{analysisTask.message}</p>
          </div>
        )}

        {/* Forensic Report */}
        {report && (
          <div className="space-y-6">
            {/* Executive Summary */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl shadow-2xl p-6 border border-slate-700">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <FileText className="w-6 h-6 mr-2 text-blue-500" />
                Executive Summary
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                {/* Risk Level Card */}
                <div className={`border rounded-lg p-4 ${getRiskLevelColor(report.risk_level)}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium opacity-75">Risk Level</span>
                    {getRiskLevelIcon(report.risk_level)}
                  </div>
                  <div className="text-2xl font-bold">
                    {report.risk_level.replace('_', ' ')}
                  </div>
                  <div className="text-xs mt-1 opacity-75">
                    Confidence: {(report.confidence_score * 100).toFixed(0)}%
                  </div>
                </div>

                {/* Numerical Risk Score */}
                <div className="border border-slate-700 bg-slate-900 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-slate-400">Numerical Risk</span>
                    <BarChart3 className="w-5 h-5 text-purple-500" />
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {(report.numerical_risk_score * 100).toFixed(1)}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">Out of 100</div>
                </div>

                {/* Textual Risk Score */}
                <div className="border border-slate-700 bg-slate-900 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-slate-400">Textual Risk</span>
                    <FileText className="w-5 h-5 text-orange-500" />
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {(report.textual_risk_score * 100).toFixed(1)}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">Out of 100</div>
                </div>

                {/* Total Flags */}
                <div className="border border-slate-700 bg-slate-900 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-slate-400">Total Flags</span>
                    <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  </div>
                  <div className="text-2xl font-bold text-white">
                    {report.executive_summary.total_anomalies + report.executive_summary.total_textual_flags}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {report.executive_summary.high_severity_anomalies + report.executive_summary.high_severity_flags} high severity
                  </div>
                </div>
              </div>

              {/* Company Info */}
              <div className="border-t border-slate-700 pt-4">
                <h3 className="text-lg font-semibold text-white mb-2">{report.company_name}</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-slate-400">CIK:</span>
                    <span className="text-white ml-2">{report.cik}</span>
                  </div>
                  <div>
                    <span className="text-slate-400">Filing Date:</span>
                    <span className="text-white ml-2">{report.filing_date}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Key Concerns */}
            {report.executive_summary.key_concerns && report.executive_summary.key_concerns.length > 0 && (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl shadow-2xl p-6 border border-slate-700">
                <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                  <AlertTriangle className="w-5 h-5 mr-2 text-yellow-500" />
                  Key Concerns
                </h3>
                <ul className="space-y-2">
                  {report.executive_summary.key_concerns.map((concern, index) => (
                    <li key={index} className="flex items-start">
                      <span className="text-yellow-500 mr-2 mt-1">•</span>
                      <span className="text-slate-300">{concern}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Recommendations */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl shadow-2xl p-6 border border-slate-700">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-green-500" />
                Actionable Recommendations
              </h3>
              <ol className="space-y-3">
                {report.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start">
                    <span className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold mr-3 mt-0.5 flex-shrink-0">
                      {index + 1}
                    </span>
                    <span className="text-slate-300">{rec}</span>
                  </li>
                ))}
              </ol>
            </div>

            {/* Detailed Findings (Collapsible) */}
            {report.detailed_findings && (
              <details className="bg-slate-800/50 backdrop-blur-sm rounded-xl shadow-2xl border border-slate-700">
                <summary className="p-6 cursor-pointer text-xl font-semibold text-white hover:text-blue-400 transition-colors">
                  Detailed Findings & Analysis
                </summary>
                <div className="px-6 pb-6 space-y-4">
                  <pre className="bg-slate-900 p-4 rounded-lg overflow-auto text-xs text-slate-300 max-h-96">
                    {JSON.stringify(report.detailed_findings, null, 2)}
                  </pre>
                </div>
              </details>
            )}

            {/* Download Report */}
            <div className="flex justify-center">
              <button
                onClick={() => {
                  const dataStr = JSON.stringify(report, null, 2);
                  const dataBlob = new Blob([dataStr], {type: 'application/json'});
                  const url = URL.createObjectURL(dataBlob);
                  const link = document.createElement('a');
                  link.href = url;
                  link.download = `forensic-report-${report.report_id}.json`;
                  link.click();
                }}
                className="flex items-center px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
              >
                <Download className="w-5 h-5 mr-2" />
                Download Full Report (JSON)
              </button>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!loading && !analysisTask && !report && companies.length === 0 && (
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl shadow-2xl p-12 border border-slate-700 text-center">
            <Brain className="w-16 h-16 text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Start Your Forensic Analysis</h3>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Search for a company to begin comprehensive AI-powered forensic auditing of their SEC 10-K filings.
              Our system analyzes both numerical and textual data to identify risks, anomalies, and suspicious patterns.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 text-center text-slate-500 text-sm">
        <p>© 2024 SEC Forensic Auditor • AI/ML-Powered Financial Forensics Platform</p>
        <p className="mt-1">Using XGBoost, SHAP, FinBERT, and Longformer for Explainable Analysis</p>
      </footer>
    </div>
  );
}

export default App;
