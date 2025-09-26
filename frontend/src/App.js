import React, { useState, useEffect } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import axios from "axios";
import DocumentUpload from "./components/DocumentUpload";
import ChatInterface from "./components/ChatInterface";
import SystemInfo from "./components/SystemInfo";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Navigation component
const Navigation = () => {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Chat Interface', icon: '💬' },
    { path: '/upload', label: 'Upload Documents', icon: '📄' },
    { path: '/system', label: 'System Info', icon: '📊' },
  ];

  return (
    <nav className="bg-white shadow-lg mb-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="text-2xl font-bold text-blue-600">🤖</div>
              <span className="text-xl font-bold text-gray-800">AGR RAG Pipeline</span>
            </Link>
          </div>
          
          <div className="flex items-center space-x-8">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  location.pathname === item.path
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <span className="text-base">{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

// Main Chat Page
const ChatPage = () => {
  const [sessionId, setSessionId] = useState(null);

  useEffect(() => {
    createNewSession();
  }, []);

  const createNewSession = async () => {
    try {
      const response = await axios.post(`${API}/sessions`);
      setSessionId(response.data.session_id);
    } catch (error) {
      console.error('Error creating session:', error);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Annual General Report Analysis
        </h1>
        <p className="text-lg text-gray-600">
          Ask questions about uploaded AGR documents and get intelligent, structured responses
        </p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Chat Interface */}
        <div className="lg:col-span-4">
          {sessionId && (
            <ChatInterface 
              sessionId={sessionId} 
              onNewSession={createNewSession}
            />
          )}
        </div>
      </div>
    </div>
  );
};

// Upload Page
const UploadPage = () => {
  const [uploadSuccess, setUploadSuccess] = useState(null);

  const handleUploadSuccess = (data) => {
    setUploadSuccess(data);
    // Clear success message after 5 seconds
    setTimeout(() => setUploadSuccess(null), 5000);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Upload AGR Documents
        </h1>
        <p className="text-lg text-gray-600">
          Upload Annual General Reports (PDF or DOCX) to enable AI-powered analysis
        </p>
      </div>

      {uploadSuccess && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center">
            <svg className="w-5 h-5 text-green-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <span className="text-green-800 font-medium">
              Successfully processed {uploadSuccess.filename} ({uploadSuccess.total_chunks} chunks created)
            </span>
          </div>
        </div>
      )}
      
      <DocumentUpload onUploadSuccess={handleUploadSuccess} />
      
      <div className="mt-8 bg-blue-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">
          📋 Document Processing Features
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
          <div className="space-y-2">
            <p><strong>✅ Automatic Section Detection:</strong> Financials, Risks, ESG, MD&A</p>
            <p><strong>✅ Smart Chunking:</strong> Intelligent text segmentation</p>
            <p><strong>✅ Vector Embeddings:</strong> Semantic search capabilities</p>
          </div>
          <div className="space-y-2">
            <p><strong>✅ Multi-Year Support:</strong> Compare across reporting periods</p>
            <p><strong>✅ Multi-Format:</strong> PDF and DOCX support</p>
            <p><strong>✅ Company Tagging:</strong> Organize by company</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// System Info Page
const SystemPage = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          System Overview
        </h1>
        <p className="text-lg text-gray-600">
          Monitor system status and manage uploaded documents
        </p>
      </div>
      
      <SystemInfo />
    </div>
  );
};

function App() {
  const [systemStatus, setSystemStatus] = useState(null);

  useEffect(() => {
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await axios.get(`${API}/`);
      setSystemStatus('connected');
    } catch (error) {
      console.error('System check failed:', error);
      setSystemStatus('error');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <BrowserRouter>
        <Navigation />
        
        {/* System Status Indicator */}
        {systemStatus && (
          <div className={`text-center py-2 text-sm ${
            systemStatus === 'connected' 
              ? 'bg-green-100 text-green-800' 
              : 'bg-red-100 text-red-800'
          }`}>
            {systemStatus === 'connected' 
              ? '🟢 System Online - AGR Pipeline Ready' 
              : '🔴 System Error - Please check backend connection'
            }
          </div>
        )}
        
        <main className="pb-8">
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/system" element={<SystemPage />} />
          </Routes>
        </main>
        
        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 py-6">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-gray-500 text-sm">
            <p>AGR Agentic RAG Pipeline - Advanced AI Analysis of Annual General Reports</p>
            <p className="mt-1">
              Powered by GPT-4, Sentence Transformers, and FAISS Vector Search
            </p>
          </div>
        </footer>
      </BrowserRouter>
    </div>
  );
}

export default App;