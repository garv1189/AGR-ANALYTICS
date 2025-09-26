import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ChatInterface = ({ sessionId, onNewSession }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [filters, setFilters] = useState({
    company: '',
    year: '',
    section: '',
    topK: 5
  });
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (sessionId) {
      loadChatHistory();
    }
  }, [sessionId]);

  const loadChatHistory = async () => {
    try {
      const response = await axios.get(`${API}/sessions/${sessionId}/messages`);
      setMessages(response.data);
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      message_type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const queryData = {
        query: inputMessage,
        company_filter: filters.company || null,
        year_filter: filters.year ? [parseInt(filters.year)] : null,
        section_filter: filters.section ? [filters.section] : null,
        top_k: filters.topK,
        session_id: sessionId
      };

      const response = await axios.post(`${API}/query`, queryData);
      const agrResponse = response.data;

      const assistantMessage = {
        id: Date.now().toString() + '_assistant',
        message_type: 'assistant',
        content: agrResponse.formatted_answer,
        timestamp: new Date().toISOString(),
        agr_response: agrResponse
      };

      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now().toString() + '_error',
        message_type: 'assistant',
        content: `Error: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatResponseType = (type) => {
    const icons = {
      table: '📊',
      chart: '📈',
      summary: '📝',
      red_flag: '🚨',
      text: '💬'
    };
    return icons[type] || '💬';
  };

  const renderMessage = (message) => {
    const isUser = message.message_type === 'user';
    const agrResponse = message.agr_response;

    return (
      <div
        key={message.id}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
      >
        <div
          className={`max-w-3xl px-4 py-3 rounded-lg ${
            isUser
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-800 border border-gray-200'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="space-y-3">
              {/* Response Type Badge */}
              {agrResponse && (
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-lg">
                    {formatResponseType(agrResponse.response_format)}
                  </span>
                  <span className="text-gray-600 capitalize">
                    {agrResponse.response_format.replace('_', ' ')} Response
                  </span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    agrResponse.confidence_score > 0.7 ? 'bg-green-100 text-green-800' :
                    agrResponse.confidence_score > 0.5 ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {(agrResponse.confidence_score * 100).toFixed(0)}% confidence
                  </span>
                </div>
              )}

              {/* Main Content */}
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>

              {/* Citations */}
              {agrResponse && agrResponse.citations && agrResponse.citations.length > 0 && (
                <div className="border-t pt-2 mt-3">
                  <p className="text-xs font-medium text-gray-600 mb-1">Sources:</p>
                  <ul className="text-xs text-gray-500 space-y-1">
                    {agrResponse.citations.map((citation, index) => (
                      <li key={index} className="flex items-start">
                        <span className="mr-1">•</span>
                        <span>{citation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Retrieved Chunks Info */}
              {agrResponse && agrResponse.retrieved_chunks && agrResponse.retrieved_chunks.length > 0 && (
                <div className="text-xs text-gray-500 border-t pt-2">
                  Retrieved {agrResponse.retrieved_chunks.length} relevant chunks from AGR documents
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-md flex flex-col h-96 lg:h-[600px]">
      {/* Header */}
      <div className="border-b px-4 py-3">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">AGR Analysis Chat</h3>
          <button
            onClick={onNewSession}
            className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors"
          >
            New Session
          </button>
        </div>

        {/* Filters */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 mt-3">
          <input
            type="text"
            placeholder="Company filter"
            value={filters.company}
            onChange={(e) => setFilters({ ...filters, company: e.target.value })}
            className="text-sm px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <input
            type="number"
            placeholder="Year filter"
            value={filters.year}
            onChange={(e) => setFilters({ ...filters, year: e.target.value })}
            className="text-sm px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <select
            value={filters.section}
            onChange={(e) => setFilters({ ...filters, section: e.target.value })}
            className="text-sm px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="">All sections</option>
            <option value="Financials">Financials</option>
            <option value="Risks">Risks</option>
            <option value="ESG">ESG</option>
            <option value="MD&A">MD&A</option>
          </select>
          <select
            value={filters.topK}
            onChange={(e) => setFilters({ ...filters, topK: parseInt(e.target.value) })}
            className="text-sm px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value={3}>Top 3 results</option>
            <option value={5}>Top 5 results</option>
            <option value={10}>Top 10 results</option>
          </select>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-8">
            <svg className="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <p className="text-lg font-medium">Start analyzing AGR documents</p>
            <p className="text-sm mt-1">Ask questions about financial data, risks, ESG metrics, or any section of uploaded reports</p>
            
            {/* Example queries */}
            <div className="mt-6 text-left max-w-md mx-auto">
              <p className="text-sm font-medium text-gray-700 mb-2">Example queries:</p>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• "What was the revenue growth in 2023?"</li>
                <li>• "Show me the key risk factors"</li>
                <li>• "Compare financial performance across years"</li>
                <li>• "Summarize ESG initiatives"</li>
              </ul>
            </div>
          </div>
        ) : (
          messages.map(renderMessage)
        )}
        
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-100 rounded-lg px-4 py-3 border border-gray-200">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                <span className="text-sm text-gray-600 ml-2">Analyzing AGR documents...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t p-4">
        <div className="flex space-x-2">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about financial data, risks, ESG metrics..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            rows="2"
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors self-end"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;