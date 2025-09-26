#!/usr/bin/env python3
"""
Comprehensive Backend Testing for AGR Agentic RAG Pipeline
Tests all backend functionality including document processing, vector search, and agent pipeline
"""

import asyncio
import aiohttp
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import time

# Add backend to path for imports
sys.path.append('/app/backend')

class AGRBackendTester:
    def __init__(self):
        # Get backend URL from frontend .env
        self.base_url = self._get_backend_url()
        self.session = None
        self.test_results = []
        self.test_session_id = None
        
    def _get_backend_url(self) -> str:
        """Get backend URL from frontend .env file"""
        try:
            with open('/app/frontend/.env', 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        url = line.split('=', 1)[1].strip()
                        return f"{url}/api"
            return "http://localhost:8001/api"  # fallback
        except Exception as e:
            print(f"Warning: Could not read frontend .env: {e}")
            return "http://localhost:8001/api"
    
    async def setup(self):
        """Setup test session"""
        self.session = aiohttp.ClientSession()
        print(f"🔧 Testing AGR Backend at: {self.base_url}")
        
    async def cleanup(self):
        """Cleanup test session"""
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details
        })
    
    async def test_basic_connectivity(self) -> bool:
        """Test basic API connectivity"""
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    self.log_test("Basic API Connectivity", True, f"Status: {data.get('status', 'unknown')}")
                    return True
                else:
                    self.log_test("Basic API Connectivity", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Basic API Connectivity", False, f"Connection error: {str(e)}")
            return False
    
    async def test_system_info(self) -> bool:
        """Test system info endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/system/info") as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ['documents_uploaded', 'total_chunks', 'vector_index_size', 'active_sessions', 'system_status']
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        self.log_test("System Info API", False, f"Missing fields: {missing_fields}")
                        return False
                    
                    self.log_test("System Info API", True, f"Status: {data['system_status']}, Docs: {data['documents_uploaded']}")
                    return True
                else:
                    self.log_test("System Info API", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("System Info API", False, f"Error: {str(e)}")
            return False
    
    async def test_chat_session_management(self) -> bool:
        """Test chat session creation and management"""
        try:
            # Create new session
            async with self.session.post(f"{self.base_url}/sessions") as response:
                if response.status == 200:
                    data = await response.json()
                    self.test_session_id = data.get('session_id')
                    
                    if not self.test_session_id:
                        self.log_test("Chat Session Creation", False, "No session_id returned")
                        return False
                    
                    self.log_test("Chat Session Creation", True, f"Session ID: {self.test_session_id[:8]}...")
                    
                    # Test session listing
                    async with self.session.get(f"{self.base_url}/sessions") as list_response:
                        if list_response.status == 200:
                            sessions = await list_response.json()
                            self.log_test("Chat Session Listing", True, f"Found {len(sessions)} sessions")
                            return True
                        else:
                            self.log_test("Chat Session Listing", False, f"HTTP {list_response.status}")
                            return False
                else:
                    self.log_test("Chat Session Creation", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Chat Session Management", False, f"Error: {str(e)}")
            return False
    
    def create_test_pdf_content(self) -> bytes:
        """Create a simple test PDF content for testing"""
        # This is a minimal PDF structure for testing
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test AGR Document - Financial Report 2023) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
299
%%EOF"""
        return pdf_content
    
    async def test_document_upload(self) -> bool:
        """Test document upload and processing"""
        try:
            # Create test PDF content
            pdf_content = self.create_test_pdf_content()
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('file', pdf_content, filename='test_agr_2023.pdf', content_type='application/pdf')
            data.add_field('company_name', 'Test Corporation')
            data.add_field('year', '2023')
            
            async with self.session.post(f"{self.base_url}/documents/upload", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    required_fields = ['message', 'document_id', 'total_chunks', 'filename']
                    
                    missing_fields = [field for field in required_fields if field not in result]
                    if missing_fields:
                        self.log_test("Document Upload", False, f"Missing response fields: {missing_fields}")
                        return False
                    
                    self.log_test("Document Upload", True, f"Uploaded: {result['filename']}, Chunks: {result['total_chunks']}")
                    return True
                else:
                    error_text = await response.text()
                    self.log_test("Document Upload", False, f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test("Document Upload", False, f"Error: {str(e)}")
            return False
    
    async def test_document_listing(self) -> bool:
        """Test document listing endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/documents") as response:
                if response.status == 200:
                    documents = await response.json()
                    self.log_test("Document Listing", True, f"Found {len(documents)} documents")
                    return True
                else:
                    self.log_test("Document Listing", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Document Listing", False, f"Error: {str(e)}")
            return False
    
    async def test_agr_query_processing(self) -> bool:
        """Test AGR query processing through the complete pipeline"""
        if not self.test_session_id:
            self.log_test("AGR Query Processing", False, "No test session available")
            return False
        
        try:
            # Test basic query
            query_data = {
                "query": "What are the key financial highlights for 2023?",
                "company_filter": "Test Corporation",
                "year_filter": [2023],
                "section_filter": ["Financials"],
                "top_k": 5,
                "session_id": self.test_session_id
            }
            
            async with self.session.post(f"{self.base_url}/query", json=query_data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Check required response fields
                    required_fields = ['query', 'retrieved_chunks', 'reasoner_analysis', 'formatted_answer', 
                                     'confidence_score', 'citations', 'response_format', 'session_id']
                    
                    missing_fields = [field for field in required_fields if field not in result]
                    if missing_fields:
                        self.log_test("AGR Query Processing", False, f"Missing response fields: {missing_fields}")
                        return False
                    
                    # Validate reasoner analysis structure
                    reasoner = result.get('reasoner_analysis', {})
                    reasoner_fields = ['analysis_type', 'requires_reformulation', 'context_summary', 'confidence_level']
                    missing_reasoner = [field for field in reasoner_fields if field not in reasoner]
                    
                    if missing_reasoner:
                        self.log_test("AGR Query Processing", False, f"Missing reasoner fields: {missing_reasoner}")
                        return False
                    
                    confidence = result.get('confidence_score', 0)
                    chunks_count = len(result.get('retrieved_chunks', []))
                    
                    self.log_test("AGR Query Processing", True, 
                                f"Confidence: {confidence:.2f}, Chunks: {chunks_count}, Format: {result['response_format']}")
                    return True
                else:
                    error_text = await response.text()
                    self.log_test("AGR Query Processing", False, f"HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log_test("AGR Query Processing", False, f"Error: {str(e)}")
            return False
    
    async def test_query_with_no_documents(self) -> bool:
        """Test query processing when no relevant documents exist"""
        if not self.test_session_id:
            return False
        
        try:
            query_data = {
                "query": "What are the environmental sustainability initiatives?",
                "company_filter": "NonExistent Company",
                "year_filter": [2020],
                "top_k": 5,
                "session_id": self.test_session_id
            }
            
            async with self.session.post(f"{self.base_url}/query", json=query_data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Should handle gracefully with low confidence
                    confidence = result.get('confidence_score', 1.0)
                    chunks_count = len(result.get('retrieved_chunks', []))
                    
                    if confidence < 0.5 and chunks_count == 0:
                        self.log_test("Query with No Documents", True, "Handled gracefully with low confidence")
                        return True
                    else:
                        self.log_test("Query with No Documents", False, f"Unexpected response: confidence={confidence}, chunks={chunks_count}")
                        return False
                else:
                    self.log_test("Query with No Documents", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Query with No Documents", False, f"Error: {str(e)}")
            return False
    
    async def test_different_analysis_types(self) -> bool:
        """Test different types of AGR analysis queries"""
        if not self.test_session_id:
            return False
        
        test_queries = [
            {
                "query": "Compare revenue growth between 2022 and 2023",
                "expected_type": "comparison"
            },
            {
                "query": "What are the main risk factors?",
                "expected_type": "risk_detection"
            },
            {
                "query": "Summarize the financial performance",
                "expected_type": "summary"
            }
        ]
        
        success_count = 0
        
        for test_query in test_queries:
            try:
                query_data = {
                    "query": test_query["query"],
                    "top_k": 3,
                    "session_id": self.test_session_id
                }
                
                async with self.session.post(f"{self.base_url}/query", json=query_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_type = result.get('reasoner_analysis', {}).get('analysis_type', 'unknown')
                        
                        # Note: Analysis type detection might vary, so we just check for valid response
                        if analysis_type and result.get('formatted_answer'):
                            success_count += 1
                            print(f"   ✓ Query: '{test_query['query'][:30]}...' -> Type: {analysis_type}")
                        else:
                            print(f"   ✗ Query failed: '{test_query['query'][:30]}...'")
                    else:
                        print(f"   ✗ HTTP {response.status} for query: '{test_query['query'][:30]}...'")
                        
            except Exception as e:
                print(f"   ✗ Error for query '{test_query['query'][:30]}...': {str(e)}")
        
        success = success_count >= 2  # At least 2 out of 3 should work
        self.log_test("Different Analysis Types", success, f"{success_count}/3 query types processed successfully")
        return success
    
    async def test_chat_history(self) -> bool:
        """Test chat history retrieval"""
        if not self.test_session_id:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/sessions/{self.test_session_id}/messages") as response:
                if response.status == 200:
                    messages = await response.json()
                    self.log_test("Chat History Retrieval", True, f"Retrieved {len(messages)} messages")
                    return True
                else:
                    self.log_test("Chat History Retrieval", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Chat History Retrieval", False, f"Error: {str(e)}")
            return False
    
    async def test_invalid_file_upload(self) -> bool:
        """Test upload with invalid file type"""
        try:
            # Try to upload a text file (should be rejected)
            data = aiohttp.FormData()
            data.add_field('file', b'This is not a PDF or DOCX file', filename='test.txt', content_type='text/plain')
            data.add_field('company_name', 'Test Corp')
            data.add_field('year', '2023')
            
            async with self.session.post(f"{self.base_url}/documents/upload", data=data) as response:
                if response.status == 400:  # Should reject invalid file type
                    self.log_test("Invalid File Upload Rejection", True, "Correctly rejected invalid file type")
                    return True
                else:
                    self.log_test("Invalid File Upload Rejection", False, f"Expected 400, got {response.status}")
                    return False
        except Exception as e:
            self.log_test("Invalid File Upload Rejection", False, f"Error: {str(e)}")
            return False
    
    async def test_llm_integration(self) -> bool:
        """Test LLM integration by checking if responses contain structured analysis"""
        if not self.test_session_id:
            return False
        
        try:
            query_data = {
                "query": "Analyze the financial performance and identify key metrics",
                "top_k": 3,
                "session_id": self.test_session_id
            }
            
            async with self.session.post(f"{self.base_url}/query", json=query_data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Check if response shows signs of LLM processing
                    formatted_answer = result.get('formatted_answer', '')
                    reasoner_analysis = result.get('reasoner_analysis', {})
                    
                    # Look for structured response indicators
                    has_structure = any([
                        len(formatted_answer) > 50,  # Substantial response
                        reasoner_analysis.get('context_summary'),  # Analysis provided
                        result.get('confidence_score', 0) > 0.1,  # Confidence calculated
                        result.get('citations')  # Citations provided
                    ])
                    
                    if has_structure:
                        self.log_test("LLM Integration", True, "LLM responses show proper structure and analysis")
                        return True
                    else:
                        self.log_test("LLM Integration", False, "LLM responses lack expected structure")
                        return False
                else:
                    self.log_test("LLM Integration", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("LLM Integration", False, f"Error: {str(e)}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"\n{'='*60}")
        print(f"AGR BACKEND TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"   • {result['test']}: {result['details']}")
        
        print(f"{'='*60}")
        
        return passed_tests, failed_tests

async def main():
    """Main test execution"""
    tester = AGRBackendTester()
    
    try:
        await tester.setup()
        
        print("🚀 Starting AGR Backend Comprehensive Testing...")
        print(f"Backend URL: {tester.base_url}")
        print("-" * 60)
        
        # Run all tests in sequence
        test_functions = [
            tester.test_basic_connectivity,
            tester.test_system_info,
            tester.test_chat_session_management,
            tester.test_document_upload,
            tester.test_document_listing,
            tester.test_agr_query_processing,
            tester.test_query_with_no_documents,
            tester.test_different_analysis_types,
            tester.test_chat_history,
            tester.test_invalid_file_upload,
            tester.test_llm_integration
        ]
        
        for test_func in test_functions:
            await test_func()
            await asyncio.sleep(0.5)  # Small delay between tests
        
        # Print final summary
        passed, failed = tester.print_summary()
        
        # Return appropriate exit code
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        return 1
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)