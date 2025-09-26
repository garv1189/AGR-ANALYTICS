#!/usr/bin/env python3
"""
Test with a more realistic AGR document content
"""

import asyncio
import aiohttp
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

async def create_realistic_agr_pdf():
    """Create a realistic AGR PDF with financial content"""
    buffer = io.BytesIO()
    
    # Create PDF with reportlab
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Page 1 - Financial Highlights
    p.drawString(100, height - 100, "ACME CORPORATION")
    p.drawString(100, height - 120, "ANNUAL GENERAL REPORT 2023")
    p.drawString(100, height - 160, "FINANCIAL HIGHLIGHTS")
    p.drawString(100, height - 200, "Revenue: $125.5 million (2023) vs $118.2 million (2022)")
    p.drawString(100, height - 220, "Net Income: $15.8 million (2023) vs $12.4 million (2022)")
    p.drawString(100, height - 240, "Total Assets: $89.3 million")
    p.drawString(100, height - 260, "Earnings per Share: $2.45")
    p.drawString(100, height - 280, "Return on Equity: 18.2%")
    
    p.drawString(100, height - 320, "RISK FACTORS")
    p.drawString(100, height - 340, "Market volatility poses significant risks to our operations.")
    p.drawString(100, height - 360, "Regulatory changes may impact our business model.")
    p.drawString(100, height - 380, "Supply chain disruptions could affect production.")
    
    p.showPage()
    
    # Page 2 - ESG Information
    p.drawString(100, height - 100, "ENVIRONMENTAL, SOCIAL & GOVERNANCE")
    p.drawString(100, height - 140, "Environmental Initiatives:")
    p.drawString(100, height - 160, "- Reduced carbon emissions by 15% in 2023")
    p.drawString(100, height - 180, "- Implemented renewable energy sources")
    p.drawString(100, height - 200, "- Achieved 85% waste recycling rate")
    
    p.drawString(100, height - 240, "Social Responsibility:")
    p.drawString(100, height - 260, "- Diversity and inclusion programs")
    p.drawString(100, height - 280, "- Employee training and development")
    p.drawString(100, height - 300, "- Community investment of $2.1 million")
    
    p.drawString(100, height - 340, "Governance:")
    p.drawString(100, height - 360, "- Independent board oversight")
    p.drawString(100, height - 380, "- Transparent reporting practices")
    p.drawString(100, height - 400, "- Strong internal controls")
    
    p.save()
    
    pdf_content = buffer.getvalue()
    buffer.close()
    return pdf_content

async def test_realistic_document():
    """Test with realistic AGR document"""
    
    # Get backend URL
    try:
        with open('/app/frontend/.env', 'r') as f:
            for line in f:
                if line.startswith('REACT_APP_BACKEND_URL='):
                    base_url = line.split('=', 1)[1].strip() + "/api"
                    break
    except:
        base_url = "http://localhost:8001/api"
    
    print(f"Testing realistic AGR document upload at: {base_url}")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Create realistic PDF
            pdf_content = await create_realistic_agr_pdf()
            print(f"Created realistic AGR PDF ({len(pdf_content)} bytes)")
            
            # Upload document
            data = aiohttp.FormData()
            data.add_field('file', pdf_content, filename='acme_agr_2023.pdf', content_type='application/pdf')
            data.add_field('company_name', 'ACME Corporation')
            data.add_field('year', '2023')
            
            async with session.post(f"{base_url}/documents/upload", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Upload successful: {result['total_chunks']} chunks created")
                    
                    # Test query with the uploaded document
                    query_data = {
                        "query": "What was the revenue growth in 2023?",
                        "company_filter": "ACME Corporation",
                        "year_filter": [2023],
                        "top_k": 5
                    }
                    
                    async with session.post(f"{base_url}/query", json=query_data) as query_response:
                        if query_response.status == 200:
                            query_result = await query_response.json()
                            chunks = len(query_result.get('retrieved_chunks', []))
                            confidence = query_result.get('confidence_score', 0)
                            print(f"✅ Query successful: {chunks} chunks retrieved, confidence: {confidence:.2f}")
                            print(f"Response format: {query_result.get('response_format')}")
                            
                            if chunks > 0:
                                print("✅ Document processing and retrieval working correctly")
                                return True
                            else:
                                print("⚠️  No chunks retrieved - may indicate processing issue")
                                return False
                        else:
                            print(f"❌ Query failed: HTTP {query_response.status}")
                            return False
                else:
                    error_text = await response.text()
                    print(f"❌ Upload failed: HTTP {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False

if __name__ == "__main__":
    # Install reportlab if not available
    try:
        import reportlab
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "reportlab"])
        import reportlab
    
    result = asyncio.run(test_realistic_document())
    print(f"\nRealistic document test: {'PASSED' if result else 'FAILED'}")