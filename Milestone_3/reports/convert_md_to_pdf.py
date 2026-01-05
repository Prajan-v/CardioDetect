#!/usr/bin/env python3
"""
CardioDetect Milestone 3 - Markdown to PDF Converter
Converts the comprehensive MILESTONE_3_TECHNICAL_REPORT.md to a professional PDF
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re

# Paths
script_dir = Path(__file__).parent
MARKDOWN_FILE = script_dir / "MILESTONE_3_COMPLETE_DOCUMENTATION.md"
OUTPUT_PDF = script_dir / "CardioDetect_Milestone3_Complete_Report.pdf"
OUTPUT_HTML = script_dir / 'temp_report.html'

def create_css():
    """Create professional CSS styling"""
    return """
    @page {
        size: A4;
        margin: 2cm;
        @top-right {
            content: "CardioDetect Milestone 3";
            font-size: 10pt;
            color: #666;
        }
        @bottom-center {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 10pt;
            color: #666;
        }
    }
    
    body {
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #1a1a1a;
        text-align: justify;
    }
    
    h1 {
        font-size: 24pt;
        color: #1e3a8a;
        margin-top: 30pt;
        margin-bottom: 18pt;
        page-break-before: always;
        font-weight: bold;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 8pt;
    }
    
    h1:first-of-type {
        page-break-before: avoid;
    }
    
    h2 {
        font-size: 18pt;
        color: #3b82f6;
        margin-top: 24pt;
        margin-bottom: 12pt;
        font-weight: bold;
    }
    
    h3 {
        font-size: 14pt;
        color: #475569;
        margin-top: 18pt;
        margin-bottom: 10pt;
        font-weight: bold;
    }
    
    h4 {
        font-size: 12pt;
        color: #64748b;
        margin-top: 14pt;
        margin-bottom: 8pt;
        font-weight: bold;
    }
    
    p {
        margin-bottom: 12pt;
        text-align: justify;
    }
    
    ul, ol {
        margin-bottom: 12pt;
        padding-left: 30pt;
    }
    
    li {
        margin-bottom: 6pt;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 18pt 0;
        page-break-inside: avoid;
    }
    
    th {
        background-color: #3b82f6;
        color: white;
        padding: 10pt;
        text-align: center;
        font-weight: bold;
        border: 1px solid #cbd5e1;
    }
    
    td {
        padding: 8pt;
        border: 1px solid #cbd5e1;
        text-align: center;
        background-color: #f8fafc;
    }
    
    tr:nth-child(even) td {
        background-color: white;
    }
    
    code {
        background-color: #f1f5f9;
        padding: 2pt 6pt;
        border-radius: 3pt;
        font-family: 'Courier', monospace;
        font-size: 10pt;
    }
    
    pre {
        background-color: #1e293b;
        color: #e2e8f0;
        padding: 12pt;
        border-radius: 6pt;
        overflow-x: auto;
        margin: 12pt 0;
        page-break-inside: avoid;
    }
    
    pre code {
        background-color: transparent;
        color: #e2e8f0;
        padding: 0;
    }
    
    blockquote {
        border-left: 4pt solid #3b82f6;
        padding-left: 15pt;
        margin: 12pt 0;
        color: #475569;
        font-style: italic;
    }
    
    hr {
        border: none;
        border-top: 2px solid #e2e8f0;
        margin: 24pt 0;
    }
    
    strong {
        font-weight: bold;
        color: #1e3a8a;
    }
    
    em {
        font-style: italic;
    }
    
    a {
        color: #3b82f6;
        text-decoration: none;
    }
    
    /* Cover page specific */
    .cover-page {
        page-break-after: always;
        text-align: center;
        padding-top: 100pt;
    }
    
    .cover-title {
        font-size: 36pt;
        color: #1e3a8a;
        font-weight: bold;
        margin-bottom: 20pt;
    }
    
    .cover-subtitle {
        font-size: 20pt;
        color: #3b82f6;
        margin-bottom: 40pt;
    }
    """

def convert_markdown_to_pdf():
    """Convert markdown file to professional PDF"""
    print(f"📄 Reading markdown file: {MARKDOWN_FILE}")
    
    # Read markdown content
    with open(MARKDOWN_FILE, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    print("🔄 Converting markdown to HTML...")
    md = markdown.Markdown(extensions=[
        'tables',
        'fenced_code',
        'codehilite',
        'toc',
        'nl2br',
        'sane_lists'
    ])
    html_body = md.convert(md_content)
    
    # Create full HTML document
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>CardioDetect Milestone 3 Report</title>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """
    
    # Save temporary HTML
    print(f"💾 Saving temporary HTML: {OUTPUT_HTML}")
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Convert HTML to PDF
    print(f"📊 Generating PDF: {OUTPUT_PDF}")
    HTML(string=html_content).write_pdf(
        OUTPUT_PDF,
        stylesheets=[CSS(string=create_css())]
    )
    
    # Clean up temporary HTML
    OUTPUT_HTML.unlink()
    
    # Get file stats
    file_size_kb = OUTPUT_PDF.stat().st_size / 1024
    
    print(f"\n✅ PDF Report generated successfully!")
    print(f"📄 Location: {OUTPUT_PDF}")
    print(f"📊 File size: {file_size_kb:.1f} KB")
    print(f"\n🎉 Report is ready to share!")
    
    return str(OUTPUT_PDF)

if __name__ == "__main__":
    convert_markdown_to_pdf()
