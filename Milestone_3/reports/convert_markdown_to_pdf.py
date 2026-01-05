"""
CardioDetect Final Report PDF Generator
Converts the Markdown report to a professional PDF using ReportLab.
"""

import re
import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor, black, white, lightgrey
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, ListFlowable, ListItem
)

# Configuration
MD_SOURCE = Path('/Users/prajanv/.gemini/antigravity/brain/2aecde2b-b8d4-44b9-a57b-cc07114102ff/CardioDetect_Final_End_to_End_Report.md')
OUTPUT_PDF = Path('/Users/prajanv/CardioDetect/Milestone_3/reports/CardioDetect_Milestone3_Complete_Report.pdf')

# Styles (Reused from existing generator for consistency)
PRIMARY_COLOR = HexColor('#1e3a8a')
ACCENT_COLOR = HexColor('#3b82f6')

def get_styles():
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='Title_Custom', parent=styles['Title'],
        fontSize=24, textColor=PRIMARY_COLOR, spaceAfter=20, alignment=TA_CENTER, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='Heading1_Custom', parent=styles['Heading1'],
        fontSize=18, textColor=PRIMARY_COLOR, spaceBefore=20, spaceAfter=12, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='Heading2_Custom', parent=styles['Heading2'],
        fontSize=14, textColor=ACCENT_COLOR, spaceBefore=15, spaceAfter=8, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='Heading3_Custom', parent=styles['Heading3'],
        fontSize=12, textColor=HexColor('#475569'), spaceBefore=10, spaceAfter=6, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='Body_Custom', parent=styles['Normal'],
        fontSize=10, alignment=TA_JUSTIFY, spaceAfter=8, leading=14
    ))
    styles.add(ParagraphStyle(
        name='Bullet_Custom', parent=styles['Normal'],
        fontSize=10, alignment=TA_JUSTIFY, leftIndent=20, spaceAfter=4, leading=14
    ))
    
    return styles

def parse_markdown_line(line):
    """Simple MD formatting to ReportLab XML"""
    # Bold
    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
    # Italic
    line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
    # Code (inline)
    line = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', line)
    return line

def create_table_from_md(lines, styles):
    """Parses a block of MD table lines and returns a ReportLab Table"""
    data = []
    for line in lines:
        # Split by | and strip whitespace
        row = [cell.strip() for cell in line.strip('|').split('|')]
        # Handle bold formatting in cells
        row = [Paragraph(parse_markdown_line(cell), styles['Body_Custom']) for cell in row]
        data.append(row)
    
    # Basic styling
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    return t

def build_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=A4,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )
    styles = get_styles()
    story = []
    
    # Read Markdown
    if not MD_SOURCE.exists():
        print(f"Error: Source file {MD_SOURCE} not found.")
        return

    with open(MD_SOURCE, 'r') as f:
        md_lines = f.readlines()

    buffer_table = []
    in_list = False
    
    for line in md_lines:
        line = line.strip()
        
        # Skip empty lines, but process table buffer if needed
        if not line:
            if buffer_table:
                story.append(create_table_from_md(buffer_table, styles))
                story.append(Spacer(1, 12))
                buffer_table = []
            continue

        # Tables
        if line.startswith('|'):
            if '---' in line: continue # Skip separator lines
            buffer_table.append(line)
            continue
        
        # Flush table if we hit non-table line
        if buffer_table:
            story.append(create_table_from_md(buffer_table, styles))
            story.append(Spacer(1, 12))
            buffer_table = []

        # Headers
        if line.startswith('# '):
            story.append(PageBreak()) # Start major sections on new page usually
            story.append(Paragraph(parse_markdown_line(line[2:]), styles['Title_Custom']))
        elif line.startswith('## '):
            story.append(Paragraph(parse_markdown_line(line[3:]), styles['Heading1_Custom']))
        elif line.startswith('### '):
            story.append(Paragraph(parse_markdown_line(line[4:]), styles['Heading2_Custom']))
        
        # Lists
        elif line.startswith('- ') or line.startswith('* '):
            text = parse_markdown_line(line[2:])
            story.append(Paragraph(f"• {text}", styles['Bullet_Custom']))
        elif re.match(r'^\d+\.', line):
            text = parse_markdown_line(line.split('.', 1)[1].strip())
            number = line.split('.', 1)[0]
            story.append(Paragraph(f"{number}. {text}", styles['Bullet_Custom']))
            
        # Normal Text
        else:
            story.append(Paragraph(parse_markdown_line(line), styles['Body_Custom']))

    # Final flush
    if buffer_table:
        story.append(create_table_from_md(buffer_table, styles))

    # Build
    doc.build(story)
    print(f"PDF Generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    build_pdf()
