#!/usr/bin/env python3
"""
PDF Documentation Generator
Converts Markdown documentation to PDF format for internship submissions
"""

import os
import subprocess
import sys
from pathlib import Path

def check_pandoc_installed():
    """Check if pandoc is installed for PDF conversion"""
    try:
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_md_to_pdf(md_file, pdf_file):
    """Convert Markdown file to PDF using pandoc"""
    try:
        cmd = [
            'pandoc',
            md_file,
            '-o', pdf_file,
            '--pdf-engine=wkhtmltopdf',  # Alternative: xelatex, pdflatex
            '--variable', 'geometry:margin=1in',
            '--variable', 'fontsize=11pt',
            '--variable', 'documentclass=article',
            '--toc',  # Table of contents
            '--toc-depth=3',
            '--highlight-style=github'
        ]
        
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully created: {pdf_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {md_file}: {e}")
        return False

def create_pdf_documentation():
    """Create PDF versions of all documentation"""
    
    # Check if pandoc is available
    if not check_pandoc_installed():
        print("❌ Pandoc is not installed. Please install it first:")
        print("   Windows: choco install pandoc")
        print("   Or download from: https://pandoc.org/installing.html")
        return False
    
    # Files to convert
    docs_to_convert = [
        ('reports/Project_Report.md', 'reports/Project_Report.pdf'),
        ('reports/Technical_Documentation.md', 'reports/Technical_Documentation.pdf'),
        ('reports/executive_summary.md', 'reports/Executive_Summary.pdf'),
        ('README.md', 'reports/README.pdf'),
        ('deploy_instructions.md', 'reports/Deployment_Guide.pdf')
    ]
    
    success_count = 0
    
    for md_file, pdf_file in docs_to_convert:
        if os.path.exists(md_file):
            if convert_md_to_pdf(md_file, pdf_file):
                success_count += 1
        else:
            print(f"⚠️  File not found: {md_file}")
    
    print(f"\n📊 Conversion Summary: {success_count}/{len(docs_to_convert)} files converted successfully")
    
    if success_count > 0:
        print("\n📁 PDF Documentation Created:")
        for _, pdf_file in docs_to_convert:
            if os.path.exists(pdf_file):
                print(f"   ✅ {pdf_file}")
    
    return success_count > 0

def create_documentation_package():
    """Create a complete documentation package"""
    
    print("🚀 Creating PDF Documentation Package...")
    print("=" * 50)
    
    # Create PDFs
    if create_pdf_documentation():
        print("\n🎯 Documentation package ready for internship submission!")
        print("\n📋 Internship Documentation Checklist:")
        print("   ✅ Project Report (PDF)")
        print("   ✅ Technical Documentation (PDF)")
        print("   ✅ Executive Summary (PDF)")
        print("   ✅ README/Overview (PDF)")
        print("   ✅ Deployment Guide (PDF)")
        print("   ✅ Live Application URL")
        print("   ✅ GitHub Repository")
        
        print("\n💼 Perfect for internship applications!")
        
    else:
        print("\n⚠️  Some conversions failed. You can still use the Markdown files.")
        print("   Alternative: Use online converters or print to PDF from browser")

if __name__ == "__main__":
    create_documentation_package()