#!/usr/bin/env python3
"""
Quick Start Example
Complete workflow from indexing to keyword mapping to analysis
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!")


def main():
    """Run the complete workflow"""
    
    # Configuration
    PDF_FOLDER = "./docs"  # Change this to your PDF folder
    INDEX_PATH = "./wire_index"
    KEYWORDS_YAML = "./keywords.yaml"
    OUTPUT_DIR = "./keyword_mappings"
    
    print("\n" + "="*80)
    print("WIRE PROCESSING - COMPLETE WORKFLOW")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  PDF Folder: {PDF_FOLDER}")
    print(f"  Index Path: {INDEX_PATH}")
    print(f"  Keywords: {KEYWORDS_YAML}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    
    # Check if PDF folder exists
    if not Path(PDF_FOLDER).exists():
        print(f"\n❌ Error: PDF folder not found: {PDF_FOLDER}")
        print(f"Please create the folder and add your PDF documents, or update PDF_FOLDER variable")
        sys.exit(1)
    
    # Check if keywords.yaml exists
    if not Path(KEYWORDS_YAML).exists():
        print(f"\n❌ Error: Keywords file not found: {KEYWORDS_YAML}")
        print(f"Please make sure keywords.yaml is in the current directory")
        sys.exit(1)
    
    # Step 1: Index PDFs (if not already indexed)
    if not Path(INDEX_PATH).exists():
        print("\n" + "="*80)
        print("No index found - will create index from PDFs")
        print("="*80)
        
        run_command(
            [
                "python", "wire_processing_indexer.py",
                "--pdf-folder", PDF_FOLDER,
                "--index-path", INDEX_PATH,
                "--action", "index"
            ],
            "Indexing PDF documents"
        )
    else:
        print("\n" + "="*80)
        print(f"Index already exists at: {INDEX_PATH}")
        print("Skipping indexing step (delete index folder to re-index)")
        print("="*80)
    
    # Step 2: Generate keyword mappings
    run_command(
        [
            "python", "keyword_document_mapper.py",
            "--index-path", INDEX_PATH,
            "--keywords-yaml", KEYWORDS_YAML,
            "--output-dir", OUTPUT_DIR,
            "--max-docs", "20",
            "--min-score", "0.05"
        ],
        "Generating keyword-to-document mappings"
    )
    
    # Step 3: Analyze mappings
    run_command(
        [
            "python", "analyze_mappings.py",
            "--mappings-dir", OUTPUT_DIR
        ],
        "Analyzing keyword mappings"
    )
    
    # Success!
    print("\n" + "="*80)
    print("✓ WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated outputs:")
    print(f"  1. Search index: {INDEX_PATH}/")
    print(f"  2. Keyword mappings: {OUTPUT_DIR}/")
    print(f"     - keyword_to_documents_*.json")
    print(f"     - document_to_keywords_*.json")
    print(f"     - keyword_document_mappings_*.csv")
    print(f"     - high_relevance_mappings_*.json")
    print(f"     - mapping_summary_*.txt")
    print(f"     - high_priority_report.txt")
    print("\nNext steps:")
    print("  - Open CSV files in Excel for analysis")
    print("  - Review summary and high_priority_report text files")
    print("  - Use JSON files for programmatic access")
    print("  - Check high_relevance_mappings for strongest matches")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
