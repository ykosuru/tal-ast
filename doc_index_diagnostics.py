"""
Index Diagnostic Tools
Verify what's actually in your index and identify issues
"""

from pathlib import Path
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Set
import re


def verify_index_coverage(
    index_dir: str = "./universal_index",
    doc_dir: str = "./your_docs"
) -> Dict[str, any]:
    """
    Check which documents are actually indexed vs what's in the directory
    
    Returns:
        Dict with coverage stats and missing files
    """
    print("\n" + "="*70)
    print("INDEX COVERAGE VERIFICATION")
    print("="*70)
    
    # Get all documents that should be indexed
    all_docs = []
    for pattern in ['**/*.pdf', '**/*.txt', '**/*.md', '**/*.docx', '**/*.doc']:
        all_docs.extend(Path(doc_dir).glob(pattern))
    
    print(f"\nDocuments in directory: {len(all_docs)}")
    
    # Load index metadata
    try:
        with open(Path(index_dir) / "metadata.pkl", 'rb') as f:
            indexed_metadata = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Index not found at: {index_dir}")
        return {'error': 'Index not found'}
    
    indexed_files = set([m['source_file'] for m in indexed_metadata])
    print(f"Documents in index: {len(indexed_files)}")
    
    # Find missing
    missing = []
    for doc in all_docs:
        if doc.name not in indexed_files:
            missing.append(doc.name)
    
    # Find indexed but not in directory (orphaned)
    doc_names = set([d.name for d in all_docs])
    orphaned = [f for f in indexed_files if f not in doc_names]
    
    # Results
    coverage_pct = (len(indexed_files) - len(orphaned)) / len(all_docs) * 100 if all_docs else 0
    
    print(f"\nCoverage: {coverage_pct:.1f}%")
    
    if missing:
        print(f"\n⚠ WARNING: {len(missing)} documents NOT indexed:")
        for doc in missing[:10]:
            print(f"  - {doc}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print("✓ All documents are indexed!")
    
    if orphaned:
        print(f"\n⚠ WARNING: {len(orphaned)} indexed files not in directory:")
        for doc in orphaned[:10]:
            print(f"  - {doc}")
    
    return {
        'total_docs': len(all_docs),
        'indexed_docs': len(indexed_files),
        'missing': missing,
        'orphaned': orphaned,
        'coverage_pct': coverage_pct
    }


def diagnose_index(index_dir: str = "./universal_index") -> Dict[str, any]:
    """
    Comprehensive analysis of index contents
    
    Returns:
        Dict with index statistics
    """
    print("\n" + "="*70)
    print("INDEX DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    try:
        # Load metadata
        with open(Path(index_dir) / "metadata.pkl", 'rb') as f:
            metadata_list = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Index not found at: {index_dir}")
        return {'error': 'Index not found'}
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Total chunks: {len(metadata_list)}")
    
    # Analyze file types
    file_types = Counter([m.get('file_type', 'unknown') for m in metadata_list])
    print(f"\n📁 File types:")
    for ft, count in file_types.most_common():
        print(f"  {ft}: {count} chunks")
    
    # Analyze unique files
    unique_files = set([m['source_file'] for m in metadata_list])
    print(f"\n📄 Unique files: {len(unique_files)}")
    
    # Chunks per file
    chunks_per_file = Counter([m['source_file'] for m in metadata_list])
    avg_chunks = sum(chunks_per_file.values()) / len(chunks_per_file)
    print(f"  Average chunks per file: {avg_chunks:.1f}")
    print(f"  Max chunks in one file: {chunks_per_file.most_common(1)[0][1]}")
    
    # Show files with most chunks
    print(f"\n📚 Files with most chunks:")
    for filename, count in chunks_per_file.most_common(5):
        print(f"  {filename}: {count} chunks")
    
    # Analyze chunk sizes
    chunk_lengths = [len(m.get('text', '')) for m in metadata_list]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    print(f"\n📏 Chunk sizes:")
    print(f"  Average: {avg_length:.0f} characters")
    print(f"  Min: {min(chunk_lengths)}")
    print(f"  Max: {max(chunk_lengths)}")
    
    # Check for requirements-related files
    req_keywords = ['requirement', 'spec', 'specification', 'rfp', 'sow', 'functional']
    req_files = set([
        m['source_file'] for m in metadata_list 
        if any(kw in m['source_file'].lower() for kw in req_keywords)
    ])
    
    print(f"\n📋 Requirements documents:")
    print(f"  Files with requirements keywords in name: {len(req_files)}")
    if req_files:
        for f in list(req_files)[:5]:
            chunk_count = chunks_per_file[f]
            print(f"  - {f} ({chunk_count} chunks)")
        if len(req_files) > 5:
            print(f"  ... and {len(req_files) - 5} more")
    
    # Check for requirements language in content
    req_lang_keywords = ['shall', 'must', 'should', 'required', 'mandatory']
    chunks_with_req_lang = 0
    for m in metadata_list:
        text_lower = m.get('text', '').lower()
        if any(kw in text_lower for kw in req_lang_keywords):
            chunks_with_req_lang += 1
    
    print(f"\n🔍 Requirements language:")
    print(f"  Chunks containing 'shall/must/should': {chunks_with_req_lang}")
    print(f"  Percentage: {chunks_with_req_lang/len(metadata_list)*100:.1f}%")
    
    # Load embeddings to check dimensions
    try:
        with open(Path(index_dir) / "embeddings.pkl", 'rb') as f:
            embeddings = pickle.load(f)
        print(f"\n🧮 Embeddings:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dimension: {embeddings.shape[1]}")
    except:
        print(f"\n⚠ Could not load embeddings")
    
    # Sample some chunks
    print(f"\n📝 Sample chunks (first 3):")
    for i, m in enumerate(metadata_list[:3]):
        print(f"\n  Chunk {i+1} from {m['source_file']}:")
        text_preview = m.get('text', '')[:150]
        print(f"  {text_preview}...")
    
    return {
        'total_chunks': len(metadata_list),
        'unique_files': len(unique_files),
        'file_types': dict(file_types),
        'avg_chunk_size': avg_length,
        'requirements_files': len(req_files),
        'chunks_with_req_lang': chunks_with_req_lang
    }


def test_document_indexed(
    filename: str,
    index_dir: str = "./universal_index"
) -> bool:
    """
    Check if a specific document is indexed and show details
    
    Args:
        filename: Name of file to check (e.g., "WireTransferSpec.pdf")
        index_dir: Index directory
        
    Returns:
        True if found, False otherwise
    """
    print("\n" + "="*70)
    print(f"CHECKING: {filename}")
    print("="*70)
    
    try:
        with open(Path(index_dir) / "metadata.pkl", 'rb') as f:
            metadata_list = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Index not found at: {index_dir}")
        return False
    
    # Find chunks for this file
    chunks = [m for m in metadata_list if filename in m['source_file']]
    
    if not chunks:
        print(f"❌ {filename} is NOT in the index!")
        
        # Show similar filenames
        all_files = set([m['source_file'] for m in metadata_list])
        similar = [f for f in all_files if filename.lower()[:10] in f.lower()]
        
        if similar:
            print(f"\nDid you mean one of these?")
            for f in similar[:5]:
                print(f"  - {f}")
        
        return False
    
    print(f"✅ {filename} is indexed")
    print(f"\n📊 Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  File type: {chunks[0].get('file_type', 'unknown')}")
    
    # Analyze chunk sizes
    chunk_sizes = [len(c.get('text', '')) for c in chunks]
    print(f"  Average chunk size: {sum(chunk_sizes)/len(chunk_sizes):.0f} characters")
    
    # Check for requirements language
    req_chunks = 0
    for c in chunks:
        text_lower = c.get('text', '').lower()
        if any(kw in text_lower for kw in ['shall', 'must', 'should']):
            req_chunks += 1
    
    print(f"  Chunks with requirements language: {req_chunks}/{len(chunks)}")
    
    # Show sample chunks
    print(f"\n📝 Sample chunks:")
    for i, c in enumerate(chunks[:3]):
        print(f"\n  Chunk {i+1} (index {chunks.index(c)}):")
        text = c.get('text', '')
        print(f"  {text[:200]}...")
    
    if len(chunks) > 3:
        print(f"\n  ... and {len(chunks) - 3} more chunks")
    
    return True


def find_requirements_docs(index_dir: str = "./universal_index") -> List[str]:
    """
    Find all documents that appear to be requirements documents
    
    Returns:
        List of requirement document filenames
    """
    print("\n" + "="*70)
    print("FINDING REQUIREMENTS DOCUMENTS")
    print("="*70)
    
    try:
        with open(Path(index_dir) / "metadata.pkl", 'rb') as f:
            metadata_list = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Index not found at: {index_dir}")
        return []
    
    # Criteria for requirements documents
    req_filename_keywords = ['requirement', 'spec', 'specification', 'rfp', 'sow', 'functional', 'design', 'technical']
    req_content_keywords = ['shall', 'must', 'should', 'required', 'mandatory']
    
    # Score each file
    file_scores = defaultdict(lambda: {'chunks': 0, 'req_chunks': 0, 'name_match': False})
    
    for m in metadata_list:
        filename = m['source_file']
        text_lower = m.get('text', '').lower()
        
        file_scores[filename]['chunks'] += 1
        
        # Check filename
        if any(kw in filename.lower() for kw in req_filename_keywords):
            file_scores[filename]['name_match'] = True
        
        # Check content
        if any(kw in text_lower for kw in req_content_keywords):
            file_scores[filename]['req_chunks'] += 1
    
    # Find likely requirements documents
    req_docs = []
    
    for filename, stats in file_scores.items():
        req_density = stats['req_chunks'] / stats['chunks'] if stats['chunks'] > 0 else 0
        
        # Criteria:
        # 1. Name matches requirements keywords, OR
        # 2. >30% of chunks contain requirements language
        if stats['name_match'] or req_density > 0.3:
            req_docs.append({
                'filename': filename,
                'chunks': stats['chunks'],
                'req_chunks': stats['req_chunks'],
                'req_density': req_density,
                'name_match': stats['name_match']
            })
    
    # Sort by likelihood
    req_docs.sort(key=lambda x: (x['name_match'], x['req_density']), reverse=True)
    
    print(f"\nFound {len(req_docs)} potential requirements documents:\n")
    
    for doc in req_docs:
        indicator = "📋" if doc['name_match'] else "📄"
        print(f"{indicator} {doc['filename']}")
        print(f"   Chunks: {doc['chunks']} | Req language: {doc['req_density']*100:.1f}%")
    
    return [d['filename'] for d in req_docs]


def compare_indexes(
    index1_dir: str = "./universal_index",
    index2_dir: str = "./universal_index_v2"
) -> Dict:
    """
    Compare two indexes (useful when re-indexing)
    """
    print("\n" + "="*70)
    print("COMPARING INDEXES")
    print("="*70)
    
    def load_index_info(index_dir):
        try:
            with open(Path(index_dir) / "metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            return {
                'chunks': len(metadata),
                'files': set([m['source_file'] for m in metadata]),
                'avg_chunk_size': sum(len(m.get('text', '')) for m in metadata) / len(metadata)
            }
        except:
            return None
    
    info1 = load_index_info(index1_dir)
    info2 = load_index_info(index2_dir)
    
    if not info1:
        print(f"❌ Could not load index 1: {index1_dir}")
        return {}
    
    if not info2:
        print(f"❌ Could not load index 2: {index2_dir}")
        return {}
    
    print(f"\n📊 Index 1: {index1_dir}")
    print(f"  Chunks: {info1['chunks']}")
    print(f"  Files: {len(info1['files'])}")
    print(f"  Avg chunk size: {info1['avg_chunk_size']:.0f} chars")
    
    print(f"\n📊 Index 2: {index2_dir}")
    print(f"  Chunks: {info2['chunks']}")
    print(f"  Files: {len(info2['files'])}")
    print(f"  Avg chunk size: {info2['avg_chunk_size']:.0f} chars")
    
    print(f"\n🔄 Comparison:")
    print(f"  Chunks: {info2['chunks'] - info1['chunks']:+d} ({info2['chunks']/info1['chunks']*100:.1f}%)")
    print(f"  Files: {len(info2['files']) - len(info1['files']):+d}")
    print(f"  Avg chunk size: {info2['avg_chunk_size'] - info1['avg_chunk_size']:+.0f} chars")
    
    # Files only in one index
    only_in_1 = info1['files'] - info2['files']
    only_in_2 = info2['files'] - info1['files']
    
    if only_in_1:
        print(f"\n⚠ {len(only_in_1)} files only in index 1:")
        for f in list(only_in_1)[:5]:
            print(f"  - {f}")
    
    if only_in_2:
        print(f"\n✨ {len(only_in_2)} files only in index 2:")
        for f in list(only_in_2)[:5]:
            print(f"  - {f}")
    
    return {
        'index1': info1,
        'index2': info2,
        'only_in_1': only_in_1,
        'only_in_2': only_in_2
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'verify':
            verify_index_coverage()
        
        elif command == 'diagnose':
            diagnose_index()
        
        elif command == 'test':
            if len(sys.argv) > 2:
                test_document_indexed(sys.argv[2])
            else:
                print("Usage: python index_diagnostics.py test <filename>")
        
        elif command == 'find-requirements':
            find_requirements_docs()
        
        elif command == 'compare':
            if len(sys.argv) > 3:
                compare_indexes(sys.argv[2], sys.argv[3])
            else:
                compare_indexes()
        
        else:
            print("Unknown command. Use: verify, diagnose, test, find-requirements, or compare")
    
    else:
        # Run all diagnostics
        verify_index_coverage()
        diagnose_index()
        find_requirements_docs()
