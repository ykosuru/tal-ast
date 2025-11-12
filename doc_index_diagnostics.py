"""
Enhanced Index Diagnostic Tools with Keyword Coverage Analysis
Analyze which documents contain specific domain keywords from keywords.yaml
"""

from pathlib import Path
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
import re
import yaml


def analyze_keyword_coverage(
    keywords_file: str = "keywords.yaml",
    index_dir: str = "./universal_index",
    min_docs: int = 1,
    show_docs: bool = True,
    export_csv: bool = False
) -> Dict:
    """
    Analyze how many documents contain each keyword from keywords.yaml
    
    Args:
        keywords_file: Path to keywords.yaml file
        index_dir: Path to index directory
        min_docs: Only show keywords found in at least this many docs
        show_docs: Show which documents contain each keyword
        export_csv: Export results to CSV file
        
    Returns:
        Dict with keyword coverage statistics
    """
    print("\n" + "="*70)
    print("KEYWORD COVERAGE ANALYSIS")
    print("="*70)
    
    # Load keywords from YAML
    try:
        with open(keywords_file, 'r') as f:
            keywords_data = yaml.safe_load(f)
        print(f"\n✓ Loaded keywords from: {keywords_file}")
    except FileNotFoundError:
        print(f"\n❌ Keywords file not found: {keywords_file}")
        print("\nCreating sample keywords.yaml...")
        create_sample_keywords_yaml(keywords_file)
        with open(keywords_file, 'r') as f:
            keywords_data = yaml.safe_load(f)
    
    # Parse keywords structure
    # Supports multiple formats:
    # 1. Flat list: ['wire', 'ach']
    # 2. Simple dict: {'payment': ['wire', 'ach']}
    # 3. Nested with priority: {'payment': {'priority': 'high', 'keywords': ['wire', 'ach']}}
    
    all_keywords = []
    keyword_categories = {}
    keyword_priorities = {}
    
    if isinstance(keywords_data, dict):
        for category, content in keywords_data.items():
            if isinstance(content, dict):
                # Nested structure with priority
                priority = content.get('priority', 'medium')
                terms = content.get('keywords', [])
                
                if terms:
                    all_keywords.extend(terms)
                    for term in terms:
                        keyword_categories[term] = category
                        keyword_priorities[term] = priority
            elif isinstance(content, list):
                # Simple list: {'payment': ['wire', 'ach']}
                all_keywords.extend(content)
                for term in content:
                    keyword_categories[term] = category
                    keyword_priorities[term] = 'medium'
            else:
                # Single term
                all_keywords.append(category)
                keyword_categories[category] = 'uncategorized'
                keyword_priorities[category] = 'medium'
    else:
        # Flat list: ['wire', 'ach', 'ofac', 'aml']
        all_keywords = keywords_data
        keyword_categories = {kw: 'uncategorized' for kw in all_keywords}
        keyword_priorities = {kw: 'medium' for kw in all_keywords}
    
    print(f"  Total keywords: {len(all_keywords)}")
    if keyword_categories:
        categories = set(keyword_categories.values())
        print(f"  Categories: {len(categories)}")
    
    # Load index
    try:
        with open(Path(index_dir) / "metadata.pkl", 'rb') as f:
            metadata_list = pickle.load(f)
        print(f"\n✓ Loaded index from: {index_dir}")
        print(f"  Total chunks: {len(metadata_list)}")
    except FileNotFoundError:
        print(f"\n❌ Index not found at: {index_dir}")
        return {}
    
    # Analyze coverage
    print(f"\n🔍 Analyzing keyword coverage...\n")
    
    keyword_stats = {}
    
    for keyword in all_keywords:
        # Find all documents containing this keyword
        matching_files = set()
        matching_chunks = []
        
        keyword_lower = keyword.lower()
        keyword_pattern = re.compile(r'\b' + re.escape(keyword_lower) + r'\b', re.IGNORECASE)
        
        for m in metadata_list:
            text = m.get('text', '')
            if keyword_pattern.search(text.lower()):
                matching_files.add(m['source_file'])
                matching_chunks.append({
                    'file': m['source_file'],
                    'chunk_idx': m.get('chunk_index', 0),
                    'text_preview': text[:150]
                })
        
        keyword_stats[keyword] = {
            'keyword': keyword,
            'category': keyword_categories.get(keyword, 'uncategorized'),
            'priority': keyword_priorities.get(keyword, 'medium'),
            'doc_count': len(matching_files),
            'chunk_count': len(matching_chunks),
            'documents': sorted(list(matching_files)),
            'sample_chunks': matching_chunks[:3]  # Keep first 3 for samples
        }
    
    # Sort by document count (descending)
    sorted_keywords = sorted(
        keyword_stats.items(),
        key=lambda x: x[1]['doc_count'],
        reverse=True
    )
    
    # Display results
    print("="*90)
    print(f"{'KEYWORD':<40} {'DOCS':<8} {'CHUNKS':<10} {'PRIORITY':<12} {'CATEGORY':<20}")
    print("="*90)
    
    found_count = 0
    missing_count = 0
    critical_missing = []
    high_missing = []
    
    for keyword, stats in sorted_keywords:
        if stats['doc_count'] >= min_docs:
            found_count += 1
            priority = stats['priority']
            category = stats['category'][:18]
            
            # Color-code by priority (for terminals that support it)
            priority_marker = {
                'critical': '🔴',
                'high': '🟡',
                'medium': '🟢'
            }.get(priority, '⚪')
            
            print(f"{keyword:<40} {stats['doc_count']:<8} {stats['chunk_count']:<10} {priority_marker} {priority:<10} {category:<20}")
            
            if show_docs and stats['documents']:
                for doc in stats['documents'][:5]:  # Show first 5 docs
                    print(f"  └─ {doc}")
                if len(stats['documents']) > 5:
                    print(f"  └─ ... and {len(stats['documents']) - 5} more")
                print()
        else:
            missing_count += 1
            if stats['priority'] == 'critical':
                critical_missing.append((keyword, stats))
            elif stats['priority'] == 'high':
                high_missing.append((keyword, stats))
    
    # Show keywords with no matches (prioritize by priority level)
    if missing_count > 0:
        print("\n" + "="*90)
        print(f"⚠ KEYWORDS NOT FOUND IN ANY DOCUMENTS ({missing_count} keywords)")
        print("="*90)
        
        if critical_missing:
            print("\n🔴 CRITICAL PRIORITY (URGENT!):")
            for keyword, stats in critical_missing:
                category = stats['category']
                print(f"  ❌ {keyword:<40} (category: {category})")
        
        if high_missing:
            print("\n🟡 HIGH PRIORITY:")
            for keyword, stats in high_missing:
                category = stats['category']
                print(f"  ❌ {keyword:<40} (category: {category})")
        
        other_missing = [
            (kw, st) for kw, st in sorted_keywords 
            if st['doc_count'] == 0 and st['priority'] not in ['critical', 'high']
        ]
        if other_missing:
            print("\n⚪ OTHER:")
            for keyword, stats in other_missing[:10]:  # Show first 10
                category = stats['category']
                print(f"  ❌ {keyword:<40} (category: {category})")
            if len(other_missing) > 10:
                print(f"  ... and {len(other_missing) - 10} more")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total keywords analyzed: {len(all_keywords)}")
    print(f"Keywords found in ≥{min_docs} doc(s): {found_count}")
    print(f"Keywords not found: {missing_count}")
    
    if found_count > 0:
        avg_docs = sum(s['doc_count'] for s in keyword_stats.values()) / len(keyword_stats)
        avg_chunks = sum(s['chunk_count'] for s in keyword_stats.values()) / len(keyword_stats)
        print(f"\nAverage docs per keyword: {avg_docs:.1f}")
        print(f"Average chunks per keyword: {avg_chunks:.1f}")
    
    # Category breakdown with priority info
    if len(set(keyword_categories.values())) > 1:
        print("\n" + "="*90)
        print("COVERAGE BY CATEGORY")
        print("="*90)
        
        category_stats = defaultdict(lambda: {
            'total': 0, 'found': 0, 'missing': 0,
            'critical': 0, 'high': 0, 'medium': 0,
            'priority': None
        })
        
        for keyword, stats in keyword_stats.items():
            category = stats['category']
            priority = stats['priority']
            
            category_stats[category]['total'] += 1
            category_stats[category][priority] += 1
            
            # Set category priority (highest priority wins)
            if not category_stats[category]['priority']:
                category_stats[category]['priority'] = priority
            elif priority == 'critical':
                category_stats[category]['priority'] = 'critical'
            elif priority == 'high' and category_stats[category]['priority'] != 'critical':
                category_stats[category]['priority'] = 'high'
            
            if stats['doc_count'] > 0:
                category_stats[category]['found'] += 1
            else:
                category_stats[category]['missing'] += 1
        
        # Sort by priority (critical first)
        priority_order = {'critical': 0, 'high': 1, 'medium': 2}
        sorted_categories = sorted(
            category_stats.items(),
            key=lambda x: (priority_order.get(x[1]['priority'], 3), x[0])
        )
        
        for category, stats in sorted_categories:
            coverage_pct = stats['found'] / stats['total'] * 100
            priority = stats['priority'] or 'medium'
            
            priority_marker = {
                'critical': '🔴',
                'high': '🟡',
                'medium': '🟢'
            }.get(priority, '⚪')
            
            print(f"\n{priority_marker} {category} (Priority: {priority}):")
            print(f"  Total keywords: {stats['total']}")
            print(f"  Found: {stats['found']} ({coverage_pct:.1f}%)")
            print(f"  Missing: {stats['missing']}")
            
            if stats['missing'] > 0 and priority in ['critical', 'high']:
                print(f"  ⚠ WARNING: {stats['missing']} {priority} priority keywords missing!")
    
    # Top documents by keyword coverage
    print("\n" + "="*70)
    print("TOP DOCUMENTS BY KEYWORD COVERAGE")
    print("="*70)
    
    doc_keyword_count = defaultdict(set)
    for keyword, stats in keyword_stats.items():
        if stats['doc_count'] > 0:
            for doc in stats['documents']:
                doc_keyword_count[doc].add(keyword)
    
    top_docs = sorted(
        doc_keyword_count.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]
    
    for doc, keywords in top_docs:
        print(f"\n{doc}")
        print(f"  Keywords found: {len(keywords)}")
        print(f"  {', '.join(sorted(list(keywords))[:10])}")
        if len(keywords) > 10:
            print(f"  ... and {len(keywords) - 10} more")
    
    # Export to CSV if requested
    if export_csv:
        export_to_csv(keyword_stats, "keyword_coverage.csv")
        print(f"\n✓ Exported results to: keyword_coverage.csv")
    
    return keyword_stats


def create_sample_keywords_yaml(filename: str = "keywords_sample.yaml"):
    """
    Create a sample keywords.yaml file with payment/banking terms
    Uses nested structure with priority levels
    """
    sample_keywords = {
        'payment_types': {
            'priority': 'high',
            'keywords': [
                'wire', 'ach', 'swift', 'fedwire', 'sepa', 'rtp', 'eft',
                'check', 'card', 'credit', 'debit', 'cash'
            ]
        },
        'payment_actions': {
            'priority': 'high',
            'keywords': [
                'transfer', 'payment', 'transaction', 'remittance', 'settlement',
                'disbursement', 'drawdown', 'advance', 'withdrawal', 'deposit'
            ]
        },
        'validation_terms': {
            'priority': 'high',
            'keywords': [
                'validate', 'validation', 'verify', 'verification', 'check',
                'screening', 'review', 'approval', 'authorization', 'confirm'
            ]
        },
        'compliance_terms': {
            'priority': 'critical',
            'keywords': [
                'ofac', 'aml', 'kyc', 'sanctions', 'compliance', 'regulation',
                'audit', 'regulatory', 'cip', 'bsa', 'fatca', 'crs'
            ]
        },
        'requirements_language': {
            'priority': 'critical',
            'keywords': [
                'shall', 'must', 'should', 'will', 'may', 'required',
                'mandatory', 'optional', 'specification', 'requirement',
                'criteria', 'constraint', 'functional', 'non-functional'
            ]
        },
        'account_terms': {
            'priority': 'medium',
            'keywords': [
                'account', 'beneficiary', 'originator', 'customer', 'party',
                'sender', 'receiver', 'payee', 'payer', 'holder'
            ]
        }
    }
    
    with open(filename, 'w') as f:
        yaml.dump(sample_keywords, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created sample keywords file: {filename}")
    print(f"  This file uses the same format as your keywords.yaml")
    print(f"  Structure: category -> priority + keywords list")


def export_to_csv(keyword_stats: Dict, filename: str = "keyword_coverage.csv"):
    """
    Export keyword coverage stats to CSV
    """
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Keyword',
            'Category',
            'Priority',
            'Document Count',
            'Chunk Count',
            'Documents'
        ])
        
        # Sort by priority first, then document count
        priority_order = {'critical': 0, 'high': 1, 'medium': 2}
        sorted_stats = sorted(
            keyword_stats.items(),
            key=lambda x: (priority_order.get(x[1].get('priority', 'medium'), 3), -x[1]['doc_count'])
        )
        
        # Write rows
        for keyword, stats in sorted_stats:
            writer.writerow([
                keyword,
                stats['category'],
                stats.get('priority', 'medium'),
                stats['doc_count'],
                stats['chunk_count'],
                '; '.join(stats['documents'])
            ])


def find_keyword_gaps(
    keywords_file: str = "keywords.yaml",
    index_dir: str = "./universal_index",
    threshold: int = 3
) -> List[str]:
    """
    Find keywords that appear in very few documents (potential gaps)
    
    Args:
        keywords_file: Path to keywords.yaml
        index_dir: Path to index
        threshold: Report keywords found in fewer than this many docs
        
    Returns:
        List of keywords with low coverage
    """
    print("\n" + "="*70)
    print(f"FINDING KEYWORD GAPS (threshold: <{threshold} documents)")
    print("="*70)
    
    # Run coverage analysis (without showing all docs)
    stats = analyze_keyword_coverage(
        keywords_file=keywords_file,
        index_dir=index_dir,
        min_docs=0,  # Show all
        show_docs=False,
        export_csv=False
    )
    
    # Find gaps
    gaps = []
    for keyword, data in stats.items():
        if data['doc_count'] < threshold:
            gaps.append({
                'keyword': keyword,
                'category': data['category'],
                'priority': data.get('priority', 'medium'),
                'doc_count': data['doc_count'],
                'documents': data['documents']
            })
    
    # Sort by priority first, then doc count (ascending)
    priority_order = {'critical': 0, 'high': 1, 'medium': 2}
    gaps.sort(key=lambda x: (priority_order.get(x['priority'], 3), x['doc_count']))
    
    print("\n" + "="*90)
    print(f"KEYWORDS WITH LOW COVERAGE (<{threshold} docs)")
    print("="*90)
    
    critical_gaps = [g for g in gaps if g['priority'] == 'critical']
    high_gaps = [g for g in gaps if g['priority'] == 'high']
    other_gaps = [g for g in gaps if g['priority'] not in ['critical', 'high']]
    
    if critical_gaps:
        print(f"\n🔴 CRITICAL PRIORITY GAPS ({len(critical_gaps)} keywords) - URGENT!")
        for gap in critical_gaps:
            print(f"\n❌ {gap['keyword']} (category: {gap['category']})")
            print(f"   Found in {gap['doc_count']} document(s)")
            if gap['documents']:
                print(f"   Documents: {', '.join(gap['documents'])}")
            else:
                print(f"   ⚠ NOT FOUND IN ANY DOCUMENTS")
    
    if high_gaps:
        print(f"\n🟡 HIGH PRIORITY GAPS ({len(high_gaps)} keywords)")
        for gap in high_gaps:
            print(f"\n❌ {gap['keyword']} (category: {gap['category']})")
            print(f"   Found in {gap['doc_count']} document(s)")
            if gap['documents']:
                print(f"   Documents: {', '.join(gap['documents'])}")
            else:
                print(f"   ⚠ NOT FOUND IN ANY DOCUMENTS")
    
    if other_gaps:
        print(f"\n⚪ MEDIUM/OTHER PRIORITY ({len(other_gaps)} keywords)")
        for gap in other_gaps[:10]:  # Show first 10
            print(f"\n❌ {gap['keyword']} (category: {gap['category']})")
            print(f"   Found in {gap['doc_count']} document(s)")
            if gap['documents']:
                print(f"   Documents: {', '.join(gap['documents'])}")
        if len(other_gaps) > 10:
            print(f"\n   ... and {len(other_gaps) - 10} more medium priority gaps")
    
    print(f"\n📊 Summary: {len(gaps)} keywords below threshold")
    
    return [g['keyword'] for g in gaps]


def compare_keyword_coverage(
    keywords_file: str = "keywords.yaml",
    index1_dir: str = "./universal_index",
    index2_dir: str = "./universal_index_v2"
):
    """
    Compare keyword coverage between two indexes
    Useful for seeing if re-indexing improved coverage
    """
    print("\n" + "="*70)
    print("COMPARING KEYWORD COVERAGE ACROSS INDEXES")
    print("="*70)
    
    # Analyze both indexes
    print(f"\n📊 Index 1: {index1_dir}")
    stats1 = analyze_keyword_coverage(
        keywords_file=keywords_file,
        index_dir=index1_dir,
        show_docs=False,
        export_csv=False
    )
    
    print(f"\n\n📊 Index 2: {index2_dir}")
    stats2 = analyze_keyword_coverage(
        keywords_file=keywords_file,
        index_dir=index2_dir,
        show_docs=False,
        export_csv=False
    )
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    improved = []
    degraded = []
    unchanged = []
    
    for keyword in stats1.keys():
        count1 = stats1[keyword]['doc_count']
        count2 = stats2[keyword]['doc_count']
        
        if count2 > count1:
            improved.append({
                'keyword': keyword,
                'old': count1,
                'new': count2,
                'delta': count2 - count1
            })
        elif count2 < count1:
            degraded.append({
                'keyword': keyword,
                'old': count1,
                'new': count2,
                'delta': count2 - count1
            })
        else:
            unchanged.append(keyword)
    
    # Sort by improvement/degradation
    improved.sort(key=lambda x: x['delta'], reverse=True)
    degraded.sort(key=lambda x: x['delta'])
    
    print(f"\n✅ Improved coverage: {len(improved)} keywords")
    if improved:
        print("\nTop improvements:")
        for item in improved[:10]:
            print(f"  {item['keyword']}: {item['old']} → {item['new']} (+{item['delta']} docs)")
    
    print(f"\n⚠ Degraded coverage: {len(degraded)} keywords")
    if degraded:
        print("\nTop degradations:")
        for item in degraded[:10]:
            print(f"  {item['keyword']}: {item['old']} → {item['new']} ({item['delta']} docs)")
    
    print(f"\n➡️ Unchanged: {len(unchanged)} keywords")
    
    # Overall summary
    avg_docs_1 = sum(s['doc_count'] for s in stats1.values()) / len(stats1)
    avg_docs_2 = sum(s['doc_count'] for s in stats2.values()) / len(stats2)
    
    print(f"\n📈 Overall average docs per keyword:")
    print(f"  Index 1: {avg_docs_1:.1f}")
    print(f"  Index 2: {avg_docs_2:.1f}")
    print(f"  Change: {avg_docs_2 - avg_docs_1:+.1f} ({(avg_docs_2/avg_docs_1 - 1)*100:+.1f}%)")


def search_keyword_contexts(
    keyword: str,
    index_dir: str = "./universal_index",
    max_results: int = 5
):
    """
    Show contexts where a specific keyword appears
    Useful for understanding how a term is used
    """
    print("\n" + "="*70)
    print(f"CONTEXTS FOR KEYWORD: '{keyword}'")
    print("="*70)
    
    # Load index
    try:
        with open(Path(index_dir) / "metadata.pkl", 'rb') as f:
            metadata_list = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Index not found at: {index_dir}")
        return
    
    # Find matches
    keyword_lower = keyword.lower()
    keyword_pattern = re.compile(r'\b' + re.escape(keyword_lower) + r'\b', re.IGNORECASE)
    
    matches = []
    for m in metadata_list:
        text = m.get('text', '')
        if keyword_pattern.search(text.lower()):
            # Find the specific sentence with the keyword
            sentences = text.split('.')
            context_sentences = []
            for sent in sentences:
                if keyword_pattern.search(sent.lower()):
                    context_sentences.append(sent.strip())
            
            matches.append({
                'file': m['source_file'],
                'chunk_idx': m.get('chunk_index', 0),
                'full_text': text,
                'context_sentences': context_sentences
            })
    
    if not matches:
        print(f"\n❌ Keyword '{keyword}' not found in any documents")
        return
    
    print(f"\n✓ Found {len(matches)} occurrences across {len(set(m['file'] for m in matches))} documents")
    print(f"\nShowing first {max_results} contexts:\n")
    
    for i, match in enumerate(matches[:max_results], 1):
        print(f"{i}. {match['file']} (chunk {match['chunk_idx']})")
        print(f"   Context:")
        for sent in match['context_sentences'][:2]:  # Show first 2 sentences
            # Highlight the keyword
            highlighted = re.sub(
                keyword_pattern,
                lambda m: f">>>{m.group()}<<<",
                sent
            )
            print(f"   • {highlighted}")
        print()


# Update the original functions from previous file
def verify_index_coverage(index_dir: str = "./universal_index", doc_dir: str = "./your_docs") -> Dict[str, any]:
    """Check which documents are actually indexed vs what's in the directory"""
    print("\n" + "="*70)
    print("INDEX COVERAGE VERIFICATION")
    print("="*70)
    all_docs = []
    for pattern in ['**/*.pdf', '**/*.txt', '**/*.md', '**/*.docx', '**/*.doc']:
        all_docs.extend(Path(doc_dir).glob(pattern))
    print(f"\nDocuments in directory: {len(all_docs)}")
    try:
        with open(Path(index_dir) / "metadata.pkl", 'rb') as f:
            indexed_metadata = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Index not found at: {index_dir}")
        return {'error': 'Index not found'}
    indexed_files = set([m['source_file'] for m in indexed_metadata])
    print(f"Documents in index: {len(indexed_files)}")
    missing = []
    for doc in all_docs:
        if doc.name not in indexed_files:
            missing.append(doc.name)
    doc_names = set([d.name for d in all_docs])
    orphaned = [f for f in indexed_files if f not in doc_names]
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
    return {'total_docs': len(all_docs), 'indexed_docs': len(indexed_files), 'missing': missing, 'orphaned': orphaned, 'coverage_pct': coverage_pct}


def diagnose_index(index_dir: str = "./universal_index") -> Dict[str, any]:
    """Comprehensive analysis of index contents"""
    print("\n" + "="*70)
    print("INDEX DIAGNOSTIC ANALYSIS")
    print("="*70)
    try:
        with open(Path(index_dir) / "metadata.pkl", 'rb') as f:
            metadata_list = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Index not found at: {index_dir}")
        return {'error': 'Index not found'}
    print(f"\n📊 Basic Statistics:")
    print(f"  Total chunks: {len(metadata_list)}")
    file_types = Counter([m.get('file_type', 'unknown') for m in metadata_list])
    print(f"\n📁 File types:")
    for ft, count in file_types.most_common():
        print(f"  {ft}: {count} chunks")
    unique_files = set([m['source_file'] for m in metadata_list])
    print(f"\n📄 Unique files: {len(unique_files)}")
    chunks_per_file = Counter([m['source_file'] for m in metadata_list])
    avg_chunks = sum(chunks_per_file.values()) / len(chunks_per_file)
    print(f"  Average chunks per file: {avg_chunks:.1f}")
    print(f"  Max chunks in one file: {chunks_per_file.most_common(1)[0][1]}")
    print(f"\n📚 Files with most chunks:")
    for filename, count in chunks_per_file.most_common(5):
        print(f"  {filename}: {count} chunks")
    chunk_lengths = [len(m.get('text', '')) for m in metadata_list]
    avg_length = sum(chunk_lengths) / len(chunk_lengths)
    print(f"\n📏 Chunk sizes:")
    print(f"  Average: {avg_length:.0f} characters")
    print(f"  Min: {min(chunk_lengths)}")
    print(f"  Max: {max(chunk_lengths)}")
    req_keywords = ['requirement', 'spec', 'specification', 'rfp', 'sow', 'functional']
    req_files = set([m['source_file'] for m in metadata_list if any(kw in m['source_file'].lower() for kw in req_keywords)])
    print(f"\n📋 Requirements documents:")
    print(f"  Files with requirements keywords in name: {len(req_files)}")
    if req_files:
        for f in list(req_files)[:5]:
            chunk_count = chunks_per_file[f]
            print(f"  - {f} ({chunk_count} chunks)")
        if len(req_files) > 5:
            print(f"  ... and {len(req_files) - 5} more")
    req_lang_keywords = ['shall', 'must', 'should', 'required', 'mandatory']
    chunks_with_req_lang = 0
    for m in metadata_list:
        text_lower = m.get('text', '').lower()
        if any(kw in text_lower for kw in req_lang_keywords):
            chunks_with_req_lang += 1
    print(f"\n🔍 Requirements language:")
    print(f"  Chunks containing 'shall/must/should': {chunks_with_req_lang}")
    print(f"  Percentage: {chunks_with_req_lang/len(metadata_list)*100:.1f}%")
    return {'total_chunks': len(metadata_list), 'unique_files': len(unique_files), 'file_types': dict(file_types), 'avg_chunk_size': avg_length, 'requirements_files': len(req_files), 'chunks_with_req_lang': chunks_with_req_lang}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'keywords':
            # Analyze keyword coverage
            keywords_file = sys.argv[2] if len(sys.argv) > 2 else "keywords.yaml"
            analyze_keyword_coverage(keywords_file=keywords_file, show_docs=True, export_csv=True)
        
        elif command == 'gaps':
            # Find keyword gaps
            keywords_file = sys.argv[2] if len(sys.argv) > 2 else "keywords.yaml"
            threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 3
            find_keyword_gaps(keywords_file=keywords_file, threshold=threshold)
        
        elif command == 'compare-keywords':
            # Compare keyword coverage between indexes
            keywords_file = sys.argv[2] if len(sys.argv) > 2 else "keywords.yaml"
            index1 = sys.argv[3] if len(sys.argv) > 3 else "./universal_index"
            index2 = sys.argv[4] if len(sys.argv) > 4 else "./universal_index_v2"
            compare_keyword_coverage(keywords_file, index1, index2)
        
        elif command == 'search':
            # Search for specific keyword contexts
            if len(sys.argv) > 2:
                keyword = sys.argv[2]
                search_keyword_contexts(keyword)
            else:
                print("Usage: python index_diagnostics_enhanced.py search <keyword>")
        
        elif command == 'verify':
            verify_index_coverage()
        
        elif command == 'diagnose':
            diagnose_index()
        
        else:
            print("Unknown command. Use: keywords, gaps, compare-keywords, search, verify, or diagnose")
    
    else:
        # Show usage
        print("\nUsage:")
        print("  python index_diagnostics_enhanced.py keywords [keywords.yaml]")
        print("  python index_diagnostics_enhanced.py gaps [keywords.yaml] [threshold]")
        print("  python index_diagnostics_enhanced.py compare-keywords [keywords.yaml] [index1] [index2]")
        print("  python index_diagnostics_enhanced.py search <keyword>")
        print("  python index_diagnostics_enhanced.py verify")
        print("  python index_diagnostics_enhanced.py diagnose")
        print("\nExamples:")
        print("  python index_diagnostics_enhanced.py keywords my_terms.yaml")
        print("  python index_diagnostics_enhanced.py gaps keywords.yaml 5")
        print("  python index_diagnostics_enhanced.py search 'wire transfer'")
