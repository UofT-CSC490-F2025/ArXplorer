"""
Generate canonical query variants from canon_papers_60.csv.
Creates test queries with misspellings, keywords, and temporal variants.
"""

import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import extract_arxiv_id, load_arxiv_metadata


# Query templates
QUERY_TEMPLATES = [
    "original {keyword} paper",
    "{keyword} paper",
    "{keyword} {year}",
    "{keyword} {year} paper",
    "seminal {keyword} work",
    "{keyword} research paper"
]


def introduce_typo(text: str) -> str:
    """
    Introduce a random typo into the text.
    
    Strategies:
    - Character swap (transposition)
    - Character deletion
    - Character duplication
    """
    if len(text) < 3:
        return text
    
    words = text.split()
    if not words:
        return text
    
    # Pick a random word to modify (prefer longer words)
    word_to_modify = random.choice([w for w in words if len(w) > 3] or words)
    word_idx = words.index(word_to_modify)
    
    typo_type = random.choice(['swap', 'delete', 'duplicate'])
    chars = list(word_to_modify)
    
    if typo_type == 'swap' and len(chars) > 1:
        # Swap two adjacent characters
        idx = random.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif typo_type == 'delete' and len(chars) > 1:
        # Delete a character
        idx = random.randint(0, len(chars) - 1)
        chars.pop(idx)
    elif typo_type == 'duplicate' and len(chars) > 0:
        # Duplicate a character
        idx = random.randint(0, len(chars) - 1)
        chars.insert(idx, chars[idx])
    
    words[word_idx] = ''.join(chars)
    return ' '.join(words)


def generate_query_variants(
    arxiv_id: str,
    keywords: List[str],
    title: str,
    year: str,
    num_variants: int = 2
) -> List[Tuple[str, str]]:
    """
    Generate query variants for a canonical paper.
    
    Args:
        arxiv_id: Paper's arXiv ID
        keywords: List of keywords from CSV
        title: Paper title
        year: Publication year
        num_variants: Total number of query variants to generate
        
    Returns:
        List of (query, variant_type) tuples
    """
    all_variants = []
    
    # 1. Misspelled title
    misspelled_title = introduce_typo(title)
    all_variants.append((misspelled_title, "title_typo"))
    
    # 2. Exact title
    all_variants.append((title, "exact_title"))
    
    # 3. Keyword-based templates
    if keywords:
        for keyword in keywords:
            for template in QUERY_TEMPLATES:
                try:
                    if '{year}' in template and year:
                        query = template.format(keyword=keyword, year=year)
                    else:
                        query = template.format(keyword=keyword)
                    all_variants.append((query, f"template_{template}"))
                except KeyError:
                    continue
    
    # Randomly select num_variants total queries from all options
    if len(all_variants) > num_variants:
        selected_variants = random.sample(all_variants, num_variants)
    else:
        selected_variants = all_variants
    
    return selected_variants


def main():
    """Generate canonical queries from canon_papers_60.csv"""
    
    # Paths
    eval_dir = Path(__file__).parent.parent
    data_dir = eval_dir / "data"
    canon_csv = data_dir / "canon_papers_60.csv"
    arxiv_jsonl = eval_dir.parent / "data" / "arxiv_300k.jsonl"
    output_file = data_dir / "canonical_queries.json"
    
    print("=" * 60)
    print("Generating Canonical Query Variants")
    print("=" * 60)
    print()
    
    # Check files exist
    if not canon_csv.exists():
        print(f"✗ Canon papers file not found: {canon_csv}")
        return
    
    if not arxiv_jsonl.exists():
        print(f"✗ ArXiv JSONL file not found: {arxiv_jsonl}")
        return
    
    # Load arXiv metadata
    metadata = load_arxiv_metadata(str(arxiv_jsonl))
    print()
    
    # Parse canon_papers_60.csv
    print(f"Reading canonical papers from {canon_csv.name}...")
    canonical_queries = []
    papers_processed = 0
    papers_missing = 0
    
    with open(canon_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = row['Entry']
            url = row['ID']
            keywords_str = row['Keywords']
            
            # Extract arXiv ID
            arxiv_id = extract_arxiv_id(url)
            
            # Parse keywords (comma-separated)
            keywords = [k.strip() for k in keywords_str.split(',')]
            
            # Get metadata
            if arxiv_id not in metadata:
                print(f"  ⚠ Paper {entry} ({arxiv_id}) not found in metadata")
                papers_missing += 1
                continue
            
            paper_meta = metadata[arxiv_id]
            title = paper_meta['title']
            year = paper_meta['year'] or "2020"  # Default year if missing
            
            # Generate 2 total variants per paper (from all options including title/typo)
            variants = generate_query_variants(
                arxiv_id, 
                keywords, 
                title, 
                year,
                num_variants=2
            )
            
            # Add to canonical queries list
            for query, variant_type in variants:
                canonical_queries.append({
                    'entry': entry,
                    'target_id': arxiv_id,
                    'target_title': title,
                    'query': query,
                    'variant_type': variant_type,
                    'keywords': keywords,
                    'year': year
                })
            
            papers_processed += 1
            print(f"  [{entry}] {title[:60]}... → {len(variants)} variants")
    
    print()
    print(f"✓ Processed {papers_processed} papers")
    print(f"  Missing from metadata: {papers_missing}")
    print(f"  Total queries generated: {len(canonical_queries)}")
    print()
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(canonical_queries, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved canonical queries to: {output_file}")
    print()
    
    # Print summary statistics
    variant_counts = {}
    for q in canonical_queries:
        vtype = q['variant_type']
        variant_counts[vtype] = variant_counts.get(vtype, 0) + 1
    
    print("Query variant breakdown:")
    for vtype, count in sorted(variant_counts.items()):
        print(f"  {vtype}: {count}")
    print()


if __name__ == "__main__":
    main()
