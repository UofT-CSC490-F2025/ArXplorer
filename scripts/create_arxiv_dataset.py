"""Create a JSONL dataset from Kaggle arXiv data.

This script reads from the Kaggle arXiv dataset and creates a JSONL file
in the same format as arxiv_300k.jsonl.

Default behavior:
- Filters to AI/ML categories: cs.ai, cs.lg, stat.ml, cs.cl, cs.cv, cs.ir, cs.ne, cs.ds
- Outputs JSONL with fields: id, title, abstract, categories, authors, primary_category, published_date, updated_date

Usage:
    # Create 300k dataset with default AI/ML categories
    python scripts/create_arxiv_dataset.py --limit 300000
    
    # Create dataset with custom categories
    python scripts/create_arxiv_dataset.py --categories cs.ai,cs.lg,stat.ml --limit 100000
    
    # Create dataset with all categories (no filter)
    python scripts/create_arxiv_dataset.py --no-filter --limit 500000
    
    # Custom input/output paths
    python scripts/create_arxiv_dataset.py --input data/kaggle_arxiv/kaggle_arxiv.json --output data/arxiv_custom.jsonl
"""

import argparse
import json
import gzip
import io
import sys
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple


# Default AI/ML categories
DEFAULT_CATEGORIES = [
    "cs.ai", "cs.lg", "stat.ml", "cs.cl", "cs.cv", "cs.ir", "cs.ne", "cs.ds"
]


def normalize_text(text: Optional[str]) -> str:
    """Normalize text by stripping and collapsing whitespace."""
    if not text:
        return ""
    return " ".join((text or "").strip().split())


def extract_categories(data: Dict) -> List[str]:
    """Extract category list from various schema formats."""
    categories = data.get("categories")
    if isinstance(categories, str):
        # Space or comma-delimited string
        tokens = categories.replace(",", " ").replace(";", " ").split()
        return [t.strip().lower() for t in tokens if t.strip()]
    elif isinstance(categories, list):
        # List of strings or objects
        result = []
        for cat in categories:
            if isinstance(cat, str):
                result.append(cat.strip().lower())
            elif isinstance(cat, dict) and "term" in cat:
                result.append(cat["term"].strip().lower())
        return result
    return []


def extract_authors(data: Dict) -> List[str]:
    """Extract author list from various schema formats."""
    authors = data.get("authors")
    
    # Check for authors_parsed first (arXiv format: [[Last, First, Middle], ...])
    if "authors_parsed" in data:
        parsed = data["authors_parsed"]
        if isinstance(parsed, list):
            names = []
            for parts in parsed:
                if isinstance(parts, (list, tuple)):
                    name = " ".join(str(p) for p in parts if p)
                    if name.strip():
                        names.append(name.strip())
            if names:
                return names
    
    # Handle authors field
    if isinstance(authors, str):
        # String format: split by comma or semicolon
        if ";" in authors:
            return [a.strip() for a in authors.split(";") if a.strip()]
        elif "," in authors:
            return [a.strip() for a in authors.split(",") if a.strip()]
        else:
            return [authors.strip()] if authors.strip() else []
    elif isinstance(authors, list):
        # List format
        result = []
        for author in authors:
            if isinstance(author, str):
                result.append(author.strip())
            elif isinstance(author, dict):
                # Check for name field
                for key in ["name", "full_name", "author"]:
                    if key in author:
                        result.append(str(author[key]).strip())
                        break
        return result
    
    return []


def extract_dates(data: Dict) -> Tuple[str, str]:
    """Extract published and updated dates."""
    published = ""
    updated = ""
    
    # Try versions array first (most reliable)
    versions = data.get("versions")
    if isinstance(versions, list) and versions:
        dates = []
        for v in versions:
            if isinstance(v, dict) and "created" in v:
                dates.append(v["created"].strip())
        if dates:
            published = dates[0]
            updated = dates[-1]
            return published, updated
    
    # Fallback to direct date fields
    for key in ["created", "submitted_date", "publish_time", "published"]:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            published = val.strip()
            break
    
    for key in ["updated", "update_date", "last_updated"]:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            updated = val.strip()
            break
    
    if not updated:
        updated = published
    
    return published, updated


def matches_categories(categories: List[str], filter_categories: Optional[List[str]]) -> bool:
    """Check if paper matches any of the filter categories."""
    if filter_categories is None:
        return True  # No filter
    
    if not categories:
        return False
    
    for cat in categories:
        for filter_cat in filter_categories:
            if cat.startswith(filter_cat):
                return True
    
    return False


def stream_json_array(fp: io.TextIOBase) -> Iterator[Dict]:
    """Stream objects from a large JSON array file without loading everything into memory."""
    # Skip to opening bracket
    ch = fp.read(1)
    while ch and ch.isspace():
        ch = fp.read(1)
    
    if ch != '[':
        return
    
    inside_string = False
    escape = False
    depth = 0
    buf = []
    
    while True:
        ch = fp.read(1)
        if not ch:
            break
        
        if inside_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                inside_string = False
            continue
        
        # Not inside string
        if ch.isspace():
            if depth > 0:
                buf.append(ch)
            continue
        
        if depth == 0:
            # Array level
            if ch == ']':
                break
            elif ch == ',':
                continue
            elif ch == '{':
                depth = 1
                buf = ['{']
            continue
        
        # Inside object
        if ch == '"':
            inside_string = True
            buf.append(ch)
        elif ch == '{':
            depth += 1
            buf.append(ch)
        elif ch == '}':
            depth -= 1
            buf.append(ch)
            if depth == 0:
                # Complete object
                try:
                    obj = json.loads(''.join(buf))
                    yield obj
                except Exception:
                    pass
                buf = []
        else:
            buf.append(ch)


def load_kaggle_arxiv(
    input_path: str,
    filter_categories: Optional[List[str]] = None
) -> Iterator[Dict]:
    """
    Load papers from Kaggle arXiv dataset.
    
    Args:
        input_path: Path to input file (JSON or JSONL)
        filter_categories: Optional list of categories to filter by
        
    Yields:
        Dict with standardized fields
    """
    opener = gzip.open if input_path.endswith('.gz') else open
    
    # Check if it's a JSON array or JSONL
    with opener(input_path, 'rt', encoding='utf-8', errors='ignore') as fp:
        # Peek at first non-whitespace character
        ch = fp.read(1)
        while ch and ch.isspace():
            ch = fp.read(1)
        is_array = (ch == '[')
    
    # Process file
    with opener(input_path, 'rt', encoding='utf-8', errors='ignore') as fp:
        if is_array:
            # Stream JSON array
            for obj in stream_json_array(fp):
                if not isinstance(obj, dict):
                    continue
                
                # Extract fields
                paper_id = obj.get("id", "").strip()
                title = normalize_text(obj.get("title"))
                abstract = normalize_text(obj.get("abstract"))
                
                if not paper_id or (not title and not abstract):
                    continue
                
                categories = extract_categories(obj)
                
                # Filter by categories
                if not matches_categories(categories, filter_categories):
                    continue
                
                authors = extract_authors(obj)
                primary_category = categories[0] if categories else ""
                published_date, updated_date = extract_dates(obj)
                
                yield {
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "categories": categories,
                    "authors": authors,
                    "primary_category": primary_category,
                    "published_date": published_date,
                    "updated_date": updated_date
                }
        else:
            # JSONL format
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                
                if not isinstance(obj, dict):
                    continue
                
                # Extract fields
                paper_id = obj.get("id", "").strip()
                title = normalize_text(obj.get("title"))
                abstract = normalize_text(obj.get("abstract"))
                
                if not paper_id or (not title and not abstract):
                    continue
                
                categories = extract_categories(obj)
                
                # Filter by categories
                if not matches_categories(categories, filter_categories):
                    continue
                
                authors = extract_authors(obj)
                primary_category = categories[0] if categories else ""
                published_date, updated_date = extract_dates(obj)
                
                yield {
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "categories": categories,
                    "authors": authors,
                    "primary_category": primary_category,
                    "published_date": published_date,
                    "updated_date": updated_date
                }


def create_dataset(
    input_path: str,
    output_path: str,
    limit: int,
    filter_categories: Optional[List[str]] = None
) -> int:
    """
    Create JSONL dataset from Kaggle arXiv data.
    
    Args:
        input_path: Input file path
        output_path: Output JSONL file path
        limit: Maximum number of papers to write
        filter_categories: Optional category filter
        
    Returns:
        Number of papers written
    """
    print(f"\n{'='*60}")
    print("CREATING ARXIV DATASET")
    print(f"{'='*60}\n")
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Limit: {limit:,}")
    
    if filter_categories:
        print(f"Categories: {', '.join(filter_categories)}")
    else:
        print("Categories: ALL (no filter)")
    
    print()
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process papers
    seen_ids = set()
    count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for paper in load_kaggle_arxiv(input_path, filter_categories):
            # Deduplicate
            if paper["id"] in seen_ids:
                continue
            seen_ids.add(paper["id"])
            
            # Write JSONL
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')
            count += 1
            
            if count % 10000 == 0:
                print(f"Processed {count:,} papers...")
            
            if count >= limit:
                break
    
    print(f"\n✓ Created dataset with {count:,} papers")
    print(f"✓ Saved to: {output_path}")
    
    # Show category breakdown
    if count > 0:
        print("\nCategory statistics:")
        category_counts = {}
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                for cat in data.get("categories", []):
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        
        for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cat}: {cnt:,}")
    
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Create JSONL dataset from Kaggle arXiv data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 300k dataset with default AI/ML categories
  python scripts/create_arxiv_dataset.py --limit 300000
  
  # Create dataset with custom categories
  python scripts/create_arxiv_dataset.py --categories cs.ai,cs.lg --limit 100000
  
  # Create dataset with all categories (no filter)
  python scripts/create_arxiv_dataset.py --no-filter --limit 500000
  
  # Custom paths
  python scripts/create_arxiv_dataset.py \\
    --input data/kaggle_arxiv/kaggle_arxiv.json \\
    --output data/arxiv_custom.jsonl \\
    --limit 200000
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/kaggle_arxiv/kaggle_arxiv.json",
        help="Input Kaggle arXiv file (JSON or JSONL)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/arxiv_300k.jsonl",
        help="Output JSONL file path"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=300000,
        help="Maximum number of papers to include"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to include (e.g., cs.ai,cs.lg,stat.ml)"
    )
    
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Include all categories (no filtering)"
    )
    
    args = parser.parse_args()
    
    # Determine category filter
    if args.no_filter:
        filter_categories = None
    elif args.categories:
        filter_categories = [c.strip().lower() for c in args.categories.split(',') if c.strip()]
    else:
        filter_categories = DEFAULT_CATEGORIES
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    try:
        create_dataset(
            input_path=args.input,
            output_path=args.output,
            limit=args.limit,
            filter_categories=filter_categories
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
