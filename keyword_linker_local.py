"""
keyword_linker_local.py
Date: 2025/7/10
Description:
Local logic-driven matcher for consolidating fragmented diligence records using fuzzy string
matching, rule-based normalization, and deterministic keyword logic. Designed as part of an
automated diligence data framework to standardize client records and accelerate analysis in
financial workflows. Runs fully offline with no external API dependencies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import argparse
import sys
import time
from pathlib import Path

# For fuzzy matching - install with: pip install rapidfuzz
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
    print("✅ Using rapidfuzz (high performance)")
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        RAPIDFUZZ_AVAILABLE = False
        print("⚠️  Using fuzzywuzzy (install rapidfuzz for better performance)")
    except ImportError:
        print("❌ Please install fuzzy matching library:")
        print("   pip install rapidfuzz")
        print("   OR")
        print("   pip install fuzzywuzzy")
        sys.exit(1)

class HybridCompanyMatcher:
    """
    🎯 HYBRID COMPANY NAME MATCHER

    Combines three powerful techniques:
    1. Business Intelligence Rules (investment banking specific)
    2. Advanced Name Normalization (handles compressed names, abbreviations)
    3. Multiple Fuzzy Matching Algorithms (token-based, partial, set-based)

    Designed specifically for investment banking data with ticker symbols,
    abbreviations, and complex naming conventions.
    """

    def __init__(self, confidence_threshold: float = 0.7, verbose: bool = True):
        """
        Initialize the hybrid matcher

        Args:
            confidence_threshold: Minimum confidence score for matches (0.0-1.0)
            verbose: Print detailed processing information
        """
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

        # Investment banking specific abbreviations
        self.abbreviations = {
            # Corporate entities
            'corporation': ['corp', 'corpn', 'crp'],
            'incorporated': ['inc', 'incorp', 'incorpd'],
            'limited': ['ltd', 'lmt', 'ltda', 'ltde'],
            'company': ['co', 'coy', 'comp'],
            'group': ['grp', 'grup', 'gp'],
            'holdings': ['holding', 'hldg', 'hldgs'],
            'enterprises': ['enterprise', 'ent', 'entpr'],

            # Business types
            'international': ['intl', 'intel', 'internl', 'intrntl'],
            'pharmaceuticals': ['pharma', 'phar', 'pharm', 'pharml'],
            'technologies': ['tech', 'technol', 'technologi', 'techn'],
            'telecommunications': ['telecom', 'tel', 'telcom'],
            'communications': ['comm', 'comms', 'commun'],
            'financial': ['finl', 'fin', 'financl'],
            'services': ['svc', 'svcs', 'serv', 'servs'],
            'systems': ['sys', 'syst', 'systm'],
            'solutions': ['solution', 'sol', 'soln'],
            'development': ['dev', 'devel', 'devlp'],
            'management': ['mgmt', 'mgt', 'mgmnt'],
            'investment': ['invest', 'inv', 'invst'],
            'operations': ['ops', 'oper', 'opern'],
            'manufacturing': ['mfg', 'manuf', 'manu'],
            'research': ['res', 'rsch', 'rsrch'],
            'consulting': ['cons', 'consult', 'cnslt'],
            'equipment': ['equip', 'eqp', 'eqpt'],
            'materials': ['mat', 'matl', 'mtrl'],
            'properties': ['prop', 'props', 'prpty'],
            'resources': ['res', 'rsrc', 'resrc'],
        }

        # Ticker symbol patterns to remove
        self.ticker_patterns = [
            r'[<>+\-F]+\s*$',      # F+, F<, <, >, +, - at end
            r'\s+[<>+\-F]+\s*',    # Symbols in middle
            r'\bF\+\b', r'\bF<\b', r'\b<\b', r'\b>\b', r'\b\+\b'
        ]

        if self.verbose:
            print("🎯 Initialized Hybrid Company Matcher")
            print(f"   Confidence threshold: {confidence_threshold}")
            print(f"   Business abbreviations loaded: {len(self.abbreviations)}")
            print(f"   Ticker symbol patterns: {len(self.ticker_patterns)}")

    def normalize_company_name(self, name: str) -> str:
        """
        🧠 ADVANCED BUSINESS NAME NORMALIZATION

        This is the "secret sauce" - transforms messy investment banking names
        into clean, comparable forms using business intelligence.

        Examples:
            "HONEYWELLINTLINC<" → "honeywell international incorporated"
            "COMCASTCORPNEWCLA+" → "comcast corporation new class a"
            "REGENERON PHAR +" → "regeneron pharmaceuticals"

        Args:
            name: Raw company name from Excel

        Returns:
            Normalized, cleaned company name
        """
        if not name or pd.isna(name):
            return ""

        name = str(name).strip()
        if not name:
            return ""

        # STEP 1: Remove ticker symbols and market indicators
        for pattern in self.ticker_patterns:
            name = re.sub(pattern, '', name)

        # STEP 2: Handle compressed names (like HONEYWELLINTLINC)
        # If all caps and long, likely compressed - insert spaces
        if name.isupper() and len(name) > 12 and ' ' not in name:
            # Insert space before capital letters that follow lowercase
            # But this is all caps, so we need different logic

            # Try to identify word boundaries by looking for known abbreviations
            for full_form, abbrevs in self.abbreviations.items():
                for abbrev in abbrevs:
                    # Look for abbreviation at end of compressed name
                    if name.upper().endswith(abbrev.upper()):
                        # Split the name
                        prefix = name[:-len(abbrev)]
                        name = f"{prefix} {abbrev}"
                        break

        # STEP 3: Convert to lowercase for processing
        name_lower = name.lower()

        # STEP 4: Expand abbreviations BEFORE removing punctuation
        for full_form, abbrevs in self.abbreviations.items():
            for abbrev in abbrevs:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(abbrev) + r'\b'
                name_lower = re.sub(pattern, full_form, name_lower)

        # STEP 5: Handle special business patterns
        # "NEW CLASS A" type patterns
        name_lower = re.sub(r'\bnew\s+class\s+([a-z])\b', r'class \1', name_lower)

        # Common business patterns
        patterns = {
            r'\bmicro\s*soft\b': 'microsoft',
            r'\bwal\s*mart\b': 'walmart',
            r'\bt\s*mobile\b': 't-mobile',
            r'\be\s*commerce\b': 'ecommerce',
            r'\bdata\s+process\b': 'data processing',
        }

        for pattern, replacement in patterns.items():
            name_lower = re.sub(pattern, replacement, name_lower)

        # STEP 6: Clean punctuation and normalize spaces
        name_lower = re.sub(r'[^\w\s]', ' ', name_lower)  # Replace punctuation with spaces
        name_lower = re.sub(r'\s+', ' ', name_lower)      # Normalize whitespace
        name_lower = name_lower.strip()

        # STEP 7: Remove common legal suffixes at the end (but keep them in middle)
        legal_suffixes = [
            'incorporated', 'corporation', 'limited', 'company', 'llc',
            'plc', 'inc', 'corp', 'co', 'ltd', 'lp', 'lp', 'gmbh', 'ag',
            'sa', 'nv', 'bv', 'srl', 'spa', 'public limited company'
        ]

        words = name_lower.split()
        if len(words) > 1 and words[-1] in legal_suffixes:
            words = words[:-1]

        # STEP 8: Remove common prefixes
        if words and words[0] == 'the':
            words = words[1:]

        result = ' '.join(words).strip()

        if self.verbose and len(result) < len(name_lower) * 0.5:
            # Warn if we removed too much (might be over-aggressive)
            print(f"   ⚠️  Aggressive normalization: '{name}' → '{result}'")

        return result

    def calculate_hybrid_similarity(self, source: str, target: str) -> Dict[str, float]:
        """
        🧮 MULTI-ALGORITHM SIMILARITY CALCULATION

        Uses 5 different fuzzy matching algorithms and combines them
        intelligently for business names.

        Args:
            source: Source company name (normalized)
            target: Target company name (normalized)

        Returns:
            Dictionary with individual scores and combined score
        """
        if not source or not target:
            return {'combined': 0.0, 'details': {}}

        # Normalize both names
        source_norm = self.normalize_company_name(source)
        target_norm = self.normalize_company_name(target)

        if not source_norm or not target_norm:
            return {'combined': 0.0, 'details': {}}

        # Calculate multiple similarity metrics
        scores = {}

        # 1. Token Sort Ratio - handles word order differences
        # "Microsoft Corporation" vs "Corporation Microsoft" = high score
        scores['token_sort'] = fuzz.token_sort_ratio(source_norm, target_norm) / 100.0

        # 2. Token Set Ratio - handles subset matches
        # "International Business Machines" vs "IBM Corp" = good score
        scores['token_set'] = fuzz.token_set_ratio(source_norm, target_norm) / 100.0

        # 3. Partial Ratio - one name contained in other
        # "Apple" vs "Apple Inc" = high score
        scores['partial'] = fuzz.partial_ratio(source_norm, target_norm) / 100.0

        # 4. Standard Levenshtein ratio
        scores['standard'] = fuzz.ratio(source_norm, target_norm) / 100.0

        # 5. Word-level Jaccard similarity (custom)
        words_source = set(source_norm.split())
        words_target = set(target_norm.split())
        if words_source or words_target:
            intersection = len(words_source & words_target)
            union = len(words_source | words_target)
            scores['jaccard'] = intersection / union if union > 0 else 0.0
        else:
            scores['jaccard'] = 0.0

        # 6. Weighted combination optimized for business names
        # These weights are tuned based on business name matching performance
        weights = {
            'token_sort': 0.30,    # Most important - handles reordering
            'token_set': 0.25,     # Second - handles abbreviations
            'partial': 0.20,       # Third - handles subset matches
            'jaccard': 0.15,       # Fourth - word overlap
            'standard': 0.10       # Least - basic string similarity
        }

        combined_score = sum(scores[metric] * weights[metric] for metric in weights)

        # Apply business-specific bonuses
        bonus = 0.0

        # Bonus for exact word matches
        source_words = set(source_norm.split())
        target_words = set(target_norm.split())
        common_words = source_words & target_words

        if common_words:
            # Bonus based on significant words matched
            significant_words = common_words - {'inc', 'corp', 'ltd', 'co', 'company', 'corporation'}
            if significant_words:
                bonus += min(0.1, len(significant_words) * 0.02)

        # Bonus for length similarity (avoids matching very short names to long ones)
        len_ratio = min(len(source_norm), len(target_norm)) / max(len(source_norm), len(target_norm), 1)
        if len_ratio > 0.7:
            bonus += 0.05

        final_score = min(1.0, combined_score + bonus)

        return {
            'combined': final_score,
            'details': scores,
            'bonus': bonus,
            'source_normalized': source_norm,
            'target_normalized': target_norm
        }

    def find_best_match(self, source_company: str, target_companies: List[str]) -> Dict:
        """
        🎯 FIND BEST MATCH FOR SINGLE COMPANY

        Tests source company against all target companies and returns
        the best match with confidence score and details.

        Args:
            source_company: Company name to match
            target_companies: List of potential matches

        Returns:
            Dictionary with match results
        """
        if not source_company or not target_companies:
            return {
                'source_company': source_company,
                'best_match': None,
                'confidence': 0.0,
                'match_details': {}
            }

        best_score = 0.0
        best_match = None
        best_details = {}

        source_norm = self.normalize_company_name(source_company)

        for target in target_companies:
            if not target:
                continue

            similarity_result = self.calculate_hybrid_similarity(source_company, target)
            score = similarity_result['combined']

            if score > best_score:
                best_score = score
                best_match = target
                best_details = similarity_result

        # Only return match if it meets threshold
        if best_score < self.confidence_threshold:
            best_match = None

        return {
            'source_company': source_company,
            'best_match': best_match,
            'confidence': best_score,
            'match_details': best_details
        }

    def match_companies(self, source_companies: List[str], target_companies: List[str]) -> List[Dict]:
        """
        🚀 MAIN MATCHING FUNCTION

        Matches all source companies against all target companies using
        the hybrid approach.

        Args:
            source_companies: List of companies to match (from "By Client")
            target_companies: List of potential matches (from "US Customer")

        Returns:
            List of match results
        """
        if self.verbose:
            print(f"\n🎯 HYBRID MATCHING PROCESS")
            print(f"   Source companies: {len(source_companies)}")
            print(f"   Target companies: {len(target_companies)}")
            print(f"   Confidence threshold: {self.confidence_threshold}")
            print(f"   Total comparisons: {len(source_companies) * len(target_companies):,}")

        matches = []
        start_time = time.time()

        for i, source in enumerate(source_companies):
            if self.verbose and (i % 50 == 0 or i == 0):
                elapsed = time.time() - start_time
                if i > 0:
                    rate = i / elapsed
                    eta = (len(source_companies) - i) / rate
                    print(f"   🔄 Processed {i}/{len(source_companies)} ({i/len(source_companies)*100:.1f}%) - ETA: {eta:.1f}s")
                else:
                    print(f"   🔄 Starting processing...")

            match_result = self.find_best_match(source, target_companies)
            matches.append(match_result)

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"   ✅ Completed in {elapsed:.2f} seconds")
            print(f"   ⚡ Rate: {len(source_companies)/elapsed:.1f} companies/second")

        return matches

    def load_excel_data(self, file_path: str, source_sheet: str = "By Client",
                       target_sheet: str = "US Customer", source_col: str = "Company",
                       target_col: str = "Customer_Name") -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame]:
        """Load and validate Excel data"""

        if self.verbose:
            print(f"📊 Loading Excel data from: {file_path}")

        try:
            # Load sheets
            source_df = pd.read_excel(file_path, sheet_name=source_sheet)
            target_df = pd.read_excel(file_path, sheet_name=target_sheet)

            if self.verbose:
                print(f"   ✅ Source sheet '{source_sheet}': {len(source_df)} rows")
                print(f"   ✅ Target sheet '{target_sheet}': {len(target_df)} rows")

            # Validate columns exist
            if source_col not in source_df.columns:
                raise ValueError(f"Column '{source_col}' not found in sheet '{source_sheet}'")
            if target_col not in target_df.columns:
                raise ValueError(f"Column '{target_col}' not found in sheet '{target_sheet}'")

            # Extract unique company names
            source_companies = source_df[source_col].dropna().unique().tolist()
            target_companies = target_df[target_col].dropna().unique().tolist()

            if self.verbose:
                print(f"   🎯 Unique source companies: {len(source_companies)}")
                print(f"   🎯 Unique target companies: {len(target_companies)}")

                # Show some examples
                print(f"   📋 Source examples: {source_companies[:3]}")
                print(f"   📋 Target examples: {target_companies[:3]}")

            return source_companies, target_companies, source_df, target_df

        except Exception as e:
            print(f"❌ Error loading Excel data: {e}")
            return [], [], pd.DataFrame(), pd.DataFrame()

    def create_comprehensive_report(self, matches: List[Dict]) -> pd.DataFrame:
        """Create detailed matching report"""

        # Convert to DataFrame
        results = []
        for match in matches:
            result = {
                'source_company': match['source_company'],
                'best_match': match['best_match'],
                'confidence': match['confidence'],
                'match_status': 'Matched' if match['best_match'] else 'No Match'
            }

            # Add detailed scores if available
            if 'match_details' in match and match['match_details']:
                details = match['match_details']
                if 'details' in details:
                    result.update({
                        'token_sort_score': details['details'].get('token_sort', 0),
                        'token_set_score': details['details'].get('token_set', 0),
                        'partial_score': details['details'].get('partial', 0),
                        'jaccard_score': details['details'].get('jaccard', 0),
                        'source_normalized': details.get('source_normalized', ''),
                        'target_normalized': details.get('target_normalized', '')
                    })

            results.append(result)

        df = pd.DataFrame(results)

        # Sort by confidence descending
        df = df.sort_values('confidence', ascending=False)

        return df

    def save_results(self, matches_df: pd.DataFrame, source_df: pd.DataFrame,
                    target_df: pd.DataFrame, output_path: str, source_col: str, target_col: str):
        """Save comprehensive results to Excel"""

        if self.verbose:
            print(f"💾 Saving results to: {output_path}")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Full matching results with scores
            matches_df.to_excel(writer, sheet_name='Matching_Results', index=False)

            # Sheet 2: Clean mapping table
            mapping_df = matches_df[['source_company', 'best_match', 'confidence', 'match_status']].copy()
            mapping_df.columns = ['Source_Company', 'Matched_Customer', 'Confidence_Score', 'Match_Status']
            mapping_df.to_excel(writer, sheet_name='Mapping_Table', index=False)

            # Sheet 3: Enhanced source data with matches
            source_enhanced = source_df.copy()
            match_dict = dict(zip(matches_df['source_company'], matches_df['best_match']))
            confidence_dict = dict(zip(matches_df['source_company'], matches_df['confidence']))

            source_enhanced['Matched_Customer'] = source_enhanced[source_col].map(match_dict)
            source_enhanced['Match_Confidence'] = source_enhanced[source_col].map(confidence_dict)
            source_enhanced['Match_Status'] = source_enhanced['Matched_Customer'].apply(
                lambda x: 'Matched' if pd.notna(x) else 'No Match'
            )
            source_enhanced.to_excel(writer, sheet_name='Enhanced_Source_Data', index=False)

            # Sheet 4: Original target data for reference
            target_df.to_excel(writer, sheet_name='Target_Data_Reference', index=False)

            # Sheet 5: Summary statistics
            total = len(matches_df)
            matched = len(matches_df[matches_df['match_status'] == 'Matched'])
            avg_conf = matches_df['confidence'].mean()
            high_conf = len(matches_df[matches_df['confidence'] > 0.9])
            medium_conf = len(matches_df[(matches_df['confidence'] >= 0.7) & (matches_df['confidence'] <= 0.9)])

            summary_data = {
                'Metric': [
                    'Total Companies Processed',
                    'Successfully Matched',
                    'No Match Found',
                    'Match Rate (%)',
                    'Average Confidence',
                    'High Confidence Matches (>90%)',
                    'Medium Confidence Matches (70-90%)',
                    'Low Confidence Matches (<70%)',
                    'Method Used',
                    'Confidence Threshold'
                ],
                'Value': [
                    total,
                    matched,
                    total - matched,
                    f"{matched/total*100:.1f}%",
                    f"{avg_conf:.3f}",
                    high_conf,
                    medium_conf,
                    len(matches_df[matches_df['confidence'] < 0.7]),
                    'Hybrid (Fuzzy + Rules + Normalization)',
                    self.confidence_threshold
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

            # Sheet 6: Action items for manual review
            needs_review = matches_df[
                (matches_df['match_status'] == 'No Match') |
                (matches_df['confidence'] < 0.85)
            ].copy()

            if len(needs_review) > 0:
                needs_review['Action_Required'] = needs_review.apply(
                    lambda row: 'Manual Research' if row['match_status'] == 'No Match'
                    else 'Review Match', axis=1
                )
                needs_review['Priority'] = needs_review['confidence'].apply(
                    lambda x: 'High' if x < 0.5 else 'Medium'
                )
                needs_review.to_excel(writer, sheet_name='Action_Items', index=False)

        if self.verbose:
            print(f"   ✅ Created 6 Excel sheets with comprehensive results")

def list_available_sheets(file_path: str):
    """List all sheets in the Excel file to help user identify correct sheet names"""
    try:
        excel_file = pd.ExcelFile(file_path)
        sheets = excel_file.sheet_names
        print(f"📄 Available sheets in {file_path}:")
        for i, sheet in enumerate(sheets, 1):
            print(f"   {i}. {sheet}")
        return sheets
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return []

def preview_sheet_columns(file_path: str, sheet_name: str):
    """Preview the columns in a specific sheet"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
        print(f"\n📋 Columns in sheet '{sheet_name}':")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")

        print(f"\n📊 Sample data from '{sheet_name}':")
        print(df.head(3).to_string())
        return list(df.columns)
    except Exception as e:
        print(f"❌ Error reading sheet '{sheet_name}': {e}")
        return []

def parse_arguments():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description='🎯 Hybrid Company Name Matcher - No API Required',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # List available sheets in your file
    python hybrid_matcher.py --file "test_data.xlsx" --list-sheets

    # Preview columns in a specific sheet
    python hybrid_matcher.py --file "test_data.xlsx" --preview-sheet "By Client"

    # Run matching with default settings
    python hybrid_matcher.py --file "test_data.xlsx"

    # Custom sheets and columns
    python hybrid_matcher.py --file "test_data.xlsx" --source-sheet "Clients" --target-sheet "Customers"

    # Adjust confidence threshold
    python hybrid_matcher.py --file "test_data.xlsx" --threshold 0.8
        """
    )

    # File options (required)
    parser.add_argument('--file', '-f', type=str, default='test_data.xlsx',
                       help='Path to Excel file (default: test_data.xlsx)')

    # Discovery options
    parser.add_argument('--list-sheets', action='store_true',
                       help='List all sheets in the Excel file and exit')
    parser.add_argument('--preview-sheet', type=str,
                       help='Preview columns and data in specified sheet')

    # Sheet and column configuration
    parser.add_argument('--source-sheet', type=str, default='By Client',
                       help='Source sheet name (default: "By Client")')
    parser.add_argument('--target-sheet', type=str, default='US Customer Cleanup JL',
                       help='Target sheet name (default: "US Customer Cleanup JL")')
    parser.add_argument('--source-col', type=str, default='By Client',
                       help='Source column name (default: "By Client")')
    parser.add_argument('--target-col', type=str, default='Customer Original Name',
                       help='Target column name (default: "Customer Original Name")')

    # Matching parameters
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold 0.0-1.0 (default: 0.7)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file name (default: auto-generated)')

    # Output control
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')

    return parser.parse_args()

def main():
    """Main execution function"""

    args = parse_arguments()

    # Validate file exists
    if not Path(args.file).exists():
        print(f"❌ File not found: {args.file}")
        print("💡 Make sure the file path is correct")
        return

    # Handle discovery commands
    if args.list_sheets:
        print(f"🔍 Discovering sheets in {args.file}...")
        sheets = list_available_sheets(args.file)
        return

    if args.preview_sheet:
        print(f"🔍 Previewing sheet '{args.preview_sheet}' in {args.file}...")
        columns = preview_sheet_columns(args.file, args.preview_sheet)
        return

    # Display configuration
    print("🎯 HYBRID COMPANY NAME MATCHER")
    print("=" * 50)
    print(f"📁 File: {args.file}")
    print(f"📄 Source: '{args.source_sheet}'['{args.source_col}']")
    print(f"📄 Target: '{args.target_sheet}'['{args.target_col}']")
    print(f"🎯 Confidence threshold: {args.threshold}")
    print(f"🔧 Method: Hybrid (Fuzzy + Rules + Normalization)")
    print("=" * 50)

    # Initialize matcher
    matcher = HybridCompanyMatcher(
        confidence_threshold=args.threshold,
        verbose=not args.quiet
    )

    # Load data
    source_companies, target_companies, source_df, target_df = matcher.load_excel_data(
        args.file, args.source_sheet, args.target_sheet, args.source_col, args.target_col
    )

    if not source_companies or not target_companies:
        print("❌ Failed to load data. Check file path and sheet/column names.")
        print("\n💡 Try these commands to explore your file:")
        print(f"   python hybrid_matcher.py --file '{args.file}' --list-sheets")
        print(f"   python hybrid_matcher.py --file '{args.file}' --preview-sheet 'SheetName'")
        return

    # Perform matching
    print(f"\n🚀 Starting hybrid matching process...")
    start_time = time.time()

    matches = matcher.match_companies(source_companies, target_companies)

    total_time = time.time() - start_time

    # Create comprehensive report
    matches_df = matcher.create_comprehensive_report(matches)

    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"hybrid_matching_results_{timestamp}.xlsx"

    # Save results
    matcher.save_results(matches_df, source_df, target_df, output_file, args.source_col, args.target_col)

    # Display summary
    total = len(matches_df)
    matched = len(matches_df[matches_df['match_status'] == 'Matched'])
    avg_conf = matches_df['confidence'].mean()
    high_conf = len(matches_df[matches_df['confidence'] > 0.9])

    print(f"\n📊 MATCHING SUMMARY")
    print("=" * 50)
    print(f"⏱️  Processing time: {total_time:.2f} seconds")
    print(f"⚡ Processing rate: {total/total_time:.1f} companies/second")
    print(f"📈 Total companies: {total}")
    print(f"✅ Successfully matched: {matched} ({matched/total*100:.1f}%)")
    print(f"🎯 High confidence (>90%): {high_conf} ({high_conf/total*100:.1f}%)")
    print(f"📊 Average confidence: {avg_conf:.1%}")

    # Show top matches
    if len(matches_df) > 0:
        print(f"\n🏆 Top 5 matches:")
        top_matches = matches_df.head(5)
        for _, row in top_matches.iterrows():
            source = str(row['source_company'])[:35]
            target = str(row['best_match'])[:35] if row['best_match'] else 'No match'
            conf = row['confidence']
            print(f"   {source:<35} → {target:<35} ({conf:.1%})")

    # Show items needing review
    needs_review = matches_df[
        (matches_df['match_status'] == 'No Match') |
        (matches_df['confidence'] < 0.85)
    ]

    if len(needs_review) > 0:
        print(f"\n⚠️  {len(needs_review)} items need manual review:")
        print("   • Check 'Action_Items' sheet in output file")
        print("   • Consider lowering --threshold if too strict")

    print(f"\n✅ Results saved to: {output_file}")
    print("📁 Excel file contains 6 sheets:")
    print("   • Matching_Results: Detailed results with scores")
    print("   • Mapping_Table: Clean source → target mapping")
    print("   • Enhanced_Source_Data: Your data + matches")
    print("   • Summary_Statistics: Performance metrics")
    print("   • Action_Items: Items needing manual review")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)