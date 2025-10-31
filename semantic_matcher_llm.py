"""
semantic_matcher_llm.py
Date: 2025/7/10
Description:
LLM-based semantic matcher that links fragmented diligence records by comparing textual similarity
between company names and metadata using Anthropic's Claude API. Designed as part of an automated
diligence data framework for streamlining client record consolidation in financial analysis workflows.
"""

import pandas as pd
import anthropic
from typing import List, Dict, Tuple
import json
import re
from pathlib import Path
import time
import argparse
import sys

class CompanyNameMatcher:
    def __init__(self, api_key: str):
        """Initialize the matcher with Claude API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"✅ Initialized CompanyNameMatcher with API key: {api_key[:12]}...")

    def list_sheets(self, file_path: str) -> List[str]:
        """List all sheet names in the Excel file"""
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

    def load_excel_data(self, file_path: str, sheet1_name: str = "By Client", sheet2_name: str = "US Customer Cleanup JL",
                       col1_name: str = "By Client", col2_name: str = "Customer Orignal Name") -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame]:
        """
        Load data from Excel file
        Handles headerless sheets by assigning default column names.
        """
        try:
            print(f"📊 Loading data from {file_path}...")
            available_sheets = self.list_sheets(file_path)

            # Try reading with header, fallback to headerless
            try:
                print(f"\n🎯 Loading SOURCE data from sheet: '{sheet1_name}'")
                sheet1 = pd.read_excel(file_path, sheet_name=sheet1_name)
                if col1_name not in sheet1.columns:
                    raise KeyError
            except Exception:
                print(f"   (No header detected in '{sheet1_name}', assigning column name '{col1_name}' to first column)")
                sheet1 = pd.read_excel(file_path, sheet_name=sheet1_name, header=None)
                sheet1.columns = [col1_name]

            try:
                print(f"\n🎯 Loading TARGET data from sheet: '{sheet2_name}'")
                sheet2 = pd.read_excel(file_path, sheet_name=sheet2_name)
                if col2_name not in sheet2.columns or 'CustomerID' not in sheet2.columns:
                    raise KeyError
            except Exception:
                print(f"   (No header detected in '{sheet2_name}', assigning columns '{col2_name}', 'CustomerID' to first two columns)")
                sheet2 = pd.read_excel(file_path, sheet_name=sheet2_name, header=None)
                sheet2.columns = [col2_name, 'CustomerID']

            # Extract company names and clean them
            companies1 = sheet1[col1_name].dropna().unique().tolist()
            companies2 = sheet2[col2_name].dropna().unique().tolist()

            print(f"\n✅ SOURCE: Loaded {len(companies1)} unique companies from '{sheet1_name}'['{col1_name}']")
            print(f"   Sample: {companies1[:3]}")
            print(f"✅ TARGET: Loaded {len(companies2)} unique companies from '{sheet2_name}'['{col2_name}']")
            print(f"   Sample: {companies2[:3]}")

            print(f"\n🔄 MATCHING PROCESS:")
            print(f"   • Each of the {len(companies1)} SOURCE companies will be matched against")
            print(f"   • ALL {len(companies2)} TARGET companies to find the best match")
            print(f"   • Result: 1:1 mapping (each source → best target or null)")

            return companies1, companies2, sheet1, sheet2
        except Exception as e:
            print(f"❌ Error loading Excel data: {e}")
            return [], [], pd.DataFrame(), pd.DataFrame()

    def clean_company_name(self, name: str) -> str:
        """Basic cleaning of company names"""
        if pd.isna(name):
            return ""
        # Remove extra spaces and convert to string
        cleaned = str(name).strip()
        return cleaned

    def batch_match_companies(self, source_companies: List[str],
                            target_companies: List[str],
                            batch_size: int = 8) -> List[Dict]:
        """Match companies in batches using Claude"""
        results = []
        total_batches = (len(source_companies) + batch_size - 1) // batch_size

        print(f"🔄 Processing {len(source_companies)} companies in {total_batches} batches...")

        # Process in batches to avoid token limits
        for i in range(0, len(source_companies), batch_size):
            batch_num = (i // batch_size) + 1
            batch = source_companies[i:i+batch_size]

            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} companies)...")

            batch_results = self._match_batch(batch, target_companies)
            results.extend(batch_results)

            # Add small delay to respect API rate limits
            time.sleep(1)

            # Show progress
            progress = (batch_num / total_batches) * 100
            print(f"✅ Batch {batch_num} completed ({progress:.1f}% done)")

        print(f"🎉 All batches completed! Processed {len(results)} companies.")
        return results

    def _match_batch(self, source_batch: List[str], target_companies: List[str]) -> List[Dict]:
        """
        Match a batch of SOURCE companies against ALL TARGET companies

        For each SOURCE company, Claude will:
        1. Look through ALL target companies
        2. Find the single best match
        3. Return confidence score (0.0-1.0)
        4. Return null if no good match found
        """

        # Create the prompt
        prompt = f"""
You are an expert at matching company names in investment banking contexts, dealing with ticker symbols, abbreviated names, and various formatting conventions.

SOURCE companies to match (find matches for these):
{json.dumps(source_batch, indent=2)}

TARGET companies to match against (search through these for matches):
{json.dumps(target_companies, indent=2)}

TASK: For each SOURCE company, find the single best TARGET company match.

Consider these investment banking naming patterns:
- Abbreviations (Corp vs Corporation, Ltd vs Limited, Inc vs Incorporated, PHAR vs Pharmaceuticals)
- Ticker symbols and market indicators (F+, F<, <, +, etc.) - ignore these
- Punctuation differences (periods, spaces, hyphens, ampersands)
- Legal suffixes (LLC, Inc, Corp, Ltd, PLC, etc.)
- Compressed names (HONEYWELLINTLINC = Honeywell International Inc)
- Common business abbreviations (INTL = International, GRP = Group, etc.)
- Case sensitivity (all caps vs proper case)

Return your response as a JSON array with this exact structure:
[
  {{
    "source_company": "exact source company name",
    "best_match": "exact target company name or null if no good match",
    "confidence": 0.95
  }}
]

IMPORTANT:
- Only match if you're reasonably confident (>70% confidence)
- Return null for best_match if no good match exists
- Each source company gets exactly one result
- Be conservative with matches - better to return null than a wrong match
"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse the JSON response
            result_text = response.content[0].text

            # Extract JSON from response (handle potential markdown formatting)
            json_match = re.search(r'\[(.*?)\]', result_text, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                matches = json.loads(json_str)
                return matches
            else:
                # Try to parse the entire response as JSON
                matches = json.loads(result_text)
                return matches

        except AttributeError as e:
            print(f"❌ Anthropic API error: {e}\nCheck that you have the latest 'anthropic' package installed and that the API usage matches the SDK version.")
            return [{"source_company": comp, "best_match": None, "confidence": 0.0} for comp in source_batch]
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response text: {result_text}")
            return [{"source_company": comp, "best_match": None, "confidence": 0.0} for comp in source_batch]
        except Exception as e:
            print(f"❌ Error in batch matching: {e}")
            return [{"source_company": comp, "best_match": None, "confidence": 0.0} for comp in source_batch]

    def create_matching_report(self, matches: List[Dict]) -> pd.DataFrame:
        """Create a DataFrame report of matches"""
        df = pd.DataFrame(matches)

        # Add match status
        df['match_status'] = df.apply(
            lambda row: 'Matched' if row['best_match'] is not None else 'No Match',
            axis=1
        )

        # Sort by confidence (descending) and then by source company name
        df = df.sort_values(['confidence', 'source_company'], ascending=[False, True])

        return df

    def save_results(self, matches_df: pd.DataFrame, source_sheet: pd.DataFrame,
                    target_sheet: pd.DataFrame, output_path: str,
                    source_col: str = "By Client", target_col: str = "Customer Orignal Name"):
        """Save comprehensive results to Excel with multiple sheets, including matched CustomerID"""
        print(f"💾 Saving results to {output_path}...")
        # Build a lookup for customer name -> CustomerID (normalize by stripping and lowering)
        customer_id_lookup = {}
        if 'CustomerID' in target_sheet.columns and target_col in target_sheet.columns:
            for _, row in target_sheet.iterrows():
                customer_name = str(row[target_col]).strip().lower()
                customer_id = row['CustomerID']
                customer_id_lookup[customer_name] = customer_id
        else:
            print("⚠️  'CustomerID' column not found in target sheet. Only matched names will be returned.")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. Main matching results (add CustomerID column, normalize best_match)
            matches_df['Matched_CustomerID'] = matches_df['best_match'].map(lambda x: customer_id_lookup.get(str(x).strip().lower()) if pd.notna(x) else None)
            matches_df[['source_company', 'best_match', 'Matched_CustomerID', 'confidence', 'match_status']].to_excel(writer, sheet_name='Matching_Results', index=False)
            # 2. Create mapping table (just matched name and CustomerID)
            mapping_df = matches_df[['source_company', 'best_match', 'Matched_CustomerID']].copy()
            mapping_df.columns = ['Source_Company', 'Matched_Customer', 'Matched_CustomerID']
            mapping_df.to_excel(writer, sheet_name='Mapping_Table', index=False)
            # 3. Enhanced source sheet with matches (just matched name and CustomerID)
            source_enhanced = source_sheet.copy()
            match_dict = dict(zip(matches_df['source_company'], matches_df['best_match']))
            id_dict = dict(zip(matches_df['source_company'], matches_df['Matched_CustomerID']))
            source_enhanced['Matched_Customer'] = source_enhanced[source_col].map(match_dict)
            source_enhanced['Matched_CustomerID'] = source_enhanced[source_col].map(id_dict)
            source_enhanced.to_excel(writer, sheet_name='By_Client_Enhanced', index=False)
            # 4. Target sheet (unchanged but included for reference)
            target_sheet.to_excel(writer, sheet_name='US_Customer_Original', index=False)
            # 5. Create summary sheet with statistics
            summary_stats = self._create_summary_stats(matches_df)
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            # 6. Create action items sheet
            action_items = self._create_action_items(matches_df)
            action_items.to_excel(writer, sheet_name='Action_Items', index=False)
        print(f"✅ Results saved successfully to: {output_path}")
        print(f"📊 Created 6 sheets: Matching_Results, Mapping_Table, By_Client_Enhanced, US_Customer_Original, Summary_Statistics, Action_Items")

    def _create_summary_stats(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics"""
        total_companies = len(matches_df)
        matched_companies = len(matches_df[matches_df['match_status'] == 'Matched'])
        unmatched_companies = total_companies - matched_companies
        avg_confidence = matches_df['confidence'].mean()

        high_confidence = len(matches_df[matches_df['confidence'] > 0.9])
        medium_confidence = len(matches_df[(matches_df['confidence'] >= 0.7) & (matches_df['confidence'] <= 0.9)])
        low_confidence = len(matches_df[matches_df['confidence'] < 0.7])

        summary_data = {
            'Metric': [
                'Total Companies Processed',
                'Successfully Matched',
                'No Match Found',
                'Match Rate (%)',
                'Average Confidence Score',
                'High Confidence Matches (>90%)',
                'Medium Confidence Matches (70-90%)',
                'Low Confidence Matches (<70%)',
                'Matches Requiring Review (<85%)'
            ],
            'Value': [
                total_companies,
                matched_companies,
                unmatched_companies,
                f"{(matched_companies/total_companies)*100:.1f}%",
                f"{avg_confidence:.3f}",
                high_confidence,
                medium_confidence,
                low_confidence,
                len(matches_df[matches_df['confidence'] < 0.85])
            ]
        }

        return pd.DataFrame(summary_data)

    def _create_action_items(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Create action items for manual review"""
        action_items = []

        # Low confidence matches
        low_confidence = matches_df[matches_df['confidence'] < 0.85]
        for _, row in low_confidence.iterrows():
            action_items.append({
                'Action_Type': 'Review Match',
                'Priority': 'Medium' if row['confidence'] > 0.7 else 'High',
                'Company': row['source_company'],
                'Suggested_Match': row['best_match'],
                'Confidence': row['confidence'],
                'Description': f"Review match with {row['confidence']:.1%} confidence"
            })

        # No matches found
        no_matches = matches_df[matches_df['match_status'] == 'No Match']
        for _, row in no_matches.iterrows():
            action_items.append({
                'Action_Type': 'Manual Research',
                'Priority': 'High',
                'Company': row['source_company'],
                'Suggested_Match': 'None found',
                'Confidence': 0.0,
                'Description': 'No automatic match found - requires manual research'
            })

        return pd.DataFrame(action_items)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LLM-Based Company Name Fuzzy Matching Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python script.py
        """
    )
    # File and basic options
    parser.add_argument('--list-sheets', action='store_true', help='List all sheets in the Excel file and exit')
    # Sheet names
    parser.add_argument('--source-sheet', type=str, default='By Client', help='Name of source sheet (default: "By Client")')
    parser.add_argument('--target-sheet', type=str, default='US Customer Cleanup JL', help='Name of target sheet (default: "US Customer Cleanup JL")')
    # Column names
    parser.add_argument('--source-col', type=str, default='By Client', help='Name of source column (default: "By Client")')
    parser.add_argument('--target-col', type=str, default='Customer Orignal Name', help='Name of target column (default: "Customer Orignal Name")')
    # Processing options
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for API calls (default: 8)')
    parser.add_argument('--output', '-o', type=str, help='Output file name (default: auto-generated)')
    # API key
    parser.add_argument('--api-key', type=str, help='Claude API key (or set CLAUDE_API_KEY environment variable)')
    return parser.parse_args()

def get_api_key(args):
    """Get API key from arguments or environment"""
    import os
    if args.api_key:
        return args.api_key
    elif os.getenv('CLAUDE_API_KEY'):
        return os.getenv('CLAUDE_API_KEY')
    else:
        # Default API key (you can remove this in production)
        return "YOUR_API_KEY_HERE"
    # SECURITY NOTE: For production use, store API key in environment variable
    # API_KEY = os.getenv('CLAUDE_API_KEY')

    # Path to your Excel file
    excel_file_path = "test_data.xlsx"  # Update this path

    print("🚀 Starting LLM-based Company Name Matching...")
    print("=" * 60)

    # Initialize matcher
    matcher = CompanyNameMatcher(API_KEY)

    # Load data
    print("\n📊 Loading Excel data...")
    source_companies, target_companies, source_sheet, target_sheet = matcher.load_excel_data(
        file_path=excel_file_path,
        sheet1_name=args.source_sheet,
        sheet2_name=args.target_sheet,
        col1_name=args.source_col,
        col2_name=args.target_col
    )

    if not source_companies or not target_companies:
        print("❌ Failed to load data. Please check your file path and sheet names.")
        return

    print(f"\n🎯 Matching {len(source_companies)} companies from '{args.source_sheet}' sheet")
    print(f"🎯 Against {len(target_companies)} companies from '{args.target_sheet}' sheet")

    # Perform matching
    print("\n🤖 Starting AI-powered matching...")
    matches = matcher.batch_match_companies(source_companies, target_companies, batch_size=args.batch_size)

    # Create comprehensive report
    print("\n📋 Creating matching report...")
    matches_df = matcher.create_matching_report(matches)

    # Save comprehensive results
    output_file = args.output if args.output else f"company_matching_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    matcher.save_results(matches_df, source_sheet, target_sheet, output_file)

    # Display summary results
    print("\n" + "=" * 60)
    print("📊 MATCHING SUMMARY")
    print("=" * 60)

    total = len(matches_df)
    matched = len(matches_df[matches_df['match_status'] == 'Matched'])
    high_conf = len(matches_df[matches_df['confidence'] > 0.9])

    print(f"Total companies processed: {total}")
    print(f"Successfully matched: {matched} ({matched/total*100:.1f}%)")
    print(f"High confidence matches: {high_conf} ({high_conf/total*100:.1f}%)")
    print(f"Average confidence: {matches_df['confidence'].mean():.1%}")

    # Show some example matches
    print(f"\n🎯 Top 5 matches:")
    top_matches = matches_df.nlargest(5, 'confidence')
    for _, row in top_matches.iterrows():
        print(f"  {row['source_company'][:30]:<30} → {str(row['best_match'])[:30]:<30} ({row['confidence']:.1%})")

    # Show items needing review
    needs_review = matches_df[matches_df['confidence'] < 0.85]
    if len(needs_review) > 0:
        print(f"\n⚠️  {len(needs_review)} matches need manual review (confidence < 85%)")
        print("Check the 'Action_Items' sheet in the output file.")

    print(f"\n✅ Complete results saved to: {output_file}")
    print("📁 Check the following sheets:")
    print("   • Matching_Results: Detailed matching results")
    print("   • Mapping_Table: Clean mapping table")
    print("   • By_Client_Enhanced: Original data with matches added")
    print("   • Action_Items: Items requiring manual review")

    return matches_df

if __name__ == "__main__":
    try:
        args = parse_arguments()
        API_KEY = get_api_key(args)
        excel_file_path = "test_data.xlsx"
        matcher = CompanyNameMatcher(API_KEY)
        if args.list_sheets:
            matcher.list_sheets(excel_file_path)
            sys.exit(0)
        source_companies, target_companies, source_sheet, target_sheet = matcher.load_excel_data(
            file_path=excel_file_path,
            sheet1_name=args.source_sheet,
            sheet2_name=args.target_sheet,
            col1_name=args.source_col,
            col2_name=args.target_col
        )
        if not source_companies or not target_companies:
            print("❌ Failed to load data. Please check your file path and sheet names.")
            sys.exit(1)
        matches = matcher.batch_match_companies(source_companies, target_companies, batch_size=args.batch_size)
        matches_df = matcher.create_matching_report(matches)
        output_file = args.output if args.output else f"company_matching_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        matcher.save_results(matches_df, source_sheet, target_sheet, output_file)
        print(f"\n✅ Complete results saved to: {output_file}")
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Use --help for usage information")
        sys.exit(1)