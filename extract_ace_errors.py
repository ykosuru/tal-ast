#!/usr/bin/env python3
"""
ACE Violation Lookup Table Builder
Extracts ACE codes from production data and maps to source/clearing/parties
"""

import json
import os
import glob
import pandas as pd
from typing import Dict, List, Any, Optional


# ============================================================================
# MAPPING CONFIGURATION - EDIT THESE MAPPINGS AS NEEDED
# ============================================================================

# Map ACE entity names (as they appear in ace[] messages) to party flag names
ACE_ENTITY_TO_FLAG = {
    "BNFBNK": "bbk_pty",           # Beneficiary Bank
    "BNF": "bnf_pty",              # Beneficiary / Creditor
    "CDTAGTINTRMY": "crdt_agt_intrmy_pty",  # Intermediary Credit Agent
    "CDTAGT": "crdt_agt_pty",      # Credit Agent
    "DBTAGT": "dbtr_agt_pty",      # Debit Agent
    "IBK": "ibk_pty",              # Instructing Bank
    "ORGBNK": "ogb_pty",           # Originating Bank
    "ORG": "org_pty",              # Originator / Debtor
    
    # Add more mappings here as you discover them in your data
    # Format: "ACE_NAME": "flag_name"
}

# Map party flags to ISO 20022 entity names
# When a flag is True, that party is active in the transaction
PARTY_FLAG_TO_ISO = {
    "bbk_pty": "beneficiaryBank",
    "bnf_pty": "cdtr",              # Creditor (Beneficiary)
    "crdt_agt_intrmy_pty": "intrmyAgt1",  # Intermediary
    "crdt_agt_pty": "cdtrAgt",       # Creditor Agent
    "dbtr_agt_pty": "dbtrAgt",       # Debtor Agent
    "ibk_pty": "instgAgt",           # Instructing Agent
    "ogb_pty": "originatingBank",
    "org_pty": "dbtr"                # Debtor (Originator)
}

# ============================================================================


def normalize_ace_entity(ace_entity: str) -> Optional[str]:
    """
    Convert ACE entity name to party flag name.
    
    Args:
        ace_entity: Entity name as it appears in ACE messages (e.g., "BNFBNK")
        
    Returns:
        Party flag name (e.g., "bnf_pty") or None if not found
    """
    # Try exact match first
    if ace_entity in ACE_ENTITY_TO_FLAG:
        return ACE_ENTITY_TO_FLAG[ace_entity]
    
    # Try case-insensitive match
    ace_upper = ace_entity.upper()
    for ace_name, flag_name in ACE_ENTITY_TO_FLAG.items():
        if ace_name.upper() == ace_upper:
            return flag_name
    
    # Not found - log it for manual mapping
    print(f"  Warning: Unknown ACE entity '{ace_entity}' - add to ACE_ENTITY_TO_FLAG mapping")
    return None


def get_iso_entity_name(ace_entity: str) -> str:
    """
    Convert ACE entity name to ISO 20022 entity name.
    
    Args:
        ace_entity: Entity name from ACE messages
        
    Returns:
        ISO entity name (e.g., "cdtr") or original if mapping not found
    """
    flag_name = normalize_ace_entity(ace_entity)
    if flag_name and flag_name in PARTY_FLAG_TO_ISO:
        return PARTY_FLAG_TO_ISO[flag_name]
    return ace_entity  # Return original if no mapping found


def parse_transaction(txn_id: str, txn_data: Dict) -> List[Dict]:
    """Parse a single transaction and extract ACE violations."""
    rows = []
    
    source_network = txn_data.get("source", "UNKNOWN")
    clearing_network = txn_data.get("clearing", "UNKNOWN")
    parties_flags = txn_data.get("parties", {})
    
    # Determine which parties are active (flags set to True)
    active_parties = []
    for flag_name, is_active in parties_flags.items():
        if is_active and flag_name in PARTY_FLAG_TO_ISO:
            active_parties.append(PARTY_FLAG_TO_ISO[flag_name])
    
    active_parties_str = ",".join(sorted(active_parties))
    
    # Process ACE violations (try both 'ace' and 'repairs' fields)
    ace_entries = txn_data.get("ace", txn_data.get("repairs", []))
    
    # Handle both list and dict formats
    if isinstance(ace_entries, dict):
        # Format: { entity: { code: {field, text} } }
        for ace_entity, violations in ace_entries.items():
            iso_entity = get_iso_entity_name(ace_entity)
            
            if isinstance(violations, dict):
                for code, details in violations.items():
                    if isinstance(details, dict):
                        row = {
                            "source": source_network,
                            "clearing": clearing_network,
                            "parties": active_parties_str,
                            "entity": iso_entity,
                            "code": code,
                            "field": details.get("field", ""),
                            "text": details.get("text", ""),
                            "severity": details.get("code", "I"),  # I/W/E
                            "txn_ids": [txn_id]
                        }
                        rows.append(row)
    
    elif isinstance(ace_entries, list):
        # Format: [ {id, field, text, entity?} ]
        for ace_entry in ace_entries:
            code = ace_entry.get("id", ace_entry.get("code", ""))
            field = ace_entry.get("field", "")
            text = ace_entry.get("text", "")
            ace_entity = ace_entry.get("entity", "UNKNOWN")
            severity = ace_entry.get("severity", ace_entry.get("code", "I"))
            
            iso_entity = get_iso_entity_name(ace_entity)
            
            row = {
                "source": source_network,
                "clearing": clearing_network,
                "parties": active_parties_str,
                "entity": iso_entity,
                "code": code,
                "field": field,
                "text": text,
                "severity": severity,
                "txn_ids": [txn_id]
            }
            rows.append(row)
    
    return rows


def parse_json_files(json_folder: str) -> pd.DataFrame:
    """Parse all JSON files in a folder and build the ACE violation matrix."""
    all_rows = []
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {json_folder}")
        return pd.DataFrame()
    
    print(f"Processing {len(json_files)} JSON files...")
    unknown_entities = set()
    
    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data_list = json.load(f)
                
                # Handle both list and dict at top level
                if isinstance(data_list, list):
                    for data in data_list:
                        if isinstance(data, dict):
                            for txn_id, txn_data in data.items():
                                txn_rows = parse_transaction(txn_id, txn_data)
                                all_rows.extend(txn_rows)
                
                elif isinstance(data_list, dict):
                    for txn_id, txn_data in data_list.items():
                        txn_rows = parse_transaction(txn_id, txn_data)
                        all_rows.extend(txn_rows)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    df = pd.DataFrame(all_rows)
    return df


def build_ace_lookup_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ACE lookup table grouped by source/clearing/parties/entity/field/code.
    Transaction IDs stored in comments field.
    """
    if df.empty:
        return df
    
    # Group by context and aggregate
    grouped = df.groupby(
        ["source", "clearing", "parties", "entity", "field", "code", "severity", "text"],
        dropna=False
    ).agg({
        "txn_ids": lambda x: list(set([item for sublist in x for item in sublist]))
    }).reset_index()
    
    # Create comments field with transaction IDs
    grouped["occurrences"] = grouped["txn_ids"].apply(len)
    grouped["comments"] = grouped["txn_ids"].apply(
        lambda ids: f"Transactions: {', '.join(ids[:10])}" + 
                    (f" ... and {len(ids)-10} more" if len(ids) > 10 else "")
    )
    
    # Drop the txn_ids list column
    grouped = grouped.drop(columns=["txn_ids"])
    
    # Reorder columns
    column_order = [
        "source", "clearing", "parties", "entity", 
        "code", "severity", "field", "text", 
        "occurrences", "comments"
    ]
    grouped = grouped[column_order]
    
    # Sort by frequency
    grouped = grouped.sort_values("occurrences", ascending=False)
    
    return grouped


if __name__ == "__main__":
    # Folder containing JSON files
    input_folder = "./prod_json"
    output_lookup = "ace_lookup_table.csv"
    
    print("=" * 80)
    print("ACE VIOLATION LOOKUP TABLE BUILDER")
    print("=" * 80)
    
    # Parse all transactions
    df_matrix = parse_json_files(input_folder)
    
    if not df_matrix.empty:
        # Build lookup table
        df_lookup = build_ace_lookup_table(df_matrix)
        df_lookup.to_csv(output_lookup, index=False)
        print(f"\n✓ ACE lookup table exported to {output_lookup} ({len(df_lookup)} unique patterns)")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Unique ACE codes: {df_lookup['code'].nunique()}")
        print(f"  Source networks: {df_lookup['source'].unique().tolist()}")
        print(f"  Clearing networks: {df_lookup['clearing'].unique().tolist()}")
        print(f"  Entities affected: {df_lookup['entity'].unique().tolist()}")
        print(f"  Total violations: {df_lookup['occurrences'].sum()}")
        
        # Show top 5 most common violations
        print(f"\nTop 5 most common violations:")
        top5 = df_lookup.head(5)
        for idx, row in top5.iterrows():
            print(f"  ACE {row['code']} ({row['severity']}): {row['text']}")
            print(f"    Field: {row['field']}, Entity: {row['entity']}")
            print(f"    Context: {row['source']}|{row['clearing']}|{row['parties']}")
            print(f"    Occurrences: {row['occurrences']}\n")
    else:
        print("\nNo data found to process")
