import json
import os
import glob
import pandas as pd
import argparse
from collections import defaultdict

# Map boolean party flags to readable Pelican party names
PARTY_MAP = {
    "bbk_pty": "BNFBNK",
    "bnf_pty": "Credit Party",
    "crdt_agt_intrmy_pty": "Intermediary Credit Agent",
    "crdt_agt_pty": "Credit Agent",
    "dbtr_agt_pty": "Debit Agent",
    "ibk_pty": "Instructing Bank",
    "ogb_pty": "Originating Bank",
    "org_pty": "Originator Party"
}

# List of entities to extract diffs from
ENTITY_DIFFS = [
    "cdtrAgt", "dbtrAgt",
    "cdtr", "dbtr",
    "cdtrAcct", "dbtrAcct",
    "instgAgt", "instdAgt",
    "rmtInf"
]

# Map entities to human-readable party codes for diffs
PARTY_ENTITY_MAP = {
    "cdtrAgt": "Credit Agent",
    "dbtrAgt": "Debit Agent",
    "cdtrAcct": "Credit Party",
    "dbtrAcct": "Debit Party",
    "cdtr": "Credit Party",
    "dbtr": "Debit Party",
    "instgAgt": "Instructing Bank",
    "instdAgt": "Instructed Bank",
    "rmtInf": "Remittance Info"
}

def parse_transaction(txn_data):
    """
    Extract ACE violations and entity diffs from a single transaction dictionary.
    Returns a list of row dictionaries.
    """
    rows = []
    source_network = txn_data.get("source")
    clearing_network = txn_data.get("clearing")
    parties_flags = txn_data.get("parties", {})
    active_parties = [PARTY_MAP[k] for k, v in parties_flags.items() if v]

    # --- ACE Violations ---
    for ace_entry in txn_data.get("ace", []):
        code = ace_entry.get("id")
        field = ace_entry.get("field")
        trigger_condition = ace_entry.get("text")

        for party in active_parties:
            row = {
                "source": source_network,
                "clearing": clearing_network,
                "code": code,
                "field": field,
                "triggerCondition": trigger_condition,
                "partyCode": party,
                "entity": None,
                "diffKey": None,
                "diffValue": None,
                "diffType": None,
                "diffDescription": None
            }
            rows.append(row)

    # --- Entity Diffs ---
    for entity in ENTITY_DIFFS:
        entity_data = txn_data.get(entity, {})
        diffs = entity_data.get("diffs", [])
        for diff in diffs:
            row = {
                "source": source_network,
                "clearing": clearing_network,
                "code": None,
                "field": None,
                "triggerCondition": None,
                "partyCode": PARTY_ENTITY_MAP.get(entity),
                "entity": entity,
                "diffKey": diff.get("key"),
                "diffValue": diff.get("val"),
                "diffType": diff.get("msg"),
                "diffDescription": diff.get("desc")
            }
            rows.append(row)

    return rows

def parse_json_files(data_dir):
    """
    Parse all JSON files recursively in a folder and return a consolidated matrix.
    Duplicate rows are merged with a count of occurrences.
    """
    all_rows = []
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files to process...")

    for file_path in json_files:
        with open(file_path, "r") as f:
            try:
                data_list = json.load(f)  # top-level is a list
                for data in data_list:
                    for txn_id, txn_data in data.items():
                        txn_rows = parse_transaction(txn_data)
                        all_rows.extend(txn_rows)
            except json.JSONDecodeError:
                print(f"Warning: failed to parse {file_path}, skipping.")

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # --- Consolidate duplicates and count ---
    grouping_cols = ["source", "clearing", "code", "field", "partyCode", "entity",
                     "diffKey", "diffValue", "diffType", "diffDescription", "triggerCondition"]
    df["count"] = 1
    df_consolidated = df.groupby(grouping_cols, dropna=False, as_index=False).count()
    df_consolidated = df_consolidated.rename(columns={"count": "occurrences"})

    # Drop extra count column created by groupby
    df_consolidated = df_consolidated[grouping_cols + ["occurrences"]]

    return df_consolidated

def main():
    parser = argparse.ArgumentParser(description="Generate ACE violation + diff matrix from production JSON files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing production JSON files.")
    parser.add_argument("--output_csv", type=str, default="ace_violation_diff_matrix.csv", help="Output CSV file name.")
    args = parser.parse_args()

    df_matrix = parse_json_files(args.data_dir)
    df_matrix.to_csv(args.output_csv, index=False)
    print(f"Consolidated violation + diff matrix exported to {args.output_csv}.")

if __name__ == "__main__":
    main()
