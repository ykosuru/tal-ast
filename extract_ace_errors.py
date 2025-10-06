import json
import os
import glob
import pandas as pd

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

def parse_transaction(txn_id, txn_data):
    """
    Parse a single transaction keyed by txndid, extracting:
    - ACE violations
    - Entity diffs (including accounts)
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
                "transactionId": txn_id,
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
                "transactionId": txn_id,
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

def parse_json_files(json_folder):
    """Parse all JSON files in a folder and build the combined ACE violation + diff matrix."""
    all_rows = []
    json_files = glob.glob(os.path.join(json_folder, "*.json"))

    for file_path in json_files:
        with open(file_path, "r") as f:
            data_list = json.load(f)  # top-level is a list
            for data in data_list:
                for txn_id, txn_data in data.items():
                    txn_rows = parse_transaction(txn_id, txn_data)
                    all_rows.extend(txn_rows)

    df = pd.DataFrame(all_rows)
    return df

if __name__ == "__main__":
    # Folder containing your production JSON files
    input_folder = "./prod_json"
    output_csv = "ace_violation_diff_matrix.csv"

    df_matrix = parse_json_files(input_folder)
    df_matrix.to_csv(output_csv, index=False)
    print(f"Violation + Diff matrix exported to {output_csv}")
