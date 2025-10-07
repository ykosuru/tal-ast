import json
import argparse  # Import the argparse library for command-line argument parsing

def parse_single_transaction(data, root_key):
    """
    Parses a single wire transfer transaction from the data dictionary,
    including both the Request and Response sections.
    
    This function walks through the nested structure of the JSON, extracting
    each piece of payment information. It is designed to be robust against
    missing keys or objects, printing 'Null' for any absent data points.
    
    Args:
        data: Dictionary containing the transaction data.
        root_key: The transaction ID key (e.g., "2025092900009602").
    """
    print(f"\n{'='*80}")
    print(f"--- Parsing Transaction: {root_key} ---")
    print(f"{'='*80}")
    
    # ---- PARSE THE REQUEST SECTION ----
    print("\n" + "="*20 + " REQUEST " + "="*20)
    
    # Safely navigate to the BasicPayment object using .get() with default empty dicts
    basic_payment = data.get(root_key, {}).get('Request', {}).get('IFML', {}).get('File', {}).get('BasicPayment', {})
    
    if not basic_payment:
        print("Request BasicPayment object not found in the provided JSON structure.")
    else:
        # ---- 1. Basic Payment Details ----
        print("\n--- Basic Payment Details ---")
        print(f"TransactionUID: {basic_payment.get('TransactionUID', 'Null')}")
        print(f"TransactionCode: {basic_payment.get('TransactionCode', 'Null')}")
        print(f"SourceCode: {basic_payment.get('SourceCode', 'Null')}")
        print(f"FundsType: {basic_payment.get('FundsType', 'Null')}")
        print(f"DbtChrg (Debit Charge): {basic_payment.get('DbtChrg', 'Null')}")
        print(f"CdtChrg (Credit Charge): {basic_payment.get('CdtChrg', 'Null')}")
        
        # ---- 2. Nested Dictionary Objects ----
        payment_dates = basic_payment.get('PaymentDates') or {}
        print(f"PaymentDates @Type: {payment_dates.get('@Type', 'Null')}")
        print(f"PaymentDates DateTime: {payment_dates.get('DateTime', 'Null')}")
        
        location_inf = basic_payment.get('LocationInf') or {}
        print(f"LocationInf IDBank: {location_inf.get('IDBank', 'Null')}")
        print(f"LocationInf Location: {location_inf.get('Location', 'Null')}")
        
        charges = basic_payment.get('Charges') or {}
        print(f"Charges @Type: {charges.get('@Type', 'Null')}")
        print(f"Charges #text: {charges.get('#text', 'Null')}")
        
        # ---- 3. List of Monetary Amounts ----
        print("\n--- Monetary Amounts ---")
        monetary_amounts = basic_payment.get('MonetaryAmount') or []
        if not monetary_amounts:
            print("MonetaryAmount: Null")
        else:
            for i, amount_info in enumerate(monetary_amounts):
                print(f" Amount [{i}]:")
                print(f"     @Type: {amount_info.get('@Type', 'Null')}")
                print(f"     Currency: {amount_info.get('Currency', 'Null')}")
                print(f"     Amount: {amount_info.get('Amount', 'Null')}")
        
        # ---- 4. Party Information ----
        party_inf = basic_payment.get('PartyInf') or {}
        if not party_inf:
            print("\nPartyInf section is missing from Request.")
        else:
            # Helper function to print address blocks
            def print_address(address_list, party_name):
                if not address_list:
                    print(f" {party_name} AddressInf: Null")
                else:
                    if not isinstance(address_list, list):
                        address_list = [address_list]
                    for addr in address_list:
                        print(f" Address (Seq {addr.get('@Seq', '?')}): {addr.get('#text', 'Null')}")
            
            # Originating Party
            print("\n--- Originating Party ---")
            orig_party = party_inf.get('OriginatingPartyInf', {}).get('BasicPartyInf') or {}
            print(f"Reference: {orig_party.get('Reference', 'Null')}")
            
            # Sending Bank
            print("\n--- Sending Bank ---")
            send_bank = party_inf.get('SendingBankInf', {}).get('BasicPartyInf') or {}
            print(f"Reference: {send_bank.get('Reference', 'Null')}")
            
            # Debit Party
            print("\n--- Debit Party ---")
            debit_party = party_inf.get('DebitPartyInf', {}).get('AccountPartyInf') or {}
            print(f"AdrBankID: {debit_party.get('AdrBankID', 'Null')}")
            print(f"ReconRef: {debit_party.get('ReconRef', 'Null')}")
            acct_id_inf_dbt = debit_party.get('AcctIDInf') or {}
            id_dbt = acct_id_inf_dbt.get('ID') or {}
            print(f"AcctIDInf Override: {acct_id_inf_dbt.get('Override', 'Null')}")
            print(f"AcctIDInf IDBank: {acct_id_inf_dbt.get('IDBank', 'Null')}")
            print(f"AcctIDInf ID @Type: {id_dbt.get('@Type', 'Null')}")
            print(f"AcctIDInf ID #text: {id_dbt.get('#text', 'Null')}")
            print_address(debit_party.get('AddressInf'), "Debit Party")
            
            # Credit Party
            print("\n--- Credit Party ---")
            credit_party = party_inf.get('CreditPartyInf', {}).get('AccountPartyInf') or {}
            print(f"ResidenceCountry: {credit_party.get('ResidenceCountry', 'Null')}")
            print(f"AdviceType: {credit_party.get('AdviceType', 'Null')}")
            print(f"Department: {credit_party.get('Department', 'Null')}")
            acct_id_inf_cdt = credit_party.get('AcctIDInf') or {}
            id_cdt = acct_id_inf_cdt.get('ID') or {}
            print(f"AcctIDInf ID @Type: {id_cdt.get('@Type', 'Null')}")
            print(f"AcctIDInf ID #text: {id_cdt.get('#text', 'Null')}")
            print_address(credit_party.get('AddressInf'), "Credit Party")
            
            # Intermediary Bank
            print("\n--- Intermediary Bank ---")
            inter_bank_wrapper = party_inf.get('IntermediaryBankInf') or {}
            print(f"@Type: {inter_bank_wrapper.get('@Type', 'Null')}")
            inter_bank = inter_bank_wrapper.get('BasicPartyInf') or {}
            print(f"AdrBankID: {inter_bank.get('AdrBankID', 'Null')}")
            print(f"Country: {inter_bank.get('Country', 'Null')}")
            id_inter = inter_bank.get('ID') or {}
            print(f"ID @Type: {id_inter.get('@Type', 'Null')}")
            print(f"ID #text: {id_inter.get('#text', 'Null')}")
            print_address(inter_bank.get('AddressInf'), "Intermediary Bank")
            
            # Beneficiary Bank
            print("\n--- Beneficiary Bank ---")
            bene_bank = party_inf.get('BeneficiaryBankInf', {}).get('BasicPartyInf') or {}
            print(f"AdrBankID: {bene_bank.get('AdrBankID', 'Null')}")
            print(f"Country: {bene_bank.get('Country', 'Null')}")
            id_bene_bank = bene_bank.get('ID') or {}
            print(f"ID @Type: {id_bene_bank.get('@Type', 'Null')}")
            print(f"ID #text: {id_bene_bank.get('#text', 'Null')}")
            print_address(bene_bank.get('AddressInf'), "Beneficiary Bank")
            
            # Beneficiary Party
            print("\n--- Beneficiary Party ---")
            bene_party_wrapper = party_inf.get('BeneficiaryPartyInf') or {}
            print(f"ChargeFlag: {bene_party_wrapper.get('ChargeFlag', 'Null')}")
            print(f"MailingCountry: {bene_party_wrapper.get('MailingCountry', 'Null')}")
            bene_party = bene_party_wrapper.get('BasicPartyInf') or {}
            print(f"ID: {bene_party.get('ID', 'Null')}")
            print_address(bene_party.get('AddressInf'), "Beneficiary Party")
    
    # ---- NEW SECTION: PARSE THE RESPONSE ----
    print("\n\n" + "="*20 + " RESPONSE " + "="*20)
    
    # Safely navigate to the Response Message object
    response_message = data.get(root_key, {}).get('Response', {}).get('IFML', {}).get('File', {}).get('Message', {})
    
    if not response_message:
        print("Response object not found in the provided JSON structure.")
    else:
        # ---- Response: Basic Payment ----
        response_basic_payment = response_message.get('BasicPayment', {})
        if response_basic_payment:
            print("\n--- Response: Basic Payment ---")
            print(f"TransactionUID: {response_basic_payment.get('TransactionUID', 'Null')}")
            
            # Handle the variable PartyInf in the response
            response_party_inf = response_basic_payment.get('PartyInf', {})
            if response_party_inf:
                # The example only shows BeneficiaryPartyInf. This can be expanded if other parties
                bene_party_inf = response_party_inf.get('BeneficiaryPartyInf', {}).get('BasicPartyInf') or {}
                if bene_party_inf:
                    print(" Beneficiary Party Info:")
                    print(f"     ID: {bene_party_inf.get('ID', 'Null')}")
            else:
                print(" PartyInf not found in Response BasicPayment.")
        
        # ---- Response: Audit Trail ----
        audit_trail = response_message.get('AuditTrail', {})
        if audit_trail:
            print("\n--- Response: Audit Trail ---")
            msg_status_list = audit_trail.get('MsgStatus', [])
            if isinstance(msg_status_list, list) and msg_status_list:
                for i, status in enumerate(msg_status_list):
                    print(f" Message Status [{i}]:")
                    print(f"     Code: {status.get('Code', 'Null')}")
                    print(f"     Severity: {status.get('Severity', 'Null')}")
                    
                    # Print only first 6 characters of InformationalData
                    info_data = status.get('InformationalData', 'Null')
                    if info_data and info_data != 'Null':
                        info_data_preview = str(info_data)[:6]
                        print(f"     InformationalData (first 6 chars): {info_data_preview}")
                    else:
                        print(f"     InformationalData: Null")
                    
                    print("\n--- ** ---")
            else:
                print(" MsgStatus not found or is empty in AuditTrail.")
        else:
            print(" MsgStatus not found or is empty in AuditTrail.")


def parse_transaction_from_json(json_string: str, limit: int = None):
    """
    Parses one or more wire transfer transactions from a JSON string.
    Handles both single transaction and multiple transactions in one file.
    
    Args:
        json_string: A string containing the transaction data in JSON format.
        limit: Maximum number of transactions to parse (None = parse all).
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in the provided file. {e}")
        return
    
    # Handle different JSON structures
    transactions_found = 0
    transactions_parsed = 0
    
    # Case 1: Data is a list of transactions
    if isinstance(data, list):
        total_transactions = len(data)
        print(f"\n{'#'*80}")
        print(f"Found {total_transactions} transaction(s) in the file (array format)")
        if limit:
            print(f"Will parse up to {limit} transaction(s)")
        print(f"{'#'*80}")
        
        for idx, transaction in enumerate(data, 1):
            # Check if we've reached the limit
            if limit and transactions_parsed >= limit:
                print(f"\nReached limit of {limit} transactions. Stopping...")
                break
                
            if isinstance(transaction, dict):
                # Each item in the list should be a dict with one transaction key
                root_key = next(iter(transaction), None)
                if root_key:
                    parse_single_transaction(transaction, root_key)
                    transactions_found += 1
                    transactions_parsed += 1
                else:
                    print(f"\nWarning: Transaction {idx} has no valid root key, skipping...")
            else:
                print(f"\nWarning: Transaction {idx} is not a dictionary, skipping...")
    
    # Case 2: Data is a dictionary with multiple transaction keys
    elif isinstance(data, dict):
        # Get all keys that look like transaction IDs
        transaction_keys = list(data.keys())
        total_transactions = len(transaction_keys)
        
        if not transaction_keys:
            print("JSON is empty or does not have any transaction keys.")
            return
        
        print(f"\n{'#'*80}")
        print(f"Found {total_transactions} transaction(s) in the file (object format)")
        if limit:
            print(f"Will parse up to {limit} transaction(s)")
        print(f"{'#'*80}")
        
        # Parse each transaction
        for root_key in transaction_keys:
            # Check if we've reached the limit
            if limit and transactions_parsed >= limit:
                print(f"\nReached limit of {limit} transactions. Stopping...")
                break
                
            parse_single_transaction(data, root_key)
            transactions_found += 1
            transactions_parsed += 1
    
    else:
        print("Error: Unexpected JSON structure. Expected a dictionary or array.")
        return
    
    # Print summary
    print(f"\n{'#'*80}")
    if limit and transactions_found > limit:
        print(f"SUMMARY: Parsed {transactions_parsed} out of {transactions_found} transaction(s)")
    else:
        print(f"SUMMARY: Successfully parsed {transactions_parsed} transaction(s)")
    print(f"{'#'*80}\n")


# ---- Main execution block ----
if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Parse a wire transfer transaction from a JSON file. The JSON file should contain transaction data."
    )
    
    # 2. Add a positional argument for the filename. This is a required argument.
    parser.add_argument(
        "json_file",
        help="The path to the JSON file to be parsed."
    )
    
    # 3. Add an optional argument to limit the number of transactions to parse
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Maximum number of transactions to parse (default: parse all)"
    )
    
    # 4. Parse the arguments provided at the command line
    args = parser.parse_args()
    
    # 5. Read the JSON file content with error handling
    try:
        # Use a 'with' statement for safe file handling (auto-closes the file)
        with open(args.json_file, 'r') as f:
            json_data_string = f.read()
        
        # 6. Call the parsing function with the file content and limit
        parse_transaction_from_json(json_data_string, limit=args.limit)
    
    except FileNotFoundError:
        print(f"Error: The file '{args.json_file}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
