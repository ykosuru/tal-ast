"""
ACE Pelican Violation Code Descriptions.
Reference data for code meanings and categories.
"""

# 6XXX Series - Warnings/Information
WARNING_CODES_6XXX = {
    # 6000-6099: BIC, NCH, IBAN extraction and validation
    '6000': 'Multiple BICs found',
    '6001': 'Multiple NCH Codes Derived',
    '6002': 'BIC not derived from BIC Plus',
    '6003': 'F23E code found in Sender to Receiver Information',
    '6004': 'Valid IBAN',
    '6005': 'NCH code extracted from Account Number line',
    '6006': 'Valid NCH code',
    '6007': 'NCH code extracted from field',
    '6008': 'NCH code extracted from IBAN',
    '6009': 'BIC derived from BIC Plus using NCH code',
    '6010': 'Domestic Account Number extracted from IBAN',
    '6014': 'Valid Domestic Account Number',
    '6015': 'ISO Country Code extracted from IBAN',
    '6016': 'Account number extracted from field',
    '6017': 'ISO Country extracted from receiver block',
    '6018': 'IBAN Derived',
    '6019': 'Account number formatted to required length',
    '6020': 'Account number greater than required length',
    '6021': 'ISO Country Code extracted from Field',
    '6022': 'BIC4 extracted from IBAN',
    '6023': 'BIC derived from BIC4',
    '6024': 'Valid BIC4',
    '6025': 'BIC4 extracted from Field',
    '6026': 'BIC4 extracted from receiver block',
    '6027': 'IBAN extracted from Field',
    '6028': 'Branch code extracted from field',
    '6029': 'Account number less than minimum length',
    '6030': 'Bank code formatted to required length',
    '6031': 'Branch code formatted to required length',
    '6032': 'Domestic Account Number extracted from field',
    '6033': 'BIC extracted from field',
    '6034': 'Name and Address derived from BIC',
    '6035': 'NCH Code derived from BIC',
    '6036': 'BIC4 derived from BIC',
    '6037': 'ISO Country Code derived from BIC',
    '6038': 'BBAN extracted from IBAN',
    '6039': 'BBAN extracted from field',
    '6040': 'NCH code extracted from BBAN',
    '6041': 'Valid BBAN',
    '6042': 'NCH code extracted from Domestic Account Number',
    '6043': 'BIC4 extracted from BBAN',
    '6044': 'BBAN found in Narrative',
    '6045': 'Domestic Account Number found in Narrative',
    '6046': 'Account Number extracted from IBAN',
    '6047': 'Country Code not in the list of IBAN related countries',
    '6048': 'ISO Country Code extracted from BIC',
    '6050': 'More than one embedded code present',
    '6051': 'FUNDID extracted from field',
    '6052': 'Head Office BIC derived from Multiple BICs found',
    '6053': 'ISO Country Code not found. IBAN derivation not performed',
    '6054': 'Fund Name derived from Fund Id',
    '6055': 'Party Identifier present, unable to delete redundant BIC',
    '6056': 'DDA Account number derived from FUNDID',
    '6057': 'Fund Name not derived from Fund id',
    '6058': 'No match found in Standard Settlement Instructions',
    '6059': 'Match found in Standard Settlement Instructions Table',
    '6060': 'Party 2 found in SSI Table',
    '6062': 'No match found in Chips Standard Settlement Instructions',
    '6063': 'Match found in Chips Standard Settlement Instructions',
    '6064': 'Message has Invalid Character(s)',
    '6065': 'OFFLINE BIC present or derived',
    '6066': 'BIC found is of type BEI',
    '6084': 'Account title mismatch',
    '6086': 'Message cannot be repaired further',
    
    # 6400-6499: Name and Address derivation
    '6400': 'Account Number found in Name and Address',
    '6401': 'CHIPS ABA found in Name and Address',
    '6402': 'Fedwire ABA found in Name and Address',
    '6403': 'CHIPS UID found in Name and Address',
    '6404': 'Account Number derived from CIF database using Name and Address',
    '6405': 'CHIPS ABA derived from CHIPS database using Name and Address',
    '6406': 'Fedwire ABA derived from Fedwire database using Name and Address',
    '6407': 'CHIPS UID derived from CHIPS database using Name and Address',
    '6408': 'Valid Account No. found in Name and Address',
    '6409': 'Valid CHIPS ABA found in Name and Address',
    '6410': 'Valid Fedwire ABA found in Name and Address',
    '6411': 'Valid CHIPS UID found in Name and Address',
    '6412': 'Address Partially matched. Push down repair required',
    '6413': 'Standard Settlement Instruction Derived',
    '6414': 'Preferred CHIPS correspondent Derived',
    '6415': 'CHIPS ABA Derived from BIC',
    '6416': 'CHIPS ABA Derived from head office BIC',
    '6417': 'BIC Derived from CHIPS ABA',
    '6418': 'CHIPS ABA found for head office BIC',
    '6419': 'CHIPS ABA not found for branch BIC',
    '6420': 'No space available for push down, Head office BIC not repaired',
    '6421': 'Push Down repair performed',
    '6422': 'Message classified as book transfer',
    '6424': 'Multiple SSI entries found',
    '6427': 'Fedwire code ineligible for funds found',
    '6428': 'Fedwire derived using SSI is inconsistent with embedded Fedwire',
    '6430': 'Product Code Found',
    '6431': 'Debit Party DDA number present',
    '6432': 'Credit Party DDA number derived',
    '6433': 'Bank ID derived',
    '6434': 'Reference pattern Found',
    '6435': 'Multiple Reference patterns Found',
    '6436': 'Unique multiple Reference patterns',
    '6437': 'Keyword Found',
    '6438': 'ISO Country Code derived',
    '6439': 'No Pattern Found',
    '6440': 'No Pattern Found',
    '6441': 'Fee Code Derived',
    '6442': 'Branch ID Derived',
    '6453': 'Multiple DDA code found',
    '6454': 'Multiple CHIPS ABA found',
    '6455': 'Multiple FEDABA Found',
    '6456': 'Multiple CHIPS UID found',
    '6473': 'No Update as Fee code derived is same as Fee code in Message',
    
    # 6500-6599: Product and Fee codes
    '6500': 'Product Code Found',
    '6501': 'Debit Party DDA NUMBER derived',
    '6502': 'Credit Party DDA NUMBER derived',
    '6503': 'DDA enrolled for the product',
    '6504': 'Bank ID derived',
    '6505': 'POSITIVE Specific Reference Pattern FOUND',
    '6506': 'Fee Parameter NOT Setup IN DATABASE',
    '6507': 'Fee Parameter Matched',
    '6508': 'Fee Parameter NOT Matched',
    '6509': 'Country Code NOT Setup IN DATABASE',
    '6510': 'Country Code Matched',
    '6511': 'Country Code NOT Matched',
    '6512': 'Fee Code derived',
    '6527': 'Charge Field Contain OUR',
    '6528': 'Negative Global Reference Pattern FOUND',
    '6529': 'Negative Specific Reference Pattern FOUND',
    
    # 6700-6799: BIC Plus derivation
    '6701': 'BIC derived from BIC Plus using Name and Address',
    '6702': 'BIC found in Name and Address',
    '6708': 'Bank Code retrieved from field',
    '6711': 'IBAN found in Account number line',
    '6712': 'Account number found in Narrative',
    '6714': 'IBAN found in Narrative',
    '6716': 'NCH code retrieved from field',
    '6717': 'NCH code derived from BIC Plus',
    '6720': 'Account number found in Account number line',
    '6750': 'Valid BIC found in Name and Address',
    '6789': 'Derived Field 58 from Field 72',
    '6791': 'Message is STP',
    '6801': 'Original message value date change to system date',
    '6802': 'Cut-off time has been overridden',
    '6803': 'Duplicate message has been released',
    '6854': 'Name and Address present',
    '6855': 'Account NUMBER present',
    '6856': 'Account NUMBER NOT present',
    '6857': 'Sending Bank Reference present',
    '6858': 'Charge flag present',
    '6859': 'Secondary wire flag NOT present',
    '6860': 'PO Box NUMBER present',
    '6868': 'Msg is not supported in current product version',
    '6869': 'FCY currency functionality turned off',
    '6874': 'MUR Enrichment done successfully',
    '6875': 'Exchange rate not ascertain or to be updated by user',
    '6923': 'Phrase inserted in Field 72',
    '6924': 'Translocation of NCH code',
    '6925': 'NCH extracted from IBAN cannot be validated',
    '6926': 'No Update as Fee code derived is same as Fee code in Message',
    '6927': 'No Space available for push down',
}

# 7XXX Series - Warnings/Information  
WARNING_CODES_7XXX = {
    # 7000-7099: Codewords and field validation
    '7000': 'BIC could not be derived',
    '7001': 'Codeword INS present in field 72 when Ordering Institution not present',
    '7002': 'Codeword ACC, INT or REC present in field 72',
    '7003': 'Codeword RCB present in field 72 when Sender\'s or Receiver\'s Correspondent not present',
    '7004': 'Codeword REC present in field 72',
    '7005': 'Codeword CHARGE present in field 72',
    '7006': 'Codeword OCMT present in field',
    '7007': 'Codeword EXCH present in field',
    '7008': 'Account number found in field 72',
    '7009': 'Codeword RFB present in field',
    '7010': 'Beneficiary account no. not present when codeword in field 72 is not BENONLY, CHEQUE or HOLD',
    '7011': 'Beneficiary account no. present when codeword in field 72 is BENONLY, CHEQUE or HOLD',
    '7012': 'Beneficiary account not valid IBAN',
    '7013': 'Codeword which can be used in field 23 are present in field 72',
    '7014': 'BIC same as that of receiver',
    '7015': 'NARRATIVE PRESENT',
    '7016': 'Account number found in Narrative and not in Account Number line',
    '7017': 'Nostro account number not present',
    '7018': 'Narrative present without formatted NCH code',
    '7019': 'Sender\'s or Receiver\'s correspondent not present',
    '7020': 'Beneficiary account no. present when codeword is not BONL or HOLD',
    '7021': 'D-A could not be done',
    '7022': 'IBAN inconsistent with field 57A',
    '7023': 'Redundant phrases found in Field 72',
    '7024': 'Name and Address could not be derived from BIC',
    '7025': 'Account number greater than required length',
    '7026': 'Extracted NCH inconsistent with Narrative present',
    '7027': 'Extracted NCH Code matches with Multiple Countries',
    '7028': 'BIC derived from BICPLUS with inappropriate Subtype',
    '7029': 'BIC PRESENT IS NOT AN IBAN BIC',
    '7030': 'IBAN DERIVATION NOT SUPPORTED FOR THE COUNTRY',
    '7031': 'MESSAGE AND BICPLUSIBAN COUNTRY CODE MISMATCH. BICPLUSIBAN COUNTRY CODE WILL BE USED TO DERIVE IBAN',
    '7033': 'Receiver\'s correspondent present and Sender\'s correspondent not present',
    '7034': 'Offline BIC present',
    '7035': 'Party Identifier present without correctly formatted FEDABA code',
    '7037': 'Account number does not match with CLABE',
    '7038': 'Multiple party information present cannot be split',
    
    # 7100-7199: STP and exception handling
    '7151': 'STP Message moved to exception queue due to STP-TO-REPAIR flag',
    
    # 7300-7399: Validation errors
    '7301': 'Settlement Date is not in allowed range',
    '7302': 'Invalid BIC',
    '7303': 'Invalid IBAN',
    '7304': 'Invalid Currency',
    '7305': 'Amount is equal to 0',
    '7306': 'Total Interbank Settlement amount is greater than allowed range',
    '7307': 'Interbank Settlement amount is greater than allowed range',
    '7345': 'Domestic Account Validation Cannot be done',
    '7355': 'Hash code generation failed',
    '7356': 'Hash code missing',
    '7358': 'Message blocked by scanning system',
    '7359': 'Failed to derive target channel',
    '7360': 'Message moved to exception queue due to amount exceeding threshold limit',
    
    # 7400-7499: Multiple codes and derivation issues
    '7450': 'Address Match Failed',
    '7451': 'Nostro Account identified',
    '7452': 'No Fields for Push Down available',
    '7453': 'Multiple DDA code found',
    '7454': 'Multiple CHIPS ABA found',
    '7455': 'Multiple FEDABA Found',
    '7456': 'Multiple CHIPS UID found',
    '7457': 'Multiple BIC Code Found',
    '7458': 'No repair for FED/CHP messages as code derived is not DDA',
    '7459': 'Fund Name derived using Fund Number not present in Narrative',
    '7460': 'BICCODE same as Credit Party',
    '7465': 'Product Code not found',
    '7466': 'Bank ID not found',
    '7467': 'ISO Country Code not derived',
    '7468': 'Multiple values for Reference pattern found',
    '7469': 'Invalid Reference pattern',
    '7470': 'Inconsistent Multiple Reference patterns',
    '7471': 'Fields to be scanned not specified',
    '7472': 'Fee Code not Derived',
    '7473': 'No update as Fee code derived is same as Fee code in message',
    '7475': 'Msg Currency is Foreign Cur And Foreign Cur Processing Mode is Non-Stp',
    
    # 7500-7599: Product and enrollment
    '7511': 'Country Code not matched',
    '7526': 'DDA NOT enrolled FOR the product',
    '7530': 'Fee Derivation Discontinued',
    '7531': 'Product NOT eligible WITH Charge Field SHA OR BEN',
    '7532': 'PRESAM Derivation Discontinued',
    
    # 7700-7799: Duplicate and routing
    '7701': 'DUPLICATE MESSAGE',
    '7706': 'Message cannot be routed',
    '7707': 'POSSIBLE DUPLICATE',
    '7737': 'Message identified as duplicate based on business parameters',
    '7753': 'B to A could not be done',
    '7783': 'As per SWF for Corporates, instructing party should not be same as account owner',
    '7784': 'As per SWF for Corporates, acc number, name & addr or BEI of ord cust is required',
    '7785': 'As per SWF for Corporates, instructing party should not be same as the creditor',
}

# 8XXX Series - Errors/Validation Failures
ERROR_CODES = {
    '8001': 'Invalid BIC',
    '8003': 'File name derivation failed',
    '8004': 'IBAN cannot be derived',
    '8005': 'Invalid BIC4',
    '8006': 'Invalid country code',
    '8007': 'Number of fractional digits of amount exceeds maximum allowable',
    '8022': 'IBAN inconsistent with Account With Institution BIC',
    '8023': 'IBAN inconsistency found in message',
    '8024': 'BBAN inconsistency found in message',
    '8025': 'Domestic Account Number inconsistency found in message',
    '8026': 'NCH inconsistency found in message',
    '8027': 'ISO Country Code inconsistency found in message',
    '8028': 'BIC4 inconsistency found in message',
    '8029': 'Account Number inconsistency found in message',
    '8030': 'IBAN derivation not supported for the country',
    '8033': 'CLABE inconsistency found in message',
    '8034': 'Forced Debit is not allowed and hence message cannot be processed',
    '8035': 'FCDA account validation failed',
    '8036': 'FCDA account name matching failed',
    '8124': 'Invalid currency',
    '8464': 'Target Channel not derived',
    '8465': 'Product code not found',
    '8472': 'Fee code not derived',
    '8851': 'Incorrect field size',
    '8852': 'Incorrect length of attribute',
    '8853': 'Incorrect number format',
    '8892': 'Invalid Account number',
    '8894': 'Invalid IBAN',
    '8895': 'Invalid NCH code',
    '8896': 'Invalid Domestic Account Number',
    '8897': 'Invalid BBAN',
    '8898': 'IBAN Check Digit calculation/validation failed',
    '8905': 'Hash code Mismatch',
    '8906': 'Message in wrong flow',
}

# 9XXX Series - Repairs/Modifications
REPAIR_CODES = {
    '9000': 'NCH code cleaned',
    '9001': 'D Field deleted',
    '9002': 'Account number cleaned of non alpha numeric characters',
    '9004': 'IBAN repaired in Account Number line',
    '9005': 'BIC replaced by derived BIC',
    '9006': 'IBAN Cleaned',
    '9007': 'Account number replaced by IBAN',
    '9008': 'IBAN to BIC repair done',
    '9009': 'Field Cleaned',
    '9010': 'Field 23E generated',
    '9012': 'IBAN Formatted in field',
    '9013': 'Name and Address repair done',
    '9014': 'BBAN cleaned',
    '9015': 'Domestic account number cleaned',
    '9017': 'Multiple party information present',
    '9018': 'Duplicate party information removed',
    '9019': 'Party Identifier cleaned of non alpha numeric characters',
    '9020': 'Party Identifier cleaned of non alpha numeric characters',
    '9021': 'FEDABA Code correctly formatted',
    '9022': 'Account number formatted to required length',
    '9023': 'Fund to Account repair done',
    '9024': 'Push Up performed',
    '9025': 'CLABE repaired',
    '9026': 'Purpose Information found in Field 72',
    '9027': 'FCDA account reformatted',
    '9028': 'NCH Code strip off',
    '9029': 'GL account insertion for FED',
    '9030': 'DDA to GL attribute type conversion',
    '9031': 'Trust account to GL number insertion',
    '9032': '8 char BIC insertion',
    '9439': 'No Pattern Found',
    '9475': 'D-A using Account No. from Name and Address',
    '9476': 'D-A using CHIPS ABA from Name and Address',
    '9477': 'D-A using FED ABA from Name and Address',
    '9478': 'D-A using CHIPS UID from Name and Address',
    '9479': 'Account Number Cleaned',
    '9480': 'Push Down of type Push 1 performed',
    '9481': 'Push Down of type Push 2 performed',
    '9482': 'Push Down of type partial Push 2 performed',
    '9483': "Receiver's duplicate information removed",
    '9484': 'Repaired using info derived from Ban to Bank Info',
    '9485': 'D-A using NCH from Name and Address',
    '9486': 'A to D performed',
    '9487': "Receiver's BIC information removed",
    '9488': 'Partial push down performed',
    '9490': 'Fee Code Updated',
    '9491': 'Tariff Fee Updated',
    '9492': 'Charge Field updated with value B',
    '9493': 'Debit Charge Field update with value N',
    '9494': 'Field updated with GTYOUR',
    '9495': 'Field updated with FCY',
    '9496': 'Field updated with FCY OUR CHARGES',
    '9497': 'Field updated with FCY GTYOUR',
    '9498': 'PRESAM SecWir updated with Y',
    '9901': 'D-A using BIC retrieved from the field',
    '9910': 'BIC repaired from BIC found in field 72',
    '9917': 'BIC repaired from BIC Plus using field 72',
    '9918': 'Attribute type auto corrected',
    '9932': 'Account number line repaired with Account number',
    '9935': 'Account number line repaired by NCH code',
    '9936': 'D-A using BIC from Name and Address',
    '9938': 'Deleted redundant phrases from Sender to Receiver information',
    '9961': 'BIC Derived from Name and Address',
    '9962': 'Account number updated',
    '9963': 'Account number added',
    '9964': 'Advice instructions modified',
    '9965': 'Charge flag modified',
    '9966': 'Secondary wire flag modified',
    '9967': 'Cleared PO Box number',
    '9968': 'Parameter values added',
    '9969': 'Related Amounts added',
    '9970': 'D-A using BIC from Name and Address',
    '9971': 'Negative ref cancels pos ref and benededuct',
    '9978': 'Numeric BIC derived using field 58',
    '9979': 'Numeric BIC derived using field 57',
    '9980': 'Alpha BIC repaired with numeric BIC',
    '9981': 'Merged BIC derived using field 58',
    '9982': 'Merged BIC derived using field 57',
    '9983': 'Exception BIC derived using field 58',
    '9984': 'Exception BIC derived using field 57',
    '9985': 'BIC from CHIPS ABA Repaired',
    '9986': 'Head Office BIC Repaired',
    '9987': 'Code Consistent',
    '9990': 'Duplicate information of CDT is removed BBI removed',
    '9991': 'Account Number pushed down to BBI from BBK',
    '9992': 'NCH code has been translocated to BBI',
    '9999': 'Field Repaired',
}

# Combined lookup
ALL_CODES = {**WARNING_CODES_6XXX, **WARNING_CODES_7XXX, **ERROR_CODES, **REPAIR_CODES}


def get_code_description(code: str) -> str:
    """Get description for a code, handling composite codes like '8004_BNPPTY'."""
    # Handle composite codes (code_party format)
    base_code = code.split('_')[0] if '_' in code else code
    return ALL_CODES.get(base_code, 'Unknown code')


def get_code_category(code: str) -> str:
    """Get category (Error, Warning, or Repair) for a code."""
    base_code = code.split('_')[0] if '_' in code else code
    if base_code in ERROR_CODES:
        return 'Error'
    elif base_code in REPAIR_CODES:
        return 'Repair'
    elif base_code in WARNING_CODES_6XXX or base_code in WARNING_CODES_7XXX:
        return 'Warning'
    return 'Unknown'


def is_target_series(code: str, series: list = None) -> bool:
    """Check if code is in target series. Default: all series."""
    if series is None:
        series = ['6', '7', '8', '9']
    base_code = code.split('_')[0] if '_' in code else code
    return any(base_code.startswith(s) for s in series)


# Code to relevant IFML fields mapping (for feature hints)
CODE_IFML_RELEVANCE = {
    '8001': ['bic', 'bic_length', 'has_bic'],  # Invalid BIC
    '8004': ['iban_country', 'has_account', 'account_type'],  # IBAN cannot be derived
    '8005': ['bic_length', 'bic'],  # Invalid BIC4
    '8006': ['country', 'bic_country', 'iban_country'],  # Invalid country code
    '8022': ['bic_iban_match', 'bic_country', 'iban_country'],  # IBAN/BIC inconsistent
    '8023': ['iban_country', 'account_type'],  # IBAN inconsistency
    '8024': ['account_type'],  # BBAN inconsistency
    '8027': ['country', 'bic_country', 'iban_country'],  # ISO Country Code inconsistency
    '8028': ['bic_length', 'bic'],  # BIC4 inconsistency
    '8029': ['has_account', 'account_type'],  # Account Number inconsistency
    '8030': ['iban_country', 'country'],  # IBAN derivation not supported
    '8892': ['has_account', 'account_type'],  # Invalid Account number
    '8894': ['account_type', 'iban_country'],  # Invalid IBAN
    '8897': ['account_type'],  # Invalid BBAN
    '8898': ['account_type', 'iban_country'],  # IBAN Check Digit failed
    '9004': ['account_type', 'iban_country'],  # IBAN repaired
    '9005': ['bic', 'has_bic'],  # BIC replaced
    '9006': ['account_type'],  # IBAN Cleaned
    '9007': ['account_type', 'iban_country'],  # Account replaced by IBAN
    '9008': ['bic', 'account_type'],  # IBAN to BIC repair
    '9013': ['address_lines', 'has_name'],  # Name and Address repair
    '9014': ['account_type'],  # BBAN cleaned
    '9017': ['present'],  # Multiple party information
    '9018': ['present'],  # Duplicate party removed
    '9032': ['bic_length'],  # 8 char BIC insertion
    '9961': ['bic', 'address_lines'],  # BIC Derived from Name and Address
}


def get_relevant_features(code: str) -> list:
    """Get IFML features relevant to a code for feature engineering hints."""
    base_code = code.split('_')[0] if '_' in code else code
    return CODE_IFML_RELEVANCE.get(base_code, [])
