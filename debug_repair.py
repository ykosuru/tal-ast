import json
from ace_repair_model import RuleLearner, RuleExecutor
from copy import deepcopy

# Load system
learner = RuleLearner.load('models')
executor = RuleExecutor(learner)

# Test payment
payment = {
    "cdtrAgt": {
        "FinInstnId": {
            "BICFI": "RBSIGB2L",
            "Nm": "RBS INTERNATIONAL (LONDON)"
        }
    }
}

print("=== ORIGINAL ===")
print(json.dumps(payment, indent=2))

# Test rule 6021 directly
result = deepcopy(payment)
print("\n=== TESTING RULE 6021 ===")

entity = result.get('cdtrAgt', {})
print(f"Got entity: {bool(entity)}")

fin_instn = entity.get('finInstnId', entity.get('FinInstnId', {}))
print(f"Got finInstnId: {bool(fin_instn)}")
print(f"finInstnId contents: {fin_instn}")

bic = fin_instn.get('bicFi', fin_instn.get('BICFI', ''))
print(f"Got BIC: '{bic}'")

if bic and len(bic) >= 6:
    country = bic[4:6].upper()
    print(f"Extracted country: {country}")
    entity['ctryOfRes'] = country
    print("✓ Added ctryOfRes")
else:
    print("✗ Failed to extract country")

print("\n=== RESULT ===")
print(json.dumps(result, indent=2))
