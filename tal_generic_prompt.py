#!/usr/bin/env python3
"""
Comprehensive TAL to Java Translation Prompt Generator
Ensures complete logic preservation and business rule translation
"""

from typing import Dict, Any, Optional, List


class ComprehensivePromptGenerator:
    """Generate extremely detailed prompts that ensure complete translation."""
    
    @staticmethod
    def generate_translation_prompt(
        context: Dict[str, Any],
        strict_mode: bool = True,
        include_validation: bool = True,
        financial_code: bool = True
    ) -> str:
        """
        Generate a comprehensive prompt that ensures COMPLETE translation.
        
        Args:
            context: Translation context from TranslationContextBuilder
            strict_mode: Enable strict translation requirements
            include_validation: Include validation checklist
            financial_code: Enable financial code specific requirements
        """
        
        prompt = f"""# TAL to Java Complete Translation Task

## 🎯 MISSION CRITICAL OBJECTIVE

You are translating production-level TAL (Transaction Application Language) code to Java.
This is NOT a code generation exercise - this is a COMPLETE, LINE-BY-LINE translation task.

**Functionality**: {context['functionality']}

**Translation Scope**: {context['summary']['total_procedures']} procedures
- Primary procedures: {context['summary']['primary_procedures']} (MUST translate completely)
- Dependency procedures: {context['summary']['total_procedures'] - context['summary']['primary_procedures']} (MUST translate completely)
- Data structures: {context['summary']['total_structures']}
- Total source code: {context['summary']['code_extraction']['total_chars']:,} characters

---

## ⚠️ CRITICAL REQUIREMENTS - READ FIRST

### What "Complete Translation" Means:

1. ✅ **Every line of TAL business logic** → Equivalent Java code
2. ✅ **Every IF/ELSE/CASE statement** → Translated exactly
3. ✅ **Every calculation** → Preserved with same precision
4. ✅ **Every validation check** → Implemented in Java
5. ✅ **Every procedure call** → Mapped to Java method call
6. ✅ **Every error handling path** → Fully implemented
7. ✅ **Every data structure** → Converted to Java class/record

### Absolutely FORBIDDEN:

❌ **NO** skeleton methods with "// TODO: implement"
❌ **NO** placeholder comments like "// Business logic goes here"
❌ **NO** simplified logic that loses details
❌ **NO** missing error handling
❌ **NO** skipped validations
❌ **NO** omitted calculations
❌ **NO** generic comments like "// Handle the rest"
❌ **NO** assumptions about "standard behavior" - translate EXACTLY what's in TAL

### Success Criteria:

Your translation will be considered COMPLETE only if:
- ✅ A Java developer can trace every line back to TAL source
- ✅ A QA engineer can verify logic equivalence line-by-line
- ✅ The code would pass a production code review
- ✅ ZERO business logic is lost, simplified, or approximated
- ✅ Another LLM reading your Java could recreate the TAL logic

---

## 📋 TRANSLATION METHODOLOGY

Follow this EXACT process for EACH procedure:

### Phase 1: ANALYZE (Before Writing Any Code)

For each TAL procedure, create a detailed analysis:

```
PROCEDURE ANALYSIS: [Procedure Name]

PURPOSE:
  [What this procedure does in 1-2 sentences]

INPUT PARAMETERS:
  - param1: [type] - [purpose and constraints]
  - param2: [type] - [purpose and constraints]

BUSINESS LOGIC BREAKDOWN:
  Section 1 (Lines X-Y): [What it does]
    - Condition 1: [Logic]
    - Condition 2: [Logic]
  
  Section 2 (Lines A-B): [What it does]
    - Calculation: [Formula]
    - Validation: [Check]
  
  Section 3 (Lines C-D): [What it does]
    - Call to: [Procedure] with [parameters]
    - Result handling: [How result is used]

CONTROL FLOW:
  [Diagram or description of the flow]
  IF condition1:
    → Action A
  ELSE IF condition2:
    → Action B
  ELSE:
    → Action C

DATA DEPENDENCIES:
  - Reads: [variables/structures]
  - Writes: [variables/structures]
  - Calls: [other procedures]

ERROR HANDLING:
  - Error code -1: [Meaning and when it occurs]
  - Error code -2: [Meaning and when it occurs]
  - Success code 1: [Meaning]

OUTPUT/SIDE EFFECTS:
  - Returns: [type and meaning]
  - Modifies: [what global state changes]
```

### Phase 2: MAP (TAL to Java Equivalents)

Create a mapping table:

```
TAL CONSTRUCT                    | JAVA EQUIVALENT
---------------------------------|----------------------------------
INT status                       | int status
STRING .name                     | String name
FIXED(2) amount                  | BigDecimal amount
ARRAY[0:9] OF INT                | int[] (size 10)
IF x = 1 THEN                    | if (x == 1) {
FOR i := 0 TO 9 DO               | for (int i = 0; i <= 9; i++) {
CALL VALIDATE(param)             | validate(param)
status := -1                     | status = -1
```

### Phase 3: IMPLEMENT (Write Complete Java Code)

Write Java code with:

1. **Class Structure**:
   ```java
   /**
    * Complete Javadoc describing the class
    * Maps to TAL file: [filename]
    */
   @Service  // or appropriate annotation
   public class [ServiceName] {
   ```

2. **Method Implementation**:
   ```java
   /**
    * [Method purpose - from TAL procedure]
    * 
    * @param param1 [description - from TAL]
    * @return [description - from TAL]
    * 
    * TAL equivalent: [PROCEDURE_NAME]
    * TAL location: [file:line]
    */
   public ReturnType methodName(ParamType param1) {
       // TAL line X-Y: [What this section does]
       [Java code implementing those TAL lines]
       
       // TAL line Z: [Specific logic]
       [Java code for that line]
   }
   ```

3. **Inline Documentation**:
   - Comment EVERY section mapping to TAL code
   - Reference TAL line numbers
   - Explain complex business rules
   - Document why certain approaches were chosen

### Phase 4: VERIFY (Self-Check)

Before considering a procedure complete, verify:

```
VERIFICATION CHECKLIST for [Procedure Name]:

Input Validation:
  [ ] TAL line __: Null/empty checks → Java equivalent
  [ ] TAL line __: Range validation → Java equivalent
  [ ] TAL line __: Type checking → Java equivalent

Business Logic:
  [ ] TAL line __: IF condition → Java if statement
  [ ] TAL line __: Calculation → Java calculation
  [ ] TAL line __: FOR loop → Java for/while loop
  [ ] TAL line __: Procedure call → Java method call
  [ ] TAL line __: Data transformation → Java equivalent

Error Handling:
  [ ] TAL line __: Error code -1 → Java exception/return
  [ ] TAL line __: Error code -2 → Java exception/return
  [ ] TAL line __: Success handling → Java success path

Data Operations:
  [ ] TAL line __: Read operation → Java getter
  [ ] TAL line __: Write operation → Java setter
  [ ] TAL line __: Structure access → Java object access

Completeness:
  [ ] Every TAL line has a Java equivalent
  [ ] No TODO comments
  [ ] No placeholder implementations
  [ ] All edge cases handled
  [ ] All error paths implemented
```

---

"""

        # Add financial code specific requirements
        if financial_code:
            prompt += """## 💰 FINANCIAL CODE CRITICAL REQUIREMENTS

This code handles financial transactions. **EVERY DETAIL IS CRITICAL**.

### Absolute Rules for Financial Code:

1. **Amount Handling** (MANDATORY):
   ```java
   // ✅ CORRECT: Use BigDecimal for ALL amounts
   BigDecimal amount = new BigDecimal("1234.56");
   BigDecimal fee = amount.multiply(new BigDecimal("0.01"));
   
   // ❌ WRONG: NEVER use double or float for money
   double amount = 1234.56;  // FORBIDDEN - loses precision
   ```

2. **Decimal Precision** (MANDATORY):
   - Preserve ALL decimal places from TAL
   - Use explicit rounding modes (RoundingMode.HALF_UP, etc.)
   - Document rounding behavior
   
   ```java
   // TAL: FIXED(2) amount  → Java:
   amount.setScale(2, RoundingMode.HALF_UP);
   ```

3. **Threshold Checks** (EXACT TRANSLATION):
   ```tal
   ! TAL code
   IF amount > 10000.00 THEN
   ```
   
   ```java
   // ✅ CORRECT: Exact threshold, BigDecimal comparison
   if (amount.compareTo(new BigDecimal("10000.00")) > 0) {
   
   // ❌ WRONG: Approximate or simplified
   if (amount > 10000) {  // Wrong: loses precision, uses double
   ```

4. **Validation Order** (PRESERVE EXACTLY):
   - TAL validation order may be security-critical
   - DO NOT reorder validations for "clarity" or "efficiency"
   - Translate in EXACT order shown in TAL

5. **Status/Error Codes** (EXACT MAPPING):
   ```tal
   ! TAL status codes
   status := -1  ! Invalid amount
   status := -2  ! Insufficient funds
   status := -3  ! Compliance failure
   status := 1   ! Success
   ```
   
   ```java
   // Option 1: Keep numeric codes if that's the pattern
   return -1;  // Invalid amount
   
   // Option 2: Use enums (better)
   return TransferStatus.INVALID_AMOUNT;
   
   // But DOCUMENT what each code means from TAL
   ```

6. **Audit Trail** (MANDATORY):
   - If TAL logs something, Java MUST log it
   - Include same level of detail
   - Preserve: timestamps, user IDs, amounts, status changes
   
   ```java
   log.info("Wire transfer processed: id={}, amount={}, status={}, user={}", 
            wireId, amount, status, userId);
   ```

7. **Compliance Checks** (NO SHORTCUTS):
   - OFAC screening: Translate exactly, don't skip
   - AML checks: Implement every rule
   - KYC requirements: All validations must be present
   - Sanctions: Preserve all list checks
   
   ```tal
   ! TAL compliance check
   IF country IN sanctions_list THEN
     CALL BLOCK_TRANSACTION;
   ```
   
   ```java
   // ✅ CORRECT: Exact translation
   if (sanctionsList.contains(country)) {
       blockTransaction();
   }
   
   // ❌ WRONG: Simplified or skipped
   // TODO: Add sanctions check  // FORBIDDEN
   ```

8. **Transaction Boundaries** (PRESERVE):
   ```tal
   ! TAL transaction pattern
   BEGIN TRANSACTION;
     CALL UPDATE_ACCOUNT;
     CALL UPDATE_LEDGER;
   COMMIT;
   ```
   
   ```java
   // Java equivalent
   @Transactional
   public void processTransfer() {
       updateAccount();
       updateLedger();
       // Commit handled by @Transactional
   }
   ```

9. **Regulatory Requirements**:
   - Preserve ALL data retention logic
   - Keep ALL audit fields
   - Maintain ALL compliance timestamps
   - Document regulatory reason for each check

---

"""

        # Add TAL language reference
        prompt += """## 📚 TAL LANGUAGE QUICK REFERENCE

### TAL Syntax → Java Translation Guide:

#### Data Types:
```tal
INT                     →  int
INT(32)                 →  int  
STRING                  →  String
STRING .ptr             →  String (pointer ignored in Java)
FIXED(n)                →  BigDecimal (n = decimal places)
REAL                    →  double (but use BigDecimal for money!)
UNSIGNED               →  int (Java doesn't have unsigned, handle carefully)
```

#### Arrays:
```tal
ARRAY[0:9] OF INT      →  int[] array = new int[10];
ARRAY[1:100] OF STRING →  String[] array = new String[100];
                          // Note: TAL arrays can start at any index
                          // Java always starts at 0
```

#### Structures:
```tal
STRUCT payment_record;  →  public class PaymentRecord {
BEGIN                   →      // Fields
  INT transaction_id;   →      private int transactionId;
  STRING .payee;        →      private String payee;
  FIXED(2) amount;      →      private BigDecimal amount;
END;                    →  }
```

#### Procedures:
```tal
PROC validate(amount);  →  public int validate(BigDecimal amount) {
  BEGIN                 →      // method body
    ...                 →      ...
  END;                  →  }
```

#### Control Flow:
```tal
IF condition THEN       →  if (condition) {
  statement;            →      statement;
                        →  }

IF x THEN               →  if (x) {
  statement1            →      statement1;
ELSE                    →  } else {
  statement2;           →      statement2;
                        →  }

FOR i := 0 TO 9 DO      →  for (int i = 0; i <= 9; i++) {
  statement;            →      statement;
                        →  }

WHILE condition DO      →  while (condition) {
  statement;            →      statement;
                        →  }

CASE status OF          →  switch (status) {
  BEGIN                 →      case -1:
    -1: action1;        →          action1; break;
    -2: action2;        →      case -2:
    OTHERWISE: default; →          action2; break;
  END;                  →      default:
                        →          default; break;
                        →  }
```

#### Operators:
```tal
:=                      →  =       (assignment)
=                       →  ==      (equality)
<>                      →  !=      (inequality)
AND                     →  &&
OR                      →  ||
NOT                     →  !
```

#### Procedure Calls:
```tal
CALL procedure(args);   →  procedure(args);
result := procedure();  →  result = procedure();
```

#### Comments:
```tal
! This is a comment     →  // This is a comment
                        →  /* This is a comment */
```

#### String Operations:
```tal
@variable               →  // Address-of operator (usually not needed in Java)
variable ':=' "text"    →  variable = "text";
```

#### Special Patterns:
```tal
BEGIN                   →  {
END                     →  }
PROC name;              →  public void name() {
  FORWARD;              →      // Forward declaration - create interface
END;                    →  }
```

---

"""

        # Add the actual translation section
        prompt += f"""## 📝 PROCEDURES TO TRANSLATE

You are translating {len(context['primary_procedures'])} procedures completely.

### Translation Requirements:
1. Start with **logic analysis** for each procedure
2. Then provide **complete Java implementation**
3. Include **verification checklist** for each
4. Add **unit test skeleton** for each

---

"""

        # Primary procedures
        prompt += f"""## 🎯 PRIMARY PROCEDURES ({len(context['primary_procedures'])} procedures)

These are the MAIN procedures that implement the {context['functionality']} functionality.
**ALL must be translated completely with FULL implementation.**

"""

        for i, proc in enumerate(context['primary_procedures'], 1):
            prompt += f"""
{'='*70}
### PROCEDURE {i}/{len(context['primary_procedures'])}: {proc['name']}
{'='*70}

**Source Location**: `{proc['file']}:{proc['line']}`
**Parameters**: {', '.join(proc['parameters']) if proc['parameters'] else 'none'}
**Return Type**: {proc['return_type'] or 'void'}
**Code Size**: {proc.get('code_length', 0):,} characters

#### Dependency Context:
"""
            deps = proc.get('dependencies', {})
            if deps.get('calls'):
                prompt += f"- **Calls**: {', '.join(deps['calls'][:10])}"
                if len(deps['calls']) > 10:
                    prompt += f" (+ {len(deps['calls']) - 10} more)"
                prompt += "\n"
            if deps.get('called_by'):
                prompt += f"- **Called by**: {', '.join(deps['called_by'][:5])}"
                if len(deps['called_by']) > 5:
                    prompt += f" (+ {len(deps['called_by']) - 5} more)"
                prompt += "\n"
            if deps.get('uses_structures'):
                prompt += f"- **Uses structures**: {', '.join(deps['uses_structures'])}\n"
            if deps.get('uses_variables'):
                prompt += f"- **Uses variables**: {', '.join(deps['uses_variables'][:10])}"
                if len(deps['uses_variables']) > 10:
                    prompt += f" (+ {len(deps['uses_variables']) - 10} more)"
                prompt += "\n"
            
            code_display = proc['code'] if proc['code'] else "// ⚠️ SOURCE CODE NOT AVAILABLE"
            
            if not proc['code']:
                prompt += f"""
#### ⚠️ WARNING: Source Code Not Available
The source code for this procedure could not be extracted.
- File: {proc['file']}
- Line: {proc['line']}
- You may need to request the source code separately or indicate this procedure needs manual extraction.

"""
            else:
                prompt += f"""
#### TAL Source Code:
```tal
{code_display}
```

#### REQUIRED OUTPUT FOR THIS PROCEDURE:

1. **LOGIC ANALYSIS**:
   ```
   PROCEDURE ANALYSIS: {proc['name']}
   
   PURPOSE:
     [Explain what this procedure does]
   
   INPUT PARAMETERS:
     [List each parameter with type and purpose]
   
   BUSINESS LOGIC BREAKDOWN:
     [Break down the logic section by section]
   
   ERROR HANDLING:
     [Document each error code and condition]
   ```

2. **COMPLETE JAVA IMPLEMENTATION**:
   ```java
   /**
    * [Complete Javadoc]
    * 
    * TAL equivalent: {proc['name']}
    * TAL location: {proc['file']}:{proc['line']}
    */
   public [ReturnType] [methodName]([parameters]) {{
       // TAL line X-Y: [Explain this section]
       [Complete implementation - NO PLACEHOLDERS]
       
       // TAL line Z: [Specific logic]
       [Implementation for that line]
   }}
   ```

3. **VERIFICATION CHECKLIST**:
   ```
   VERIFICATION for {proc['name']}:
   [ ] All input validations translated
   [ ] All business logic preserved
   [ ] All error paths implemented
   [ ] All calculations present
   [ ] All procedure calls mapped
   [ ] No TODO or placeholder comments
   ```

4. **UNIT TEST SKELETON**:
   ```java
   @Test
   public void test_{proc['name']}_[scenario]() {{
       // Given: [test setup]
       // When: [call method]
       // Then: [verify behavior]
   }}
   ```

"""

        # Dependency procedures
        if context['dependency_procedures']:
            prompt += f"""
---

## 🔗 DEPENDENCY PROCEDURES ({len(context['dependency_procedures'])} procedures)

These procedures are called by or call the primary procedures.
**ALL must be translated completely** (not as stubs or placeholders).

"""
            
            # Show first 10 in detail, summarize rest
            for i, proc in enumerate(context['dependency_procedures'][:10], 1):
                is_external = proc.get('is_external', False)
                external_note = " (EXTERNAL - may need interface)" if is_external else ""
                
                prompt += f"""
### DEPENDENCY {i}: {proc['name']}{external_note}

**Location**: `{proc['file']}:{proc['line']}`
**Parameters**: {', '.join(proc['parameters']) if proc['parameters'] else 'none'}
**Return**: {proc['return_type'] or 'void'}

#### TAL Code:
```tal
{proc['code'][:800] if proc['code'] else '// Code not available'}
{'...\n[Truncated - see full context for complete code]' if proc['code'] and len(proc['code']) > 800 else ''}
```

**Translation Required**: Complete implementation (not a stub)

"""
            
            if len(context['dependency_procedures']) > 10:
                prompt += f"""
### Additional Dependencies ({len(context['dependency_procedures']) - 10} more):

"""
                for proc in context['dependency_procedures'][10:]:
                    prompt += f"- **{proc['name']}**: {proc['file']}:{proc['line']}\n"
                
                prompt += "\nAll must be translated completely. Request full code if truncated.\n"

        # Data structures
        if context['structures']:
            prompt += f"""
---

## 📊 DATA STRUCTURES ({len(context['structures'])} structures)

Translate these TAL structures to Java classes or records.

"""
            for struct in context['structures']:
                prompt += f"""
### Structure: {struct['name']}

#### TAL Definition:
```tal
{struct['code'] if struct['code'] else '// Structure definition not available'}
```

#### Fields ({len(struct.get('fields', []))}):
"""
                if struct.get('fields'):
                    for field in struct['fields']:
                        prompt += f"- `{field.get('name')}`: {field.get('type', 'unknown')}"
                        if field.get('description'):
                            prompt += f" - {field['description']}"
                        prompt += "\n"
                
                prompt += f"""
#### Required Java Translation:
```java
/**
 * {struct['name']} data structure
 * TAL equivalent: {struct['name']} STRUCT
 */
public class {struct['name']} {{
    // Translate all fields with appropriate Java types
    // Add getters/setters or use record (Java 14+)
    // Add validation if needed
}}
```

"""

        # Call graph
        if context.get('call_graph'):
            prompt += """
---

## 🔄 CALL GRAPH

Understanding the call relationships helps maintain correct behavior:

```
"""
            for proc_name, deps in list(context['call_graph'].items())[:15]:
                prompt += f"{proc_name}:\n"
                if deps.get('calls'):
                    prompt += "  Calls:\n"
                    for callee in deps['calls'][:5]:
                        prompt += f"    → {callee}\n"
                    if len(deps['calls']) > 5:
                        prompt += f"    → ... and {len(deps['calls']) - 5} more\n"
                
                if deps.get('called_by'):
                    prompt += "  Called by:\n"
                    for caller in deps['called_by'][:5]:
                        prompt += f"    ← {caller}\n"
                    if len(deps['called_by']) > 5:
                        prompt += f"    ← ... and {len(deps['called_by']) - 5} more\n"
                prompt += "\n"
            
            prompt += "```\n\n"

        # Output format and requirements
        prompt += """
---

## 📤 REQUIRED OUTPUT FORMAT

Provide your translation in this EXACT structure:

### Part 1: Overview
```
TRANSLATION OVERVIEW

Functionality: [name]
Total Procedures Translated: [number]
Total Classes Created: [number]
Key Design Decisions:
  - [Decision 1 and rationale]
  - [Decision 2 and rationale]
```

### Part 2: For Each Procedure

Repeat this for EVERY procedure:

```
═══════════════════════════════════════════════════════════════
PROCEDURE: [TAL_PROCEDURE_NAME]
═══════════════════════════════════════════════════════════════

STEP 1: LOGIC ANALYSIS
────────────────────────────────────────────────────────────────
PURPOSE: [What it does]

INPUT PARAMETERS:
  - param1: type - [purpose and constraints]

BUSINESS LOGIC:
  Section 1 (TAL lines X-Y): [What it does]
  Section 2 (TAL lines A-B): [What it does]

ERROR HANDLING:
  - Error -1: [Condition and meaning]
  - Error -2: [Condition and meaning]


STEP 2: JAVA IMPLEMENTATION
────────────────────────────────────────────────────────────────
```java
package com.company.[domain].[module];

import java.math.BigDecimal;
// ... all imports

/**
 * [Complete Javadoc description]
 * 
 * <p>TAL equivalent: [PROCEDURE_NAME]
 * <p>TAL location: [file]:[line]
 * 
 * @author Generated from TAL
 */
public class [ClassName] {
    
    /**
     * [Method description matching TAL procedure purpose]
     *
     * @param [param] [description from TAL]
     * @return [description from TAL]
     * @throws [Exception] if [condition from TAL error handling]
     */
    public [ReturnType] [methodName]([parameters]) {
        // TAL lines X-Y: [What this section does]
        [Complete implementation]
        
        // TAL line Z: [Specific logic]
        [Implementation]
        
        // Continue for EVERY line of TAL code
    }
}
```

STEP 3: VERIFICATION
────────────────────────────────────────────────────────────────
COMPLETENESS CHECK:
  [✓] All TAL lines translated
  [✓] All conditions implemented
  [✓] All error paths present
  [✓] All calculations accurate
  [✓] No placeholders or TODOs

TRACEABILITY:
  TAL Line X → Java Line Y: [Mapping]
  TAL Line A → Java Line B: [Mapping]


STEP 4: UNIT TEST SKELETON
────────────────────────────────────────────────────────────────
```java
@Test
public void test_[scenarioDescription]() {
    // Given: [Test setup based on TAL input parameters]
    
    // When: [Call the method]
    
    // Then: [Verify based on TAL expected outputs]
}

@Test
public void test_[errorScenario]() {
    // Given: [Setup for error condition from TAL]
    
    // When/Then: [Verify error handling]
}
```
```

### Part 3: Supporting Classes

For each data structure:
```java
/**
 * [Structure description]
 * TAL equivalent: [STRUCT_NAME]
 */
public class [ClassName] {
    [Complete implementation with all fields]
}
```

### Part 4: Summary and Migration Notes

```
TRANSLATION SUMMARY
═══════════════════════════════════════════════════════════════

✓ Procedures Translated: [X/X]
✓ Data Structures: [X/X]
✓ Total Lines of Code: ~[number]

DESIGN DECISIONS:
1. [Key decision and why]
2. [Another decision and rationale]

MIGRATION NOTES:
⚠ [Important thing to watch out for]
⚠ [Another gotcha]
⚠ [Deployment consideration]

ASSUMPTIONS:
• [Any assumptions made during translation]
• [Dependencies that need to be in place]

VERIFICATION REQUIRED:
□ Manual code review of [specific complex sections]
□ Integration testing for [specific flows]
□ Performance testing for [specific operations]
```

---

## ✅ FINAL CHECKLIST

Before submitting your translation, verify:

**Completeness**:
- [ ] EVERY procedure has complete implementation (no TODOs)
- [ ] EVERY TAL line has a Java equivalent
- [ ] EVERY condition/branch is translated
- [ ] EVERY calculation is present
- [ ] EVERY error path is implemented

**Correctness**:
- [ ] Data types match precision requirements
- [ ] Thresholds/constants are exact (not approximated)
- [ ] Logic flow matches TAL exactly
- [ ] Error codes/returns are preserved

**Quality**:
- [ ] Javadoc for all public methods/classes
- [ ] Inline comments map to TAL lines
- [ ] Code follows Java conventions
- [ ] No compiler warnings expected

**Financial Code** (if applicable):
- [ ] BigDecimal used for all amounts
- [ ] Rounding explicitly specified
- [ ] Validation order preserved
- [ ] Audit logging present

**Traceability**:
- [ ] Each method references TAL source location
- [ ] Complex sections have TAL line references
- [ ] Design decisions are documented

---

## 🚀 BEGIN TRANSLATION

Now translate ALL procedures completely following the methodology above.

Remember:
- ✅ Analyze before implementing
- ✅ Implement every line
- ✅ Verify completeness
- ✅ Add test skeletons
- ✅ Document everything

**START WITH PROCEDURE 1 and work through ALL procedures systematically.**
"""

        return prompt
    
    @staticmethod
    def save_prompt(prompt: str, output_file: str):
        """Save prompt to file."""
        from pathlib import Path
        
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        print(f"✓ Comprehensive translation prompt saved to: {output_file}")
        print(f"  Prompt size: {len(prompt):,} characters")
        print(f"  Estimated tokens: ~{len(prompt) // 4:,}")


def example_usage():
    """Show how to use the comprehensive prompt generator."""
    
    # Mock context for demonstration
    context = {
        'functionality': 'wire_transfer_processing',
        'summary': {
            'primary_procedures': 3,
            'total_procedures': 8,
            'total_structures': 2,
            'code_extraction': {
                'total_chars': 15000
            }
        },
        'primary_procedures': [
            {
                'name': 'PROCESS_WIRE_TRANSFER',
                'file': 'wire.tal',
                'line': 100,
                'parameters': ['wire_id', 'amount', 'dest_account'],
                'return_type': 'INT',
                'code': 'PROC PROCESS_WIRE_TRANSFER(wire_id, amount, dest_account);\n  BEGIN\n    ...\n  END;',
                'code_length': 500,
                'dependencies': {
                    'calls': ['VALIDATE_WIRE', 'CHECK_LIMITS'],
                    'called_by': ['MAIN_HANDLER'],
                    'uses_structures': ['WIRE_RECORD'],
                    'uses_variables': ['transaction_log']
                }
            }
        ],
        'dependency_procedures': [],
        'structures': [],
        'call_graph': {}
    }
    
    # Generate comprehensive prompt
    generator = ComprehensivePromptGenerator()
    prompt = generator.generate_translation_prompt(
        context,
        strict_mode=True,
        include_validation=True,
        financial_code=True
    )
    
    # Save it
    generator.save_prompt(prompt, "./output/comprehensive_translation_prompt.md")
    
    print("\n✓ Comprehensive prompt generated!")
    print("\nThis prompt ensures:")
    print("  1. Complete logic analysis before coding")
    print("  2. Line-by-line TAL to Java mapping")
    print("  3. Full implementation (no placeholders)")
    print("  4. Verification checklist for each procedure")
    print("  5. Financial code precision requirements")
    print("  6. Unit test skeletons")


if __name__ == "__main__":
    example_usage()
