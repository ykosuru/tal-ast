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
IF x = 1 THEN                    | if (x == 1) {{
FOR i := 0 TO 9 DO               | for (int i = 0; i <= 9; i++) {{
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
   public class [ServiceName] {{
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
   public ReturnType methodName(ParamType param1) {{
       // TAL line X-Y: [What this section does]
       [Java code implementing those TAL lines]
       
       // TAL line Z: [Specific logic]
       [Java code for that line]
   }}
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
   if (amount.compareTo(new BigDecimal("10000.00")) > 0) {{
   
   // ❌ WRONG: Approximate or simplified
   if (amount > 10000) {{  // Wrong: loses precision, uses double
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
   log.info("Wire transfer processed: id={{}}, amount={{}}, status={{}}, user={{}}", 
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
   if (sanctionsList.contains(country)) {{
       blockTransaction();
   }}
   
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
   public void processTransfer() {{
       updateAccount();
       updateLedger();
       // Commit handled by @Transactional
   }}
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
UNSIGNED                →  int (Java doesn't have unsigned, handle carefully)
```

#### Arrays:
```tal
ARRAY[0:9] OF INT       →  int[] array = new int[10];
ARRAY[1:100] OF STRING  →  String[] array = new String[100];
                           // Note: TAL arrays can start at any index
                           // Java always starts at 0
```

#### Structures:
```tal
STRUCT payment_record;  →  public class PaymentRecord {{
BEGIN                   →      // Fields
  INT transaction_id;   →      private int transactionId;
  STRING .payee;        →      private String payee;
  FIXED(2) amount;      →      private BigDecimal amount;
END;                    →  }}
```

#### Procedures:
```tal
PROC validate(amount);  →  public int validate(BigDecimal amount) {{
  BEGIN                 →      // method body
    ...                 →      ...
  END;                  →  }}
```

#### Control Flow:
```tal
IF condition THEN       →  if (condition) {{
  statement;            →      statement;
                        →  }}

IF x THEN               →  if (x) {{
  statement1            →      statement1;
ELSE                    →  }} else {{
  statement2;           →      statement2;
                        →  }}

FOR i := 0 TO 9 DO      →  for (int i = 0; i <= 9; i++) {{
  statement;            →      statement;
                        →  }}

WHILE condition DO      →  while (condition) {{
  statement;            →      statement;
                        →  }}

CASE status OF          →  switch (status) {{
  BEGIN                 →      case -1:
    -1: action1;        →          action1; break;
    -2: action2;        →      case -2:
    OTHERWISE: default; →          action2; break;
  END;                  →      default:
                        →          default; break;
                        →  }}
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
$len(str)               →  str.length()
```

#### Special TAL Constructs:
```tal
BEGIN                   →  {{
END                     →  }}
PROC name;              →  public void name() {{
  FORWARD;              →      // Forward declaration - create interface
END;                    →  }}
send(), receive()       →  // OS (Guardian) functions - modernize
```

#### Domain-Specific Context:
- **db** or **dbtr** refers to **debtor**
- **cr** or **crdtr** refers to **creditors**
- **acct** refers to **account**
- **trn** is **payment transaction**
- **^** represents delimiter or word separators in TAL
- **FAIN** is the older payment message type
- **GSMOS** refers to OFAC/sanctions
- **Host messages and MQs** are no longer relevant

---

"""

        # Add the actual translation section
        prompt += f"""## 📝 PROCEDURES TO TRANSLATE

You are translating {len(context['primary_procedures'])} primary procedures completely.

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
public class [ClassName] {{
    
    /**
     * [Method description matching TAL procedure purpose]
     *
     * @param [param] [description from TAL]
     * @return [description from TAL]
     * @throws [Exception] if [condition from TAL error handling]
     */
    public [ReturnType] [methodName]([parameters]) {{
        // TAL lines X-Y: [What this section does]
        [Complete implementation]
        
        // TAL line Z: [Specific logic]
        [Implementation]
        
        // Continue for EVERY line of TAL code
    }}
}}
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
public void test_[scenarioDescription]() {{
    // Given: [Test setup based on TAL input parameters]
    
    // When: [Call the method]
    
    // Then: [Verify based on TAL expected outputs]
}}

@Test
public void test_[errorScenario]() {{
    // Given: [Setup for error condition from TAL]
    
    // When/Then: [Verify error handling]
}}
```
```

### Part 3: Supporting Classes

For each data structure:
```java
/**
 * [Structure description]
 * TAL equivalent: [STRUCT_NAME]
 */
public class [ClassName] {{
    [Complete implementation with all fields]
}}
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

@staticmethod
    def generate_documentation_prompt(
        context: Dict[str, Any],
        include_architecture: bool = True,
        include_design: bool = True,
        system_name: str = "TAL Payment System",
        system_context: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive documentation prompt for architecture and design docs.
        Documents EXISTING TAL implementation using payment domain terminology.
        Output format: Structurizr DSL (C4 Model)
        
        Args:
            context: Translation context from TranslationContextBuilder
            include_architecture: Generate architecture-level documentation
            include_design: Generate design-level documentation
            system_name: Name of the system being documented
            system_context: Additional context about the system
        """
        
        prompt = f"""# TAL Payment System Documentation Task

## 🎯 DOCUMENTATION OBJECTIVE

You are analyzing and documenting an **EXISTING TAL payment processing system** based on the source code provided.

**Your Role**: Payment systems architect and technical writer who understands both legacy TAL systems and modern payment processing.

**System Being Documented**: {system_name}
**Functionality**: {context['functionality']}

**Analysis Scope**:
- {context['summary']['total_procedures']} TAL procedures to analyze
- {context['summary']['total_structures']} data structures to document
- {context['summary']['primary_procedures']} primary payment capabilities
- ~{context['summary']['code_extraction']['total_chars']:,} characters of TAL source code

"""

        if system_context:
            prompt += f"""**Additional Context**:
{system_context}

"""

        prompt += """---

## 📐 DOCUMENTATION REQUIREMENTS

**CRITICAL**: You are documenting what the TAL code CURRENTLY DOES, not designing a new system.

Produce **TWO COMPREHENSIVE DOCUMENTS**:

### Document 1: High-Level Architecture Documentation
**Format**: Business-focused architecture document with Structurizr DSL diagrams
**Audience**: Business stakeholders, architects, payment operations leaders
**Purpose**: Understand payment capabilities, business flows, and system integration
**Focus**: 
- What payment operations are supported (wire transfers, ACH, clearing, settlement, etc.)
- How payment messages flow through the system
- Which payment standards are implemented (ISO 20022, SWIFT, FedWire, etc.)
- Integration with payment networks and core banking systems
- Risk, compliance, and regulatory controls (AML, OFAC, KYC)

### Document 2: Low-Level Design Documentation
**Format**: Technical design document with detailed Structurizr DSL component models
**Audience**: Developers, QA engineers, system maintainers
**Purpose**: Understand technical implementation details and business logic
**Focus**:
- How TAL procedures implement each payment operation
- Data structures and their relationships
- Algorithms for payment validation, calculation, and processing
- Error handling and exception flows
- Business rules and decision logic
- Technical integration patterns

### Output Format: Structurizr DSL + Markdown

- **Primary format**: Structurizr DSL for architectural diagrams
- **Supporting format**: Markdown for textual explanations
- Both documents must be complete, professional, and production-ready

---

## 💳 PAYMENT DOMAIN KNOWLEDGE REQUIREMENTS

Use your understanding of payment systems to properly identify and document:

### Payment Message Standards
- **ISO 20022**: pacs.008 (payment initiation), pacs.002 (status), camt.054 (debit/credit notification)
- **SWIFT MT**: MT103 (single customer credit), MT202 (financial institution transfer)
- **FedWire**: Fedwire Funds Service message formats
- **ACH**: NACHA file formats, batching, settlement
- **SEPA**: SEPA Credit Transfer (SCT), SEPA Instant Credit Transfer (SCT Inst)

### Payment Operations
- **Origination**: Payment initiation, validation, authorization
- **Clearing**: Interbank clearing, netting, bilateral/multilateral settlement
- **Settlement**: Real-time gross settlement (RTGS), deferred net settlement (DNS)
- **Reconciliation**: Account reconciliation, exception handling
- **Returns**: Payment returns, rejections, recalls
- **Repair**: Payment repair, amendment, correction

### Compliance & Risk
- **AML/CFT**: Anti-money laundering, sanctions screening, transaction monitoring
- **KYC**: Customer identification, beneficial ownership, PEP screening
- **OFAC**: Sanctions list screening (SDN, consolidated lists)
- **Fraud Detection**: Real-time fraud scoring, velocity checks, pattern analysis
- **Regulatory Reporting**: CTR, SAR, FBAR, regulatory filings

### Payment Types
- **Wire Transfers**: Domestic, international, same-day, next-day
- **ACH Payments**: ACH credit, ACH debit, same-day ACH
- **Real-Time Payments**: FedNow, RTP, instant payments
- **Card Payments**: Card-on-file, authorization, capture, settlement
- **Direct Debits**: Recurring payments, mandates, pre-authorization

### Technical Concepts
- **Message Queuing**: MQ, message routing, guaranteed delivery
- **Database**: Transaction management, consistency, locking
- **Batch Processing**: EOD processing, cutoff times, batch settlement
- **API Integration**: REST, SOAP, synchronous/asynchronous patterns
- **Error Handling**: Retry logic, dead letter queues, exception management

---

## 🏗️ DOCUMENT 1: HIGH-LEVEL ARCHITECTURE

### Required Contents:

#### 1. Executive Summary (Markdown)
```markdown
# {system_name} - Architecture Documentation

## System Overview
[2-3 paragraph description of what this payment system does, 
using proper payment terminology]

## Business Capabilities
[List of payment capabilities supported - be specific with payment types]

## Critical Success Factors
[What makes this system critical to payment operations]
```

#### 2. Business Capability Model (Markdown)
Map TAL procedures to payment business capabilities:

```markdown
## Payment Business Capabilities

### Payment Origination
**Capabilities**:
- Wire transfer initiation (domestic & international)
- ACH payment origination
- Payment validation & enrichment
- Duplicate detection

**TAL Procedures**:
- [PROCEDURE_NAME]: [What it does in payment terms]
- [PROCEDURE_NAME]: [What it does in payment terms]

### Payment Validation & Compliance
**Capabilities**:
- OFAC sanctions screening
- AML transaction monitoring
- Payment limit enforcement
- Regulatory hold processing

**TAL Procedures**:
- [List relevant procedures]

### Payment Processing & Settlement
[Continue for each business area...]
```

#### 3. Payment Message Flows (Markdown + Structurizr)
Document the payment flows:

```markdown
## Payment Message Flows

### Wire Transfer Flow
[Describe the flow from initiation to settlement]

**Message Standards**: ISO 20022 pacs.008, pacs.002
**Processing Steps**:
1. Payment received from originating channel
2. Validation: format, limits, account verification
3. Compliance screening: OFAC, AML rules
4. Enrichment: BIC routing, intermediary bank resolution
5. Submission to payment network (SWIFT/FedWire)
6. Status tracking and notification

**TAL Implementation**:
- Entry point: [PROCEDURE_NAME]
- Validation: [PROCEDURE_NAME]
- Compliance: [PROCEDURE_NAME]
- Network submission: [PROCEDURE_NAME]
```

Then provide Structurizr dynamic view showing the flow.

#### 4. System Context Diagram (Structurizr DSL)
Show the system's place in the payment ecosystem:

```structurizr
workspace {{
    name "{system_name} Architecture"
    description "High-level architecture of existing TAL payment system"
    
    model {{
        # Payment Users
        person wireInitiator "Wire Transfer Operator" "Initiates wire transfers"
        person achOperator "ACH Operator" "Manages ACH payments"
        person complianceOfficer "Compliance Officer" "Monitors compliance"
        person opsManager "Operations Manager" "System oversight"
        
        # External Payment Systems
        softwareSystem swiftNetwork "SWIFT Network" "International payment messaging" "External"
        softwareSystem fedwire "FedWire" "US domestic wire transfers" "External"
        softwareSystem achNetwork "ACH Network" "Automated Clearing House" "External"
        softwareSystem coreBanking "Core Banking System" "Account management and ledger" "External"
        softwareSystem ofacService "OFAC Screening Service" "Sanctions list screening" "External"
        
        # This TAL Payment System
        softwareSystem talPaymentSystem "{system_name}" "Existing TAL payment processing system" {{
            
            # Document containers based on actual TAL implementation
            # Analyze the code to determine functional groupings
            
            container paymentOrigination "Payment Origination" "Handles payment initiation and validation" "TAL Processes" {{
                tags "PaymentProcessing" "Critical"
            }}
            
            container complianceEngine "Compliance Engine" "Sanctions and AML screening" "TAL Processes" {{
                tags "Compliance" "Critical"
            }}
            
            container paymentProcessor "Payment Processor" "Core payment processing and routing" "TAL Processes" {{
                tags "PaymentProcessing" "Critical"
            }}
            
            container networkInterface "Network Interface" "SWIFT/FedWire/ACH integration" "TAL Processes" {{
                tags "Integration" "Critical"
            }}
            
            container paymentDatabase "Payment Database" "Transaction data, audit logs" "NonStop SQL" {{
                tags "Database" "Critical"
            }}
            
            container messageQueue "Message Queue" "Async processing queue" "TMF/Pathway" {{
                tags "Middleware"
            }}
        }}
        
        # Relationships - based on actual TAL code flows
        wireInitiator -> talPaymentSystem.paymentOrigination "Initiates wire transfer" "Terminal/UI"
        
        talPaymentSystem.paymentOrigination -> talPaymentSystem.complianceEngine "Screen payment" "Procedure call"
        talPaymentSystem.complianceEngine -> ofacService "Check sanctions lists" "External call"
        
        talPaymentSystem.paymentOrigination -> talPaymentSystem.paymentProcessor "Submit for processing"
        talPaymentSystem.paymentProcessor -> talPaymentSystem.paymentDatabase "Store transaction"
        talPaymentSystem.paymentProcessor -> talPaymentSystem.networkInterface "Route to network"
        
        talPaymentSystem.networkInterface -> swiftNetwork "Send MT103/pacs.008"
        talPaymentSystem.networkInterface -> fedwire "Send FedWire message"
        
        talPaymentSystem.paymentProcessor -> coreBanking "Debit/credit accounts" "IPC"
    }}
    
    views {{
        systemContext talPaymentSystem "PaymentSystemContext" {{
            include *
            autoLayout lr
            title "Payment System Context - Integration with Payment Networks"
        }}
        
        container talPaymentSystem "PaymentContainers" {{
            include *
            autoLayout tb
            title "Payment Processing Components (TAL Implementation)"
        }}
        
        styles {{
            element "PaymentProcessing" {{ background #1168bd; color #ffffff }}
            element "Compliance" {{ background #ff6b6b; color #ffffff }}
            element "Integration" {{ background #6b9bd1; color #ffffff }}
            element "External" {{ background #cccccc }}
            element "Critical" {{ border solid; thickness 3 }}
        }}
        
        theme default
    }}
}}
```

#### 5. Integration Architecture (Markdown)
Document all external integrations:

```markdown
## External System Integrations

### SWIFT Network Integration
**Purpose**: International wire transfers
**Protocol**: SWIFT Alliance Access, SWIFTNet
**Message Types**: MT103, MT202, MT910, MT940
**Implementation**: [Describe TAL procedures handling SWIFT]
**Key TAL Procedures**:
- [PROCEDURE]: Formats MT103 messages
- [PROCEDURE]: Parses MT910 confirmations

### Core Banking Integration
**Purpose**: Account debit/credit, balance verification
**Protocol**: [IPC/Pathway/TMF]
**Operations**: Balance inquiry, account update, hold management
**Implementation**: [Describe TAL procedures]

[Continue for each integration...]
```

#### 6. Data Architecture (Markdown)
Document key data structures in payment terms:

```markdown
## Payment Data Model

### Payment Transaction
**TAL Structure**: [STRUCT_NAME]
**Purpose**: Core payment transaction record
**Lifecycle**: Created → Validated → Screened → Processed → Settled
**Key Attributes**:
- Transaction ID: Unique identifier
- Payment Type: Wire/ACH/RTP
- Amount & Currency: With precision handling
- Originator: Debtor party information
- Beneficiary: Creditor party information
- Status: Current processing status
- Network Reference: SWIFT UETR / FedWire IMAD

[Continue for each major data structure...]
```

---

## 🔧 DOCUMENT 2: LOW-LEVEL DESIGN

### Required Contents:

#### 1. Technical Overview (Markdown)
```markdown
# {system_name} - Design Documentation

## Technical Architecture
[Describe the TAL implementation architecture - how procedures are organized,
how they interact, data flow patterns]

## Technology Stack
- **Language**: TAL (Transaction Application Language)
- **Platform**: HP NonStop (Tandem)
- **Database**: [SQL/MX, Enscribe]
- **Messaging**: [TMF, Pathway, OSS]
- **File System**: [Guardian, OSS]

## Design Principles
[What principles are evident in the code? Modularity, error handling, etc.]
```

#### 2. Component Design (Structurizr DSL)

Why Structurizr DSL?
✅ Machine-readable architecture diagrams
✅ Version controllable documentation
✅ Automatic rendering of multiple views
✅ Clear separation of concerns (Context → Container → Component → Code)
✅ Supports deployment and dynamic views

---

## 📚 STRUCTURIZR DSL QUICK REFERENCE

### Basic Structure:
```structurizr
workspace {
    name "System Name"
    description "System description"
    
    model {
        # Define people, systems, containers, components
        
        person user "User" "Description"
        
        softwareSystem system "System" "Description" {
            container api "API" "Description" "Technology"
            container database "Database" "Description" "Technology"
        }
        
        # Define relationships
        user -> system "Uses"
        api -> database "Reads/Writes" "Protocol"
    }
    
    views {
        systemContext system "SystemContext" {
            include *
        }
        
        container system "Containers" {
            include *
        }
        
        component api "Components" {
            include *
        }
        
        theme default
    }
}
```

### Key Elements:

**People & External Systems**:
```structurizr
person user "User Name" "Role description"
softwareSystem externalSystem "Name" "What it does" "External"
```

**Software System & Containers**:
```structurizr
softwareSystem mySystem "System Name" "Purpose" {
    container webApp "Web Application" "Description" "Technology" {
        component controller "Controller" "Handles HTTP" "Spring MVC"
        component service "Service" "Business logic" "Java"
        component repository "Repository" "Data access" "Spring Data"
    }
    
    container database "Database" "Stores data" "PostgreSQL"
    container messageQueue "Message Queue" "Async processing" "RabbitMQ"
}
```

**Relationships**:
```structurizr
# Format: source -> destination "Description" "Technology/Protocol"
user -> webApp "Submits payments" "HTTPS"
controller -> service "Calls"
service -> repository "Uses"
repository -> database "Reads/Writes" "JDBC"
service -> messageQueue "Publishes events" "AMQP"
```

**Tags for Styling**:
```structurizr
container api "API" "Description" "Spring Boot" {
    tags "Backend" "Critical"
}
```

**Deployment View**:
```structurizr
deploymentEnvironment "Production" {
    deploymentNode "AWS" {
        deploymentNode "EC2" {
            containerInstance webApp
        }
        deploymentNode "RDS" {
            containerInstance database
        }
    }
}
```

---

## 🏗️ LEVEL 1: ARCHITECTURE DOCUMENTATION

### Requirements for Architecture Documentation:

1. **System Context** (C4 Level 1):
   - Show the system in its environment
   - Identify ALL external actors (users, systems, services)
   - Show high-level interactions
   - Clarify system boundaries

2. **Container View** (C4 Level 2):
   - Break down system into major containers (applications, databases, file systems)
   - Show technology choices
   - Identify integration protocols
   - Map data storage

3. **Key Architecture Decisions**:
   - Why TAL to Java?
   - Technology stack rationale
   - Integration patterns
   - Data persistence strategy

4. **Business Capability Mapping**:
   - Map TAL procedures to business capabilities
   - Show which containers/components support which capabilities
   - Identify critical vs supporting capabilities

### Architecture Documentation Template:

```structurizr
workspace {{
    name "{system_name}"
    description "Architecture documentation for {context['functionality']}"
    
    !docs docs/architecture
    !adrs adrs
    
    model {{
        # EXTERNAL ACTORS
        # ===============
        # Document WHO uses this system
        
        person endUser "End User" "Description of user role"
        person backOfficeUser "Back Office User" "Operations staff"
        person administrator "System Administrator" "Manages the system"
        
        # EXTERNAL SYSTEMS
        # ================
        # Document WHAT systems this integrates with
        
        softwareSystem legacyTalSystem "Legacy TAL System" "Original system being replaced" "Legacy" {{
            tags "Legacy" "Deprecated"
        }}
        
        softwareSystem coreBanking "Core Banking System" "Handles accounts" "External"
        softwareSystem complianceSystem "Compliance System" "OFAC/AML checks" "External"
        softwareSystem reportingSystem "Reporting System" "Business intelligence" "External"
        
        # THIS SYSTEM
        # ===========
        softwareSystem modernPaymentSystem "{system_name}" "Modernized payment processing" {{
            
            # CONTAINERS (Major architectural components)
            # ===========================================
            
            container paymentApi "Payment API" "REST API for payment operations" "Spring Boot" {{
                tags "Backend" "Critical"
            }}
            
            container paymentProcessor "Payment Processor" "Async payment processing" "Spring Batch" {{
                tags "Backend" "Critical" "Batch"
            }}
            
            container validationEngine "Validation Engine" "Business rule validation" "Drools" {{
                tags "Backend" "Rules"
            }}
            
            container webPortal "Web Portal" "User interface" "React" {{
                tags "Frontend"
            }}
            
            container database "Payment Database" "Stores payment data" "PostgreSQL" {{
                tags "Database" "Critical"
            }}
            
            container cache "Cache" "Performance optimization" "Redis" {{
                tags "Cache"
            }}
            
            container messageQueue "Message Queue" "Async communication" "RabbitMQ" {{
                tags "Middleware"
            }}
            
            container auditLog "Audit Log" "Compliance audit trail" "Elasticsearch" {{
                tags "Database" "Compliance"
            }}
        }}
        
        # RELATIONSHIPS (Integration points)
        # ==================================
        
        # User interactions
        endUser -> webPortal "Submits payments, views status" "HTTPS"
        backOfficeUser -> paymentApi "Manages payments, runs reports" "HTTPS"
        administrator -> paymentApi "System configuration" "HTTPS"
        
        # System interactions
        webPortal -> paymentApi "API calls" "REST/HTTPS"
        paymentApi -> database "Reads/Writes payment data" "JDBC"
        paymentApi -> cache "Caches frequently accessed data" "Redis Protocol"
        paymentApi -> messageQueue "Publishes payment events" "AMQP"
        paymentApi -> validationEngine "Validates business rules" "Java"
        
        messageQueue -> paymentProcessor "Consumes payment events" "AMQP"
        paymentProcessor -> database "Updates payment status" "JDBC"
        paymentProcessor -> auditLog "Writes audit records" "HTTP"
        
        # External integrations
        paymentApi -> coreBanking "Account verification, balance checks" "REST/HTTPS"
        validationEngine -> complianceSystem "OFAC/AML screening" "SOAP/HTTPS"
        paymentProcessor -> reportingSystem "Sends payment metrics" "JMS"
        
        # Legacy migration path
        legacyTalSystem -> database "Data migration" "JDBC" {{
            tags "Migration"
        }}
    }}
    
    views {{
        # SYSTEM CONTEXT VIEW
        # ===================
        systemContext modernPaymentSystem "SystemContext" {{
            include *
            autoLayout lr
            title "[System Context] {system_name}"
            description "Shows the system in its environment with external actors and systems"
        }}
        
        # CONTAINER VIEW
        # ==============
        container modernPaymentSystem "Containers" {{
            include *
            autoLayout lr
            title "[Container View] {system_name}"
            description "Shows the major technical building blocks"
        }}
        
        # FILTERED VIEWS
        # ==============
        container modernPaymentSystem "CriticalPath" {{
            include modernPaymentSystem.paymentApi
            include modernPaymentSystem.database
            include modernPaymentSystem.validationEngine
            include coreBanking
            include complianceSystem
            include endUser
            autoLayout lr
            title "[Critical Payment Path]"
            description "Shows the critical components for payment processing"
        }}
        
        # DYNAMIC VIEW (Payment Flow)
        # ===========================
        dynamic modernPaymentSystem "PaymentSubmission" "Payment submission flow" {{
            endUser -> modernPaymentSystem.webPortal "1. Submits payment"
            modernPaymentSystem.webPortal -> modernPaymentSystem.paymentApi "2. POST /api/payments"
            modernPaymentSystem.paymentApi -> modernPaymentSystem.validationEngine "3. Validate business rules"
            modernPaymentSystem.validationEngine -> complianceSystem "4. Check OFAC/AML"
            modernPaymentSystem.paymentApi -> modernPaymentSystem.database "5. Save payment"
            modernPaymentSystem.paymentApi -> modernPaymentSystem.messageQueue "6. Publish payment.created event"
            modernPaymentSystem.messageQueue -> modernPaymentSystem.paymentProcessor "7. Process payment"
            modernPaymentSystem.paymentProcessor -> coreBanking "8. Execute transfer"
            autoLayout lr
        }}
        
        # DEPLOYMENT VIEW
        # ===============
        deployment modernPaymentSystem "Production" "Production" {{
            deploymentNode "AWS Cloud" {{
                deploymentNode "ECS Cluster" {{
                    deploymentNode "API Container" {{
                        containerInstance modernPaymentSystem.paymentApi
                    }}
                    deploymentNode "Processor Container" {{
                        containerInstance modernPaymentSystem.paymentProcessor
                    }}
                }}
                
                deploymentNode "RDS" {{
                    containerInstance modernPaymentSystem.database
                }}
                
                deploymentNode "ElastiCache" {{
                    containerInstance modernPaymentSystem.cache
                }}
                
                deploymentNode "Amazon MQ" {{
                    containerInstance modernPaymentSystem.messageQueue
                }}
            }}
        }}
        
        # STYLING
        # =======
        styles {{
            element "External" {{
                background #cccccc
                color #000000
            }}
            element "Legacy" {{
                background #ff6666
                color #ffffff
            }}
            element "Critical" {{
                background #ff0000
                color #ffffff
            }}
            element "Database" {{
                shape Cylinder
            }}
            element "Batch" {{
                background #3366ff
            }}
            relationship "Migration" {{
                style dashed
                color #ff6666
            }}
        }}
        
        theme default
    }}
}}
```

---

## 🔧 LEVEL 2: DESIGN DOCUMENTATION

### Requirements for Design Documentation:

1. **Component View** (C4 Level 3):
   - Break down containers into components
   - Show internal architecture patterns (MVC, layered, hexagonal)
   - Map TAL procedures to Java components
   - Document data flow through components

2. **Business Logic Mapping**:
   - Map each TAL procedure to Java component(s)
   - Show how business capabilities are implemented
   - Document service boundaries

3. **Data Model**:
   - Show key entities
   - Document relationships
   - Map TAL structures to Java classes

4. **Integration Details**:
   - API contracts
   - Message formats
   - Error handling patterns

### Design Documentation Template:

```structurizr
workspace {{
    name "{system_name} - Design"
    description "Detailed design documentation for developers"
    
    model {{
        softwareSystem modernPaymentSystem "{system_name}" {{
            
            container paymentApi "Payment API" "REST API" "Spring Boot" {{
                
                # PRESENTATION LAYER
                # ==================
                component paymentController "Payment Controller" "Handles HTTP requests" "Spring MVC RestController" {{
                    technology "Spring MVC"
                    tags "Controller" "RestAPI"
                }}
                
                component validationController "Validation Controller" "Handles validation requests" "Spring MVC RestController"
                
                # APPLICATION/SERVICE LAYER
                # =========================
                component paymentService "Payment Service" "Core payment business logic" "Spring Service" {{
                    tags "Service" "BusinessLogic" "Critical"
                    # Maps to TAL procedures: PROCESS_PAYMENT, VALIDATE_PAYMENT
                }}
                
                component validationService "Validation Service" "Business rule validation" "Spring Service" {{
                    # Maps to TAL procedures: VALIDATE_AMOUNT, CHECK_LIMITS, VERIFY_ACCOUNT
                }}
                
                component complianceService "Compliance Service" "OFAC/AML checks" "Spring Service" {{
                    tags "Service" "Compliance"
                    # Maps to TAL procedures: CHECK_OFAC, RUN_AML_RULES
                }}
                
                component accountService "Account Service" "Account operations" "Spring Service" {{
                    # Maps to TAL procedures: GET_ACCOUNT_BALANCE, VERIFY_ACCOUNT
                }}
                
                # DOMAIN LAYER
                # ============
                component paymentDomain "Payment Domain" "Payment entities and business rules" "Java Domain Model" {{
                    tags "Domain" "Entity"
                }}
                
                component validationRules "Validation Rules" "Business validation rules" "Domain Logic"
                
                # INFRASTRUCTURE LAYER
                # ====================
                component paymentRepository "Payment Repository" "Payment data access" "Spring Data JPA" {{
                    tags "Repository" "DataAccess"
                }}
                
                component accountRepository "Account Repository" "Account data access" "Spring Data JPA"
                
                component cacheManager "Cache Manager" "Caching abstraction" "Spring Cache"
                
                component messagingGateway "Messaging Gateway" "Message publishing" "Spring AMQP"
                
                # EXTERNAL INTEGRATION
                # ====================
                component coreBankingClient "Core Banking Client" "Integration with core banking" "REST Client" {{
                    tags "Integration" "External"
                }}
                
                component complianceClient "Compliance Client" "Integration with compliance system" "SOAP Client" {{
                    tags "Integration" "External" "Compliance"
                }}
                
                # CROSS-CUTTING
                # ==============
                component auditLogger "Audit Logger" "Audit trail logging" "Custom Component" {{
                    tags "CrossCutting" "Compliance"
                }}
                
                component errorHandler "Error Handler" "Global exception handling" "Spring ControllerAdvice"
                
                # COMPONENT RELATIONSHIPS
                # =======================
                
                # Request flow
                paymentController -> paymentService "Calls"
                validationController -> validationService "Calls"
                
                # Service interactions
                paymentService -> validationService "Validates payment"
                paymentService -> complianceService "Compliance checks"
                paymentService -> accountService "Account operations"
                paymentService -> paymentDomain "Uses domain objects"
                
                validationService -> validationRules "Applies rules"
                validationService -> accountService "Verifies account"
                
                complianceService -> complianceClient "External OFAC check"
                accountService -> coreBankingClient "Get account details"
                
                # Data access
                paymentService -> paymentRepository "Persists payments"
                accountService -> accountRepository "Account queries"
                paymentRepository -> cacheManager "Uses cache"
                
                # Event publishing
                paymentService -> messagingGateway "Publishes events"
                
                # Cross-cutting
                paymentService -> auditLogger "Logs operations"
                complianceService -> auditLogger "Logs compliance checks"
                paymentController -> errorHandler "Exception handling"
            }}
            
            container database "Payment Database" "PostgreSQL" {{
                
                component paymentTable "Payment Table" "Stores payment records" "Table"
                component accountTable "Account Table" "Cached account data" "Table"
                component auditTable "Audit Table" "Audit trail" "Table"
                component validationTable "Validation Rules" "Business rules" "Table"
            }}
            
            container paymentProcessor "Payment Processor" "Spring Batch" {{
                
                component paymentListener "Payment Listener" "Consumes payment events" "Spring AMQP Listener"
                component batchProcessor "Batch Processor" "Processes payments in batch" "Spring Batch"
                component settlementJob "Settlement Job" "End-of-day settlement" "Scheduled Job"
                component reconciliationJob "Reconciliation Job" "Payment reconciliation" "Scheduled Job"
                
                paymentListener -> batchProcessor "Triggers"
                batchProcessor -> paymentRepository "Updates status"
                settlementJob -> batchProcessor "Runs daily"
            }}
        }}
        
        # Repository relationships to database
        paymentApi.paymentRepository -> database.paymentTable "CRUD operations" "SQL"
        paymentApi.accountRepository -> database.accountTable "Read operations" "SQL"
        paymentApi.auditLogger -> database.auditTable "Insert audit records" "SQL"
        paymentApi.validationService -> database.validationTable "Reads rules" "SQL"
    }}
    
    views {{
        # COMPONENT VIEW - Payment API
        # ============================
        component paymentApi "PaymentAPI_Components" {{
            include *
            autoLayout tb
            title "[Component] Payment API Internal Structure"
            description "Shows the internal components and their relationships"
        }}
        
        # FILTERED COMPONENT VIEWS
        # ========================
        component paymentApi "ServiceLayer" {{
            include ->paymentApi.paymentService->
            include ->paymentApi.validationService->
            include ->paymentApi.complianceService->
            include ->paymentApi.accountService->
            autoLayout lr
            title "[Service Layer] Business Logic Components"
        }}
        
        component paymentApi "DataAccessLayer" {{
            include ->paymentApi.paymentRepository->
            include ->paymentApi.accountRepository->
            include ->paymentApi.cacheManager->
            include ->database.*
            autoLayout lr
            title "[Data Access] Repository and Database Components"
        }}
        
        # DYNAMIC VIEWS (Sequence flows)
        # ==============================
        dynamic paymentApi "ProcessPayment" "Payment processing sequence" {{
            paymentController -> paymentService "1. processPayment(request)"
            paymentService -> validationService "2. validatePayment(payment)"
            validationService -> validationRules "3. applyRules(payment)"
            paymentService -> complianceService "4. checkCompliance(payment)"
            complianceService -> complianceClient "5. checkOFAC(payerName)"
            paymentService -> accountService "6. verifyAccount(accountId)"
            accountService -> coreBankingClient "7. getAccountBalance(accountId)"
            paymentService -> paymentDomain "8. createPaymentEntity(data)"
            paymentService -> paymentRepository "9. save(payment)"
            paymentService -> auditLogger "10. logPaymentCreated(payment)"
            paymentService -> messagingGateway "11. publishPaymentCreatedEvent(payment)"
            autoLayout lr
        }}
        
        # STYLES
        # ======
        styles {{
            element "Controller" {{
                background #85bbf0
                shape RoundedBox
            }}
            element "Service" {{
                background #438dd5
                color #ffffff
                shape Hexagon
            }}
            element "Repository" {{
                background #6cb33e
                shape Cylinder
            }}
            element "Domain" {{
                background #f4a261
            }}
            element "Integration" {{
                background #cccccc
                shape Component
            }}
            element "CrossCutting" {{
                background #8b4513
                color #ffffff
            }}
            element "Critical" {{
                border solid
                thickness 3
            }}
            element "Compliance" {{
                background #ff6b6b
                color #ffffff
            }}
        }}
        
        theme default
    }}
}}
```

---

## 📊 YOUR TASK: ANALYZE AND DOCUMENT

Based on the TAL code provided, create complete Structurizr DSL documentation.

### Provided Context:

**Procedures to Document**:
"""

        # List procedures to document
        for i, proc in enumerate(context['primary_procedures'][:10], 1):
            prompt += f"\n{i}. **{proc['name']}** - {proc['file']}:{proc['line']}"
            if proc.get('parameters'):
                prompt += f"\n   Parameters: {', '.join(proc['parameters'])}"
            deps = proc.get('dependencies', {})
            if deps.get('calls'):
                prompt += f"\n   Calls: {', '.join(deps['calls'][:5])}"
            prompt += "\n"
        
        if len(context['primary_procedures']) > 10:
            prompt += f"\n... and {len(context['primary_procedures']) - 10} more procedures\n"

        prompt += f"""
**Data Structures**:
"""
        for struct in context['structures'][:5]:
            prompt += f"- {struct['name']}: {len(struct.get('fields', []))} fields\n"
        
        if len(context['structures']) > 5:
            prompt += f"- ... and {len(context['structures']) - 5} more structures\n"

        prompt += """
---

## 📤 REQUIRED OUTPUT

Provide your documentation in this structure:

### Part 1: Architecture Overview

```
ARCHITECTURE DOCUMENTATION
══════════════════════════════════════════════════════════════

SYSTEM OVERVIEW:
  Name: [System name]
  Purpose: [What it does]
  Context: [Where it fits in the enterprise]

KEY ARCHITECTURAL DECISIONS:
  1. [Decision]: [Rationale]
  2. [Decision]: [Rationale]

CONTAINERS IDENTIFIED:
  - [Container 1]: [Purpose] - [Technology]
  - [Container 2]: [Purpose] - [Technology]
  
EXTERNAL INTEGRATIONS:
  - [System 1]: [Integration type] - [Protocol]
  - [System 2]: [Integration type] - [Protocol]

BUSINESS CAPABILITY MAPPING:
  Capability 1 → Container X, Component Y
  Capability 2 → Container Z, Component W
```

### Part 2: Complete Structurizr DSL

```structurizr
workspace {
    name "..."
    description "..."
    
    model {
        # Complete model following the templates above
        # Include:
        # - All actors (users, administrators)
        # - All external systems
        # - Software system with containers
        # - Components within critical containers
        # - All relationships
    }
    
    views {
        # System context view
        # Container view
        # Component views for key containers
        # Dynamic views for key flows
        # Deployment view (if applicable)
        # Styles
    }
}
```

### Part 3: TAL to Component Mapping

```
PROCEDURE → COMPONENT MAPPING
══════════════════════════════════════════════════════════════

TAL Procedure: [PROCEDURE_NAME_1]
→ Java Component: [ComponentName]
→ Container: [ContainerName]
→ Layer: [Service/Repository/Controller]
→ Responsibilities:
    - [Responsibility 1]
    - [Responsibility 2]
→ Dependencies:
    - Calls: [Other components]
    - Uses: [Data structures]

TAL Procedure: [PROCEDURE_NAME_2]
→ Java Component: [ComponentName]
...
```

### Part 4: Data Model Documentation

```
DATA MODEL OVERVIEW
══════════════════════════════════════════════════════════════

Entity: [EntityName]
  TAL Structure: [STRUCT_NAME]
  Fields:
    - field1: type (from TAL: FIELD1)
    - field2: type (from TAL: FIELD2)
  Relationships:
    - hasMany: [OtherEntity]
    - belongsTo: [ParentEntity]
  Usage:
    - Used by: [Component1, Component2]
    - Persisted in: [database.tableName]
```

### Part 5: Integration Specifications

```
INTEGRATION DETAILS
══════════════════════════════════════════════════════════════

External System: [SystemName]
  Protocol: [REST/SOAP/JMS]
  Operations:
    - Operation1: [Description]
      Request: [Format]
      Response: [Format]
    - Operation2: [Description]
  
  Error Handling:
    - Error1: [How it's handled]
    - Error2: [How it's handled]
  
  Security:
    - Authentication: [Method]
    - Authorization: [Method]
```

### Part 6: Design Patterns & Best Practices

```
DESIGN PATTERNS APPLIED
══════════════════════════════════════════════════════════════

Pattern: [Pattern Name]
  Location: [Component/Layer]
  Rationale: [Why this pattern]
  Example: [How it's used]

Pattern: [Pattern Name]
  ...
```

---

## ✅ DOCUMENTATION CHECKLIST

Before submitting, verify:

**Architecture Documentation**:
- [ ] System context clearly shows boundaries
- [ ] All external actors identified
- [ ] All containers have clear purposes and technologies
- [ ] Integration points are documented
- [ ] Business capabilities are mapped

**Design Documentation**:
- [ ] Components within containers are identified
- [ ] Component responsibilities are clear
- [ ] TAL procedures are mapped to components
- [ ] Data flow is documented
- [ ] Key sequences are shown in dynamic views

**Structurizr DSL**:
- [ ] Valid Structurizr DSL syntax
- [ ] Can be rendered without errors
- [ ] Views are properly defined
- [ ] Styles enhance readability
- [ ] Relationships have descriptions

**Traceability**:
- [ ] TAL procedures → Java components
- [ ] TAL structures → Java entities
- [ ] Business capabilities → Implementation
- [ ] External dependencies → Integration components

**Completeness**:
- [ ] Architecture addresses Level 1 (Context & Container)
- [ ] Design addresses Level 2 (Component & Code)
- [ ] All major TAL procedures are accounted for
- [ ] Data model is documented
- [ ] Integration patterns are clear

---

## 🚀 BEGIN DOCUMENTATION

Analyze the TAL code and create comprehensive architecture and design documentation in Structurizr DSL format.

**Remember**:
- ✅ Clear separation: Architecture vs Design
- ✅ Complete Structurizr DSL (valid syntax)
- ✅ Map every TAL procedure to a component
- ✅ Show data flows
- ✅ Document integrations
- ✅ Include dynamic views for key scenarios

**START WITH** the architecture overview, then provide the complete Structurizr workspace.
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
        
        print(f"✓ Comprehensive prompt saved to: {output_file}")
        print(f"  Prompt size: {len(prompt):,} characters")
        print(f"  Estimated tokens: ~{len(prompt) // 4:,}")


# Example usage
if __name__ == "__main__":
    # Example context structure (would come from TranslationContextBuilder)
    example_context = {
        'functionality': 'Payment Processing',
        'summary': {
            'total_procedures': 25,
            'primary_procedures': 5,
            'total_structures': 8,
            'code_extraction': {
                'total_chars': 45000
            }
        },
        'primary_procedures': [
            {
                'name': 'PROCESS_PAYMENT',
                'file': 'payment.tal',
                'line': 100,
                'parameters': ['payment_data', 'user_id'],
                'return_type': 'INT',
                'code': '! TAL code here...',
                'code_length': 2500,
                'dependencies': {
                    'calls': ['VALIDATE_PAYMENT', 'CHECK_COMPLIANCE'],
                    'called_by': ['MAIN_HANDLER'],
                    'uses_structures': ['PAYMENT_RECORD'],
                    'uses_variables': ['status', 'amount']
                }
            }
        ],
        'dependency_procedures': [],
        'structures': [
            {
                'name': 'PAYMENT_RECORD',
                'code': 'STRUCT payment_record...',
                'fields': [
                    {'name': 'transaction_id', 'type': 'INT'},
                    {'name': 'amount', 'type': 'FIXED(2)'}
                ]
            }
        ],
        'call_graph': {}
    }
    
    generator = ComprehensivePromptGenerator()
    
    # Generate translation prompt
    translation_prompt = generator.generate_translation_prompt(example_context)
    generator.save_prompt(translation_prompt, "/tmp/translation_prompt.md")
    
    # Generate documentation prompt
    doc_prompt = generator.generate_documentation_prompt(
        example_context,
        system_name="Modern Payment System",
        system_context="Replacing legacy TAL payment processing with modern Java microservices"
    )
    generator.save_prompt(doc_prompt, "/tmp/documentation_prompt.md")

