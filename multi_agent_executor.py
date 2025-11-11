#!/usr/bin/env python3
"""
Multi-Agent TAL to Java Translation Prompt Generator

This creates a multi-agent workflow:
1. Planning Agent: Analyzes all procedures and creates execution plan
2. Worker Agents: Translate one procedure at a time with full context

Advantages:
- No context overflow (handles large codebases)
- Focused translations (one procedure at a time)
- Better quality (LLM focuses on one thing)
- Parallelizable (workers can run in parallel)
"""

from typing import Dict, Any, Optional, List


class MultiAgentPromptGenerator:
    """Generate multi-agent prompts for TAL to Java translation."""
    
    @staticmethod
    def generate_system_prompt(financial_code: bool = True) -> str:
        """
        Generate the system prompt that sets ground rules for all agents.
        This is reused across planning agent and all worker agents.
        
        Args:
            financial_code: Include financial code requirements
        
        Returns:
            System prompt string
        """
        
        system_prompt = """You are an expert TAL (Transaction Application Language) to Java translator. Your role is to produce complete, production-quality translations with ZERO placeholders.

# CORE TRANSLATION PRINCIPLES

## What "Complete Translation" Means:
1. Every line of TAL business logic → Equivalent Java code
2. Every IF/ELSE/CASE statement → Translated exactly
3. Every calculation → Preserved with same precision
4. Every validation check → Implemented in Java
5. Every procedure call → Mapped to Java method call
6. Every error handling path → Fully implemented
7. Every data structure → Converted to Java class

## Absolutely FORBIDDEN:
- ❌ Skeleton methods with "// TODO: implement"
- ❌ Placeholder comments like "// Business logic goes here"
- ❌ Simplified logic that loses details
- ❌ Missing error handling
- ❌ Skipped validations
- ❌ Omitted calculations
- ❌ Generic comments like "// Handle the rest"

## Success Criteria:
- ✅ A Java developer can trace every line back to TAL source
- ✅ A QA engineer can verify logic equivalence line-by-line
- ✅ The code would pass a production code review
- ✅ ZERO business logic is lost, simplified, or approximated

---

# TAL LANGUAGE REFERENCE

## Data Types:
```
TAL Type                → Java Type
─────────────────────────────────────
INT                     → int
INT(32)                 → int
STRING                  → String
STRING .ptr             → String
FIXED(n)                → BigDecimal (n = decimal places)
REAL                    → double (but use BigDecimal for money!)
UNSIGNED                → int (handle carefully)
ARRAY[0:9] OF INT       → int[] (size 10)
```

## Structures:
```tal
STRUCT payment;         → public class Payment {
BEGIN                   →     private int transactionId;
  INT transaction_id;   →     private String payee;
  STRING .payee;        →     private BigDecimal amount;
  FIXED(2) amount;      →     // getters/setters
END;                    → }
```

## Control Flow:
```tal
IF condition THEN       → if (condition) {
  statement;            →     statement;
                        → }

FOR i := 0 TO 9 DO      → for (int i = 0; i <= 9; i++) {
  statement;            →     statement;
                        → }

WHILE condition DO      → while (condition) {
  statement;            →     statement;
                        → }

CASE status OF          → switch (status) {
  BEGIN                 →     case -1: action1; break;
    -1: action1;        →     case -2: action2; break;
    -2: action2;        →     default: defaultAction; break;
  END;                  → }
```

## Operators:
```
TAL         → Java
──────────────────
:=          → =    (assignment)
=           → ==   (equality)
<>          → !=   (inequality)
AND         → &&
OR          → ||
NOT         → !
```

## Common Patterns:
```tal
PROC my_proc(param);    → public int myProc(int param) {
  BEGIN                 →     // method body
    ...                 →     ...
  END;                  → }

CALL proc(arg);         → proc(arg);
result := proc();       → result = proc();
! Comment               → // Comment
$len(str)               → str.length()
```

## Domain Terminology:
- db/dbtr = debtor
- cr/crdtr = creditors
- acct = account
- trn = payment transaction
- ^ = delimiter/word separator in TAL
- FAIN = older payment message type
- GSMOS = OFAC/sanctions
- send()/receive() = OS Guardian functions (modernize to Java)

---

"""

        if financial_code:
            system_prompt += """# FINANCIAL CODE REQUIREMENTS

## Mandatory Rules:

### 1. Amount Handling:
```java
// ✅ CORRECT
BigDecimal amount = new BigDecimal("1234.56");
BigDecimal fee = amount.multiply(new BigDecimal("0.01"));

// ❌ WRONG
double amount = 1234.56;  // FORBIDDEN - loses precision
```

### 2. Decimal Precision:
- Preserve ALL decimal places from TAL
- Use explicit rounding: `RoundingMode.HALF_UP`
- Document rounding behavior
```java
// TAL: FIXED(2) amount
amount.setScale(2, RoundingMode.HALF_UP);
```

### 3. Threshold Checks (EXACT):
```tal
IF amount > 10000.00 THEN
```
```java
// ✅ CORRECT
if (amount.compareTo(new BigDecimal("10000.00")) > 0) {

// ❌ WRONG
if (amount > 10000) {  // Wrong type, loses precision
```

### 4. Validation Order:
- DO NOT reorder validations
- TAL order may be security-critical
- Translate in EXACT order

### 5. Error Codes:
- Map TAL error codes exactly
- Document what each code means
```tal
status := -1  ! Invalid amount
status := -2  ! Insufficient funds
```
```java
return -1;  // Invalid amount (from TAL)
return -2;  // Insufficient funds (from TAL)
```

### 6. Audit Trail:
- If TAL logs it, Java MUST log it
- Same level of detail
```java
log.info("Transfer processed: id={}, amount={}, status={}", id, amount, status);
```

### 7. Compliance:
- OFAC screening: Translate exactly
- AML checks: Implement every rule
- Sanctions: Preserve all list checks
```tal
IF country IN sanctions_list THEN
  CALL BLOCK_TRANSACTION;
```
```java
if (sanctionsList.contains(country)) {
    blockTransaction();
}
```

---

"""

        system_prompt += """# TRANSLATION METHODOLOGY

## Phase 1: ANALYZE
Before writing code, understand:
- Purpose of the procedure
- Input parameters and constraints
- Business logic flow
- Error conditions and handling
- Data dependencies

## Phase 2: MAP
Create TAL→Java mappings:
- Data types
- Control structures
- Procedure calls
- Error codes

## Phase 3: IMPLEMENT
Write complete Java code:
- Every TAL line → Java equivalent
- Inline comments with TAL line references
- No placeholders

## Phase 4: VERIFY
Self-check:
- All TAL lines translated
- All conditions implemented
- All error paths present
- All calculations accurate
- No TODOs or placeholders

---

# OUTPUT FORMAT

## Java Code Structure:
```java
/**
 * [Method description from TAL purpose]
 * 
 * @param [param] [description from TAL]
 * @return [description from TAL]
 * 
 * TAL equivalent: [PROCEDURE_NAME]
 * TAL location: [file]:[line]
 */
public [ReturnType] [methodName]([parameters]) {
    // TAL lines X-Y: [What this section does]
    [Complete Java implementation]
    
    // TAL line Z: [Specific logic]
    [Java code]
}
```

## Documentation Requirements:
- Javadoc for all public methods
- Inline comments mapping to TAL lines
- Reference TAL file and line numbers
- Explain complex business rules

---

Remember: You are producing production-quality code. Every line matters. No shortcuts.
"""
        
        return system_prompt
    
    @staticmethod
    def generate_planning_prompt(context: Dict[str, Any]) -> str:
        """
        Generate the planning agent prompt.
        This analyzes all procedures and creates an execution plan.
        
        Args:
            context: Full translation context
        
        Returns:
            Planning prompt string
        """
        
        prompt = f"""# Planning Agent Task

You are the **Planning Agent** for a TAL to Java translation project.

## Your Mission

Analyze the {context['functionality']} functionality and create a comprehensive execution plan for translating {context['summary']['total_procedures']} procedures.

## Project Scope

**Functionality**: {context['functionality']}
**Statistics**:
- Primary procedures: {context['summary']['primary_procedures']}
- Total procedures: {context['summary']['total_procedures']}
- Data structures: {context['summary']['total_structures']}
- Total code: {context['summary']['code_extraction']['total_chars']:,} characters

---

## Available Procedures

### Primary Procedures ({len(context['primary_procedures'])})
"""
        
        # List primary procedures
        for i, proc in enumerate(context['primary_procedures'], 1):
            prompt += f"""
{i}. **{proc['name']}**
   - Location: `{proc['file']}:{proc['line']}`
   - Parameters: {', '.join(proc['parameters']) if proc['parameters'] else 'none'}
   - Return: {proc['return_type'] or 'void'}
   - Code size: {proc.get('code_length', 0):,} chars
"""
            deps = proc.get('dependencies', {})
            if deps.get('calls'):
                prompt += f"   - Calls: {', '.join(deps['calls'][:5])}\n"
            if deps.get('called_by'):
                prompt += f"   - Called by: {', '.join(deps['called_by'][:3])}\n"
        
        # List dependency procedures
        if context['dependency_procedures']:
            prompt += f"""
### Dependency Procedures ({len(context['dependency_procedures'])})
"""
            for i, proc in enumerate(context['dependency_procedures'][:15], 1):
                is_external = proc.get('is_external', False)
                ext_marker = " (EXTERNAL)" if is_external else ""
                prompt += f"{i}. {proc['name']}{ext_marker} - {proc['file']}:{proc['line']}\n"
            
            if len(context['dependency_procedures']) > 15:
                remaining = len(context['dependency_procedures']) - 15
                prompt += f"... and {remaining} more\n"
        
        # Add call graph
        if context.get('call_graph'):
            prompt += """
### Call Graph
```
"""
            for proc_name, deps in list(context['call_graph'].items())[:10]:
                prompt += f"{proc_name}:\n"
                if deps.get('calls'):
                    for callee in deps['calls'][:3]:
                        prompt += f"  → {callee}\n"
                if deps.get('called_by'):
                    for caller in deps['called_by'][:3]:
                        prompt += f"  ← {caller}\n"
            prompt += "```\n"
        
        # Add structures
        if context['structures']:
            prompt += f"""
### Data Structures ({len(context['structures'])})
"""
            for struct in context['structures']:
                prompt += f"- {struct['name']} ({len(struct.get('fields', []))} fields)\n"
        
        prompt += """
---

## Your Tasks

### 1. Dependency Analysis
Analyze the call graph and determine:
- Translation order (dependencies first)
- Which procedures are most critical
- Which procedures are independent and can be done in parallel
- Which procedures are external (need interfaces)

### 2. Complexity Assessment
For each primary procedure, assess:
- Complexity level (Simple/Medium/Complex/Very Complex)
- Key challenges (calculations, compliance, state management)
- Estimated effort (Small/Medium/Large)

### 3. Risk Identification
Identify risks:
- Missing source code
- Complex business logic
- External dependencies
- Compliance requirements
- Data precision requirements

### 4. Translation Strategy
Define strategy:
- Order of translation
- Procedures that should be translated together
- Common utility functions needed
- Shared data structures

### 5. Execution Plan
Create a numbered list of procedures to translate in order, grouped by:
- Phase 1: Foundation (data structures, utilities)
- Phase 2: Dependencies (called procedures)
- Phase 3: Primary (main functionality)
- Phase 4: Integration (external interfaces)

---

## Output Format

```json
{
  "translation_order": [
    {
      "phase": "Phase 1: Foundation",
      "procedures": [
        {
          "name": "PROCEDURE_NAME",
          "type": "structure|utility|dependency|primary",
          "complexity": "Simple|Medium|Complex|Very Complex",
          "dependencies": ["PROC1", "PROC2"],
          "rationale": "Why this order",
          "estimated_effort": "Small|Medium|Large",
          "key_challenges": ["challenge1", "challenge2"]
        }
      ]
    }
  ],
  "shared_components": {
    "data_structures": ["STRUCT1", "STRUCT2"],
    "utilities": ["UTIL1", "UTIL2"],
    "interfaces": ["INTERFACE1"]
  },
  "risks": [
    {
      "procedure": "PROC_NAME",
      "risk": "Description",
      "mitigation": "How to handle"
    }
  ],
  "recommendations": [
    "Recommendation 1",
    "Recommendation 2"
  ]
}
```

---

## Important Notes

- Be thorough in your analysis
- Consider dependencies carefully
- Identify ALL risks upfront
- Provide clear rationale for translation order
- This plan will guide the worker agents

Begin your analysis now.
"""
        
        return prompt
    
    @staticmethod
    def generate_worker_prompt(
        context: Dict[str, Any],
        procedure: Dict[str, Any],
        procedure_index: int,
        total_procedures: int,
        previous_translations: List[str] = None,
        plan_summary: str = None
    ) -> str:
        """
        Generate a worker agent prompt for translating ONE procedure.
        
        Args:
            context: Full translation context
            procedure: The specific procedure to translate
            procedure_index: Current procedure number (1-based)
            total_procedures: Total number of procedures
            previous_translations: List of already translated procedure names
            plan_summary: Summary from planning agent
        
        Returns:
            Worker prompt string
        """
        
        prompt = f"""# Worker Agent Task - Procedure {procedure_index}/{total_procedures}

You are a **Worker Agent** translating ONE procedure as part of a larger translation project.

## Project Context

**Functionality**: {context['functionality']}
**Your task**: Translate procedure **{procedure['name']}**
**Progress**: Procedure {procedure_index} of {total_procedures}

"""
        
        if plan_summary:
            prompt += f"""## Planning Context

{plan_summary}

"""
        
        if previous_translations:
            prompt += f"""## Already Translated

The following procedures have been translated and are available:
"""
            for prev in previous_translations:
                prompt += f"- {prev}\n"
            prompt += "\nYou can reference these in your translation.\n\n"
        
        prompt += f"""---

## Procedure to Translate: {procedure['name']}

**Location**: `{procedure['file']}:{procedure['line']}`
**Parameters**: {', '.join(procedure['parameters']) if procedure['parameters'] else 'none'}
**Return Type**: {procedure['return_type'] or 'void'}
**Code Size**: {procedure.get('code_length', 0):,} characters

"""
        
        # Add dependency context
        deps = procedure.get('dependencies', {})
        if deps:
            prompt += """### Dependencies

"""
            if deps.get('calls'):
                prompt += f"**Calls**: {', '.join(deps['calls'][:10])}\n"
                if len(deps['calls']) > 10:
                    prompt += f"... and {len(deps['calls']) - 10} more\n"
            
            if deps.get('called_by'):
                prompt += f"**Called by**: {', '.join(deps['called_by'][:5])}\n"
                if len(deps['called_by']) > 5:
                    prompt += f"... and {len(deps['called_by']) - 5} more\n"
            
            if deps.get('uses_structures'):
                prompt += f"**Uses structures**: {', '.join(deps['uses_structures'])}\n"
            
            if deps.get('uses_variables'):
                prompt += f"**Uses variables**: {', '.join(deps['uses_variables'][:10])}\n"
            
            prompt += "\n"
        
        # Add the TAL code
        if procedure.get('code'):
            prompt += f"""### TAL Source Code

```tal
{procedure['code']}
```

"""
        else:
            prompt += """### ⚠️ Source Code Not Available

The source code for this procedure could not be extracted. Please note this in your output and provide what you can based on the metadata.

"""
        
        # Add call graph context
        proc_name = procedure['name']
        if proc_name in context.get('call_graph', {}):
            graph = context['call_graph'][proc_name]
            prompt += """### Call Graph for This Procedure

```
"""
            prompt += f"{proc_name}:\n"
            if graph.get('calls'):
                prompt += "  Calls:\n"
                for callee in graph['calls'][:8]:
                    prompt += f"    → {callee}\n"
            if graph.get('called_by'):
                prompt += "  Called by:\n"
                for caller in graph['called_by'][:8]:
                    prompt += f"    ← {caller}\n"
            prompt += "```\n\n"
        
        # Add related structures
        related_structs = [s for s in context['structures'] 
                          if s['name'] in deps.get('uses_structures', [])]
        if related_structs:
            prompt += """### Related Data Structures

"""
            for struct in related_structs:
                prompt += f"""#### {struct['name']}

```tal
{struct['code'] if struct['code'] else '// Structure definition not available'}
```

"""
        
        prompt += """---

## Your Task

Translate this ONE procedure completely following the 4-phase methodology:

### Phase 1: ANALYZE
```
PROCEDURE ANALYSIS: {proc_name}

PURPOSE:
  [What this procedure does in 1-2 sentences]

INPUT PARAMETERS:
  [List each parameter with type and purpose]

BUSINESS LOGIC:
  Section 1 (TAL lines X-Y): [What it does]
  Section 2 (TAL lines A-B): [What it does]
  ...

ERROR HANDLING:
  - Error code -1: [Meaning]
  - Error code -2: [Meaning]
  ...

DATA DEPENDENCIES:
  - Reads: [variables/structures]
  - Writes: [variables/structures]
  - Calls: [other procedures]
```

### Phase 2: MAP
```
TAL CONSTRUCT           | JAVA EQUIVALENT
────────────────────────|──────────────────
[TAL construct]         | [Java equivalent]
...
```

### Phase 3: IMPLEMENT
```java
/**
 * [Complete Javadoc]
 * 
 * TAL equivalent: {proc_name}
 * TAL location: {procedure['file']}:{procedure['line']}
 */
public [ReturnType] [methodName]([parameters]) {{
    // TAL lines X-Y: [Explanation]
    [Complete Java implementation]
    
    // NO PLACEHOLDERS
    // NO TODOs
    // EVERY line of TAL → Java
}}
```

### Phase 4: VERIFY
```
VERIFICATION CHECKLIST for {proc_name}:

Input Validation:
  [ ] TAL line __: Validation → Java equivalent

Business Logic:
  [ ] TAL line __: Logic → Java implementation

Error Handling:
  [ ] TAL line __: Error handling → Java

Completeness:
  [ ] Every TAL line has Java equivalent
  [ ] No TODO comments
  [ ] No placeholders
  [ ] All edge cases handled
```

---

## Output Format

Provide your translation in this EXACT structure:

```
═══════════════════════════════════════════════════════════════
PROCEDURE: {proc_name} ({procedure_index}/{total_procedures})
═══════════════════════════════════════════════════════════════

PHASE 1: ANALYSIS
────────────────────────────────────────────────────────────────
[Your analysis here]

PHASE 2: MAPPING
────────────────────────────────────────────────────────────────
[Your TAL→Java mappings here]

PHASE 3: IMPLEMENTATION
────────────────────────────────────────────────────────────────
```java
[Complete Java code here]
```

PHASE 4: VERIFICATION
────────────────────────────────────────────────────────────────
[Your verification checklist here]

TRACEABILITY:
  TAL Line X → Java Line Y: [Mapping]
  ...
```

---

## Important Reminders

- Focus ONLY on this procedure ({proc_name})
- Translate COMPLETELY (no placeholders)
- Reference already-translated procedures when needed
- Follow the 4-phase methodology
- Include ALL error handling
- Map EVERY TAL line to Java

"""
        
        # Add reminder about what's coming
        remaining = total_procedures - procedure_index
        if remaining > 0:
            prompt += f"""---

## Next Steps

After you complete this procedure, the next worker agent will translate the remaining {remaining} procedure(s).
Your work will be used as reference for subsequent translations.

"""
        else:
            prompt += """---

## Final Procedure

This is the LAST procedure in the translation. After this, all procedures will be translated!

"""
        
        prompt += """Begin your translation now.
"""
        
        return prompt
    
    @staticmethod
    def generate_consolidation_prompt(
        context: Dict[str, Any],
        all_translations: List[Dict[str, Any]]
    ) -> str:
        """
        Generate prompt for consolidation agent to combine all translations.
        
        Args:
            context: Full translation context
            all_translations: List of all translated procedures
        
        Returns:
            Consolidation prompt string
        """
        
        prompt = f"""# Consolidation Agent Task

You are the **Consolidation Agent**. All procedures have been translated individually by worker agents.

## Your Mission

Combine all translations into a cohesive, production-ready Java implementation.

## Project Summary

**Functionality**: {context['functionality']}
**Total procedures translated**: {len(all_translations)}
**Total structures**: {len(context['structures'])}

---

## Translated Procedures

"""
        
        for i, translation in enumerate(all_translations, 1):
            prompt += f"{i}. {translation['procedure_name']} ({translation.get('line_count', '?')} lines)\n"
        
        prompt += """
---

## Your Tasks

### 1. Package Structure
Create appropriate package structure:
```
com.company.{domain}.{module}/
├── service/          (main service classes)
├── model/            (data structures)
├── dto/              (data transfer objects)
├── repository/       (if needed)
└── util/             (utilities)
```

### 2. Class Organization
Organize procedures into appropriate classes:
- Service classes for business logic
- Model classes for data structures
- Utility classes for helpers
- Interface definitions for external dependencies

### 3. Import Statements
Add all necessary imports:
```java
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
// ... etc
```

### 4. Spring Annotations (if applicable)
Add appropriate annotations:
```java
@Service
@Repository
@Component
@Transactional
// ... etc
```

### 5. Dependency Injection
Set up proper dependency injection:
```java
private final SomeService someService;

public MyService(SomeService someService) {
    this.someService = someService;
}
```

### 6. Error Handling
Consolidate error handling:
- Custom exception classes
- Exception handling patterns
- Error codes mapping

### 7. Configuration
Create configuration classes if needed:
```java
@Configuration
public class PaymentConfig {
    // Configuration beans
}
```

### 8. Unit Test Skeleton
Provide comprehensive test skeleton:
```java
@Test
public void testProcedureName_scenario() {
    // Given
    // When
    // Then
}
```

---

## Output Format

### Part 1: Project Structure
```
src/main/java/
  com/company/{domain}/{module}/
    ├── service/
    │   ├── {MainService}.java
    │   └── {HelperService}.java
    ├── model/
    │   ├── {DataClass1}.java
    │   └── {DataClass2}.java
    └── ...

src/test/java/
  com/company/{domain}/{module}/
    └── service/
        └── {MainService}Test.java
```

### Part 2: Complete Java Files

For each file:
```java
package com.company.{domain}.{module}.{subpackage};

// Imports

/**
 * Complete class-level Javadoc
 * 
 * Translated from TAL files: {list of files}
 */
public class {ClassName} {
    [Complete implementation combining all relevant procedures]
}
```

### Part 3: Configuration Files

Any needed configuration:
- application.properties
- application.yml
- Configuration classes

### Part 4: Test Files

Complete test skeleton for all classes

### Part 5: Documentation

README.md with:
- Overview of translated functionality
- Class responsibilities
- Key design decisions
- Migration notes
- Known issues or TODOs (if any)

---

## Quality Checklist

Before submitting, verify:

- [ ] All procedures are included
- [ ] Proper package structure
- [ ] All imports present
- [ ] Appropriate annotations
- [ ] Dependency injection configured
- [ ] Error handling consistent
- [ ] Logging in place
- [ ] Javadoc complete
- [ ] Test skeleton comprehensive
- [ ] No TODO or placeholder code
- [ ] Code follows Java conventions
- [ ] BigDecimal used for financial amounts (if applicable)

---

Begin consolidation now. Provide complete, production-ready Java code.
"""
        
        return prompt
    
    @staticmethod
    def save_prompts(
        system_prompt: str,
        planning_prompt: str,
        worker_prompts: List[Dict[str, str]],
        consolidation_prompt: str,
        output_dir: str,
        functionality: str
    ):
        """
        Save all prompts to files.
        
        Args:
            system_prompt: System prompt (reused across all agents)
            planning_prompt: Planning agent prompt
            worker_prompts: List of worker prompts with metadata
            consolidation_prompt: Consolidation agent prompt
            output_dir: Directory to save prompts
            functionality: Functionality name
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save system prompt
        system_file = output_path / f"{functionality}_00_system_prompt.md"
        with open(system_file, 'w', encoding='utf-8') as f:
            f.write("# System Prompt (Reused for All Agents)\n\n")
            f.write(system_prompt)
        
        # Save planning prompt
        planning_file = output_path / f"{functionality}_01_planning_agent.md"
        with open(planning_file, 'w', encoding='utf-8') as f:
            f.write(planning_prompt)
        
        # Save worker prompts
        for i, worker in enumerate(worker_prompts, 1):
            worker_file = output_path / f"{functionality}_02_worker_{i:03d}_{worker['procedure_name']}.md"
            with open(worker_file, 'w', encoding='utf-8') as f:
                f.write(worker['prompt'])
        
        # Save consolidation prompt
        consolidation_file = output_path / f"{functionality}_03_consolidation_agent.md"
        with open(consolidation_file, 'w', encoding='utf-8') as f:
            f.write(consolidation_prompt)
        
        # Save execution manifest
        manifest_file = output_path / f"{functionality}_EXECUTION_MANIFEST.md"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            f.write(f"""# Execution Manifest: {functionality}

## Multi-Agent Translation Workflow

### Phase 1: System Setup
**File**: `{functionality}_00_system_prompt.md`
**Usage**: Use this as the SYSTEM PROMPT for ALL agents (planning, workers, consolidation)

### Phase 2: Planning
**File**: `{functionality}_01_planning_agent.md`
**Agent**: Planning Agent
**Action**: Analyze all procedures and create execution plan
**Output**: JSON with translation order and strategy

### Phase 3: Worker Agents (One per procedure)
""")
            for i, worker in enumerate(worker_prompts, 1):
                f.write(f"""
**Worker {i}**: `{functionality}_02_worker_{i:03d}_{worker['procedure_name']}.md`
- Procedure: {worker['procedure_name']}
- System Prompt: Use `{functionality}_00_system_prompt.md`
- Action: Translate this ONE procedure completely
- Output: Complete Java implementation with verification
""")
            
            f.write(f"""
### Phase 4: Consolidation
**File**: `{functionality}_03_consolidation_agent.md`
**Agent**: Consolidation Agent
**Action**: Combine all translations into production-ready code
**Output**: Complete Java project structure

---

## Execution Order

1. **Setup**: Load system prompt into your LLM configuration
2. **Plan**: Run planning agent → Get translation order
3. **Translate**: Run worker agents in the order determined by planner
   - Each worker focuses on ONE procedure
   - Each worker has full context (dependencies, call graph)
   - Workers can potentially run in parallel
4. **Consolidate**: Run consolidation agent with all worker outputs
5. **Review**: Verify final output with quality checklist

---

## File Statistics

- System prompt: {len(system_prompt):,} characters
- Planning prompt: {len(planning_prompt):,} characters
- Worker prompts: {len(worker_prompts)} files
- Total workers: {len(worker_prompts)}
- Consolidation prompt: {len(consolidation_prompt):,} characters

---

## Advantages of This Approach

✅ **No Context Overflow**: Each worker handles one procedure
✅ **Better Focus**: LLM focuses on one task at a time
✅ **Higher Quality**: More attention to detail per procedure
✅ **Parallelizable**: Workers can run simultaneously
✅ **Scalable**: Can handle large codebases (100+ procedures)
✅ **Traceable**: Clear audit trail of what was translated when
✅ **Recoverable**: Can restart from any failed worker

---

## Notes

- Always use the system prompt as the base for all agents
- Follow the execution order for best results
- Save outputs from each agent for traceability
- The planner's output informs the worker order
""")
        
        print(f"✓ Multi-agent prompts saved to: {output_dir}/")
        print(f"  System prompt: {functionality}_00_system_prompt.md")
        print(f"  Planning agent: {functionality}_01_planning_agent.md")
        print(f"  Worker agents: {len(worker_prompts)} files")
        print(f"  Consolidation: {functionality}_03_consolidation_agent.md")
        print(f"  Manifest: {functionality}_EXECUTION_MANIFEST.md")
        print(f"\nTotal files: {3 + len(worker_prompts)} prompts")


def generate_multi_agent_prompts(
    context: Dict[str, Any],
    output_dir: str,
    financial_code: bool = True,
    plan_summary: str = None
) -> Dict[str, Any]:
    """
    Generate all multi-agent prompts for a translation context.
    
    Args:
        context: Translation context from TranslationContextBuilder
        output_dir: Directory to save prompts
        financial_code: Include financial code requirements
        plan_summary: Optional summary from planning agent
    
    Returns:
        Dictionary with file paths and statistics
    """
    from pathlib import Path
    
    functionality = context['functionality']
    
    print(f"\n{'='*70}")
    print(f"GENERATING MULTI-AGENT PROMPTS: {functionality}")
    print(f"{'='*70}\n")
    
    # Generate system prompt (reused by all agents)
    print("Step 1: Generating system prompt...")
    system_prompt = MultiAgentPromptGenerator.generate_system_prompt(
        financial_code=financial_code
    )
    print(f"  ✓ System prompt: {len(system_prompt):,} characters")
    
    # Generate planning agent prompt
    print("\nStep 2: Generating planning agent prompt...")
    planning_prompt = MultiAgentPromptGenerator.generate_planning_prompt(context)
    print(f"  ✓ Planning prompt: {len(planning_prompt):,} characters")
    
    # Generate worker prompts (one per primary procedure)
    print(f"\nStep 3: Generating worker agent prompts...")
    worker_prompts = []
    primary_procedures = context['primary_procedures']
    total_primary = len(primary_procedures)
    
    # Track which procedures have been "translated" (for context)
    previous_translations = []
    
    for i, proc in enumerate(primary_procedures, 1):
        print(f"  Worker {i}/{total_primary}: {proc['name']}")
        
        worker_prompt = MultiAgentPromptGenerator.generate_worker_prompt(
            context=context,
            procedure=proc,
            procedure_index=i,
            total_procedures=total_primary,
            previous_translations=previous_translations[:],  # Copy
            plan_summary=plan_summary
        )
        
        worker_prompts.append({
            'procedure_name': proc['name'],
            'prompt': worker_prompt,
            'index': i
        })
        
        # Add to previous translations for next worker's context
        previous_translations.append(proc['name'])
    
    print(f"  ✓ Generated {len(worker_prompts)} worker prompts")
    
    # Generate consolidation prompt
    print("\nStep 4: Generating consolidation agent prompt...")
    
    # Create mock translation list for consolidation prompt
    all_translations = [
        {
            'procedure_name': proc['name'],
            'line_count': proc.get('code_length', 0) // 50  # Rough estimate
        }
        for proc in primary_procedures
    ]
    
    consolidation_prompt = MultiAgentPromptGenerator.generate_consolidation_prompt(
        context=context,
        all_translations=all_translations
    )
    print(f"  ✓ Consolidation prompt: {len(consolidation_prompt):,} characters")
    
    # Save all prompts
    print(f"\nStep 5: Saving all prompts to {output_dir}...")
    MultiAgentPromptGenerator.save_prompts(
        system_prompt=system_prompt,
        planning_prompt=planning_prompt,
        worker_prompts=worker_prompts,
        consolidation_prompt=consolidation_prompt,
        output_dir=output_dir,
        functionality=functionality
    )
    
    # Generate statistics
    total_prompt_size = (
        len(system_prompt) + 
        len(planning_prompt) + 
        sum(len(w['prompt']) for w in worker_prompts) +
        len(consolidation_prompt)
    )
    
    result = {
        'functionality': functionality,
        'output_dir': output_dir,
        'files': {
            'system_prompt': f"{functionality}_00_system_prompt.md",
            'planning_prompt': f"{functionality}_01_planning_agent.md",
            'worker_prompts': [
                f"{functionality}_02_worker_{w['index']:03d}_{w['procedure_name']}.md"
                for w in worker_prompts
            ],
            'consolidation_prompt': f"{functionality}_03_consolidation_agent.md",
            'manifest': f"{functionality}_EXECUTION_MANIFEST.md"
        },
        'statistics': {
            'system_prompt_chars': len(system_prompt),
            'planning_prompt_chars': len(planning_prompt),
            'worker_count': len(worker_prompts),
            'total_worker_chars': sum(len(w['prompt']) for w in worker_prompts),
            'avg_worker_chars': sum(len(w['prompt']) for w in worker_prompts) // len(worker_prompts) if worker_prompts else 0,
            'consolidation_chars': len(consolidation_prompt),
            'total_chars': total_prompt_size,
            'estimated_tokens': total_prompt_size // 4
        },
        'procedures': [w['procedure_name'] for w in worker_prompts]
    }
    
    print(f"\n{'='*70}")
    print("MULTI-AGENT PROMPT GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nStatistics:")
    print(f"  Total prompts: {3 + len(worker_prompts)} files")
    print(f"  Worker agents: {len(worker_prompts)}")
    print(f"  Total size: {total_prompt_size:,} characters (~{total_prompt_size // 4:,} tokens)")
    print(f"  Avg worker size: {result['statistics']['avg_worker_chars']:,} characters")
    print(f"\nExecution order:")
    print(f"  1. Use system prompt for ALL agents")
    print(f"  2. Run planning agent")
    print(f"  3. Run {len(worker_prompts)} worker agents (can parallelize)")
    print(f"  4. Run consolidation agent")
    print(f"\nSee {functionality}_EXECUTION_MANIFEST.md for detailed instructions")
    print(f"{'='*70}\n")
    
    return result


def example_usage():
    """Example usage of the multi-agent prompt generator."""
    
    # Mock context (in real use, this comes from TranslationContextBuilder)
    context = {
        'functionality': 'wire_transfer',
        'summary': {
            'primary_procedures': 3,
            'total_procedures': 10,
            'total_structures': 2,
            'code_extraction': {
                'success': 9,
                'failed': 1,
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
                'code': '''PROC PROCESS_WIRE_TRANSFER(wire_id, amount, dest_account);
  BEGIN
    INT status;
    FIXED(2) transfer_amount;
    
    ! Validate amount
    IF amount <= 0 THEN
      RETURN -1;  ! Invalid amount
    
    ! Check limits
    CALL CHECK_TRANSFER_LIMITS(amount, status);
    IF status <> 0 THEN
      RETURN -2;  ! Limit exceeded
    
    ! Validate account
    CALL VALIDATE_ACCOUNT(dest_account, status);
    IF status <> 0 THEN
      RETURN -3;  ! Invalid account
    
    ! Process transfer
    transfer_amount := amount;
    CALL EXECUTE_TRANSFER(wire_id, transfer_amount, dest_account, status);
    
    RETURN status;
  END;
''',
                'code_length': 500,
                'dependencies': {
                    'calls': ['CHECK_TRANSFER_LIMITS', 'VALIDATE_ACCOUNT', 'EXECUTE_TRANSFER'],
                    'called_by': ['WIRE_HANDLER'],
                    'uses_structures': ['WIRE_RECORD'],
                    'uses_variables': []
                }
            },
            {
                'name': 'VALIDATE_ACCOUNT',
                'file': 'wire.tal',
                'line': 200,
                'parameters': ['account_id', 'status'],
                'return_type': 'void',
                'code': '''PROC VALIDATE_ACCOUNT(account_id, status);
  BEGIN
    ! Check if account exists
    IF account_id = "" THEN
      status := -1;
      RETURN;
    
    ! Check account status
    CALL GET_ACCOUNT_STATUS(account_id, status);
    
    RETURN;
  END;
''',
                'code_length': 200,
                'dependencies': {
                    'calls': ['GET_ACCOUNT_STATUS'],
                    'called_by': ['PROCESS_WIRE_TRANSFER'],
                    'uses_structures': [],
                    'uses_variables': []
                }
            },
            {
                'name': 'EXECUTE_TRANSFER',
                'file': 'wire.tal',
                'line': 300,
                'parameters': ['wire_id', 'amount', 'dest_account', 'status'],
                'return_type': 'void',
                'code': '''PROC EXECUTE_TRANSFER(wire_id, amount, dest_account, status);
  BEGIN
    ! Log transfer
    CALL LOG_TRANSACTION(wire_id, amount, dest_account);
    
    ! Execute transfer
    CALL UPDATE_BALANCES(amount, dest_account, status);
    IF status <> 0 THEN
      RETURN;
    
    ! Update wire record
    CALL UPDATE_WIRE_STATUS(wire_id, 1);  ! 1 = completed
    
    status := 0;  ! Success
    RETURN;
  END;
''',
                'code_length': 300,
                'dependencies': {
                    'calls': ['LOG_TRANSACTION', 'UPDATE_BALANCES', 'UPDATE_WIRE_STATUS'],
                    'called_by': ['PROCESS_WIRE_TRANSFER'],
                    'uses_structures': ['WIRE_RECORD'],
                    'uses_variables': []
                }
            }
        ],
        'dependency_procedures': [
            {
                'name': 'CHECK_TRANSFER_LIMITS',
                'file': 'limits.tal',
                'line': 50,
                'parameters': ['amount', 'status'],
                'return_type': 'void',
                'code': '! Code here',
                'code_length': 150,
                'is_external': False,
                'dependencies': {}
            }
        ],
        'structures': [
            {
                'name': 'WIRE_RECORD',
                'fields': [
                    {'name': 'wire_id', 'type': 'INT'},
                    {'name': 'amount', 'type': 'FIXED(2)'},
                    {'name': 'dest_account', 'type': 'STRING'}
                ],
                'code': 'STRUCT WIRE_RECORD; BEGIN INT wire_id; FIXED(2) amount; STRING .dest_account; END;'
            }
        ],
        'variables': [],
        'call_graph': {
            'PROCESS_WIRE_TRANSFER': {
                'calls': ['CHECK_TRANSFER_LIMITS', 'VALIDATE_ACCOUNT', 'EXECUTE_TRANSFER'],
                'called_by': ['WIRE_HANDLER']
            },
            'VALIDATE_ACCOUNT': {
                'calls': ['GET_ACCOUNT_STATUS'],
                'called_by': ['PROCESS_WIRE_TRANSFER']
            },
            'EXECUTE_TRANSFER': {
                'calls': ['LOG_TRANSACTION', 'UPDATE_BALANCES', 'UPDATE_WIRE_STATUS'],
                'called_by': ['PROCESS_WIRE_TRANSFER']
            }
        }
    }
    
    # Generate multi-agent prompts
    result = generate_multi_agent_prompts(
        context=context,
        output_dir='./multi_agent_prompts',
        financial_code=True
    )
    
    print("\n✓ Example complete!")
    print(f"\nGenerated files in: {result['output_dir']}/")
    print("\nTo use:")
    print("  1. Load system prompt into LLM")
    print("  2. Run planning agent")
    print("  3. Run each worker agent")
    print("  4. Run consolidation agent")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'example':
        example_usage()
    else:
        print("Multi-Agent Prompt Generator")
        print("\nUsage:")
        print("  python multi_agent_prompt_generator.py example")
        print("\nOr import and use programmatically:")
        print("  from multi_agent_prompt_generator import generate_multi_agent_prompts")
        print("  result = generate_multi_agent_prompts(context, './output')")
