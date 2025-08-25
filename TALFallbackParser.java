import java.util.*;
import java.util.regex.*;
import java.io.*;

/**
 * TAL Fallback Parser - Robust regex-based parsing for when grammar fails
 * Updated to work with refactored class names
 */
public class TALFallbackParser {
    
    // Analysis results - updated to use refactored classes
    private List<TALProcedure> procedures = new ArrayList<>();
    private List<TALDataItem> extractedDataItems = new ArrayList<>();
    private List<TALStatement> sqlStatements = new ArrayList<>();
    private List<TALStatement> copyStatements = new ArrayList<>();
    private List<TALStatement> callStatements = new ArrayList<>();
    private List<TALStatement> systemStatements = new ArrayList<>();
    private Map<String, Integer> statementCounts = new HashMap<>();
    private Map<String, Integer> callReferences = new HashMap<>();
    private List<String> parseWarnings = new ArrayList<>();
    
    // Current parsing context
    private TALProcedure currentProcedure;
    private Set<String> processedProcedures = new HashSet<>();
    
    // Keywords and patterns
    private static final Set<String> TAL_KEYWORDS = Set.of(
        "PROC", "SUBPROC", "INT", "STRING", "REAL", "FIXED", "BYTE", "CHAR",
        "IF", "THEN", "ELSE", "WHILE", "FOR", "RETURN", "BEGIN", "END", 
        "FORWARD", "STRUCT", "MAIN", "INTERRUPT", "RESIDENT", "CALLABLE",
        "CALL", "MOVE", "SCAN", "RSCAN", "BITDEPOSIT", "ASSIGN", "TO",
        "CASE", "OF", "OTHERWISE", "UNTIL", "DO", "DOWNTO", "BY", "GOTO"
    );
    
    // Compiled patterns for performance
    private static final Pattern PROC_PATTERN = Pattern.compile(
        "(?:^|\\s)(?:(\\w+(?:\\([^)]*\\))?)\\s+)?(PROC|SUBPROC)\\s+(\\w+)\\s*\\(([^)]*)\\)(?:\\s+(\\w+(?:,\\s*\\w+)*))?", 
        Pattern.CASE_INSENSITIVE
    );
    
    private static final Pattern DATA_PATTERN = Pattern.compile(
        "\\b(INT|STRING|REAL|FIXED|BYTE|CHAR|TIMESTAMP|STRUCT|UNSIGNED|EXTADDR|SGADDR)(?:\\([^)]*\\))?\\s+(\\w+)(?:\\s*:=\\s*[^;]+)?[;\\s]*", 
        Pattern.CASE_INSENSITIVE
    );
    
    private static final Pattern CALL_PATTERN = Pattern.compile(
        "\\b(?:CALL\\s+)?(\\w+)\\s*\\([^)]*\\)(?:\\s*->\\s*(\\w+))?", 
        Pattern.CASE_INSENSITIVE
    );
    
    private static final Pattern ASSIGNMENT_PATTERN = Pattern.compile(
        "(\\w+(?:\\.\\w+|\\[\\d+\\])*?)\\s*:=\\s*(.+)", 
        Pattern.CASE_INSENSITIVE
    );
    
    private static final Pattern IF_PATTERN = Pattern.compile(
        "\\bIF\\s+(.+?)\\s+THEN\\b", 
        Pattern.CASE_INSENSITIVE
    );
    
    private static final Pattern WHILE_PATTERN = Pattern.compile(
        "\\bWHILE\\s+(.+?)\\s+DO\\b", 
        Pattern.CASE_INSENSITIVE
    );
    
    private static final Pattern FOR_PATTERN = Pattern.compile(
        "\\bFOR\\s+(\\w+)\\s*:=\\s*(.+?)\\s+(?:TO|DOWNTO)\\s+(.+?)(?:\\s+BY\\s+(.+?))?\\s+DO\\b", 
        Pattern.CASE_INSENSITIVE
    );
    
    public TALFallbackParser() {
        System.out.println("TAL Fallback Parser initialized for regex-based analysis");
    }
    
    // =====================================================================
    // MAIN PARSING METHOD
    // =====================================================================
    
    public boolean parseTALSource(String[] sourceLines) {
        System.out.println("Starting fallback regex-based parsing of " + sourceLines.length + " lines...");
        
        int proceduresFound = 0;
        int dataItemsFound = 0;
        int statementsFound = 0;
        
        for (int i = 0; i < sourceLines.length; i++) {
            String line = sourceLines[i].trim();
            int lineNumber = i + 1;
            
            if (line.isEmpty() || line.startsWith("!")) continue;
            
            try {
                // Parse procedures
                if (isProcedureDeclaration(line)) {
                    TALProcedure proc = parseProcedureLine(line, lineNumber);
                    if (proc != null && !isDuplicateProcedure(proc.getName())) {
                        procedures.add(proc);
                        processedProcedures.add(proc.getName().toUpperCase());
                        currentProcedure = proc;
                        proceduresFound++;
                        System.out.println("Fallback: Found procedure: " + proc.getName() + " at line " + lineNumber);
                    }
                }
                
                // Parse data declarations
                if (isDataDeclaration(line)) {
                    TALDataItem dataItem = parseDataLine(line, lineNumber);
                    if (dataItem != null) {
                        extractedDataItems.add(dataItem);
                        dataItemsFound++;
                        System.out.println("Fallback: Found data item: " + dataItem.getName() + " (" + dataItem.getDataType() + ") at line " + lineNumber);
                    }
                }
                
                // Parse statements
                List<TALStatement> statements = parseStatements(line, lineNumber);
                for (TALStatement stmt : statements) {
                    categorizeAndAddStatement(stmt);
                    statementsFound++;
                }
                
                // Extract call references from any line
                extractCallReferences(line);
                
            } catch (Exception e) {
                parseWarnings.add("Error parsing line " + lineNumber + ": " + e.getMessage());
            }
        }
        
        System.out.println("Fallback parsing completed:");
        System.out.println("  - Procedures: " + proceduresFound);
        System.out.println("  - Data items: " + dataItemsFound);
        System.out.println("  - Statements: " + statementsFound);
        System.out.println("  - Warnings: " + parseWarnings.size());
        
        return proceduresFound > 0 || dataItemsFound > 0 || statementsFound > 0;
    }
    
    // =====================================================================
    // PROCEDURE PARSING
    // =====================================================================
    
    private boolean isProcedureDeclaration(String line) {
        return PROC_PATTERN.matcher(line).find() &&
               !line.toUpperCase().trim().startsWith("FORWARD") &&
               line.contains("(") && line.contains(")");
    }
    
    private TALProcedure parseProcedureLine(String line, int lineNumber) {
        Matcher matcher = PROC_PATTERN.matcher(line);
        if (!matcher.find()) return null;
        
        TALProcedure procedure = new TALProcedure();
        
        String returnType = matcher.group(1);
        String procType = matcher.group(2);
        String procName = matcher.group(3);
        String parameters = matcher.group(4);
        String attributes = matcher.group(5);
        
        procedure.setName(procName);
        procedure.setLineNumber(lineNumber);
        procedure.setEndLineNumber(lineNumber);
        procedure.setReasoningInfo("Found via fallback regex parsing");
        procedure.setContextScore(25.0);
        
        if (returnType != null && !returnType.trim().isEmpty()) {
            procedure.setReturnType(returnType.trim());
        }
        
        if (parameters != null && !parameters.trim().isEmpty()) {
            List<String> paramList = Arrays.asList(parameters.split(","));
            procedure.setParameters(paramList.stream()
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .collect(ArrayList::new, ArrayList::add, ArrayList::addAll));
        }
        
        List<String> attrList = new ArrayList<>();
        attrList.add("TYPE:" + procType.toUpperCase());
        
        if (attributes != null && !attributes.trim().isEmpty()) {
            String[] attrs = attributes.split(",");
            for (String attr : attrs) {
                attrList.add(attr.trim().toUpperCase());
            }
        }
        
        // Check for common procedure attributes in the line
        String upperLine = line.toUpperCase();
        if (upperLine.contains("MAIN")) attrList.add("MAIN");
        if (upperLine.contains("INTERRUPT")) attrList.add("INTERRUPT");
        if (upperLine.contains("RESIDENT")) attrList.add("RESIDENT");
        if (upperLine.contains("CALLABLE")) attrList.add("CALLABLE");
        if (upperLine.contains("PRIV")) attrList.add("PRIV");
        if (upperLine.contains("VARIABLE")) attrList.add("VARIABLE");
        if (upperLine.contains("EXTENSIBLE")) attrList.add("EXTENSIBLE");
        
        procedure.setAttributes(attrList);
        
        return procedure;
    }
    
    private boolean isDuplicateProcedure(String procName) {
        return processedProcedures.contains(procName.toUpperCase());
    }
    
    // =====================================================================
    // DATA PARSING
    // =====================================================================
    
    private boolean isDataDeclaration(String line) {
        return DATA_PATTERN.matcher(line).find() && !isProcedureDeclaration(line);
    }
    
    private TALDataItem parseDataLine(String line, int lineNumber) {
        Matcher matcher = DATA_PATTERN.matcher(line);
        if (!matcher.find()) return null;
        
        TALDataItem dataItem = new TALDataItem();
        dataItem.setDataType(matcher.group(1));
        dataItem.setName(matcher.group(2));
        dataItem.setLineNumber(lineNumber);
        dataItem.setDefinition(line);
        dataItem.setSection(currentProcedure != null ? currentProcedure.getName() : "GLOBAL");
        
        return dataItem;
    }
    
    // =====================================================================
    // STATEMENT PARSING
    // =====================================================================
    
    private List<TALStatement> parseStatements(String line, int lineNumber) {
        List<TALStatement> statements = new ArrayList<>();
        
        // Try different statement patterns
        statements.addAll(parseCallStatements(line, lineNumber));
        statements.addAll(parseAssignmentStatements(line, lineNumber));
        statements.addAll(parseControlFlowStatements(line, lineNumber));
        statements.addAll(parseFileOperationStatements(line, lineNumber));
        statements.addAll(parseMoveStatements(line, lineNumber));
        statements.addAll(parseOtherStatements(line, lineNumber));
        
        return statements;
    }
    
    private List<TALStatement> parseCallStatements(String line, int lineNumber) {
        List<TALStatement> statements = new ArrayList<>();
        
        Matcher callMatcher = CALL_PATTERN.matcher(line);
        while (callMatcher.find()) {
            String calledProc = callMatcher.group(1);
            if (!isKeywordOrReserved(calledProc)) {
                TALStatement stmt = createStatement("CALL", line, lineNumber);
                statements.add(stmt);
                incrementStatementCount("CALL");
                callReferences.merge(calledProc.toUpperCase(), 1, Integer::sum);
            }
        }
        
        // Handle explicit CALL statements
        if (line.toUpperCase().contains("CALL ")) {
            Pattern explicitCall = Pattern.compile("\\bCALL\\s+(\\w+)", Pattern.CASE_INSENSITIVE);
            Matcher matcher = explicitCall.matcher(line);
            if (matcher.find()) {
                String calledProc = matcher.group(1);
                if (!isKeywordOrReserved(calledProc)) {
                    TALStatement stmt = createStatement("CALL", line, lineNumber);
                    statements.add(stmt);
                    incrementStatementCount("CALL");
                    callReferences.merge(calledProc.toUpperCase(), 1, Integer::sum);
                }
            }
        }
        
        return statements;
    }
    
    private List<TALStatement> parseAssignmentStatements(String line, int lineNumber) {
        List<TALStatement> statements = new ArrayList<>();
        
        Matcher assignMatcher = ASSIGNMENT_PATTERN.matcher(line);
        if (assignMatcher.find()) {
            TALStatement stmt = createStatement("ASSIGNMENT", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("ASSIGNMENT");
            
            // Extract variable reference
            String varName = assignMatcher.group(1);
            if (!isKeywordOrReserved(varName)) {
                callReferences.merge("VAR_" + varName.toUpperCase(), 1, Integer::sum);
            }
        }
        
        return statements;
    }
    
    private List<TALStatement> parseControlFlowStatements(String line, int lineNumber) {
        List<TALStatement> statements = new ArrayList<>();
        
        // IF statements
        Matcher ifMatcher = IF_PATTERN.matcher(line);
        if (ifMatcher.find()) {
            TALStatement stmt = createStatement("IF", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("IF");
            extractConditionalReferences(ifMatcher.group(1));
        }
        
        // WHILE statements
        Matcher whileMatcher = WHILE_PATTERN.matcher(line);
        if (whileMatcher.find()) {
            TALStatement stmt = createStatement("WHILE", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("WHILE");
            extractConditionalReferences(whileMatcher.group(1));
        }
        
        // FOR statements
        Matcher forMatcher = FOR_PATTERN.matcher(line);
        if (forMatcher.find()) {
            TALStatement stmt = createStatement("FOR", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("FOR");
            
            String loopVar = forMatcher.group(1);
            if (!isKeywordOrReserved(loopVar)) {
                callReferences.merge("VAR_" + loopVar.toUpperCase(), 1, Integer::sum);
            }
        }
        
        // CASE statements
        if (line.toUpperCase().contains("CASE ")) {
            TALStatement stmt = createStatement("CASE", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("CASE");
        }
        
        // RETURN statements
        if (line.toUpperCase().matches(".*\\bRETURN\\b.*")) {
            TALStatement stmt = createStatement("RETURN", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("RETURN");
        }
        
        // GOTO statements
        if (line.toUpperCase().contains("GOTO ")) {
            TALStatement stmt = createStatement("GOTO", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("GOTO");
        }
        
        return statements;
    }
    
    private List<TALStatement> parseFileOperationStatements(String line, int lineNumber) {
        List<TALStatement> statements = new ArrayList<>();
        String upperLine = line.toUpperCase();
        
        if (upperLine.contains("READ") || upperLine.contains("WRITE") || 
            upperLine.contains("OPEN") || upperLine.contains("CLOSE") ||
            upperLine.contains("WRITEREAD")) {
            TALStatement stmt = createStatement("FILE_OP", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("FILE_OP");
        }
        
        return statements;
    }
    
    private List<TALStatement> parseMoveStatements(String line, int lineNumber) {
        List<TALStatement> statements = new ArrayList<>();
        String upperLine = line.toUpperCase();
        
        if (upperLine.contains("MOVE ")) {
            TALStatement stmt = createStatement("MOVE", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("MOVE");
        }
        
        if (upperLine.contains("SCAN ") || upperLine.contains("RSCAN ")) {
            TALStatement stmt = createStatement("SCAN", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("SCAN");
        }
        
        if (upperLine.contains("STRINGMOVE")) {
            TALStatement stmt = createStatement("STRINGMOVE", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("STRINGMOVE");
        }
        
        return statements;
    }
    
    private List<TALStatement> parseOtherStatements(String line, int lineNumber) {
        List<TALStatement> statements = new ArrayList<>();
        String upperLine = line.toUpperCase();
        
        if (upperLine.contains("BITDEPOSIT")) {
            TALStatement stmt = createStatement("BITDEPOSIT", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("BITDEPOSIT");
        }
        
        // SQL-like statements
        if (upperLine.contains("SELECT ") || upperLine.contains("INSERT ") || 
            upperLine.contains("UPDATE ") || upperLine.contains("DELETE ")) {
            TALStatement stmt = createStatement("SQL", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("SQL");
        }
        
        // Copy statements
        if (upperLine.contains("COPY ")) {
            TALStatement stmt = createStatement("COPY", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("COPY");
        }
        
        // Compiler directives
        if (line.startsWith("?") || upperLine.contains("SOURCE") || upperLine.contains("LIST")) {
            TALStatement stmt = createStatement("COMPILER_DIRECTIVE", line, lineNumber);
            statements.add(stmt);
            incrementStatementCount("COMPILER_DIRECTIVE");
        }
        
        return statements;
    }
    
    private void categorizeAndAddStatement(TALStatement stmt) {
        String type = stmt.getType(); // Use direct getter since we're now using the correct class
        
        if (type == null) {
            systemStatements.add(stmt);
            return;
        }
        
        switch (type.toUpperCase()) {
            case "CALL":
                callStatements.add(stmt);
                break;
            case "SQL":
                sqlStatements.add(stmt);
                break;
            case "COPY":
                copyStatements.add(stmt);
                break;
            default:
                systemStatements.add(stmt);
                break;
        }
    }
    
    // =====================================================================
    // HELPER METHODS
    // =====================================================================
    
    private TALStatement createStatement(String type, String content, int lineNumber) {
        TALStatement stmt = new TALStatement();
        stmt.setType(type);
        stmt.setContent(content.trim());
        stmt.setLineNumber(lineNumber);
        
        // Set context
        String procedureName = currentProcedure != null ? currentProcedure.getName() : "GLOBAL";
        stmt.setContext(procedureName);
        
        return stmt;
    }
    
    private void incrementStatementCount(String statementType) {
        statementCounts.merge(statementType, 1, Integer::sum);
    }
    
    private void extractCallReferences(String content) {
        if (content == null || content.trim().isEmpty()) return;
        
        Pattern[] callPatterns = {
            Pattern.compile("\\bCALL\\s+([A-Za-z_$][A-Za-z0-9_]*)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("\\b([A-Za-z_][A-Za-z0-9_]*)\\s*\\(", Pattern.CASE_INSENSITIVE),
            Pattern.compile("\\$([A-Za-z_][A-Za-z0-9_]*)", Pattern.CASE_INSENSITIVE)
        };
        
        for (Pattern pattern : callPatterns) {
            Matcher matcher = pattern.matcher(content);
            while (matcher.find()) {
                String calledProc = matcher.group(1);
                String upperProc = calledProc.toUpperCase();
                
                if (!isKeywordOrReserved(calledProc)) {
                    callReferences.merge(upperProc, 1, Integer::sum);
                }
            }
        }
    }
    
    private void extractConditionalReferences(String condition) {
        if (condition == null) return;
        
        // Extract variables used in conditional expressions
        Pattern condPattern = Pattern.compile("\\b([A-Za-z_][A-Za-z0-9_]*)\\s*(?:[<>=!]=?|<>)", Pattern.CASE_INSENSITIVE);
        Matcher matcher = condPattern.matcher(condition);
        
        while (matcher.find()) {
            String varName = matcher.group(1);
            if (!isKeywordOrReserved(varName)) {
                callReferences.merge("COND_" + varName.toUpperCase(), 1, Integer::sum);
            }
        }
    }
    
    private boolean isKeywordOrReserved(String identifier) {
        if (identifier == null || identifier.trim().isEmpty()) return true;
        return TAL_KEYWORDS.contains(identifier.toUpperCase());
    }
    
    // =====================================================================
    // GETTERS FOR RESULTS - Updated return types
    // =====================================================================
    
    public List<TALProcedure> getProcedures() { return procedures; }
    public List<TALDataItem> getExtractedDataItems() { return extractedDataItems; }
    public List<TALStatement> getSqlStatements() { return sqlStatements; }
    public List<TALStatement> getCopyStatements() { return copyStatements; }
    public List<TALStatement> getCallStatements() { return callStatements; }
    public List<TALStatement> getSystemStatements() { return systemStatements; }
    public Map<String, Integer> getStatementCounts() { return statementCounts; }
    public Map<String, Integer> getCallReferences() { return callReferences; }
    public List<String> getParseWarnings() { return parseWarnings; }
    
    // =====================================================================
    // MERGE WITH GRAMMAR RESULTS - Fixed compatibility
    // =====================================================================
    
    public void mergeWithGrammarResults(TALStatementVisitor grammarVisitor) {
        System.out.println("Merging fallback results with grammar results...");
        
        // Merge procedures (avoid duplicates) - now using compatible types
        Set<String> existingProcNames = grammarVisitor.getProcedures().stream()
            .map(p -> p.getName().toUpperCase())
            .collect(java.util.stream.Collectors.toSet());
            
        for (TALProcedure proc : procedures) {
            if (!existingProcNames.contains(proc.getName().toUpperCase())) {
                grammarVisitor.getProcedures().add(proc);
                System.out.println("Merged fallback procedure: " + proc.getName());
            }
        }
        
        // Merge data items - now compatible
        grammarVisitor.getExtractedDataItems().addAll(extractedDataItems);
        
        // Merge statements - now compatible
        grammarVisitor.getSqlStatements().addAll(sqlStatements);
        grammarVisitor.getCopyStatements().addAll(copyStatements);
        grammarVisitor.getCallStatements().addAll(callStatements);
        grammarVisitor.getSystemStatements().addAll(systemStatements);
        
        // Merge counts
        for (Map.Entry<String, Integer> entry : statementCounts.entrySet()) {
            grammarVisitor.getStatementCounts().merge(entry.getKey(), entry.getValue(), Integer::sum);
        }
        
        // Merge call references
        for (Map.Entry<String, Integer> entry : callReferences.entrySet()) {
            grammarVisitor.getCallReferences().merge(entry.getKey(), entry.getValue(), Integer::sum);
        }
        
        // Merge warnings
        grammarVisitor.getParseWarnings().addAll(parseWarnings);
        
        System.out.println("Merge completed successfully");
    }
    
    // =====================================================================
    // UTILITY METHODS FOR ANALYSIS
    // =====================================================================
    
    public void printAnalysisReport() {
        System.out.println("\n=== FALLBACK PARSER ANALYSIS REPORT ===");
        System.out.println("Procedures found: " + procedures.size());
        System.out.println("Data items found: " + extractedDataItems.size());
        System.out.println("Total statements: " + (sqlStatements.size() + copyStatements.size() + 
                                                   callStatements.size() + systemStatements.size()));
        System.out.println("Call references: " + callReferences.size());
        System.out.println("Parse warnings: " + parseWarnings.size());
        
        if (!procedures.isEmpty()) {
            System.out.println("\nProcedures:");
            for (TALProcedure proc : procedures) {
                System.out.println("  - " + proc.getName() + 
                    " (line " + proc.getLineNumber() + 
                    ", type: " + (proc.getReturnType() != null ? proc.getReturnType() : "untyped") + ")");
            }
        }
        
        if (!statementCounts.isEmpty()) {
            System.out.println("\nStatement counts:");
            statementCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .forEach(entry -> System.out.println("  " + entry.getKey() + ": " + entry.getValue()));
        }
        
        if (!callReferences.isEmpty()) {
            System.out.println("\nTop call references:");
            callReferences.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(10)
                .forEach(entry -> System.out.println("  " + entry.getKey() + ": " + entry.getValue() + " refs"));
        }
        
        if (!parseWarnings.isEmpty()) {
            System.out.println("\nFirst 5 parse warnings:");
            parseWarnings.stream()
                .limit(5)
                .forEach(warning -> System.out.println("  " + warning));
            if (parseWarnings.size() > 5) {
                System.out.println("  ... and " + (parseWarnings.size() - 5) + " more warnings");
            }
        }
    }
}

