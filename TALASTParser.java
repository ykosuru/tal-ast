import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
import java.io.*;
import java.util.*;
import java.util.regex.*;
import java.util.stream.Collectors;

/**
 * Enhanced TAL AST Parser - Clean integration with semantic analyzer and statement visitor
 * Primary orchestration class that coordinates all parsing activities
 * Version 2.0 - Cleaned up, deduplicated, and properly integrated
 */
public class TALASTParser {
    
    // Core parsing state
    private String[] sourceLines;
    private String programName;
    private CommonTokenStream tokenStream;
    private int totalLinesProcessed = 0;
    private long parseStartTime;
    private List<String> parseWarnings = new ArrayList<>();
    
    // Primary analysis engine - semantic analyzer does the heavy lifting
    private TALSemanticAnalyzer semanticAnalyzer;
    
    // Secondary analysis engine - statement visitor for backup/supplementary analysis
    private TALStatementVisitor statementVisitor;
    
    // Parse statistics
    private Map<String, Integer> parseMethodStats = new HashMap<>();
    private Map<String, Long> performanceMetrics = new HashMap<>();
    
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java TALASTParser <tal-file> [--debug-tokens|--grammar-only|--profile]");
            System.exit(1);
        }
        
        try {
            TALASTParser parser = new TALASTParser();
            String talFile = args[0];
            
            // Parse command line options
            boolean debugTokens = false;
            boolean grammarOnly = false;
            boolean profile = false;
            
            for (int i = 1; i < args.length; i++) {
                switch (args[i]) {
                    case "--debug-tokens":
                        debugTokens = true;
                        break;
                    case "--grammar-only":
                        grammarOnly = true;
                        break;
                    case "--profile":
                        profile = true;
                        break;
                }
            }
            
            if (debugTokens) {
                parser.debugTokenGeneration(talFile);
                return;
            }
            
            System.out.println("Starting Enhanced TAL Parser v2.0 for: " + talFile);
            long startTime = System.currentTimeMillis();
            
            TALSemanticAnalysisResult result = parser.parseWithEnhancedGrammar(talFile, grammarOnly, profile);
            
            long endTime = System.currentTimeMillis();
            parser.printResults(result, endTime - startTime);
            
            String astFilename = talFile + ".enhanced.ast";
            saveEnhancedAST(result, astFilename);
            System.out.println("Enhanced analysis complete! AST saved to: " + astFilename);
            
        } catch (Exception e) {
            System.err.println("Error in enhanced TAL analysis: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    public TALASTParser() {
        System.out.println("Enhanced TAL Parser v2.0 initialized with integrated grammar-semantic analysis");
        
        // Initialize primary semantic analyzer
        this.semanticAnalyzer = new TALSemanticAnalyzer(this);
        
        // Initialize secondary statement visitor for backup analysis
        this.statementVisitor = new TALStatementVisitor(this);
        
        // Initialize parse statistics
        this.parseMethodStats.put("GRAMMAR_SUCCESS", 0);
        this.parseMethodStats.put("GRAMMAR_PARTIAL", 0);
        this.parseMethodStats.put("GRAMMAR_FAILED", 0);
        this.parseMethodStats.put("RECOVERY_SUCCESS", 0);
    }
    
    // =====================================================================
    // MAIN PARSING ORCHESTRATION - Enhanced and Clean
    // =====================================================================
    
    public TALSemanticAnalysisResult parseWithEnhancedGrammar(String filename, boolean grammarOnly, boolean profile) throws Exception {
        parseStartTime = System.currentTimeMillis();
        sourceLines = readSourceLines(filename);
        totalLinesProcessed = sourceLines.length;
        
        System.out.println("Enhanced Grammar Processing - Lines: " + totalLinesProcessed);
        
        // Primary parsing attempt with enhanced grammar and semantic analysis
        boolean grammarSuccess = attemptPrimaryGrammarParsing(filename, sourceLines, profile);
        
        if (grammarSuccess) {
            parseMethodStats.put("GRAMMAR_SUCCESS", parseMethodStats.get("GRAMMAR_SUCCESS") + 1);
            System.out.println("✓ Enhanced grammar parsing succeeded completely");
        } else {
            parseMethodStats.put("GRAMMAR_FAILED", parseMethodStats.get("GRAMMAR_FAILED") + 1);
            System.out.println("✗ Enhanced grammar parsing encountered issues");
            
            if (!grammarOnly) {
                System.out.println("Attempting recovery and supplementary parsing...");
                boolean recoverySuccess = attemptRecoveryParsing(sourceLines, profile);
                if (recoverySuccess) {
                    parseMethodStats.put("RECOVERY_SUCCESS", parseMethodStats.get("RECOVERY_SUCCESS") + 1);
                }
            }
        }
        
        // Build comprehensive result from primary semantic analyzer
        TALSemanticAnalysisResult result = buildFinalResult(filename, profile);
        
        return result;
    }
    
    private boolean attemptPrimaryGrammarParsing(String filename, String[] sourceToParse, boolean profile) {
        try {
            long startTime = System.currentTimeMillis();
            
            // Prepare source content
            String sourceContent = String.join("\n", sourceToParse);
            CharStream input = CharStreams.fromString(sourceContent);
            
            // Create enhanced lexer and parser with error handling
            TALLexer lexer = new TALLexer(input);
            setupErrorHandling(lexer);
            
            this.tokenStream = new CommonTokenStream(lexer);
            TALParser parser = new TALParser(tokenStream);
            setupErrorHandling(parser);
            
            // Configure analyzers with token stream
            semanticAnalyzer.setTokenStream(tokenStream);
            statementVisitor.setTokenStream(tokenStream);
            
            if (profile) {
                System.out.println("Profiling: Setup completed in " + 
                    (System.currentTimeMillis() - startTime) + "ms");
            }
            
            // Generate parse tree
            long parseTreeStartTime = System.currentTimeMillis();
            ParseTree tree = parser.program();
            
            if (profile) {
                System.out.println("Profiling: Parse tree generation in " + 
                    (System.currentTimeMillis() - parseTreeStartTime) + "ms");
            }
            
            if (tree != null) {
                System.out.println("Parse tree generated successfully");
                
                // Primary semantic analysis
                long semanticStartTime = System.currentTimeMillis();
                semanticAnalyzer.visit(tree);
                
                // Secondary analysis with statement visitor for supplementary data
                System.out.println("Running supplementary statement visitor analysis...");
                statementVisitor.visit(tree);
                
                if (profile) {
                    System.out.println("Profiling: Primary semantic analysis in " + 
                        (System.currentTimeMillis() - semanticStartTime) + "ms");
                }
                
                // NOW evaluate success AFTER semantic analysis is complete
                TALSemanticAnalysisResult currentResult = semanticAnalyzer.getAnalysisResult();
                int proceduresFound = currentResult.getProcedures().size();
                int dataItemsFound = currentResult.getDataItems().size();
                int businessRulesExtracted = currentResult.getBusinessRules().size();
                int warningsCount = currentResult.getParseWarnings().size();
                
                System.out.println("Primary parsing results:");
                System.out.println("  Procedures: " + proceduresFound);
                System.out.println("  Data items: " + dataItemsFound);
                System.out.println("  Business rules: " + businessRulesExtracted);
                System.out.println("  Parse warnings: " + warningsCount);
                
                // not happy with this success check
                boolean success = 
                    // Found procedures and reasonable data items
                    (proceduresFound > 0 && dataItemsFound >= 5 && warningsCount <= 30) ||
                    // Found lots of data items even with warnings
                    (dataItemsFound >= 15 && warningsCount <= 50) ||
                    // Basic success - found something significant
                    (proceduresFound + dataItemsFound >= 8) ||
                    // Fallback for complex files
                    (dataItemsFound >= 3 && warningsCount <= 100);
                
                System.out.println("Success evaluation: warnings=" + warningsCount + 
                                   ", dataItems=" + dataItemsFound + 
                                   ", success=" + success);
                
                // Store performance metrics
                long totalTime = System.currentTimeMillis() - startTime;
                performanceMetrics.put("TOTAL_PARSE_TIME", totalTime);
                performanceMetrics.put("PROCEDURES_FOUND", (long) proceduresFound);
                performanceMetrics.put("DATA_ITEMS_FOUND", (long) dataItemsFound);
                performanceMetrics.put("BUSINESS_RULES_EXTRACTED", (long) businessRulesExtracted);
                
                return success;
            }
            else {
                System.out.println("Parse tree generation failed");
                return false;
            }
            
        } catch (Exception e) {
            System.err.println("Primary grammar parsing exception: " + e.getMessage());
            parseWarnings.add("Primary parsing failed: " + e.getMessage());
            return false;
        }
    }
    
    private boolean attemptRecoveryParsing(String[] sourceLines, boolean profile) {
        System.out.println("Attempting comprehensive recovery parsing with statement visitor...");
        
        try {
            long recoveryStartTime = System.currentTimeMillis();
            
            // Get current state from semantic analyzer
            TALSemanticAnalysisResult semanticResult = semanticAnalyzer.getAnalysisResult();
            
            // Track what we find during recovery
            int proceduresFound = 0;
            int dataItemsFound = 0;
            int literalsFound = 0;
            int structInstancesFound = 0;
            int callReferencesFound = 0;
            
            System.out.println("Starting line-by-line recovery analysis on " + sourceLines.length + " lines...");
            
            // Enhanced line-by-line analysis
            for (int i = 0; i < sourceLines.length; i++) {
                String line = sourceLines[i].trim();
                int lineNumber = i + 1;
                
                // Skip empty lines, comments, and preprocessor directives
                if (line.isEmpty() || line.startsWith("!") || line.startsWith("?")) {
                    continue;
                }
                
                try {
                    // 1. Check for procedure declarations
                    if (isProcedureDeclaration(line)) {
                        TALProcedure proc = parseRecoveredProcedure(line, lineNumber);
                        if (proc != null && !isDuplicate(proc, semanticResult.getProcedures())) {
                            proc.setReasoningInfo("Recovery parsing - procedure declaration");
                            proc.setContextScore(70.0); // Higher confidence for clear procedure patterns
                            semanticResult.getProcedures().add(proc);
                            proceduresFound++;
                            System.out.println("  Recovered procedure: " + proc.getName() + " at line " + lineNumber);
                        }
                    }
                    
                    // 2. Check for standard data declarations
                    if (isDataDeclaration(line)) {
                        TALDataItem dataItem = parseRecoveredDataItem(line, lineNumber);
                        if (dataItem != null && !isDuplicate(dataItem, semanticResult.getDataItems())) {
                            dataItem.setDefinition("Recovery: " + line);
                            semanticResult.getDataItems().add(dataItem);
                            dataItemsFound++;
                        }
                    }

                    List<TALDataItem> multiVarItems = parseMultiVariableLine(line, lineNumber);
                    for (TALDataItem item : multiVarItems) {
                        if (!isDuplicate(item, semanticResult.getDataItems())) {
                            semanticResult.getDataItems().add(item);
                            dataItemsFound++;
                        }
                    }
                    
                    // 3. Check for LITERAL declarations
                    if (isLiteralDeclaration(line)) {
                        TALDataItem literalItem = parseRecoveredLiteral(line, lineNumber);
                        if (literalItem != null && !isDuplicate(literalItem, semanticResult.getDataItems())) {
                            literalItem.setDefinition("Recovery: " + line);
                            semanticResult.getDataItems().add(literalItem);
                            literalsFound++;
                            System.out.println("  Recovered literal: " + literalItem.getName() + " at line " + lineNumber);
                        }
                    }
                    
                    // 4. Check for STRUCT instance declarations
                    if (isStructInstanceDeclaration(line)) {
                        TALDataItem structInstance = parseRecoveredStructInstance(line, lineNumber);
                        if (structInstance != null && !isDuplicate(structInstance, semanticResult.getDataItems())) {
                            structInstance.setDefinition("Recovery: " + line);
                            semanticResult.getDataItems().add(structInstance);
                            structInstancesFound++;
                            System.out.println("  Recovered struct instance: " + structInstance.getName() + " at line " + lineNumber);
                        }
                    }
                    
                    // 5. Check for struct member access patterns
                    if (isStructMemberAccess(line)) {
                        TALDataItem memberAccess = parseStructMemberAccess(line, lineNumber);
                        if (memberAccess != null && !isDuplicate(memberAccess, semanticResult.getDataItems())) {
                            memberAccess.setDefinition("Recovery: " + line);
                            semanticResult.getDataItems().add(memberAccess);
                            dataItemsFound++;
                        }
                    }
                    
                    // 6. Check for bit field operations
                    if (isBitFieldOperation(line)) {
                        TALDataItem bitFieldItem = parseBitFieldOperation(line, lineNumber);
                        if (bitFieldItem != null && !isDuplicate(bitFieldItem, semanticResult.getDataItems())) {
                            bitFieldItem.setDefinition("Recovery: " + line);
                            semanticResult.getDataItems().add(bitFieldItem);
                            dataItemsFound++;
                        }
                    }
                    
                    // 7. Check for pointer operations
                    if (isPointerOperation(line)) {
                        TALDataItem pointerItem = parsePointerOperation(line, lineNumber);
                        if (pointerItem != null && !isDuplicate(pointerItem, semanticResult.getDataItems())) {
                            pointerItem.setDefinition("Recovery: " + line);
                            semanticResult.getDataItems().add(pointerItem);
                            dataItemsFound++;
                        }
                    }
                    
                    // 8. Check for string move operations
                    if (isStringMoveOperation(line)) {
                        extractStringMovePatterns(line, semanticResult, lineNumber);
                    }
                    
                    // 9. Enhanced call reference extraction
                    int callsBefore = semanticResult.getCallReferences().size();
                    extractCallReferences(line, semanticResult);
                    int callsAfter = semanticResult.getCallReferences().size();
                    if (callsAfter > callsBefore) {
                        callReferencesFound += (callsAfter - callsBefore);
                    }
                    
                    // 10. Extract business rules and patterns
                    extractBusinessRulesFromLine(line, semanticResult, lineNumber);


                    
                } catch (Exception e) {
                    // Log individual line parsing errors but continue
                    if (profile) {
                        System.out.println("    Error parsing line " + lineNumber + ": " + e.getMessage());
                    }
                }
            }
            
            // Post-processing: Look for patterns across multiple lines
            performMultiLinePatternAnalysis(sourceLines, semanticResult);
            
            if (profile) {
                System.out.println("Profiling: Recovery parsing completed in " + 
                    (System.currentTimeMillis() - recoveryStartTime) + "ms");
            }
            
            // Report recovery statistics
            int totalRecovered = proceduresFound + dataItemsFound + literalsFound + structInstancesFound;
            System.out.println("\n=== RECOVERY PARSING RESULTS ===");
            System.out.println("Procedures recovered: " + proceduresFound);
            System.out.println("Data items recovered: " + dataItemsFound);
            System.out.println("Literals recovered: " + literalsFound);
            System.out.println("Struct instances recovered: " + structInstancesFound);
            System.out.println("Call references found: " + callReferencesFound);
            System.out.println("Total items recovered: " + totalRecovered);
            
            int finalTotal = semanticResult.getProcedures().size() + semanticResult.getDataItems().size();
            System.out.println("Final total structural elements: " + finalTotal);
            
            return totalRecovered > 0 || callReferencesFound > 0;
            
        } catch (Exception e) {
            System.err.println("Recovery parsing failed: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    // Helper method for multi-line pattern analysis
    private void performMultiLinePatternAnalysis(String[] sourceLines, TALSemanticAnalysisResult result) {
        System.out.println("Performing multi-line pattern analysis...");
        
        // Look for procedure bodies that span multiple lines
        for (int i = 0; i < sourceLines.length - 5; i++) {
            String currentLine = sourceLines[i].trim();
            
            // If we find a procedure declaration, try to find its END
            if (isProcedureDeclaration(currentLine)) {
                int endLine = findProcedureEnd(sourceLines, i);
                if (endLine > i) {
                    updateProcedureEndLine(result, i + 1, endLine + 1);
                }
            }
            
            // Look for STRUCT definitions that span multiple lines
            if (currentLine.toUpperCase().contains("STRUCT") && currentLine.contains("BEGIN")) {
                int structEnd = findStructEnd(sourceLines, i);
                if (structEnd > i) {
                    extractStructFields(sourceLines, i, structEnd, result);
                }
            }
        }
    }
    
    // Helper method to find procedure end
    private int findProcedureEnd(String[] sourceLines, int startLine) {
        int beginCount = 0;
        int endCount = 0;
        
        for (int i = startLine; i < sourceLines.length; i++) {
            String line = sourceLines[i].trim().toUpperCase();
            if (line.contains("BEGIN")) beginCount++;
            if (line.contains("END")) endCount++;
            
            if (beginCount > 0 && beginCount == endCount) {
                return i;
            }
        }
        return -1;
    }
    
    // Helper method to find struct end
    private int findStructEnd(String[] sourceLines, int startLine) {
        for (int i = startLine + 1; i < sourceLines.length; i++) {
            String line = sourceLines[i].trim().toUpperCase();
            if (line.startsWith("END")) {
                return i;
            }
        }
        return -1;
    }
    
    // Helper method to update procedure end line
    private void updateProcedureEndLine(TALSemanticAnalysisResult result, int startLine, int endLine) {
        for (TALProcedure proc : result.getProcedures()) {
            if (proc.getLineNumber() == startLine) {
                proc.setEndLineNumber(endLine);
                break;
            }
        }
    }
    
    // Helper method to extract struct fields
    private void extractStructFields(String[] sourceLines, int startLine, int endLine, TALSemanticAnalysisResult result) {
        for (int i = startLine + 1; i < endLine; i++) {
            String line = sourceLines[i].trim();
            if (!line.isEmpty() && !line.startsWith("!") && isDataDeclaration(line)) {
                TALDataItem field = parseRecoveredDataItem(line, i + 1);
                if (field != null) {
                    field.setSection("STRUCT_FIELD");
                    field.setDefinition("Struct field: " + line);
                    if (!isDuplicate(field, result.getDataItems())) {
                        result.getDataItems().add(field);
                    }
                }
            }
        }
    }
    
    // Helper method to extract business rules from individual lines
    private void extractBusinessRulesFromLine(String line, TALSemanticAnalysisResult result, int lineNumber) {
        String upperLine = line.toUpperCase();
        
        // Look for conditional logic patterns
        if (upperLine.contains("IF") || upperLine.contains("CASE") || upperLine.contains("WHILE")) {
            BusinessRule rule = new BusinessRule();
            rule.setRuleType("CONDITIONAL_LOGIC");
            rule.setDescription("Conditional logic at line " + lineNumber);
            rule.setLineNumber(lineNumber);
            rule.setSourceCode(line);
            result.getBusinessRules().add(rule);
        }
        
        // Look for validation patterns
        if (upperLine.contains("VALIDATE") || upperLine.contains("CHECK") || upperLine.contains("VERIFY")) {
            BusinessRule rule = new BusinessRule();
            rule.setRuleType("VALIDATION");
            rule.setDescription("Validation logic at line " + lineNumber);
            rule.setLineNumber(lineNumber);
            rule.setSourceCode(line);
            result.getBusinessRules().add(rule);
        }
        
        // Look for error handling patterns
        if (upperLine.contains("ERROR") || upperLine.contains("ABEND") || upperLine.contains("TRAP")) {
            BusinessRule rule = new BusinessRule();
            rule.setRuleType("ERROR_HANDLING");
            rule.setDescription("Error handling at line " + lineNumber);
            rule.setLineNumber(lineNumber);
            rule.setSourceCode(line);
            result.getBusinessRules().add(rule);
        }
    }
    
    // Helper method to extract string move patterns
    private void extractStringMovePatterns(String line, TALSemanticAnalysisResult result, int lineNumber) {
        if (line.contains("':='") || line.contains("MOVL") || line.contains("MOVR")) {
            // Add to statement counts
            result.getStatementCounts().merge("STRING_MOVE", 1, Integer::sum);
            
            // Extract variable names involved in string moves
            Pattern varPattern = Pattern.compile("([A-Za-z_][A-Za-z0-9_^]*)\\s*(?:\\[[^\\]]*\\])?\\s*':='");
            Matcher matcher = varPattern.matcher(line);
            while (matcher.find()) {
                String varName = matcher.group(1);
                if (!isKeywordOrReserved(varName)) {
                    result.getCallReferences().merge("VAR_" + varName.toUpperCase(), 1, Integer::sum);
                }
            }
        }
    }
    
    private TALDataItem parseRecoveredLiteral(String line, int lineNumber) {
        Pattern pattern = Pattern.compile("\\bLITERAL\\s+([A-Za-z_][A-Za-z0-9_^]*)\\s*=\\s*(.+)", Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(line);
        
        if (matcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType("LITERAL");
            dataItem.setName(matcher.group(1));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(line.trim());
            dataItem.setSection("LITERALS");
            return dataItem;
        }
        
        return null;
    }
    
    private TALDataItem parseRecoveredStructInstance(String line, int lineNumber) {
        Pattern pattern = Pattern.compile("\\bSTRUCT\\s+([.*]?[A-Za-z_][A-Za-z0-9_^]*)\\s*\\(([A-Za-z_][A-Za-z0-9_^]*)\\)", Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(line);
        
        if (matcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType("STRUCT");
            dataItem.setName(matcher.group(1));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(line.trim());
            dataItem.setSection("STRUCT_INSTANCES");
            return dataItem;
        }
        
        return null;
    }
    // =====================================================================
    // ERROR HANDLING SETUP
    // =====================================================================
    
    private void setupErrorHandling(TALLexer lexer) {
        lexer.removeErrorListeners();
        lexer.addErrorListener(new BaseErrorListener() {
            @Override
            public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
                                  int line, int charPositionInLine, String msg, RecognitionException e) {
                String error = "Lexer error at line " + line + ":" + charPositionInLine + " - " + msg;
                parseWarnings.add(error);
                
                // Log only first few errors to avoid spam
                if (parseWarnings.size() <= 5) {
                    System.out.println("LEXER: " + error);
                }
            }
        });
    }
    
    private void setupErrorHandling(TALParser parser) {
        parser.removeErrorListeners();
        parser.addErrorListener(new BaseErrorListener() {
            // In setupErrorHandling for parser:
            @Override
            public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
                                int line, int charPositionInLine, String msg, RecognitionException e) {
                String error = "Parser error at line " + line + ":" + charPositionInLine + " - " + msg;
                
                // Classify error types for better reporting
                if (msg.contains("missing ';'") || msg.contains("extraneous input")) {
                    // These are now expected with flexible parsing - reduce severity
                    parseWarnings.add("Warning: " + error);
                } else {
                    parseWarnings.add(error);
                }
                
                // Log fewer errors to avoid spam
                if (parseWarnings.stream().filter(w -> w.contains("Parser error")).count() <= 15) {
                    System.out.println("PARSER: " + error);
                }
            }
        });
        
        // Enhanced error recovery strategy
        parser.setErrorHandler(new DefaultErrorStrategy() {
            @Override
            public void recover(Parser recognizer, RecognitionException e) {
                super.recover(recognizer, e);
            }
        });
    }
    
    // =====================================================================
    // RESULT BUILDING AND INTEGRATION
    // =====================================================================
    
    private TALSemanticAnalysisResult buildFinalResult(String filename, boolean profile) {
        long resultBuildStart = System.currentTimeMillis();
        
        // Get the primary result from semantic analyzer
        TALSemanticAnalysisResult result = semanticAnalyzer.getAnalysisResult();
        
        System.out.println("DEBUG: Semantic analyzer found:");
        System.out.println("  - Procedures: " + result.getProcedures().size());
        System.out.println("  - Data items: " + result.getDataItems().size());
        System.out.println("  - Call references: " + result.getCallReferences().size());
        System.out.println("  - Business rules: " + result.getBusinessRules().size());
        
        // Set metadata
        result.setProgramName(determineProgramName(filename));
        result.setAnalysisMethod("ENHANCED_GRAMMAR_SEMANTIC");
        result.setParseTimestamp(new Date());
        result.setSourceLinesProcessed(totalLinesProcessed);
        
        // Merge warnings from all sources
        List<String> allWarnings = new ArrayList<>(parseWarnings);
        allWarnings.addAll(result.getParseWarnings());
        if (statementVisitor != null) {
            allWarnings.addAll(statementVisitor.getParseWarnings());
        }
        result.setParseWarnings(allWarnings.stream().distinct().collect(Collectors.toList()));
        
        // Add performance metrics
        result.getPerformanceMetrics().putAll(performanceMetrics);
        result.getParseMethodStats().putAll(parseMethodStats);
        
        // Merge supplementary data from statement visitor
        mergeSupplementaryData(result);
        
        if (profile) {
            long resultBuildTime = System.currentTimeMillis() - resultBuildStart;
            System.out.println("Profiling: Final result building completed in " + resultBuildTime + "ms");
        }
        
        return result;
    }

    private void mergeSupplementaryData(TALSemanticAnalysisResult result) {
        // Merge any additional findings from statement visitor that weren't captured
        // by the semantic analyzer (this provides redundancy and completeness)
        
        System.out.println("DEBUG: Merging supplementary data from statement visitor");
        System.out.println("DEBUG: Statement visitor found " + statementVisitor.getExtractedDataItems().size() + " data items");
        System.out.println("DEBUG: Statement visitor found " + statementVisitor.getProcedures().size() + " procedures");
        
        // Merge additional call references
        Map<String, Integer> visitorCalls = statementVisitor.getCallReferences();
        for (Map.Entry<String, Integer> entry : visitorCalls.entrySet()) {
            result.getCallReferences().merge(entry.getKey(), entry.getValue(), Integer::sum);
        }
        
        // Merge additional statement counts
        Map<String, Integer> visitorStatements = statementVisitor.getStatementCounts();
        for (Map.Entry<String, Integer> entry : visitorStatements.entrySet()) {
            result.getStatementCounts().merge(entry.getKey(), entry.getValue(), Integer::sum);
        }
        
        // Add any procedures found by visitor but missed by semantic analyzer
        for (TALProcedure visitorProc : statementVisitor.getProcedures()) {
            if (!isDuplicate(visitorProc, result.getProcedures())) {
                visitorProc.setReasoningInfo("Found by supplementary statement visitor");
                visitorProc.setContextScore(60.0); // Lower confidence for visitor-only findings
                result.getProcedures().add(visitorProc);
                System.out.println("DEBUG: Added procedure from visitor: " + visitorProc.getName());
            }
        }
        
        // Add any data items found by visitor but missed by semantic analyzer
        for (TALDataItem visitorData : statementVisitor.getExtractedDataItems()) {
            if (!isDuplicate(visitorData, result.getDataItems())) {
                result.getDataItems().add(visitorData);
                System.out.println("DEBUG: Added data item from visitor: " + visitorData.getName() + " (" + visitorData.getDataType() + ")");
            }
        }
        
        System.out.println("DEBUG: After merging, total data items: " + result.getDataItems().size());
    }
    
    // =====================================================================
    // RECOVERY PARSING HELPERS
    // =====================================================================
    
    private boolean isProcedureDeclaration(String line) {
        return line.matches(".*\\b(?:PROC|proc|SUBPROC|subproc)\\b.*") &&
               !line.toUpperCase().trim().startsWith("FORWARD") &&
               (line.contains("(") || line.matches(".*\\b(?:PROC|SUBPROC)\\s+[A-Za-z_][A-Za-z0-9_]*\\b.*"));
    }
    

    private boolean isLiteralDeclaration(String line) {
        return line.matches(".*\\bLITERAL\\s+[A-Za-z_][A-Za-z0-9_^]*\\s*=.*");
    }
    
    private boolean isStructInstanceDeclaration(String line) {
        return line.matches(".*\\bSTRUCT\\s+[.*]?[A-Za-z_][A-Za-z0-9_^]*\\s*\\([A-Za-z_][A-Za-z0-9_^]*\\).*");
    }
    
    // Enhanced data declaration pattern to include pointers
    private boolean isDataDeclaration(String line) {
        return line.matches(".*\\b(?:INT(?:\\([^)]*\\))?|STRING(?:\\([^)]*\\))?|REAL(?:\\([^)]*\\))?|FIXED(?:\\([^)]*\\))?|BYTE|CHAR|TIMESTAMP|STRUCT|UNSIGNED(?:\\(\\d+\\))?|EXTADDR|SGADDR|BOOLEAN)\\b.*") &&
            !isProcedureDeclaration(line) &&
            !line.toUpperCase().trim().startsWith("FORWARD") &&
            !line.trim().startsWith("?") &&
            !line.trim().startsWith("!") ||
            // Add pointer declaration patterns
            line.matches(".*\\b(?:STRING|INT|CHAR)\\s+\\.[A-Za-z_][A-Za-z0-9_^]*.*") ||
            // Add struct instance patterns  
            line.matches(".*\\bSTRUCT\\s+[.*]?[A-Za-z_][A-Za-z0-9_^]*\\s*\\([A-Za-z_][A-Za-z0-9_^]*\\).*") ||
            // Add literal patterns
            line.matches(".*\\bLITERAL\\s+[A-Za-z_][A-Za-z0-9_^]*\\s*=.*");
    }
    
    private boolean isStructMemberAccess(String line) {
        return line.matches(".*\\bSTRUCT\\s*\\.\\s*[A-Za-z_][A-Za-z0-9_]*.*") ||
               line.matches(".*\\b[A-Za-z_][A-Za-z0-9_]*\\s*\\.\\s*[A-Za-z_][A-Za-z0-9_]*.*");
    }

    private boolean isPreprocessorDirective(String line) {
        return line.trim().startsWith("?");
    }
    
    private TALProcedure parseRecoveredProcedure(String line, int lineNumber) {
        String procName = extractProcedureName(line);
        if (procName == null) return null;
        
        TALProcedure procedure = new TALProcedure();
        procedure.setName(procName);
        procedure.setLineNumber(lineNumber);
        procedure.setEndLineNumber(lineNumber + 5); // Estimate
        procedure.setReasoningInfo("Recovery parsing - line-by-line analysis");
        procedure.setContextScore(50.0); // Medium confidence for recovery
        
        // Extract additional details
        procedure.setReturnType(extractReturnType(line));
        procedure.setAttributes(extractAttributes(line));
        procedure.setParameters(extractParameters(line));
        
        return procedure;
    }
    
    private TALDataItem parseRecoveredDataItem(String line, int lineNumber) {
        String trimmedLine = line.trim();
        
        // run through each pattern
        // 0. Multi-variable pointer declarations with array
        Pattern multiPointerArrayPattern = Pattern.compile(
            "\\b(STRING|INT|CHAR)\\s+\\.(\\w+(?:\\^\\w+)*)\\[([^\\]]+)\\]\\s*,\\s*((?:\\.\\w+(?:\\^\\w+)*(?:\\s*,\\s*)*)+)",
            Pattern.CASE_INSENSITIVE
        );
        Matcher multiPointerMatcher = multiPointerArrayPattern.matcher(trimmedLine);
        if (multiPointerMatcher.find()) {
            // Return the array variable first, then handle pointers separately
            TALDataItem arrayItem = new TALDataItem();
            arrayItem.setDataType(multiPointerMatcher.group(1) + " ARRAY");
            arrayItem.setName("." + multiPointerMatcher.group(2));
            arrayItem.setLineNumber(lineNumber);
            arrayItem.setDefinition(trimmedLine);
            arrayItem.setSection("MULTI_POINTER_ARRAYS");
            
            // Also parse the additional pointer variables from group(4)
            // This would require a separate method to handle multiple items per line
            return arrayItem;
        }
        // 1. Complex array with multiple pointer variables like "string .cw[0:$len(...)], .begin^ptr, .end^ptr, .ptr;"
        Pattern complexArrayPattern = Pattern.compile(
            "\\b(STRING|INT|CHAR|BYTE)\\s+\\.(\\w+(?:\\^\\w+)*)\\[([^\\]]+)\\](?:\\s*,\\s*\\.\\w+(?:\\^\\w+)*)*",
            Pattern.CASE_INSENSITIVE
        );
        Matcher complexArrayMatcher = complexArrayPattern.matcher(trimmedLine);
        if (complexArrayMatcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType(complexArrayMatcher.group(1) + " ARRAY");
            dataItem.setName("." + complexArrayMatcher.group(2));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(trimmedLine);
            dataItem.setSection("COMPLEX_ARRAYS");
            return dataItem;
        }
        
        // 2. STRUCT instance declarations like "data_packet_def .variable_name" or "struct .name (type)"
        Pattern structInstancePattern = Pattern.compile(
            "\\b(?:STRUCT\\s+)?([A-Za-z_][A-Za-z0-9_^]*)\\s+\\.(\\w+(?:\\^\\w+)*)",
            Pattern.CASE_INSENSITIVE
        );
        Matcher structInstanceMatcher = structInstancePattern.matcher(trimmedLine);
        if (structInstanceMatcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType(structInstanceMatcher.group(1));
            dataItem.setName("." + structInstanceMatcher.group(2));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(trimmedLine);
            dataItem.setSection("STRUCT_INSTANCES");
            return dataItem;
        }
        
        // 3. STRUCT with template like "struct .EXT data^stack (codeword^stack^def)"
        Pattern structWithTemplatePattern = Pattern.compile(
            "\\bSTRUCT\\s+\\.(\\w+)\\s+(\\w+(?:\\^\\w+)*)\\s*\\((\\w+(?:\\^\\w+)*)\\)",
            Pattern.CASE_INSENSITIVE
        );
        Matcher structTemplateMatcher = structWithTemplatePattern.matcher(trimmedLine);
        if (structTemplateMatcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType("STRUCT");
            dataItem.setName("." + structTemplateMatcher.group(1) + " " + structTemplateMatcher.group(2));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(trimmedLine);
            dataItem.setSection("STRUCT_TEMPLATES");
            return dataItem;
        }
        
        // 4. Simple pointer declarations like "STRING .ptr1, .ptr2;"
        Pattern pointerPattern = Pattern.compile(
            "\\b(INT(?:\\(\\d+\\))?|STRING(?:\\(\\d+\\))?|REAL(?:\\(\\d+\\))?|FIXED(?:\\(\\d+(?:,\\d+)?\\))?|BYTE|CHAR|TIMESTAMP|UNSIGNED(?:\\(\\d+\\))?|BOOLEAN)\\s+\\.(\\w+(?:\\^\\w+)*)",
            Pattern.CASE_INSENSITIVE
        );
        Matcher pointerMatcher = pointerPattern.matcher(trimmedLine);
        if (pointerMatcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType(pointerMatcher.group(1) + " POINTER");
            dataItem.setName("." + pointerMatcher.group(2));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(trimmedLine);
            dataItem.setSection("POINTERS");
            return dataItem;
        }
        
        // 5. Array declarations like "STRING array[0:255];" or "INT .array[10]"
        Pattern arrayPattern = Pattern.compile(
            "\\b(INT(?:\\(\\d+\\))?|STRING(?:\\(\\d+\\))?|REAL(?:\\(\\d+\\))?|FIXED(?:\\(\\d+(?:,\\d+)?\\))?|BYTE|CHAR|TIMESTAMP|UNSIGNED(?:\\(\\d+\\))?|BOOLEAN)\\s+([.*]?)(\\w+(?:\\^\\w+)*)\\[([^\\]]+)\\]",
            Pattern.CASE_INSENSITIVE
        );
        Matcher arrayMatcher = arrayPattern.matcher(trimmedLine);
        if (arrayMatcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType(arrayMatcher.group(1) + " ARRAY");
            String prefix = arrayMatcher.group(2);
            String name = arrayMatcher.group(3);
            dataItem.setName(prefix + name);
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(trimmedLine);
            dataItem.setSection("ARRAYS");
            return dataItem;
        }
        
        // 6. STRUCT field declarations (inside struct definitions)
        if (trimmedLine.contains("STRUCT") || isWithinStructDefinition(trimmedLine)) {
            Pattern structFieldPattern = Pattern.compile(
                "\\b(INT(?:\\(\\d+\\))?|STRING(?:\\(\\d+\\))?|REAL(?:\\(\\d+\\))?|FIXED(?:\\(\\d+(?:,\\d+)?\\))?|BYTE|CHAR|TIMESTAMP|UNSIGNED(?:\\(\\d+\\))?|BOOLEAN)\\s+(\\w+(?:\\^\\w+)*)",
                Pattern.CASE_INSENSITIVE
            );
            Matcher structFieldMatcher = structFieldPattern.matcher(trimmedLine);
            if (structFieldMatcher.find()) {
                TALDataItem dataItem = new TALDataItem();
                dataItem.setDataType(structFieldMatcher.group(1));
                dataItem.setName(structFieldMatcher.group(2));
                dataItem.setLineNumber(lineNumber);
                dataItem.setDefinition(trimmedLine);
                dataItem.setSection("STRUCT_FIELDS");
                return dataItem;
            }
        }
        
        // 7. LITERAL declarations like "literal APPLEN = $len(...)"
        Pattern literalPattern = Pattern.compile(
            "\\bLITERAL\\s+(\\w+(?:\\^\\w+)*)\\s*=\\s*(.+)",
            Pattern.CASE_INSENSITIVE
        );
        Matcher literalMatcher = literalPattern.matcher(trimmedLine);
        if (literalMatcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType("LITERAL");
            dataItem.setName(literalMatcher.group(1));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(trimmedLine);
            dataItem.setSection("LITERALS");
            return dataItem;
        }
        
        // 8. Function parameters from procedure definitions
        if (trimmedLine.contains("(") && (trimmedLine.contains("PROC") || trimmedLine.contains("proc"))) {
            Pattern paramPattern = Pattern.compile(
                "(\\w+(?:\\^\\w+)*)\\s*,|\\b(\\w+(?:\\^\\w+)*)\\s*\\)",
                Pattern.CASE_INSENSITIVE
            );
            Matcher paramMatcher = paramPattern.matcher(trimmedLine);
            if (paramMatcher.find()) {
                TALDataItem dataItem = new TALDataItem();
                dataItem.setDataType("PARAMETER");
                String paramName = paramMatcher.group(1) != null ? paramMatcher.group(1) : paramMatcher.group(2);
                dataItem.setName(paramName);
                dataItem.setLineNumber(lineNumber);
                dataItem.setDefinition(trimmedLine);
                dataItem.setSection("PARAMETERS");
                return dataItem;
            }
        }
        
        // 9. Regular typed variable declarations
        Pattern typedVariablePattern = Pattern.compile(
            "\\b(INT(?:\\([^)]*\\))?|STRING(?:\\([^)]*\\))?|REAL(?:\\([^)]*\\))?|FIXED(?:\\([^)]*\\))?|BYTE|CHAR|TIMESTAMP|STRUCT|UNSIGNED(?:\\(\\d+\\))?|EXTADDR|SGADDR|BOOLEAN)\\s+(?:[.*]\\s*)?(\\w+(?:\\^\\w+)*)",
            Pattern.CASE_INSENSITIVE
        );
        Matcher typedVariableMatcher = typedVariablePattern.matcher(trimmedLine);
        if (typedVariableMatcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType(typedVariableMatcher.group(1));
            dataItem.setName(typedVariableMatcher.group(2));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(trimmedLine);
            dataItem.setSection("VARIABLES");
            return dataItem;
        }
        
        // 10. Generic identifier patterns (fallback for untyped variables)
        Pattern genericPattern = Pattern.compile(
            "\\b(\\w+(?:\\^\\w+)*)\\s*(?::=|=)",
            Pattern.CASE_INSENSITIVE
        );
        Matcher genericMatcher = genericPattern.matcher(trimmedLine);
        if (genericMatcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType("UNKNOWN");
            dataItem.setName(genericMatcher.group(1));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(trimmedLine);
            dataItem.setSection("UNTYPED");
            return dataItem;
        }
        
        return null;
    }
    
    // Helper method to detect if we're inside a STRUCT definition
    private boolean isWithinStructDefinition(String line) {
        return line.trim().matches(".*\\b(INT|STRING|REAL|FIXED|BYTE|CHAR|TIMESTAMP|BOOLEAN)\\s+\\w+.*") &&
               !line.toUpperCase().contains("PROC") &&
               !line.trim().startsWith("!");
    }

    private boolean isBitFieldOperation(String line) {
        return line.contains(".<") || line.matches(".*<\\d+:\\d+>.*") || 
               line.contains("BITFIELD") || line.contains("BITDEPOSIT");
    }
    
    private boolean isPointerOperation(String line) {
        return line.contains("@") || line.matches(".*\\.[A-Za-z_][A-Za-z0-9_^]*\\s*:=.*") ||
               line.contains("ADDRESS") || line.matches(".*\\*[A-Za-z_][A-Za-z0-9_^]*.*");
    }
    
    private boolean isStringMoveOperation(String line) {
        return line.contains("':='") || line.contains("MOVL") || line.contains("MOVR") ||
               line.contains("FOR") && line.contains("BYTES");
    }
    
    private TALDataItem parseBitFieldOperation(String line, int lineNumber) {
        Pattern pattern = Pattern.compile("([A-Za-z_][A-Za-z0-9_^]*).*<(\\d+):(\\d+)>");
        Matcher matcher = pattern.matcher(line);
        
        if (matcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType("BIT_FIELD");
            dataItem.setName(matcher.group(1));
            dataItem.setLineNumber(lineNumber);
            dataItem.setSection("BIT_OPERATIONS");
            return dataItem;
        }
        return null;
    }
    
    private TALDataItem parsePointerOperation(String line, int lineNumber) {
        Pattern pattern = Pattern.compile("([A-Za-z_][A-Za-z0-9_^]*)\\s*:=\\s*@([A-Za-z_][A-Za-z0-9_^]*)");
        Matcher matcher = pattern.matcher(line);
        
        if (matcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType("POINTER");
            dataItem.setName(matcher.group(1));
            dataItem.setLineNumber(lineNumber);
            dataItem.setSection("POINTER_OPERATIONS");
            return dataItem;
        }
        return null;
    }
    
    private TALDataItem parseStructMemberAccess(String line, int lineNumber) {
        Pattern pattern = Pattern.compile("([A-Za-z_][A-Za-z0-9_^]*)\\.([A-Za-z_][A-Za-z0-9_^]*)");
        Matcher matcher = pattern.matcher(line);
        
        if (matcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType("STRUCT_MEMBER");
            dataItem.setName(matcher.group(1) + "." + matcher.group(2));
            dataItem.setLineNumber(lineNumber);
            dataItem.setSection("STRUCT_ACCESS");
            return dataItem;
        }
        return null;
    }


    private List<TALDataItem> parseMultiVariableLine(String line, int lineNumber) {
        List<TALDataItem> items = new ArrayList<>();
        
        // Pattern for declarations like "int var1, var2, var3;"
        Pattern multiVarPattern = Pattern.compile(
            "\\b(INT(?:\\([^)]*\\))?|STRING(?:\\([^)]*\\))?|REAL(?:\\([^)]*\\))?|FIXED(?:\\([^)]*\\))?|BYTE|CHAR|TIMESTAMP|UNSIGNED(?:\\(\\d+\\))?|BOOLEAN)\\s+(.+);?", 
            Pattern.CASE_INSENSITIVE
        );
        
        Matcher matcher = multiVarPattern.matcher(line);
        if (matcher.find()) {
            String dataType = matcher.group(1);
            String variableList = matcher.group(2);
            
            // Split on commas and parse each variable
            String[] variables = variableList.split(",");
            for (String var : variables) {
                var = var.trim();
                
                // Extract variable name (handle pointers, arrays, etc.)
                Pattern varPattern = Pattern.compile("([.*]?)([A-Za-z_][A-Za-z0-9_^]*)(\\[[^\\]]*\\])?");
                Matcher varMatcher = varPattern.matcher(var);
                
                if (varMatcher.find()) {
                    TALDataItem dataItem = new TALDataItem();
                    String prefix = varMatcher.group(1);
                    String name = varMatcher.group(2);
                    String arrayPart = varMatcher.group(3);
                    
                    if (arrayPart != null) {
                        dataItem.setDataType(dataType + " ARRAY");
                    } else if (!prefix.isEmpty()) {
                        dataItem.setDataType(dataType + " POINTER");
                    } else {
                        dataItem.setDataType(dataType);
                    }
                    
                    dataItem.setName(prefix + name);
                    dataItem.setLineNumber(lineNumber);
                    dataItem.setDefinition("Multi-var: " + line);
                    dataItem.setSection("MULTI_DECLARATIONS");
                    items.add(dataItem);
                }
            }
        }
        
        return items;
    }
    
    // =====================================================================
    // UTILITY METHODS
    // =====================================================================
    

    private String[] readSourceLines(String filename) throws IOException {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        }
        return lines.toArray(new String[0]);
    }
    
    private String determineProgramName(String filename) {
        String baseName = new File(filename).getName();
        if (baseName.contains(".")) {
            baseName = baseName.substring(0, baseName.lastIndexOf('.'));
        }
        return baseName;
    }
    
    private String extractProcedureName(String line) {
        Pattern[] patterns = {
            Pattern.compile("(?:[A-Z_][A-Z0-9_]*(?:\\([^)]*\\))?\\s+)?(?:PROC|proc)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(", Pattern.CASE_INSENSITIVE),
            Pattern.compile("(?:SUBPROC|subproc)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(", Pattern.CASE_INSENSITIVE),
            Pattern.compile("\\b(?:PROC|proc|SUBPROC|subproc)\\s+([A-Za-z_][A-Za-z0-9_]*)", Pattern.CASE_INSENSITIVE)
        };
        
        for (Pattern pattern : patterns) {
            Matcher matcher = pattern.matcher(line);
            if (matcher.find()) {
                return matcher.group(1);
            }
        }
        return null;
    }
    
    private String extractReturnType(String line) {
        Pattern pattern = Pattern.compile("^\\s*([A-Z_][A-Z0-9_]*(?:\\([^)]*\\))?)\\s+(?:PROC|proc)", Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(line);
        if (matcher.find()) {
            String type = matcher.group(1);
            if (!type.matches("(?i)(FORWARD|EXTERNAL|STRUCT)")) {
                return type;
            }
        }
        return null;
    }
    
    private List<String> extractAttributes(String line) {
        List<String> attributes = new ArrayList<>();
        String[] attrs = {"MAIN", "INTERRUPT", "RESIDENT", "CALLABLE", "PRIV", "VARIABLE", "EXTENSIBLE", "SHARED", "REENTRANT"};
        
        for (String attr : attrs) {
            if (line.toUpperCase().contains(attr)) {
                attributes.add(attr);
            }
        }
        return attributes;
    }
    
    private List<String> extractParameters(String line) {
        Pattern paramPattern = Pattern.compile("\\(([^)]*)\\)");
        Matcher matcher = paramPattern.matcher(line);
        
        if (matcher.find()) {
            String paramString = matcher.group(1).trim();
            if (!paramString.isEmpty()) {
                return Arrays.asList(paramString.split(","));
            }
        }
        return new ArrayList<>();
    }
    
    private void extractCallReferences(String content, TALSemanticAnalysisResult result) {
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
                if (!isKeywordOrReserved(calledProc)) {
                    result.getCallReferences().merge(calledProc.toUpperCase(), 1, Integer::sum);
                }
            }
        }
    }
    
    private boolean isKeywordOrReserved(String identifier) {
        if (identifier == null || identifier.trim().isEmpty()) return true;
        
        Set<String> keywords = Set.of(
            "PROC", "SUBPROC", "INT", "STRING", "REAL", "FIXED", "BYTE", "CHAR", "BOOLEAN",
            "IF", "THEN", "ELSE", "WHILE", "FOR", "RETURN", "BEGIN", "END", 
            "FORWARD", "STRUCT", "MAIN", "INTERRUPT", "RESIDENT", "CALLABLE", "SHARED",
            "TIMESTAMP", "EXTADDR", "SGADDR", "UNSIGNED", "EXTENSIBLE"
        );
        
        return keywords.contains(identifier.toUpperCase());
    }
    
    private <T> boolean isDuplicate(T item, List<T> existingList) {
        if (item instanceof TALProcedure) {
            TALProcedure proc = (TALProcedure) item;
            return existingList.stream().anyMatch(p -> 
                p instanceof TALProcedure && 
                ((TALProcedure) p).getName() != null && 
                ((TALProcedure) p).getName().equals(proc.getName()) &&
                Math.abs(((TALProcedure) p).getLineNumber() - proc.getLineNumber()) < 3);
        } else if (item instanceof TALDataItem) {
            TALDataItem data = (TALDataItem) item;
            return existingList.stream().anyMatch(d -> 
                d instanceof TALDataItem && 
                ((TALDataItem) d).getName().equals(data.getName()) &&
                Math.abs(((TALDataItem) d).getLineNumber() - data.getLineNumber()) < 2);
        }
        return false;
    }
    
    // =====================================================================
    // DEBUG AND PROFILING METHODS
    // =====================================================================
    
    public void debugTokenGeneration(String filename) throws Exception {
        System.out.println("\n=== ENHANCED DEBUG TOKEN GENERATION FOR " + filename + " ===");
        
        String[] lines = readSourceLines(filename);
        String sourceContent = String.join("\n", lines);
        
        System.out.println("Source content (" + sourceContent.length() + " chars)");
        System.out.println("First 300 chars: '" + sourceContent.substring(0, Math.min(300, sourceContent.length())) + "'");
        
        CharStream input = CharStreams.fromString(sourceContent);
        TALLexer lexer = new TALLexer(input);
        
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        tokens.fill();
        
        System.out.println("Total tokens: " + tokens.size());
        
        // Token analysis
        Map<String, Integer> tokenTypeStats = new HashMap<>();
        List<Token> importantTokens = new ArrayList<>();
        
        for (int i = 0; i < tokens.size(); i++) {
            Token token = tokens.get(i);
            String tokenName = getTokenName(token.getType());
            
            tokenTypeStats.merge(tokenName, 1, Integer::sum);
            
            // Collect important tokens
            String text = token.getText().toUpperCase();
            if (text.contains("PROC") || text.equals("MAIN") || text.equals("STRUCT") ||
                text.equals("IF") || text.equals("CALL") || text.startsWith("$") ||
                text.contains("GLOBALS") || text.startsWith("?")) 
            {
                importantTokens.add(token);
            }
        }
        
        System.out.println("Important tokens found: " + importantTokens.size());
        for (Token token : importantTokens.stream().limit(10).collect(Collectors.toList())) {
            String tokenName = getTokenName(token.getType());
            System.out.printf("  Line %d:%d Type=%s Text='%s'%n",
                token.getLine(), token.getCharPositionInLine(), 
                tokenName, escapeString(token.getText()));
        }
        
        System.out.println("\nTop token types:");
        tokenTypeStats.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .limit(8)
            .forEach(entry -> System.out.println("  " + entry.getKey() + ": " + entry.getValue()));
    }
    
    private String getTokenName(int tokenType) {
        try {
            String name = TALLexer.VOCABULARY.getSymbolicName(tokenType);
            if (name == null) {
                name = TALLexer.VOCABULARY.getLiteralName(tokenType);
            }
            return name != null ? name : "UNKNOWN_" + tokenType;
        } catch (Exception e) {
            return "TOKEN_" + tokenType;
        }
    }
    
    private static String escapeString(String str) {
        if (str == null) return "";
        return str.replace("\\", "\\\\")
                 .replace("\"", "\\\"")
                 .replace("\n", "\\n")
                 .replace("\r", "\\r")
                 .replace("\t", "\\t");
    }
    
    // =====================================================================
    // RESULTS DISPLAY
    // =====================================================================
    
    private void printResults(TALSemanticAnalysisResult result, long totalTime) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("ENHANCED TAL ANALYSIS RESULTS");
        System.out.println("=".repeat(60));
        
        System.out.println("Program: " + result.getProgramName());
        System.out.println("Analysis Method: " + result.getAnalysisMethod());
        System.out.println("Total Processing Time: " + totalTime + "ms");
        System.out.println("Source Lines Processed: " + result.getSourceLinesProcessed());
        
        if (result.getSourceLinesProcessed() > 0 && totalTime > 0) {
            double linesPerSecond = (double) result.getSourceLinesProcessed() / totalTime * 1000;
            System.out.println("Processing Rate: " + Math.round(linesPerSecond) + " lines/sec");
        }
        
        System.out.println("\n" + "-".repeat(40));
        System.out.println("STRUCTURAL ANALYSIS");
        System.out.println("-".repeat(40));
        System.out.println("Procedures found: " + result.getProcedures().size());
        System.out.println("Data items: " + result.getDataItems().size());
        System.out.println("Call references: " + result.getCallReferences().size());
        
        System.out.println("\n" + "-".repeat(40));
        System.out.println("SEMANTIC ANALYSIS");
        System.out.println("-".repeat(40));
        System.out.println("Business rules extracted: " + result.getBusinessRules().size());
        System.out.println("Bit field operations: " + result.getBitFieldOperations().size());
        System.out.println("Pointer operations: " + result.getPointerOperations().size());
        System.out.println("SQL operations: " + result.getSqlOperations().size());
        
        System.out.println("\n" + "-".repeat(40));
        System.out.println("STATEMENT ANALYSIS");
        System.out.println("-".repeat(40));
        int totalStatements = result.getStatementCounts().values().stream().mapToInt(Integer::intValue).sum();
        System.out.println("Total statements analyzed: " + totalStatements);
        System.out.println("SQL statements: " + result.getSqlStatements().size());
        System.out.println("Call statements: " + result.getCallStatements().size());
        System.out.println("System statements: " + result.getSystemStatements().size());
        
        System.out.println("\n" + "-".repeat(40));
        System.out.println("QUALITY METRICS");
        System.out.println("-".repeat(40));
        System.out.println("Parse warnings: " + result.getParseWarnings().size());
        
        if (result.getParseMethodStats() != null && !result.getParseMethodStats().isEmpty()) {
            System.out.println("Parse method statistics:");
            result.getParseMethodStats().forEach((method, count) -> 
                System.out.println("  " + method + ": " + count));
        }
        
        if (!result.getProcedures().isEmpty()) {
            System.out.println("\nTop procedures by confidence:");
            result.getProcedures().stream()
                .sorted((p1, p2) -> Double.compare(p2.getContextScore(), p1.getContextScore()))
                .limit(5)
                .forEach(proc -> System.out.println("  - " + proc.getName() + 
                    " (line " + proc.getLineNumber() + 
                    ", confidence: " + String.format("%.1f", proc.getContextScore()) + ")"));
        }
        
        if (!result.getParseWarnings().isEmpty()) {
            System.out.println("\nParse warnings (first 3):");
            result.getParseWarnings().stream()
                .limit(3)
                .forEach(warning -> System.out.println("  " + warning));
        }
        
        System.out.println("\n" + "=".repeat(60));
    }
    
    // =====================================================================
    // AST SAVING
    // =====================================================================
    
    private static void saveEnhancedAST(TALSemanticAnalysisResult result, String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("(TAL-ENHANCED-ANALYSIS \"" + escapeString(result.getProgramName()) + "\"");
            writer.println("  (METADATA");
            writer.println("    (ANALYSIS-VERSION \"2.0-CLEAN\")");
            writer.println("    (ANALYSIS-METHOD \"" + result.getAnalysisMethod() + "\")");
            writer.println("    (TIMESTAMP \"" + result.getParseTimestamp() + "\")");
            writer.println("    (SOURCE-LINES " + result.getSourceLinesProcessed() + ")");
            writer.println("    (PROCEDURES-COUNT " + result.getProcedures().size() + ")");
            writer.println("    (BUSINESS-RULES-COUNT " + result.getBusinessRules().size() + ")");
            writer.println("  )");
            
            // Write procedures
            if (!result.getProcedures().isEmpty()) {
                writer.println("  (PROCEDURES");
                for (TALProcedure proc : result.getProcedures()) {
                    writer.println("    (PROCEDURE \"" + escapeString(proc.getName()) + "\"");
                    writer.println("      (CONFIDENCE-SCORE " + proc.getContextScore() + ")");
                    writer.println("      (START-LINE " + proc.getLineNumber() + ")");
                    if (proc.getReturnType() != null) {
                        writer.println("      (RETURN-TYPE \"" + escapeString(proc.getReturnType()) + "\")");
                    }
                    writer.println("    )");
                }
                writer.println("  )");
            }

            if (!result.getDataItems().isEmpty()) {
                writer.println("  (DATA-ITEMS");
                for (TALDataItem item : result.getDataItems().stream().limit(20).collect(Collectors.toList())) {
                    writer.println("    (DATA-ITEM \"" + escapeString(item.getName()) + "\"");
                    writer.println("      (TYPE \"" + escapeString(item.getDataType()) + "\")");
                    writer.println("      (LINE " + item.getLineNumber() + ")");
                    writer.println("    )");
                }
                writer.println("  )");
            }

            if (!result.getCallReferences().isEmpty()) {
                writer.println("  (CALL-REFERENCES");
                result.getCallReferences().entrySet().stream()
                    .limit(10)
                    .forEach(entry -> writer.println("    (\"" + escapeString(entry.getKey()) + "\" " + entry.getValue() + ")"));
                writer.println("  )");
            }
            
            
            // Write business rules summary
            if (!result.getBusinessRules().isEmpty()) {
                writer.println("  (BUSINESS-RULES-SUMMARY");
                Map<String, Long> ruleTypes = result.getBusinessRules().stream()
                    .collect(Collectors.groupingBy(BusinessRule::getRuleType, Collectors.counting()));
                ruleTypes.forEach((type, count) -> 
                    writer.println("    (\"" + escapeString(type) + "\" " + count + ")"));
                writer.println("  )");
            }
            
            writer.println(")");
            
        } catch (IOException e) {
            System.err.println("Failed to write AST file: " + e.getMessage());
        }
    }
}
