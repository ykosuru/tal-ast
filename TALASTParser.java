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
                
                // Gather results and evaluate success
                TALSemanticAnalysisResult result = semanticAnalyzer.getAnalysisResult();
                
                int proceduresFound = result.getProcedures().size();
                int dataItemsFound = result.getDataItems().size();
                int businessRulesExtracted = result.getBusinessRules().size();
                int warningsCount = result.getParseWarnings().size();
                
                System.out.println("Primary parsing results:");
                System.out.println("  Procedures: " + proceduresFound);
                System.out.println("  Data items: " + dataItemsFound);
                System.out.println("  Business rules: " + businessRulesExtracted);
                System.out.println("  Parse warnings: " + warningsCount);
                
                // Store performance metrics
                long totalTime = System.currentTimeMillis() - startTime;
                performanceMetrics.put("TOTAL_PARSE_TIME", totalTime);
                performanceMetrics.put("PROCEDURES_FOUND", (long) proceduresFound);
                performanceMetrics.put("BUSINESS_RULES_EXTRACTED", (long) businessRulesExtracted);
                
                // Success criteria: found substantial content with manageable warnings
                boolean success = (proceduresFound > 0 || dataItemsFound > 3) && warningsCount < 100;
                
                return success;
            } else {
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
        System.out.println("Attempting recovery parsing with statement visitor...");
        
        try {
            long recoveryStartTime = System.currentTimeMillis();
            
            // Get current state from semantic analyzer
            TALSemanticAnalysisResult semanticResult = semanticAnalyzer.getAnalysisResult();
            
            // Use statement visitor for supplementary analysis
            for (int i = 0; i < sourceLines.length; i++) {
                String line = sourceLines[i].trim();
                int lineNumber = i + 1;
                
                if (line.isEmpty() || line.startsWith("!")) continue;
                
                try {
                    // Recover critical constructs that may have been missed
                    if (isProcedureDeclaration(line)) {
                        TALProcedure proc = parseRecoveredProcedure(line, lineNumber);
                        if (proc != null && !isDuplicate(proc, semanticResult.getProcedures())) {
                            semanticResult.getProcedures().add(proc);
                        }
                    }
                    
                    if (isDataDeclaration(line)) {
                        TALDataItem dataItem = parseRecoveredDataItem(line, lineNumber);
                        if (dataItem != null && !isDuplicate(dataItem, semanticResult.getDataItems())) {
                            semanticResult.getDataItems().add(dataItem);
                        }
                    }
                    
                    // Extract call references for completeness
                    extractCallReferences(line, semanticResult);
                    
                } catch (Exception e) {
                    // Continue on individual line errors during recovery
                }
            }
            
            if (profile) {
                System.out.println("Profiling: Recovery parsing completed in " + 
                    (System.currentTimeMillis() - recoveryStartTime) + "ms");
            }
            
            int recoveredItems = semanticResult.getProcedures().size() + semanticResult.getDataItems().size();
            System.out.println("Recovery parsing added items, total structural elements: " + recoveredItems);
            
            return recoveredItems > 0;
            
        } catch (Exception e) {
            System.err.println("Recovery parsing failed: " + e.getMessage());
            return false;
        }
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
            @Override
            public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
                                  int line, int charPositionInLine, String msg, RecognitionException e) {
                String error = "Parser error at line " + line + ":" + charPositionInLine + " - " + msg;
                parseWarnings.add(error);
                
                // Log only first 10 parser errors to avoid spam
                if (parseWarnings.stream().filter(w -> w.contains("Parser error")).count() <= 10) {
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
        
        // Set metadata
        result.setProgramName(determineProgramName(filename));
        result.setAnalysisMethod("ENHANCED_GRAMMAR_SEMANTIC");
        result.setParseTimestamp(new Date());
        result.setSourceLinesProcessed(totalLinesProcessed);
        
        // Merge warnings from all sources
        List<String> allWarnings = new ArrayList<>(parseWarnings);
        allWarnings.addAll(result.getParseWarnings());
        allWarnings.addAll(statementVisitor.getParseWarnings());
        result.setParseWarnings(allWarnings.stream().distinct().collect(Collectors.toList()));
        
        // Add performance metrics
        result.getPerformanceMetrics().putAll(performanceMetrics);
        result.getParseMethodStats().putAll(parseMethodStats);
        
        // Merge any supplementary data from statement visitor if needed
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
    
    private boolean isDataDeclaration(String line) {
        return line.matches(".*\\b(?:INT(?:\\([^)]*\\))?|STRING(?:\\([^)]*\\))?|REAL(?:\\([^)]*\\))?|FIXED(?:\\([^)]*\\))?|BYTE|CHAR|TIMESTAMP|STRUCT|UNSIGNED(?:\\(\\d+\\))?|EXTADDR|SGADDR|BOOLEAN)\\b.*") &&
               !isProcedureDeclaration(line) &&
               !line.toUpperCase().trim().startsWith("FORWARD");
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
        Pattern pattern = Pattern.compile(
            "\\b(INT(?:\\(\\d+\\))?|STRING(?:\\(\\d+\\))?|REAL(?:\\(\\d+\\))?|FIXED(?:\\(\\d+(?:,\\d+)?\\))?|BYTE|CHAR|TIMESTAMP|STRUCT|UNSIGNED(?:\\(\\d+\\))?|EXTADDR|SGADDR|BOOLEAN)\\s+(?:[.*]\\s*)?([A-Za-z_][A-Za-z0-9_]*)", 
            Pattern.CASE_INSENSITIVE
        );
        Matcher matcher = pattern.matcher(line);
        
        if (matcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType(matcher.group(1));
            dataItem.setName(matcher.group(2));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(line);
            dataItem.setSection("GLOBAL");
            return dataItem;
        }
        
        return null;
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
                text.equals("IF") || text.equals("CALL") || text.startsWith("$")) {
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
