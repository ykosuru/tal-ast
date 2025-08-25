import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.tree.*;
import java.util.*;
import java.util.regex.*;

/**
 * Dedicated TAL Statement Visitor for grammar-based parsing only
 * Works in conjunction with TALSemanticAnalyzer for comprehensive analysis
 * Focuses on statement extraction and basic structural analysis
 */
public class TALStatementVisitor extends TALBaseVisitor<Void> {
    
    // Reference to main parser for accessing shared state
    private TALASTParser mainParser;
    
    // Analysis results - these feed into the main semantic analyzer
    private List<TALProcedure> procedures = new ArrayList<>();
    private List<TALDataItem> extractedDataItems = new ArrayList<>();
    private List<TALStatement> sqlStatements = new ArrayList<>();
    private List<TALStatement> copyStatements = new ArrayList<>();
    private List<TALStatement> callStatements = new ArrayList<>();
    private List<TALStatement> systemStatements = new ArrayList<>();
    private Map<String, Integer> statementCounts = new HashMap<>();
    private Map<String, Integer> callReferences = new HashMap<>();
    protected List<String> parseWarnings = new ArrayList<>();
    
    // Current parsing context
    private TALProcedure currentProcedure;
    private CommonTokenStream tokenStream;
    private Set<String> processedProcedures = new HashSet<>();
    
    public TALStatementVisitor(TALASTParser mainParser) {
        this.mainParser = mainParser;
    }
    
    // Setters for context
    public void setCurrentProcedure(TALProcedure procedure) {
        this.currentProcedure = procedure;
    }
    
    public void setTokenStream(CommonTokenStream tokenStream) {
        this.tokenStream = tokenStream;
    }
    
    // Getters for ALL results
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
    // ROBUST VISITOR PATTERN - Handle all grammar contexts gracefully
    // =====================================================================
    
    @Override
    public Void visitChildren(RuleNode node) {
        if (node instanceof ParserRuleContext) {
            ParserRuleContext ctx = (ParserRuleContext) node;
            try {
                handleGenericContext(ctx);
            } catch (Exception e) {
                String error = "Error processing context at line " + getLineNumber(ctx) + ": " + e.getMessage();
                parseWarnings.add(error);
                // Continue processing despite errors
            }
        }
        return super.visitChildren(node);
    }
    
    private void handleGenericContext(ParserRuleContext ctx) {
        String ruleName = getRuleName(ctx);
        String content = getFullText(ctx).trim();
        
        if (content.isEmpty()) return;
        
        // Handle different rule types based on name patterns
        if (ruleName != null) {
            if (ruleName.toLowerCase().contains("procedure")) {
                handleProcedureContext(ctx, content);
            } else if (ruleName.toLowerCase().contains("data")) {
                handleDataContext(ctx, content);
            } else if (ruleName.toLowerCase().contains("statement") || 
                       ruleName.toLowerCase().contains("stmt")) {
                handleStatementContext(ctx, content, ruleName);
            }
        }
    }
    
    private String getRuleName(ParserRuleContext ctx) {
        try {
            return ctx.getClass().getSimpleName().replace("Context", "");
        } catch (Exception e) {
            return "UnknownRule";
        }
    }
    
    protected int getLineNumber(ParserRuleContext ctx) {
        if (ctx.getStart() != null) {
            return ctx.getStart().getLine();
        }
        return 0;
    }
    
    // =====================================================================
    // PROCEDURE AND DATA VISITOR METHODS
    // =====================================================================
    
    @Override
    public Void visitProcedureDeclaration(TALParser.ProcedureDeclarationContext ctx) {
        if (ctx.procHeader() != null) {
            handleProcedureHeader(ctx.procHeader(), ctx);
        }
        return super.visitProcedureDeclaration(ctx);
    }
    
    @Override
    public Void visitDataDeclaration(TALParser.DataDeclarationContext ctx) {
        try {
            handleDataDeclaration(ctx);
        } catch (Exception e) {
            parseWarnings.add("Error handling data declaration at line " + getLineNumber(ctx) + ": " + e.getMessage());
        }
        return super.visitDataDeclaration(ctx);
    }
    
    // =====================================================================
    // COMPREHENSIVE STATEMENT VISITOR METHODS  
    // =====================================================================
    
    @Override
    public Void visitAssignStmt(TALParser.AssignStmtContext ctx) {
        handleStatementSafely(ctx, "ASSIGNMENT", () -> {
            if (ctx.assignmentStatement() != null) {
                handleAssignmentStatement(ctx.assignmentStatement());
            }
        });
        return super.visitAssignStmt(ctx);
    }
    
    @Override
    public Void visitCallStmt(TALParser.CallStmtContext ctx) {
        handleStatementSafely(ctx, "CALL", () -> {
            if (ctx.callStatement() != null) {
                handleCallStatement(ctx.callStatement());
            }
        });
        return super.visitCallStmt(ctx);
    }
    
    @Override
    public Void visitIfStmt(TALParser.IfStmtContext ctx) {
        handleStatementSafely(ctx, "IF", () -> {
            if (ctx.ifStatement() != null) {
                handleIfStatement(ctx.ifStatement());
            }
        });
        return super.visitIfStmt(ctx);
    }
    
    @Override
    public Void visitWhileStmt(TALParser.WhileStmtContext ctx) {
        handleStatementSafely(ctx, "WHILE", () -> {
            if (ctx.whileStatement() != null) {
                handleWhileStatement(ctx.whileStatement());
            }
        });
        return super.visitWhileStmt(ctx);
    }
    
    @Override
    public Void visitForStmt(TALParser.ForStmtContext ctx) {
        handleStatementSafely(ctx, "FOR", () -> {
            if (ctx.forStatement() != null) {
                handleForStatement(ctx.forStatement());
            }
        });
        return super.visitForStmt(ctx);
    }
    
    @Override
    public Void visitCaseStmt(TALParser.CaseStmtContext ctx) {
        handleStatementSafely(ctx, "CASE", () -> {
            if (ctx.caseStatement() != null) {
                handleCaseStatement(ctx.caseStatement());
            }
        });
        return super.visitCaseStmt(ctx);
    }
    
    @Override
    public Void visitReturnStmt(TALParser.ReturnStmtContext ctx) {
        handleStatementSafely(ctx, "RETURN", () -> {
            if (ctx.returnStatement() != null) {
                handleReturnStatement(ctx.returnStatement());
            }
        });
        return super.visitReturnStmt(ctx);
    }
    
    @Override
    public Void visitLocalDeclStmt(TALParser.LocalDeclStmtContext ctx) {
        handleStatementSafely(ctx, "LOCAL_DECLARATION", () -> {
            if (ctx.localDeclarationStatement() != null) {
                handleLocalDeclarationStatement(ctx.localDeclarationStatement());
            }
        });
        return super.visitLocalDeclStmt(ctx);
    }
    
    @Override
    public Void visitInlineCommentItem(TALParser.InlineCommentItemContext ctx) {
        // Handle inline comments within structure definitions
        try {
            String commentText = getFullText(ctx);
            if (commentText != null && !commentText.trim().isEmpty()) {
                // Store inline comment for later processing if needed
                System.out.println("Grammar: Found inline comment: " + commentText.trim());
            }
        } catch (Exception e) {
            parseWarnings.add("Error processing inline comment at line " + getLineNumber(ctx) + ": " + e.getMessage());
        }
        return super.visitInlineCommentItem(ctx);
    }
    
    // =====================================================================
    // SAFE STATEMENT HANDLING
    // =====================================================================
    
    private void handleStatementSafely(ParserRuleContext ctx, String statementType, Runnable handler) {
        try {
            handler.run();
            incrementStatementCount(statementType);
        } catch (Exception e) {
            String error = "Error handling " + statementType + " statement at line " + getLineNumber(ctx) + ": " + e.getMessage();
            parseWarnings.add(error);
        }
    }
    
    // =====================================================================
    // PROCEDURE HANDLER METHODS
    // =====================================================================
    
    private void handleProcedureHeader(TALParser.ProcHeaderContext headerCtx, TALParser.ProcedureDeclarationContext procCtx) {
        try {
            TALProcedure procedure = new TALProcedure();
            
            // Extract line information
            Token startToken = procCtx.getStart();
            Token stopToken = procCtx.getStop();
            if (startToken != null) {
                procedure.setLineNumber(startToken.getLine());
            }
            if (stopToken != null && startToken != null) {
                int endLine = Math.max(stopToken.getLine(), startToken.getLine());
                procedure.setEndLineNumber(endLine);
            }
            
            // Handle typed vs untyped headers
            if (headerCtx.typedProcHeader() != null) {
                handleTypedProcHeader(headerCtx.typedProcHeader(), procedure);
            } else if (headerCtx.untypedProcHeader() != null) {
                handleUntypedProcHeader(headerCtx.untypedProcHeader(), procedure);
            }
            
            // Only add if we successfully extracted a name and haven't seen it before
            if (procedure.getName() != null && !procedure.getName().isEmpty() && 
                !processedProcedures.contains(procedure.getName().toUpperCase())) {
                
                procedure.setReasoningInfo("Found via enhanced grammar parsing");
                procedure.setContextScore(75.0); // Higher score for grammar-based parsing
                
                procedures.add(procedure);
                processedProcedures.add(procedure.getName().toUpperCase());
                this.currentProcedure = procedure;
                
                System.out.println("Grammar: Added procedure: " + procedure.getName() + 
                    " (line " + procedure.getLineNumber() + 
                    ", type: " + (procedure.getReturnType() != null ? procedure.getReturnType() : "untyped") + ")");
            }
            
        } catch (Exception e) {
            parseWarnings.add("Error handling procedure header: " + e.getMessage());
        }
    }
    
    private void handleTypedProcHeader(TALParser.TypedProcHeaderContext ctx, TALProcedure procedure) {
        // Extract return type
        if (ctx.typeSpecification() != null) {
            String returnType = extractTypeFromContext(ctx.typeSpecification());
            procedure.setReturnType(returnType);
        }
        
        // Extract procedure name
        if (ctx.procName() != null && ctx.procName().IDENTIFIER() != null) {
            String procName = ctx.procName().IDENTIFIER().getText();
            procedure.setName(procName);
        }
        
        // Extract parameters - with null safety for grammar compatibility
        if (ctx.formalParamList() != null) {
            List<String> params = extractParametersFromContext(ctx.formalParamList());
            procedure.setParameters(params);
        }
        
        // Extract attributes
        if (ctx.procAttributeList() != null) {
            List<String> attributes = extractAttributesFromContext(ctx.procAttributeList());
            procedure.setAttributes(attributes);
        }
    }
    
    private void handleUntypedProcHeader(TALParser.UntypedProcHeaderContext ctx, TALProcedure procedure) {
        // Extract procedure name
        if (ctx.procName() != null && ctx.procName().IDENTIFIER() != null) {
            String procName = ctx.procName().IDENTIFIER().getText();
            procedure.setName(procName);
        }
        
        // Extract parameters - with null safety for grammar compatibility
        if (ctx.formalParamList() != null) {
            List<String> params = extractParametersFromContext(ctx.formalParamList());
            procedure.setParameters(params);
        }
        
        // Extract attributes
        if (ctx.procAttributeList() != null) {
            List<String> attributes = extractAttributesFromContext(ctx.procAttributeList());
            procedure.setAttributes(attributes);
        }
        
        // Determine if it's PROC or SUBPROC - store in attributes
        String text = getFullText(ctx);
        if (text.toLowerCase().contains("subproc")) {
            List<String> attrs = procedure.getAttributes();
            if (attrs == null) attrs = new ArrayList<>();
            attrs.add("TYPE:SUBPROC");
            procedure.setAttributes(attrs);
        } else {
            List<String> attrs = procedure.getAttributes();
            if (attrs == null) attrs = new ArrayList<>();
            attrs.add("TYPE:PROC");
            procedure.setAttributes(attrs);
        }
    }
    
    private String extractTypeFromContext(TALParser.TypeSpecificationContext ctx) {
        if (ctx.dataType() != null) {
            return getFullText(ctx.dataType()).trim();
        } else if (ctx.forwardTypeName() != null) {
            return getFullText(ctx.forwardTypeName()).trim();
        }
        return null;
    }
    
    private List<String> extractParametersFromContext(TALParser.FormalParamListContext ctx) {
        List<String> parameters = new ArrayList<>();
        
        if (ctx != null) {
            try {
                // Try to access formal parameters using reflection for grammar compatibility
                java.lang.reflect.Method[] methods = ctx.getClass().getMethods();
                for (java.lang.reflect.Method method : methods) {
                    if (method.getName().equals("formalParam") || 
                        method.getName().equals("formalParameter") ||
                        method.getName().equals("parameter") ||
                        method.getName().equals("param")) {
                        try {
                            Object result = method.invoke(ctx);
                            if (result instanceof List) {
                                @SuppressWarnings("unchecked")
                                List<ParserRuleContext> paramList = (List<ParserRuleContext>) result;
                                for (ParserRuleContext paramCtx : paramList) {
                                    if (paramCtx != null) {
                                        String paramText = getFullText(paramCtx);
                                        if (paramText != null && !paramText.trim().isEmpty()) {
                                            parameters.add(paramText.trim());
                                        }
                                    }
                                }
                            }
                            break;
                        } catch (Exception e) {
                            // Continue trying other methods
                        }
                    }
                }
                
                // If no formal parameters found via reflection, try to extract from text
                if (parameters.isEmpty()) {
                    String fullText = getFullText(ctx);
                    if (fullText != null && fullText.contains("(") && fullText.contains(")")) {
                        // Extract parameter list from parentheses
                        Pattern paramPattern = Pattern.compile("\\(([^)]*)\\)");
                        Matcher matcher = paramPattern.matcher(fullText);
                        if (matcher.find()) {
                            String paramStr = matcher.group(1).trim();
                            if (!paramStr.isEmpty()) {
                                String[] params = paramStr.split(",");
                                for (String param : params) {
                                    parameters.add(param.trim());
                                }
                            }
                        }
                    }
                }
                
            } catch (Exception e) {
                parseWarnings.add("Error extracting parameters: " + e.getMessage());
            }
        }
        
        return parameters;
    }
    
    private List<String> extractAttributesFromContext(TALParser.ProcAttributeListContext ctx) {
        List<String> attributes = new ArrayList<>();
        
        if (ctx != null) {
            try {
                // Try to access attributes using reflection for grammar compatibility
                java.lang.reflect.Method[] methods = ctx.getClass().getMethods();
                for (java.lang.reflect.Method method : methods) {
                    if (method.getName().equals("procAttribute") || 
                        method.getName().equals("attribute") ||
                        method.getName().equals("attr")) {
                        try {
                            Object result = method.invoke(ctx);
                            if (result instanceof List) {
                                @SuppressWarnings("unchecked")
                                List<ParserRuleContext> attrList = (List<ParserRuleContext>) result;
                                for (ParserRuleContext attrCtx : attrList) {
                                    if (attrCtx != null) {
                                        String attrText = getFullText(attrCtx);
                                        if (attrText != null && !attrText.trim().isEmpty()) {
                                            attributes.add(attrText.trim());
                                        }
                                    }
                                }
                            }
                            break;
                        } catch (Exception e) {
                            // Continue trying other methods
                        }
                    }
                }
                
                // If no attributes found via reflection, try to extract from text
                if (attributes.isEmpty()) {
                    String fullText = getFullText(ctx);
                    if (fullText != null) {
                        // Common TAL attributes
                        String[] commonAttrs = {"MAIN", "INTERRUPT", "RESIDENT", "CALLABLE", "PRIV", "VARIABLE", "EXTENSIBLE", "SHARED", "REENTRANT"};
                        for (String attr : commonAttrs) {
                            if (fullText.toUpperCase().contains(attr)) {
                                attributes.add(attr);
                            }
                        }
                    }
                }
                
            } catch (Exception e) {
                parseWarnings.add("Error extracting attributes: " + e.getMessage());
            }
        }
        
        return attributes;
    }
    
    // =====================================================================
    // DATA AND STATEMENT HANDLERS
    // =====================================================================
    
    private void handleDataDeclaration(TALParser.DataDeclarationContext ctx) {
        String content = getFullText(ctx).trim();
        int lineNumber = getLineNumber(ctx);
        
        TALDataItem dataItem = parseDataFromContent(content, lineNumber);
        if (dataItem != null) {
            extractedDataItems.add(dataItem);
            System.out.println("Grammar: Added data item: " + dataItem.getName() + 
                " (type: " + dataItem.getDataType() + ", line: " + lineNumber + ")");
        }
    }
    
    private void handleProcedureContext(ParserRuleContext ctx, String content) {
        try {
            TALProcedure procedure = new TALProcedure();
            
            // Set line numbers
            Token startToken = ctx.getStart();
            Token stopToken = ctx.getStop();
            if (startToken != null) {
                procedure.setLineNumber(startToken.getLine());
            }
            if (stopToken != null && startToken != null) {
                int endLine = Math.max(stopToken.getLine(), startToken.getLine());
                procedure.setEndLineNumber(endLine);
            }
            
            // Try to extract procedure name from content
            String procName = extractProcedureName(content);
            if (procName != null && !procName.isEmpty() && 
                !processedProcedures.contains(procName.toUpperCase())) {
                
                procedure.setName(procName);
                procedure.setReasoningInfo("Found via generic grammar context");
                procedure.setContextScore(50.0);
                
                this.currentProcedure = procedure;
                procedures.add(procedure);
                processedProcedures.add(procName.toUpperCase());
                System.out.println("Grammar: Added procedure: " + procName + " (generic context)");
            }
        } catch (Exception e) {
            parseWarnings.add("Error handling procedure context: " + e.getMessage());
        }
    }
    
    private void handleDataContext(ParserRuleContext ctx, String content) {
        try {
            int lineNumber = getLineNumber(ctx);
            TALDataItem dataItem = parseDataFromContent(content, lineNumber);
            if (dataItem != null) {
                extractedDataItems.add(dataItem);
                System.out.println("Grammar: Added data item via context: " + dataItem.getName());
            }
        } catch (Exception e) {
            parseWarnings.add("Error handling data context: " + e.getMessage());
        }
    }
    
    private void handleStatementContext(ParserRuleContext ctx, String content, String ruleName) {
        try {
            String statementType = determineStatementType(content, ruleName);
            TALStatement stmt = createStatement(statementType, content, ctx.getStart());
            
            // Categorize statement
            if (statementType.equals("CALL")) {
                callStatements.add(stmt);
                extractCallReferences(content);
            } else if (statementType.equals("SQL") || content.toUpperCase().contains("SELECT") || 
                      content.toUpperCase().contains("INSERT") || content.toUpperCase().contains("UPDATE")) {
                sqlStatements.add(stmt);
            } else if (statementType.equals("COPY")) {
                copyStatements.add(stmt);
            } else {
                systemStatements.add(stmt);
            }
            
            incrementStatementCount(statementType);
            
        } catch (Exception e) {
            parseWarnings.add("Error handling statement context: " + e.getMessage());
        }
    }
    
    // =====================================================================
    // SPECIFIC STATEMENT HANDLER METHODS - Simplified for Statement Visitor
    // =====================================================================
    
    private void handleAssignmentStatement(TALParser.AssignmentStatementContext ctx) {
        String content = getFullText(ctx);
        if (content == null || content.trim().isEmpty()) return;
        
        TALStatement stmt = createStatement("ASSIGNMENT", content, ctx.getStart());
        systemStatements.add(stmt);
        
        // Extract variable references
        extractVariableReferences(content);
        
        System.out.println("Statement: Found assignment at line " + getLineNumber(ctx));
    }
    
    private void handleCallStatement(TALParser.CallStatementContext ctx) {
        String content = getFullText(ctx);
        if (content == null || content.trim().isEmpty()) return;
        
        TALStatement stmt = createStatement("CALL", content, ctx.getStart());
        callStatements.add(stmt);
        
        // Extract procedure call references
        extractCallReferences(content);
        
        System.out.println("Statement: Found call statement at line " + getLineNumber(ctx));
    }
    
    private void handleIfStatement(TALParser.IfStatementContext ctx) {
        String content = getFullText(ctx);
        if (content == null || content.trim().isEmpty()) return;
        
        TALStatement stmt = createStatement("IF", content, ctx.getStart());
        systemStatements.add(stmt);
        
        // Extract condition expressions
        extractConditionalReferences(content);
        
        System.out.println("Statement: Found if statement at line " + getLineNumber(ctx));
    }
    
    private void handleWhileStatement(TALParser.WhileStatementContext ctx) {
        String content = getFullText(ctx);
        if (content == null || content.trim().isEmpty()) return;
        
        TALStatement stmt = createStatement("WHILE", content, ctx.getStart());
        systemStatements.add(stmt);
        
        System.out.println("Statement: Found while statement at line " + getLineNumber(ctx));
    }
    
    private void handleForStatement(TALParser.ForStatementContext ctx) {
        String content = getFullText(ctx);
        if (content == null || content.trim().isEmpty()) return;
        
        TALStatement stmt = createStatement("FOR", content, ctx.getStart());
        systemStatements.add(stmt);
        
        System.out.println("Statement: Found for statement at line " + getLineNumber(ctx));
    }
    
    private void handleCaseStatement(TALParser.CaseStatementContext ctx) {
        String content = getFullText(ctx);
        if (content == null || content.trim().isEmpty()) return;
        
        TALStatement stmt = createStatement("CASE", content, ctx.getStart());
        systemStatements.add(stmt);
        
        System.out.println("Statement: Found case statement at line " + getLineNumber(ctx));
    }
    
    private void handleReturnStatement(TALParser.ReturnStatementContext ctx) {
        String content = getFullText(ctx);
        if (content == null || content.trim().isEmpty()) return;
        
        TALStatement stmt = createStatement("RETURN", content, ctx.getStart());
        systemStatements.add(stmt);
        
        System.out.println("Statement: Found return statement at line " + getLineNumber(ctx));
    }
    
    private void handleLocalDeclarationStatement(TALParser.LocalDeclarationStatementContext ctx) {
        String content = getFullText(ctx);
        if (content == null || content.trim().isEmpty()) return;
        
        int lineNumber = getLineNumber(ctx);
        System.out.println("Statement: Found local declaration at line " + lineNumber);
        System.out.println("DEBUG: Statement visitor processing content: '" + content + "'");
        
        // Extract data item using the improved parsing logic
        TALDataItem dataItem = parseDataFromContentImproved(content, lineNumber);
        if (dataItem != null) {
            dataItem.setSection(currentProcedure != null ? currentProcedure.getName() : "LOCAL");
            extractedDataItems.add(dataItem);
            System.out.println("Statement: Successfully extracted data item: " + dataItem.getName() + " (" + dataItem.getDataType() + ")");
        } else {
            System.out.println("DEBUG: Failed to extract data item from: '" + content + "'");
        }
    }
    
    private TALDataItem parseDataFromContentImproved(String content, int lineNumber) {
        System.out.println("DEBUG: Statement visitor attempting to parse: '" + content + "'");
        
        // Handle TAL's compact syntax where types and names run together
        String[] dataTypes = {"INT", "STRING", "REAL", "FIXED", "BYTE", "CHAR", "TIMESTAMP", "STRUCT", "UNSIGNED", "EXTADDR", "SGADDR"};
        
        for (String dataType : dataTypes) {
            System.out.println("DEBUG: Checking for data type: " + dataType);
            
            // Pattern 1: TAL pointer syntax: TYPE.variable_name
            Pattern dotPattern = Pattern.compile("\\b" + dataType + "\\.([A-Za-z_][A-Za-z0-9_]*)", Pattern.CASE_INSENSITIVE);
            Matcher dotMatcher = dotPattern.matcher(content);
            if (dotMatcher.find()) {
                String varName = dotMatcher.group(1);
                System.out.println("DEBUG: Found TAL pointer - type: " + dataType + " POINTER, name: ." + varName);
                
                TALDataItem dataItem = new TALDataItem();
                dataItem.setDataType(dataType + " POINTER");
                dataItem.setName("." + varName);
                dataItem.setLineNumber(lineNumber);
                dataItem.setDefinition(content.trim());
                return dataItem;
            }
            
            // Pattern 2: Compact syntax: TYPEvariable_name (no space)
            Pattern compactPattern = Pattern.compile("\\b" + dataType + "([A-Za-z_][A-Za-z0-9_]*)", Pattern.CASE_INSENSITIVE);
            Matcher compactMatcher = compactPattern.matcher(content);
            if (compactMatcher.find()) {
                String varName = compactMatcher.group(1);
                System.out.println("DEBUG: Found compact syntax - type: " + dataType + ", name: " + varName);
                
                TALDataItem dataItem = new TALDataItem();
                dataItem.setDataType(dataType);
                dataItem.setName(varName);
                dataItem.setLineNumber(lineNumber);
                dataItem.setDefinition(content.trim());
                return dataItem;
            }
            
            // Pattern 3: Standard syntax: TYPE variable_name (with space)
            Pattern standardPattern = Pattern.compile("\\b" + dataType + "\\s+([A-Za-z_][A-Za-z0-9_]*)", Pattern.CASE_INSENSITIVE);
            Matcher standardMatcher = standardPattern.matcher(content);
            if (standardMatcher.find()) {
                String varName = standardMatcher.group(1);
                System.out.println("DEBUG: Found standard syntax - type: " + dataType + ", name: " + varName);
                
                TALDataItem dataItem = new TALDataItem();
                dataItem.setDataType(dataType);
                dataItem.setName(varName);
                dataItem.setLineNumber(lineNumber);
                dataItem.setDefinition(content.trim());
                return dataItem;
            }
        }
        
        System.out.println("DEBUG: No data type patterns matched in content: '" + content + "'");
        return null;
    }
    
    private void handleLocalDataDeclaration(TALParser.SimpleVariableDeclarationContext ctx) {
        String content = getFullText(ctx);
        int lineNumber = getLineNumber(ctx);
        
        TALDataItem dataItem = parseDataFromContent(content, lineNumber);
        if (dataItem != null) {
            // Mark as local to current procedure
            dataItem.setSection(currentProcedure != null ? currentProcedure.getName() : "LOCAL");
            extractedDataItems.add(dataItem);
            System.out.println("Grammar: Added local data item: " + dataItem.getName());
        }
    }
    
    // =====================================================================
    // HELPER METHODS
    // =====================================================================
    
    private TALStatement createStatement(String type, String content, Token startToken) {
        TALStatement stmt = new TALStatement();
        stmt.setType(type);
        stmt.setContent(content.trim());
        stmt.setLineNumber(startToken != null ? startToken.getLine() : 0);
        
        // Set context if available
        String procedureName = currentProcedure != null ? currentProcedure.getName() : "GLOBAL";
        stmt.setContext(procedureName);
        
        return stmt;
    }
    
    private String determineStatementType(String content, String ruleName) {
        String upperContent = content.toUpperCase();
        
        // Check content patterns first
        if (upperContent.contains("CALL ")) return "CALL";
        if (upperContent.contains("SELECT ") || upperContent.contains("INSERT ") || 
            upperContent.contains("UPDATE ") || upperContent.contains("DELETE ")) return "SQL";
        if (upperContent.contains("COPY ")) return "COPY";
        if (upperContent.contains("IF ")) return "IF";
        if (upperContent.contains("WHILE ")) return "WHILE";
        if (upperContent.contains("FOR ")) return "FOR";
        if (upperContent.contains("RETURN")) return "RETURN";
        if (upperContent.contains(":=")) return "ASSIGNMENT";
        
        // Use rule name as fallback
        if (ruleName != null) {
            String upperRule = ruleName.toUpperCase();
            if (upperRule.contains("CALL")) return "CALL";
            if (upperRule.contains("IF")) return "IF";
            if (upperRule.contains("WHILE")) return "WHILE";
            if (upperRule.contains("FOR")) return "FOR";
            if (upperRule.contains("ASSIGN")) return "ASSIGNMENT";
        }
        
        return "UNKNOWN";
    }
    
    private TALDataItem parseDataFromContent(String content, int lineNumber) {
        Pattern pattern = Pattern.compile("\\b(INT(?:\\(\\d+\\))?|STRING(?:\\(\\d+\\))?|REAL(?:\\(\\d+\\))?|FIXED(?:\\(\\d+(?:,\\d+)?\\))?|BYTE|CHAR|TIMESTAMP|STRUCT|UNSIGNED(?:\\(\\d+\\))?|EXTADDR|SGADDR)\\s+([A-Za-z_][A-Za-z0-9_]*)", Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(content);
        
        if (matcher.find()) {
            TALDataItem dataItem = new TALDataItem();
            dataItem.setDataType(matcher.group(1));
            dataItem.setName(matcher.group(2));
            dataItem.setLineNumber(lineNumber);
            dataItem.setDefinition(content);
            dataItem.setSection(currentProcedure != null ? currentProcedure.getName() : "GLOBAL");
            return dataItem;
        }
        
        return null;
    }
    
    private String extractProcedureName(String line) {
        // Multiple patterns to handle various cases
        Pattern[] patterns = {
            // Type PROC name(params)
            Pattern.compile("(?:[A-Z_][A-Z0-9_]*(?:\\([^)]*\\))?\\s+)?(?:PROC|proc)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(", Pattern.CASE_INSENSITIVE),
            // SUBPROC name(params)
            Pattern.compile("(?:SUBPROC|subproc)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*\\(", Pattern.CASE_INSENSITIVE),
            // Simple PROC name
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
    
    private void extractVariableReferences(String content) {
        // Extract variable names from assignment statements
        Pattern varPattern = Pattern.compile("\\b([A-Za-z_][A-Za-z0-9_]*)\\s*:=", Pattern.CASE_INSENSITIVE);
        Matcher matcher = varPattern.matcher(content);
        
        while (matcher.find()) {
            String varName = matcher.group(1);
            if (!isKeywordOrReserved(varName)) {
                callReferences.merge("VAR_" + varName.toUpperCase(), 1, Integer::sum);
            }
        }
    }
    
    private void extractConditionalReferences(String content) {
        // Extract variables used in conditional expressions
        Pattern condPattern = Pattern.compile("\\b([A-Za-z_][A-Za-z0-9_]*)\\s*(?:[<>=!]=?|<>)", Pattern.CASE_INSENSITIVE);
        Matcher matcher = condPattern.matcher(content);
        
        while (matcher.find()) {
            String varName = matcher.group(1);
            if (!isKeywordOrReserved(varName)) {
                callReferences.merge("COND_" + varName.toUpperCase(), 1, Integer::sum);
            }
        }
    }
    
    private boolean isKeywordOrReserved(String identifier) {
        if (identifier == null || identifier.trim().isEmpty()) return true;
        
        Set<String> keywords = Set.of(
            "PROC", "SUBPROC", "INT", "STRING", "REAL", "FIXED", "BYTE", "CHAR",
            "IF", "THEN", "ELSE", "WHILE", "FOR", "RETURN", "BEGIN", "END", 
            "FORWARD", "STRUCT", "MAIN", "INTERRUPT", "RESIDENT", "CALLABLE",
            "CALL", "MOVE", "SCAN", "RSCAN", "BITDEPOSIT", "ASSIGN", "TO",
            "CASE", "OF", "OTHERWISE", "UNTIL", "DO", "DOWNTO", "BY"
        );
        
        return keywords.contains(identifier.toUpperCase());
    }
    
    String getFullText(ParserRuleContext ctx) {
        if (ctx == null) return "";
        
        Token startToken = ctx.getStart();
        Token stopToken = ctx.getStop();
        
        if (startToken != null && stopToken != null && tokenStream != null) {
            try {
                return tokenStream.getText(startToken, stopToken);
            } catch (Exception e) {
                return ctx.getText();
            }
        }
        
        return ctx.getText();
    }
}

