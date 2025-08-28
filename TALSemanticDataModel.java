import java.util.*;

// =====================================================================
// CORE SEMANTIC CONTEXT AND ANALYSIS CLASSES
// =====================================================================

/**
 * Represents semantic parsing context with enhanced type safety
 */
class SemanticContext {
    public enum Type {
        GLOBAL,
        PROCEDURE,
        BLOCK,
        STRUCTURE,
        CONDITIONAL,
        LOOP,
        STRUCT
    }

    private String name;
    private Type type;
    private int startLine;
    private int endLine;
    private SemanticContext parent;
    private Map<String, Object> attributes;
    private ProcedureSemantics procedureSemantics;

    public SemanticContext(String name, Type type) {
        this.name = name;
        this.type = type;
        this.attributes = new HashMap<>();
        this.startLine = -1;
        this.endLine = -1;
    }

    public SemanticContext(String name, Type type, int startLine) {
        this(name, type);
        this.startLine = startLine;
    }

    // Getters and setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public Type getType() { return type; }
    public void setType(Type type) { this.type = type; }

    public int getStartLine() { return startLine; }
    public void setStartLine(int startLine) { this.startLine = startLine; }

    public int getEndLine() { return endLine; }
    public void setEndLine(int endLine) { this.endLine = endLine; }

    public SemanticContext getParent() { return parent; }
    public void setParent(SemanticContext parent) { this.parent = parent; }

    public Map<String, Object> getAttributes() { return attributes; }
    public void setAttributes(Map<String, Object> attributes) { this.attributes = attributes; }

    public ProcedureSemantics getProcedureSemantics() { return procedureSemantics; }
    public void setProcedureSemantics(ProcedureSemantics procedureSemantics) { 
        this.procedureSemantics = procedureSemantics; 
    }

    // Utility methods
    public void setAttribute(String key, Object value) {
        attributes.put(key, value);
    }

    public Object getAttribute(String key) {
        return attributes.get(key);
    }

    public String getFullPath() {
        if (parent == null) {
            return name;
        }
        return parent.getFullPath() + "." + name;
    }

    public boolean isWithinLine(int lineNumber) {
        return startLine <= lineNumber && (endLine == -1 || lineNumber <= endLine);
    }

    @Override
    public String toString() {
        return String.format("SemanticContext{name='%s', type=%s, lines=%d-%d}", 
            name, type, startLine, endLine);
    }
}

/**
 * Represents procedure-level semantic information
 */
class ProcedureSemantics {
    private String purpose;
    private int complexity;
    private String dataAccess; // "READ_ONLY", "WRITE_ONLY", "READ_WRITE"
    private List<String> sideEffects = new ArrayList<>();
    private List<String> businessFunctions = new ArrayList<>();
    private String returnSemantics;
    private Map<String, String> parameterSemantics = new HashMap<>();
    
    public String getPurpose() { return purpose; }
    public void setPurpose(String purpose) { this.purpose = purpose; }
    
    public int getComplexity() { return complexity; }
    public void setComplexity(int complexity) { this.complexity = complexity; }
    
    public String getDataAccess() { return dataAccess; }
    public void setDataAccess(String dataAccess) { this.dataAccess = dataAccess; }
    
    public List<String> getSideEffects() { return sideEffects; }
    public void setSideEffects(List<String> sideEffects) { this.sideEffects = sideEffects; }
    
    public List<String> getBusinessFunctions() { return businessFunctions; }
    public void setBusinessFunctions(List<String> businessFunctions) { this.businessFunctions = businessFunctions; }
    
    public String getReturnSemantics() { return returnSemantics; }
    public void setReturnSemantics(String returnSemantics) { this.returnSemantics = returnSemantics; }
    
    public Map<String, String> getParameterSemantics() { return parameterSemantics; }
    public void setParameterSemantics(Map<String, String> parameterSemantics) { this.parameterSemantics = parameterSemantics; }
}

// =====================================================================
// CORE TAL STRUCTURAL ELEMENTS
// =====================================================================

/**
 * Unified procedure representation
 */
class TALProcedure {
    private String name;
    private String returnType;
    private List<String> attributes = new ArrayList<>();
    private List<String> parameters = new ArrayList<>();
    private int lineNumber;
    private int endLineNumber;
    private String reasoningInfo;
    private double contextScore;
    
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getReturnType() { return returnType; }
    public void setReturnType(String returnType) { this.returnType = returnType; }
    
    public List<String> getAttributes() { return attributes; }
    public void setAttributes(List<String> attributes) { this.attributes = attributes; }
    
    public List<String> getParameters() { return parameters; }
    public void setParameters(List<String> parameters) { this.parameters = parameters; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public int getEndLineNumber() { return endLineNumber; }
    public void setEndLineNumber(int endLineNumber) { this.endLineNumber = endLineNumber; }
    
    public String getReasoningInfo() { return reasoningInfo; }
    public void setReasoningInfo(String reasoningInfo) { this.reasoningInfo = reasoningInfo; }
    
    public double getContextScore() { return contextScore; }
    public void setContextScore(double contextScore) { this.contextScore = contextScore; }
}

/**
 * Unified statement representation
 */
class TALStatement {
    private String type;
    private String content;
    private int lineNumber;
    private String context;
    
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getContext() { return context; }
    public void setContext(String context) { this.context = context; }
}

/**
 * Data item representation
 */
class TALDataItem {
    private String name;
    private String dataType;
    private int lineNumber;
    private String definition;
    private String section;
    
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getDataType() { return dataType; }
    public void setDataType(String dataType) { this.dataType = dataType; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getDefinition() { return definition; }
    public void setDefinition(String definition) { this.definition = definition; }
    
    public String getSection() { return section; }
    public void setSection(String section) { this.section = section; }
}

/**
 * File descriptor representation
 */
class TALFileDescriptor {
    private String name;
    private String type;
    private int lineNumber;
    
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
}

/**
 * Comment representation
 */
class TALComment {
    private String content;
    private int lineNumber;
    private String type; // "HEADER", "INLINE", "BLOCK"
    
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
}

// =====================================================================
// SQL AND DATABASE OPERATIONS
// =====================================================================

/**
 * SQL operation representation
 */
class SqlOperation {
    private String type;
    private String sqlType;
    private int lineNumber;
    private String rawContent;
    private String businessPurpose;
    private List<String> tables = new ArrayList<>();
    private List<String> columns = new ArrayList<>();
    
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    
    public String getSqlType() { return sqlType; }
    public void setSqlType(String sqlType) { this.sqlType = sqlType; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getRawContent() { return rawContent; }
    public void setRawContent(String rawContent) { this.rawContent = rawContent; }
    
    public String getBusinessPurpose() { return businessPurpose; }
    public void setBusinessPurpose(String businessPurpose) { this.businessPurpose = businessPurpose; }
    
    public List<String> getTables() { return tables; }
    public void setTables(List<String> tables) { this.tables = tables; }
    
    public List<String> getColumns() { return columns; }
    public void setColumns(List<String> columns) { this.columns = columns; }
}

// =====================================================================
// PREPROCESSOR DIRECTIVES
// =====================================================================

/**
 * Preprocessor directive representation
 */
class PreprocessorDirective {
    private String type;
    private String directive;
    private int lineNumber;
    private String condition;
    private String symbolName;
    private String symbolValue;
    private boolean result;
    private String includePath;
    
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    
    public String getDirective() { return directive; }
    public void setDirective(String directive) { this.directive = directive; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getCondition() { return condition; }
    public void setCondition(String condition) { this.condition = condition; }
    
    public String getSymbolName() { return symbolName; }
    public void setSymbolName(String symbolName) { this.symbolName = symbolName; }
    
    public String getSymbolValue() { return symbolValue; }
    public void setSymbolValue(String symbolValue) { this.symbolValue = symbolValue; }
    
    public boolean getResult() { return result; }
    public void setResult(boolean result) { this.result = result; }
    
    public String getIncludePath() { return includePath; }
    public void setIncludePath(String includePath) { this.includePath = includePath; }
}

// =====================================================================
// PROCEDURE REGISTRY AND METADATA
// =====================================================================

/**
 * Procedure registry information
 */
class ProcedureInfo {
    private String name;
    private String returnType;
    private List<String> parameters = new ArrayList<>();
    private List<String> attributes = new ArrayList<>();
    private int lineNumber;
    private String context;
    private boolean isSystemProcedure;
    private String businessPurpose;
    
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getReturnType() { return returnType; }
    public void setReturnType(String returnType) { this.returnType = returnType; }
    
    public List<String> getParameters() { return parameters; }
    public void setParameters(List<String> parameters) { this.parameters = parameters; }
    
    public List<String> getAttributes() { return attributes; }
    public void setAttributes(List<String> attributes) { this.attributes = attributes; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getContext() { return context; }
    public void setContext(String context) { this.context = context; }
    
    public boolean isSystemProcedure() { return isSystemProcedure; }
    public void setSystemProcedure(boolean systemProcedure) { this.isSystemProcedure = systemProcedure; }
    
    public String getBusinessPurpose() { return businessPurpose; }
    public void setBusinessPurpose(String businessPurpose) { this.businessPurpose = businessPurpose; }
}

/**
 * Variable information and usage patterns
 */
class VariableInfo {
    private String name;
    private String dataType;
    private String scope;
    private String context;
    private String purpose;
    private List<String> usagePatterns = new ArrayList<>();
    private boolean isPointer;
    private boolean isArray;
    private String businessMeaning;
    private int declarationLine;
    
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public String getDataType() { return dataType; }
    public void setDataType(String dataType) { this.dataType = dataType; }
    
    public String getScope() { return scope; }
    public void setScope(String scope) { this.scope = scope; }
    
    public String getContext() { return context; }
    public void setContext(String context) { this.context = context; }
    
    public String getPurpose() { return purpose; }
    public void setPurpose(String purpose) { this.purpose = purpose; }
    
    public List<String> getUsagePatterns() { return usagePatterns; }
    public void setUsagePatterns(List<String> usagePatterns) { this.usagePatterns = usagePatterns; }
    
    public boolean isPointer() { return isPointer; }
    public void setPointer(boolean pointer) { this.isPointer = pointer; }
    
    public boolean isArray() { return isArray; }
    public void setArray(boolean array) { this.isArray = array; }
    
    public String getBusinessMeaning() { return businessMeaning; }
    public void setBusinessMeaning(String businessMeaning) { this.businessMeaning = businessMeaning; }
    
    public int getDeclarationLine() { return declarationLine; }
    public void setDeclarationLine(int declarationLine) { this.declarationLine = declarationLine; }
}

// =====================================================================
// SEMANTIC OPERATIONS
// =====================================================================

/**
 * Procedure call operation
 */
class ProcedureCall {
    private String procedureName;
    private int lineNumber;
    private String rawContent;
    private boolean systemCall;
    private List<String> parameters = new ArrayList<>();
    private String businessPurpose;
    
    public String getProcedureName() { return procedureName; }
    public void setProcedureName(String procedureName) { this.procedureName = procedureName; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getRawContent() { return rawContent; }
    public void setRawContent(String rawContent) { this.rawContent = rawContent; }
    
    public boolean isSystemCall() { return systemCall; }
    public void setSystemCall(boolean systemCall) { this.systemCall = systemCall; }
    
    public List<String> getParameters() { return parameters; }
    public void setParameters(List<String> parameters) { this.parameters = parameters; }
    
    public String getBusinessPurpose() { return businessPurpose; }
    public void setBusinessPurpose(String businessPurpose) { this.businessPurpose = businessPurpose; }
}

/**
 * Bit field operation representation
 */
class BitFieldOperation {
    private int lineNumber;
    private String rawContent;
    private String operation; // "EXTRACT_BITS" or "SET_BITS"
    private String targetVariable;
    private String sourceVariable;
    private String sourceValue;
    private int startBit;
    private int endBit;
    private int bitWidth;
    private String businessPurpose;
    private String modernEquivalent;
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getRawContent() { return rawContent; }
    public void setRawContent(String rawContent) { this.rawContent = rawContent; }
    
    public String getOperation() { return operation; }
    public void setOperation(String operation) { this.operation = operation; }
    
    public String getTargetVariable() { return targetVariable; }
    public void setTargetVariable(String targetVariable) { this.targetVariable = targetVariable; }
    
    public String getSourceVariable() { return sourceVariable; }
    public void setSourceVariable(String sourceVariable) { this.sourceVariable = sourceVariable; }
    
    public String getSourceValue() { return sourceValue; }
    public void setSourceValue(String sourceValue) { this.sourceValue = sourceValue; }
    
    public int getStartBit() { return startBit; }
    public void setStartBit(int startBit) { this.startBit = startBit; }
    
    public int getEndBit() { return endBit; }
    public void setEndBit(int endBit) { this.endBit = endBit; }
    
    public int getBitWidth() { return bitWidth; }
    public void setBitWidth(int bitWidth) { this.bitWidth = bitWidth; }
    
    public String getBusinessPurpose() { return businessPurpose; }
    public void setBusinessPurpose(String businessPurpose) { this.businessPurpose = businessPurpose; }
    
    public String getModernEquivalent() { return modernEquivalent; }
    public void setModernEquivalent(String modernEquivalent) { this.modernEquivalent = modernEquivalent; }
    
    @Override
    public String toString() {
        return String.format("BitFieldOp{%s: %s[%d:%d] (%s)}", 
            operation, targetVariable, startBit, endBit, businessPurpose);
    }
}

/**
 * Pointer operation representation
 */
class PointerOperation {
    private int lineNumber;
    private String rawContent;
    private String operation; // "POINTER_ASSIGN", "POINTER_DEREFERENCE"
    private String targetVariable;
    private String sourceVariable;
    private String accessType; // "ADDRESS_COPY", "VALUE_STORE", "VALUE_LOAD"
    private String businessPurpose;
    private String memorySemantics;
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getRawContent() { return rawContent; }
    public void setRawContent(String rawContent) { this.rawContent = rawContent; }
    
    public String getOperation() { return operation; }
    public void setOperation(String operation) { this.operation = operation; }
    
    public String getTargetVariable() { return targetVariable; }
    public void setTargetVariable(String targetVariable) { this.targetVariable = targetVariable; }
    
    public String getSourceVariable() { return sourceVariable; }
    public void setSourceVariable(String sourceVariable) { this.sourceVariable = sourceVariable; }
    
    public String getAccessType() { return accessType; }
    public void setAccessType(String accessType) { this.accessType = accessType; }
    
    public String getBusinessPurpose() { return businessPurpose; }
    public void setBusinessPurpose(String businessPurpose) { this.businessPurpose = businessPurpose; }
    
    public String getMemorySemantics() { return memorySemantics; }
    public void setMemorySemantics(String memorySemantics) { this.memorySemantics = memorySemantics; }
    
    @Override
    public String toString() {
        return String.format("PointerOp{%s: %s -> %s (%s)}", 
            operation, sourceVariable, targetVariable, businessPurpose);
    }
}

// =====================================================================
// BUSINESS LOGIC AND RULES
// =====================================================================

/**
 * Business rule representation
 */
class BusinessRule {
    private String ruleType; // "DATA_PROCESSING", "BUSINESS_PROCESS", "VALIDATION", "DATA_PERSISTENCE"
    private String context;
    private int lineNumber;
    private String description;
    private String businessLogic;
    private String purpose;
    private String sourceCode;
    private String modernEquivalent;
    private List<String> conditions = new ArrayList<>();
    private List<String> actions = new ArrayList<>();
    private String priority; // "HIGH", "MEDIUM", "LOW"
    
    public String getRuleType() { return ruleType; }
    public void setRuleType(String ruleType) { this.ruleType = ruleType; }
    
    public String getContext() { return context; }
    public void setContext(String context) { this.context = context; }
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    
    public String getBusinessLogic() { return businessLogic; }
    public void setBusinessLogic(String businessLogic) { this.businessLogic = businessLogic; }
    
    public String getPurpose() { return purpose; }
    public void setPurpose(String purpose) { this.purpose = purpose; }
    
    public String getModernEquivalent() { return modernEquivalent; }
    public void setModernEquivalent(String modernEquivalent) { this.modernEquivalent = modernEquivalent; }
    
    public List<String> getConditions() { return conditions; }
    public void setConditions(List<String> conditions) { this.conditions = conditions; }
    
    public List<String> getActions() { return actions; }
    public void setActions(List<String> actions) { this.actions = actions; }
    
    public String getPriority() { return priority; }
    public void setPriority(String priority) { this.priority = priority; }
    
    public void setSourceCode(String sourceCode) {
        this.sourceCode = sourceCode;
    }
    public String toString() {
        return String.format("BusinessRule{%s: %s (%s)}", 
            ruleType, description, purpose);
    }
}

/**
 * Control flow pattern representation
 */
class ControlFlowPattern {
    private int lineNumber;
    private String rawContent;
    private String patternType; // "IF", "WHILE", "FOR", "CASE"
    private String conditionType; // "COMPARISON", "RANGE_CHECK", "NULL_CHECK"
    private List<String> conditions = new ArrayList<>();
    private String businessLogic;
    private List<String> actions = new ArrayList<>();
    private String errorHandling;
    
    public int getLineNumber() { return lineNumber; }
    public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
    
    public String getRawContent() { return rawContent; }
    public void setRawContent(String rawContent) { this.rawContent = rawContent; }
    
    public String getPatternType() { return patternType; }
    public void setPatternType(String patternType) { this.patternType = patternType; }
    
    public String getConditionType() { return conditionType; }
    public void setConditionType(String conditionType) { this.conditionType = conditionType; }
    
    public List<String> getConditions() { return conditions; }
    public void setConditions(List<String> conditions) { this.conditions = conditions; }
    
    public String getBusinessLogic() { return businessLogic; }
    public void setBusinessLogic(String businessLogic) { this.businessLogic = businessLogic; }
    
    public List<String> getActions() { return actions; }
    public void setActions(List<String> actions) { this.actions = actions; }
    
    public String getErrorHandling() { return errorHandling; }
    public void setErrorHandling(String errorHandling) { this.errorHandling = errorHandling; }
    
    @Override
    public String toString() {
        return String.format("ControlFlow{%s: %s conditions (%s)}", 
            patternType, conditions.size(), businessLogic);
    }
}

// =====================================================================
// COMPREHENSIVE ANALYSIS RESULT
// =====================================================================

/**
 * Unified semantic analysis result class - the single source of truth
 */
class TALSemanticAnalysisResult {
    private String programName;
    private String analysisMethod;
    private Date parseTimestamp;
    private int sourceLinesProcessed;
    
    // Core structural elements
    private List<TALProcedure> procedures = new ArrayList<>();
    private List<TALDataItem> dataItems = new ArrayList<>();
    private List<TALFileDescriptor> fileDescriptors = new ArrayList<>();
    private List<TALComment> headerComments = new ArrayList<>();
    private List<TALComment> inlineComments = new ArrayList<>();
    private Map<Integer, List<TALComment>> commentsByLine = new HashMap<>();
    
    // Statement collections
    private List<TALStatement> sqlStatements = new ArrayList<>();
    private List<TALStatement> copyStatements = new ArrayList<>();
    private List<TALStatement> callStatements = new ArrayList<>();
    private List<TALStatement> systemStatements = new ArrayList<>();
    
    // Enhanced semantic analysis results
    private List<ProcedureCall> procedureCalls = new ArrayList<>();
    private List<BitFieldOperation> bitFieldOperations = new ArrayList<>();
    private List<PointerOperation> pointerOperations = new ArrayList<>();
    private List<BusinessRule> businessRules = new ArrayList<>();
    private List<ControlFlowPattern> controlFlowPatterns = new ArrayList<>();
    
    // Database and SQL operations
    private List<SqlOperation> sqlOperations = new ArrayList<>();
    private List<PreprocessorDirective> preprocessorDirectives = new ArrayList<>();
    
    // Registry and metadata
    private Map<String, VariableInfo> variableRegistry = new HashMap<>();
    private Map<String, ProcedureInfo> procedureRegistry = new HashMap<>();
    
    // Metrics and statistics
    private Map<String, Integer> statementCounts = new HashMap<>();
    private Map<String, Integer> callReferences = new HashMap<>();
    private List<String> parseWarnings = new ArrayList<>();
    private Map<String, Long> performanceMetrics = new HashMap<>();
    private Map<String, Integer> parseMethodStats = new HashMap<>();
    
    // Getters and setters for all fields
    public String getProgramName() { return programName; }
    public void setProgramName(String programName) { this.programName = programName; }
    
    public String getAnalysisMethod() { return analysisMethod; }
    public void setAnalysisMethod(String analysisMethod) { this.analysisMethod = analysisMethod; }
    
    public Date getParseTimestamp() { return parseTimestamp; }
    public void setParseTimestamp(Date parseTimestamp) { this.parseTimestamp = parseTimestamp; }
    
    public int getSourceLinesProcessed() { return sourceLinesProcessed; }
    public void setSourceLinesProcessed(int sourceLinesProcessed) { this.sourceLinesProcessed = sourceLinesProcessed; }
    
    public List<TALProcedure> getProcedures() { return procedures; }
    public void setProcedures(List<TALProcedure> procedures) { this.procedures = procedures; }
    
    public List<TALDataItem> getDataItems() { return dataItems; }
    public void setDataItems(List<TALDataItem> dataItems) { this.dataItems = dataItems; }
    
    public List<TALFileDescriptor> getFileDescriptors() { return fileDescriptors; }
    public void setFileDescriptors(List<TALFileDescriptor> fileDescriptors) { this.fileDescriptors = fileDescriptors; }
    
    public List<TALComment> getHeaderComments() { return headerComments; }
    public void setHeaderComments(List<TALComment> headerComments) { this.headerComments = headerComments; }
    
    public List<TALComment> getInlineComments() { return inlineComments; }
    public void setInlineComments(List<TALComment> inlineComments) { this.inlineComments = inlineComments; }
    
    public Map<Integer, List<TALComment>> getCommentsByLine() { return commentsByLine; }
    public void setCommentsByLine(Map<Integer, List<TALComment>> commentsByLine) { this.commentsByLine = commentsByLine; }
    
    public List<TALStatement> getSqlStatements() { return sqlStatements; }
    public void setSqlStatements(List<TALStatement> sqlStatements) { this.sqlStatements = sqlStatements; }
    
    public List<TALStatement> getCopyStatements() { return copyStatements; }
    public void setCopyStatements(List<TALStatement> copyStatements) { this.copyStatements = copyStatements; }
    
    public List<TALStatement> getCallStatements() { return callStatements; }
    public void setCallStatements(List<TALStatement> callStatements) { this.callStatements = callStatements; }
    
    public List<TALStatement> getSystemStatements() { return systemStatements; }
    public void setSystemStatements(List<TALStatement> systemStatements) { this.systemStatements = systemStatements; }
    
    public List<ProcedureCall> getProcedureCalls() { return procedureCalls; }
    public void setProcedureCalls(List<ProcedureCall> procedureCalls) { this.procedureCalls = procedureCalls; }
    
    public List<BitFieldOperation> getBitFieldOperations() { return bitFieldOperations; }
    public void setBitFieldOperations(List<BitFieldOperation> bitFieldOperations) { this.bitFieldOperations = bitFieldOperations; }
    
    public List<PointerOperation> getPointerOperations() { return pointerOperations; }
    public void setPointerOperations(List<PointerOperation> pointerOperations) { this.pointerOperations = pointerOperations; }
    
    public List<BusinessRule> getBusinessRules() { return businessRules; }
    public void setBusinessRules(List<BusinessRule> businessRules) { this.businessRules = businessRules; }
    
    public List<ControlFlowPattern> getControlFlowPatterns() { return controlFlowPatterns; }
    public void setControlFlowPatterns(List<ControlFlowPattern> controlFlowPatterns) { this.controlFlowPatterns = controlFlowPatterns; }
    
    public List<SqlOperation> getSqlOperations() { return sqlOperations; }
    public void setSqlOperations(List<SqlOperation> sqlOperations) { this.sqlOperations = sqlOperations; }
    
    public List<PreprocessorDirective> getPreprocessorDirectives() { return preprocessorDirectives; }
    public void setPreprocessorDirectives(List<PreprocessorDirective> preprocessorDirectives) { this.preprocessorDirectives = preprocessorDirectives; }
    
    public Map<String, VariableInfo> getVariableRegistry() { return variableRegistry; }
    public void setVariableRegistry(Map<String, VariableInfo> variableRegistry) { this.variableRegistry = variableRegistry; }
    
    public Map<String, ProcedureInfo> getProcedureRegistry() { return procedureRegistry; }
    public void setProcedureRegistry(Map<String, ProcedureInfo> procedureRegistry) { this.procedureRegistry = procedureRegistry; }
    
    public Map<String, Integer> getStatementCounts() { return statementCounts; }
    public void setStatementCounts(Map<String, Integer> statementCounts) { this.statementCounts = statementCounts; }
    
    public Map<String, Integer> getCallReferences() { return callReferences; }
    public void setCallReferences(Map<String, Integer> callReferences) { this.callReferences = callReferences; }
    
    public List<String> getParseWarnings() { return parseWarnings; }
    public void setParseWarnings(List<String> parseWarnings) { this.parseWarnings = parseWarnings; }
    
    public Map<String, Long> getPerformanceMetrics() { return performanceMetrics; }
    public void setPerformanceMetrics(Map<String, Long> performanceMetrics) { this.performanceMetrics = performanceMetrics; }
    
    public Map<String, Integer> getParseMethodStats() { return parseMethodStats; }
    public void setParseMethodStats(Map<String, Integer> parseMethodStats) { this.parseMethodStats = parseMethodStats; }
}

