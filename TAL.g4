grammar TAL;

// ----------------------
// Top-level Program Structure
// ----------------------
program
    : sourceItem* EOF
    ;

sourceItem
    : namePart
    | globalDeclarationItem
    | blockDeclaration
    | procedureDeclaration
    | directiveLine
    | moduleImport
    | pragmaDirective
    | sourceAssignment
    | commentedSourceAssignment
    | preprocessorDirective
    | talSqlStatement
    ;

namePart: NAME IDENTIFIER SEMI;

// Handle source assignments like "source = TANDEM_EXTDECSO"
sourceAssignment
    : SOURCE (ASSIGN | SIMPLE_EQ) IDENTIFIER SEMI?
    ;

// Handle commented source assignments like "!source = TANDEM_EXTDECSO(...)"
commentedSourceAssignment
    : TAL_INLINE_COMMENT SEMI?
    ;

// Enhanced preprocessor directive handling
preprocessorDirective
    : conditionalCompilation
    | defineDirective
    | includeDirective
    | pragmaDirective
    ;

conditionalCompilation
    : QUESTION_MARK IF preprocessorExpression SEMI preprocessorBody (QUESTION_MARK ELSE preprocessorBody)? QUESTION_MARK ENDIF SEMI
    | QUESTION_MARK IFNOT preprocessorExpression SEMI preprocessorBody (QUESTION_MARK ELSE preprocessorBody)? QUESTION_MARK ENDIF SEMI
    ;

defineDirective
    : QUESTION_MARK DEFINE IDENTIFIER (ASSIGN | SIMPLE_EQ) preprocessorExpression SEMI
    | QUESTION_MARK DEFINE IDENTIFIER LPAREN identifierList? RPAREN (ASSIGN | SIMPLE_EQ) preprocessorExpression SEMI
    ;

includeDirective
    : QUESTION_MARK INCLUDE STRING_LITERAL SEMI
    | QUESTION_MARK SOURCE STRING_LITERAL SEMI
    ;

preprocessorBody
    : sourceItem*
    ;

preprocessorExpression
    : preprocessorTerm ((AND | OR) preprocessorTerm)*
    ;

preprocessorTerm
    : IDENTIFIER
    | preprocessorComparison
    | LPAREN preprocessorExpression RPAREN
    | NOT preprocessorTerm
    ;

preprocessorComparison
    : IDENTIFIER (SIMPLE_EQ | NEQ) (STRING_LITERAL | INT_LITERAL | IDENTIFIER)
    ;

identifierList
    : IDENTIFIER (COMMA IDENTIFIER)*
    ;

// TAL/SQL embedded statements
talSqlStatement
    : sqlExecStatement
    | sqlFetchStatement
    | sqlOpenStatement
    | sqlCloseStatement
    | sqlDeclareStatement
    ;

sqlExecStatement
    : EXEC SQL sqlCommand SEMI
    ;

sqlFetchStatement
    : EXEC SQL FETCH IDENTIFIER INTO variableList SEMI
    ;

sqlOpenStatement
    : EXEC SQL OPEN IDENTIFIER SEMI
    ;

sqlCloseStatement
    : EXEC SQL CLOSE IDENTIFIER SEMI
    ;

sqlDeclareStatement
    : EXEC SQL DECLARE IDENTIFIER CURSOR FOR sqlSelectStatement SEMI
    ;

sqlCommand
    : sqlSelectStatement
    | sqlInsertStatement
    | sqlUpdateStatement
    | sqlDeleteStatement
    | sqlCommitStatement
    | sqlRollbackStatement
    ;

sqlSelectStatement
    : SELECT sqlSelectList FROM sqlTableList (WHERE sqlWhereClause)? (ORDER BY sqlOrderList)?
    ;

sqlInsertStatement
    : INSERT INTO IDENTIFIER (LPAREN identifierList RPAREN)? VALUES LPAREN expressionList RPAREN
    ;

sqlUpdateStatement
    : UPDATE IDENTIFIER SET sqlAssignmentList (WHERE sqlWhereClause)?
    ;

sqlDeleteStatement
    : DELETE FROM IDENTIFIER (WHERE sqlWhereClause)?
    ;

sqlCommitStatement
    : COMMIT (WORK)?
    ;

sqlRollbackStatement
    : ROLLBACK (WORK)?
    ;

sqlSelectList
    : MUL | sqlSelectItem (COMMA sqlSelectItem)*
    ;

sqlSelectItem
    : expression (AS IDENTIFIER)?
    ;

sqlTableList
    : IDENTIFIER (COMMA IDENTIFIER)*
    ;

sqlWhereClause
    : sqlCondition ((AND | OR) sqlCondition)*
    ;

sqlCondition
    : expression comparisonOperator expression
    | expression IN LPAREN expressionList RPAREN
    | expression BETWEEN expression AND expression
    | expression IS (NOT)? NIL
    ;

sqlOrderList
    : sqlOrderItem (COMMA sqlOrderItem)*
    ;

sqlOrderItem
    : IDENTIFIER (ASC | DESC)?
    ;

sqlAssignmentList
    : sqlAssignment (COMMA sqlAssignment)*
    ;

sqlAssignment
    : IDENTIFIER SIMPLE_EQ expression
    ;

comparisonOperator
    : SIMPLE_EQ | NEQ | LT | LE | GT | GE
    ;

variableList
    : variableExpr (COMMA variableExpr)*
    ;

expressionList
    : expression (COMMA expression)*
    ;

// Keywords that can appear as identifiers in certain contexts
keywordAsIdentifier
    : OPEN | TIME | BEGINNING | STOP | READ | WRITE | ACCESS | SHARE | INPUT | OUTPUT
    ;

pragmaDirective
    : PRAGMA IDENTIFIER (LPAREN pragmaArgList? RPAREN)? SEMI
    ;
pragmaArgList
    : pragmaArg (COMMA pragmaArg)*
    ;
pragmaArg
    : IDENTIFIER | STRING_LITERAL | INT_LITERAL | INTEGER_VALUE
    ;

// ----------------------
// Global Declarations
// ----------------------
globalDeclarationItem
    : dataDeclaration
    | literalDeclaration
    | defineDeclaration
    | forwardDeclaration
    | externalDeclaration
    | equivalencedVarDeclaration
    | constSection
    | typeSection
    | varSection
    | trapDeclaration
    | guardianDeclaration
    | enscribeDeclaration
    ;

// Guardian file system declarations
guardianDeclaration
    : GUARDIAN fileDeclaration SEMI
    ;

fileDeclaration
    : IDENTIFIER (PERIOD IDENTIFIER)* (LBRACK expression RBRACK)?
    ;

// Enscribe file declarations
enscribeDeclaration
    : ENSCRIBE fileAttribute* fileDeclaration SEMI
    ;

fileAttribute
    : RECORD_SIZE SIMPLE_EQ expression
    | BLOCK_SIZE SIMPLE_EQ expression
    | FILE_TYPE SIMPLE_EQ IDENTIFIER
    | ACCESS_METHOD SIMPLE_EQ IDENTIFIER
    | ORGANIZATION SIMPLE_EQ IDENTIFIER
    ;

constSection
    : CONST (constDef SEMI?)+
    ;
constDef
    : IDENTIFIER (SIMPLE_EQ | ASSIGN) expression
    ;

typeSection
    : TYPE (typeDef SEMI?)+
    ;
typeDef
    : IDENTIFIER (SIMPLE_EQ | ASSIGN) typeSpec
    ;

varSection
    : VAR (varDef SEMI?)+
    ;
varDef
    : identList COLON typeSpec (ASSIGN expression)?
    ;

blockDeclaration
    : BLOCK blockName SEMI globalDeclarationItem* END BLOCK SEMI?
    | BLOCK PRIVATE SEMI globalDeclarationItem* END BLOCK SEMI?
    ;

blockName: IDENTIFIER;

// DEFINE declaration with HASH delimiter
defineDeclaration
    : DEFINE IDENTIFIER (LPAREN parameterList? RPAREN)? ASSIGN expression HASH SEMI?
    ;

// Trap declaration for exception handling
trapDeclaration
    : TRAP IDENTIFIER SEMI
    ;

// ----------------------
// Enhanced Data Declarations
// ----------------------
dataDeclaration
    : simpleVariableDeclaration
    | arrayDeclaration
    | structureDeclaration
    | pointerDeclaration
    | readOnlyArrayDeclaration
    | structurePointerDeclaration
    | systemGlobalPointerDeclaration
    | structVariableDeclaration
    | talPointerDeclaration
    | baseAddressEquivDeclaration
    | extendedAddressDeclaration
    | stackGroupDeclaration
    ;

// Extended addressing declarations
extendedAddressDeclaration
    : typeSpecification EXTADDR IDENTIFIER (ASSIGN extendedAddress)? (COMMA EXTADDR IDENTIFIER (ASSIGN extendedAddress)?)* SEMI
    ;

extendedAddress
    : IDENTIFIER COLON IDENTIFIER
    | STRING_LITERAL
    ;

// Stack group declarations
stackGroupDeclaration
    : typeSpecification SGADDR IDENTIFIER (ASSIGN stackGroupAddress)? (COMMA SGADDR IDENTIFIER (ASSIGN stackGroupAddress)?)* SEMI
    ;

stackGroupAddress
    : IDENTIFIER COLON IDENTIFIER
    | STRING_LITERAL
    ;

// Base-address equivalencing declaration
baseAddressEquivDeclaration
    : typeSpecification (GCONTROL | LCONTROL | SCONTROL | SGCONTROL) IDENTIFIER (ASSIGN initialization)? SEMI
    ;

// Enhanced TAL pointer declarations with proper PERIOD prefix support
talPointerDeclaration
    : typeSpecification PERIOD IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)? (COMMA PERIOD IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)?)* SEMI
    | IDENTIFIER PERIOD IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)? (COMMA PERIOD IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)?)* SEMI
    | typeSpecification DOT_EXT IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)? (COMMA DOT_EXT IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)?)* SEMI
    | typeSpecification DOT_SG IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)? (COMMA DOT_SG IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)?)* SEMI
    ;

simpleVariableDeclaration
    : typeSpecification IDENTIFIER (ASSIGN initialization)? (COMMA IDENTIFIER (ASSIGN initialization)?)* SEMI
    ;

arrayDeclaration
    : typeSpecification standardIndirectSymbol? IDENTIFIER LBRACK indexRange RBRACK (ASSIGN initialization)? (COMMA standardIndirectSymbol? IDENTIFIER LBRACK indexRange RBRACK (ASSIGN initialization)?)* SEMI
    ;

// STRUCT variable declarations
structVariableDeclaration
    : STRUCT IDENTIFIER IDENTIFIER SEMI
    | STRUCT IDENTIFIER (PERIOD IDENTIFIER)+ SEMI
    | IDENTIFIER PERIOD IDENTIFIER SEMI
    ;

structureDeclaration
    : STRUCT IDENTIFIER (LPAREN MUL RPAREN)? (LBRACK indexRange RBRACK)? SEMI structureBody
    ;

structureBody: BEGIN structureItem* END SEMI?;

structureItem
    : fieldDeclaration
    | nestedStructureDeclaration
    | fillerDeclaration
    | equivalencedFieldDeclaration
    | pointerDeclaration
    | structurePointerDeclaration
    | structPointerFieldDeclaration
    | inlineCommentItem
    | unionDeclaration
    ;

// Union declarations within structures
unionDeclaration
    : UNION BEGIN unionItem+ END SEMI
    ;

unionItem
    : fieldDeclaration
    | structureDeclaration
    ;

// Support for inline comments in structure definitions
inlineCommentItem
    : TAL_INLINE_COMMENT
    ;

structPointerFieldDeclaration
    : typeSpecification PERIOD IDENTIFIER SEMI
    | typeSpecification DOT_EXT IDENTIFIER SEMI
    | typeSpecification DOT_SG IDENTIFIER SEMI
    ;

fieldDeclaration: typeSpecification IDENTIFIER (LBRACK indexRange RBRACK)? (COMMA IDENTIFIER (LBRACK indexRange RBRACK)?)* SEMI;
nestedStructureDeclaration: STRUCT IDENTIFIER (LBRACK indexRange RBRACK)? SEMI structureBody;
fillerDeclaration: FILLER expression SEMI;
equivalencedFieldDeclaration: typeSpecification IDENTIFIER (LBRACK expression RBRACK)? ASSIGN IDENTIFIER SEMI;

pointerDeclaration
    : typeSpecification indirection IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)? (COMMA indirection IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)?)* SEMI
    | typeSpecification MUL IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)? (COMMA MUL IDENTIFIER (LBRACK indexRange RBRACK)? (ASSIGN initialization)?)* SEMI
    ;

structurePointerDeclaration
    : typeSpecification indirection IDENTIFIER LPAREN IDENTIFIER RPAREN (ASSIGN initialization)? (COMMA indirection IDENTIFIER LPAREN IDENTIFIER RPAREN (ASSIGN initialization)?)* SEMI
    ;

systemGlobalPointerDeclaration
    : typeSpecification SGINDIRECT IDENTIFIER (ASSIGN initialization)? (COMMA SGINDIRECT IDENTIFIER (ASSIGN initialization)?)* SEMI
    ;

readOnlyArrayDeclaration
    : typeSpecification IDENTIFIER (LBRACK indexRange RBRACK)? ASSIGN PCONTROL ASSIGN initialization (COMMA IDENTIFIER (LBRACK indexRange RBRACK)? ASSIGN PCONTROL ASSIGN initialization)* SEMI
    ;

equivalencedVarDeclaration
    : typeSpecification IDENTIFIER ASSIGN equivalencedReference (LBRACK expression RBRACK)? offsetSpec? (COMMA IDENTIFIER ASSIGN equivalencedReference (LBRACK expression RBRACK)? offsetSpec?)* SEMI
    ;

equivalencedReference: IDENTIFIER | SGCONTROL | GCONTROL | LCONTROL | SCONTROL;
offsetSpec: (PLUS | MINUS) expression;

literalDeclaration
    : LITERAL IDENTIFIER ASSIGN expression (COMMA IDENTIFIER ASSIGN expression)* SEMI
    ;

// ----------------------
// Enhanced Type System
// ----------------------
typeSpec
    : baseType                        #baseTypeSpec
    | arrayType                       #arrayTypeSpec
    | unionType                       #unionTypeSpec
    ;

unionType
    : UNION baseType (COMMA baseType)*
    ;

typeSpecification
    : dataType
    | forwardTypeName
    ;

baseType
    : simpleType
    | IDENTIFIER
    | stringType
    | recordType
    | pointerType
    | LPAREN typeSpec RPAREN
    ;

simpleType
    : INT
    | UINT
    | SHORT
    | USHORT
    | LONG
    | ULONG
    | BOOL
    | CHAR
    | BYTE
    ;

pointerType
    : REF typeSpec
    | BXOR typeSpec
    ;

arrayType
    : baseType LBRACK expression? RBRACK
    ;

stringType
    : STRING (LBRACK expression RBRACK)?
    | STRING LPAREN expression RPAREN
    ;

recordType
    : RECORD fieldDecl+ END
    ;
fieldDecl
    : identList COLON typeSpec SEMI
    ;

// Enhanced TAL data type support with all variations
dataType
    : STRING                                                          #stringDataType
    | STRING LPAREN INTEGER_VALUE RPAREN                            #sizedStringDataType  
    | INT                                                            #intDataType
    | INT32                                                          #int32DataType
    | INT64                                                          #int64DataType
    | FIXED (LPAREN INTEGER_VALUE (COMMA INTEGER_VALUE)? RPAREN)?   #fixedDataType
    | REAL                                                           #realDataType
    | REAL64                                                         #real64DataType
    | UNSIGNED LPAREN (INT_LITERAL | INTEGER_VALUE | TAL_LIT_BINARY | TAL_LIT_OCTAL | TAL_LIT_HEX) RPAREN #unsignedDataType
    | BYTE                                                           #byteDataType
    | CHAR                                                           #charDataType
    | TIMESTAMP                                                      #timestampDataType
    | EXTADDR                                                        #extaddrDataType
    | SGADDR                                                         #sgaddrDataType
    | TEMPLATE LPAREN IDENTIFIER RPAREN                            #templateDataType
    | REAL32                                                         #real32DataType
    | BOOLEAN                                                        #booleanDataType
    ;

forwardTypeName: IDENTIFIER;
indirection: PERIOD | EXTINDIRECT | SGINDIRECT | DOT_EXT | DOT_SG;
structureReferral: LPAREN IDENTIFIER RPAREN;
indexRange: lowerBound COLON upperBound;
lowerBound: expression;
upperBound: expression;

initialization: expression | constantList;
constantList: (repetitionFactor MUL)? LBRACK constantListItem (COMMA constantListItem)* RBRACK;
constantListItem: constantExpr | STRING_LITERAL;
repetitionFactor: expression;

identList
    : IDENTIFIER (COMMA IDENTIFIER)*
    ;

// ----------------------
// Enhanced Procedures
// ----------------------

procedureDeclaration
    : procHeader SEMI procBody?
    ;

procHeader
    : typedProcHeader
    | untypedProcHeader
    ;

typedProcHeader 
    : typeSpecification (PROC_UPPER | PROC_LOWER) procName formalParamList? procAttributeList?
    ;

untypedProcHeader 
    : (PROC_UPPER | PROC_LOWER | SUBPROC_UPPER | SUBPROC_LOWER) procName formalParamList? procAttributeList?
    ;

procName
    : IDENTIFIER  
    ;

// Enhanced parameter parsing with better error recovery
formalParamList
    : LPAREN RPAREN                                    #emptyParams
    | LPAREN formalParam (COMMA formalParam)* RPAREN  #nonEmptyParams
    ;

// Restructured formalParam to avoid conflicts - order matters for disambiguation
formalParam
    : VAR identList COLON typeSpec                     #varParameter
    | REF identList COLON typeSpec                     #refParameter  
    | dataType pointerOrIndirect IDENTIFIER structureReferral? #typedPointerParam
    | dataType IDENTIFIER (LBRACK expression? RBRACK)?         #typedParam
    | STRUCT IDENTIFIER IDENTIFIER                     #structParameter
    | IDENTIFIER pointerOrIndirect IDENTIFIER structureReferral? #forwardPointerParam
    | IDENTIFIER IDENTIFIER                            #forwardParam
    | IDENTIFIER                                       #simpleParam
    | MUL                                             #anyParam
    ;

pointerOrIndirect
    : MUL | PERIOD | DOT_EXT | DOT_SG
    ;

procAttributeList: procAttribute (COMMA procAttribute)*;

procAttribute
    : MAIN
    | INTERRUPT
    | RESIDENT
    | CALLABLE
    | PRIV
    | VARIABLE
    | EXTENSIBLE (LBRACE INTEGER_VALUE RBRACE)?
    | LANGUAGE IDENTIFIER  
    | FORWARD
    | EXTERNAL
    | SUBPROC_SPECIFIC
    | SHARED
    | REENTRANT
    | SAVEABEND
    ;

// Enhanced procedure body handling with error recovery
procBody
    : procBodyContent
    | FORWARD SEMI
    | EXTERNAL SEMI
    | errorRecovery
    ;

errorRecovery
    : SEMI  // Simple error recovery - consume semicolon and continue
    ;

procBodyContent
    : (localDeclarationStatement)*
      BEGIN
      (statement | errorStatement)*
      END SEMI?
    ;

errorStatement
    : SEMI  // Simplified error recovery within statements
    ;

// Enhanced statement handling within procedures
statementWithComments
    : inlineComment* statement inlineComment*
    ;

inlineComment
    : TAL_INLINE_COMMENT
    | TAL_BLOCK_COMMENT
    ;

// Simplified declaration handling for procedure context  
declarationOrStatement
    : globalDeclarationItem
    | labelDeclaration
    | entryPointDeclaration
    | subprocedureDeclaration
    | forwardSubprocedureDeclaration
    ;

// Case-insensitive subprocedure handling
subprocedureDeclaration: (SUBPROC_UPPER | SUBPROC_LOWER) procName formalParamList? SEMI procBody;
entryPointDeclaration: ENTRY IDENTIFIER SEMI;
labelDeclaration: LABEL IDENTIFIER (COMMA IDENTIFIER)* SEMI;
forwardSubprocedureDeclaration: (SUBPROC_UPPER | SUBPROC_LOWER) procName formalParamList? SEMI FORWARD SEMI;

statementList: statement+;

// ----------------------
// Forward and External Declarations
// ----------------------
forwardDeclaration
    : FORWARD (PROC_UPPER | PROC_LOWER) procName formalParamList? SEMI            #forwardProcDeclaration
    | FORWARD STRUCT IDENTIFIER SEMI                                              #forwardStructDeclaration
    | FORWARD IDENTIFIER SEMI                                                     #forwardTypeDeclaration
    | FORWARD typeDeclaration SEMI                                                #forwardTypeDeclWithType
    ;

typeDeclaration: dataType IDENTIFIER | forwardTypeName IDENTIFIER;

externalDeclaration
    : EXTERNAL (PROC_UPPER | PROC_LOWER) IDENTIFIER languageSpecifier? SEMI
    | EXTERNAL STRUCT IDENTIFIER languageSpecifier? SEMI
    | EXTERNAL identList (COLON typeSpec)? SEMI
    ;

languageSpecifier: LANGUAGE languageNameChoice;
languageNameChoice: IDENTIFIER | COBOL85 | FORTRAN | PASCAL | UNSPECIFIED | C | TAL;

// ----------------------
// Module Import
// ----------------------
moduleImport: IMPORT moduleIdentifier SEMI | IMPORT moduleIdentifier LPAREN importedItems RPAREN SEMI;
moduleIdentifier: IDENTIFIER (PERIOD IDENTIFIER)*;
importedItems: IDENTIFIER (COMMA IDENTIFIER)* | MUL;

// ----------------------
// Enhanced Statements with Error Recovery
// ----------------------
statement
    : assignmentStatement SEMI                       #assignStmt
    | localDeclarationStatement                      #localDeclStmt
    | bitDepositStatement                            #bitDepositStmt
    | bitFieldAssignmentStatement SEMI               #bitFieldAssignStmt
    | pointerAssignmentStatement SEMI                #pointerAssignStmt
    | pointerDereferenceStatement SEMI               #pointerDerefStmt
    | stringMoveStatement SEMI                       #stringMoveStmt
    | moveStatement SEMI                             #moveStmt
    | scanStatement SEMI                             #scanStmt
    | rscanStatement SEMI                            #rscanStmt
    | callStatement SEMI                             #callStmt
    | ifStatement                                    #ifStmt
    | caseStatement                                  #caseStmt
    | whileStatement                                 #whileStmt
    | doUntilStatement SEMI                          #doUntilStmt
    | forStatement                                   #forStmt
    | gotoStatement SEMI                             #gotoStmt
    | returnStatement                                #returnStmt
    | assertStatement SEMI                           #assertStmt
    | useStatement SEMI                              #useStmt
    | dropStatement SEMI                             #dropStmt
    | stackStatement SEMI                            #stackStmt
    | storeStatement SEMI                            #storeStmt
    | codeStatement SEMI                             #codeStmt
    | labeledStatement                               #labeledStmt
    | expressionStatement SEMI                       #exprStmt
    | blockStatement                                 #nestedBlockStmt
    | SEMI                                           #emptyStmt
    | block                                          #blockStmt
    | varSection                                     #localVarSection
    | constSection                                   #localConstSection
    | typeSection                                    #localTypeSection
    | fileOperationStatement SEMI                    #fileOpStmt
    | trapHandlingStatement SEMI                     #trapStmt
    | processControlStatement SEMI                   #processStmt
    | guardianFileStatement SEMI                     #guardianFileStmt
    | enscribeFileStatement SEMI                     #enscribeFileStmt
    | talSqlStatement                                #talSqlStmt
    ;

// Guardian file operations
guardianFileStatement
    : guardianOpenStatement
    | guardianCloseStatement  
    | guardianReadStatement
    | guardianWriteStatement
    | guardianControlStatement
    ;

guardianOpenStatement
    : OPEN LPAREN fileExpr (COMMA openFlags)? RPAREN
    ;

guardianCloseStatement
    : CLOSE LPAREN fileExpr RPAREN
    ;

guardianReadStatement
    : READ LPAREN fileExpr COMMA variableExpr (COMMA expression)? RPAREN
    ;

guardianWriteStatement
    : WRITE LPAREN fileExpr COMMA expression (COMMA expression)? RPAREN
    ;

guardianControlStatement
    : CONTROL LPAREN fileExpr COMMA expression (COMMA parameterList)? RPAREN
    ;

openFlags
    : openFlag (COMMA openFlag)*
    ;

openFlag
    : expression
    | IDENTIFIER
    ;

// Enscribe file operations
enscribeFileStatement
    : enscribeOpenStatement
    | enscribeCloseStatement
    | enscribeReadStatement  
    | enscribeWriteStatement
    | enscribePositionStatement
    | enscribeKeyStatement
    ;

enscribeOpenStatement
    : OPEN fileExpr (FOR openMode)? (ACCESS accessMode)? (SHARE shareMode)? (NOWAIT)?
    ;

enscribeCloseStatement
    : CLOSE fileExpr (NOWAIT)?
    ;

enscribeReadStatement
    : READ fileExpr INTO variableExpr (maxLength)? (actualLength)? (NOWAIT)?
    ;

enscribeWriteStatement
    : WRITE expression TO fileExpr (actualLength)? (NOWAIT)?
    ;

enscribePositionStatement
    : POSITION fileExpr TO expression (NOWAIT)?
    ;

enscribeKeyStatement
    : KEYPOSITION fileExpr TO expression (NOWAIT)?
    ;

// Enhanced assignment statements for TAL with string moves
assignmentStatement
    : variableExpr ASSIGN expression                 #simpleAssign
    | variableExpr STRINGMOVE expression             #stringAssign
    | variableExpr LBRACK expression RBRACK STRINGMOVE expression #arrayStringAssign
    | variableExpr QUOTED_STRINGMOVE expression      #quotedStringAssign
    ;

expressionStatement
    : expression                                     #exprOnly
    ;

blockStatement
    : BEGIN statement* END                           #nestedBlock
    ;

block
    : BEGIN statement* END SEMI?                     #simpleBlock
    ;

// Enhanced bit field assignments with proper token handling
bitFieldAssignmentStatement
    : variableExpr BITFIELD_OPEN bitPosition (COLON bitPosition)? GT ASSIGN expression
    ;

// Enhanced pointer assignments for TAL with @ operator
pointerAssignmentStatement
    : ADDRESS variableExpr ASSIGN ADDRESS variableExpr
    | variableExpr ASSIGN ADDRESS variableExpr
    | ADDRESS variableExpr ASSIGN variableExpr
    | variableExpr (PERIOD IDENTIFIER | LBRACK expression RBRACK)+ ASSIGN ADDRESS variableExpr
    | variableExpr DOT_EXT IDENTIFIER ASSIGN ADDRESS variableExpr
    | variableExpr DOT_SG IDENTIFIER ASSIGN ADDRESS variableExpr
    ;

// Enhanced pointer dereference for TAL
pointerDereferenceStatement
    : PERIOD variableExpr ASSIGN expression
    | DOT_EXT variableExpr ASSIGN expression
    | DOT_SG variableExpr ASSIGN expression
    ;

// Enhanced string move statements
stringMoveStatement
    : variableExpr STRINGMOVE expression
    | variableExpr LBRACK expression RBRACK STRINGMOVE expression
    | variableExpr LBRACK expression RBRACK QUOTED_STRINGMOVE expression
    | MOVL_CALL LPAREN variableExpr COMMA expression COMMA expression RPAREN
    | MOVR_CALL LPAREN variableExpr COMMA expression COMMA expression RPAREN
    ;

moveStatement
    : MOVE variableExpr TO variableExpr
    | MOVE LPAREN variableExpr COMMA variableExpr COMMA expression RPAREN
    ;

// Enhanced bit deposit statements
bitDepositStatement
    : BITDEPOSIT expression TO variableExpr BITFIELD_OPEN bitPosition (COLON bitPosition)? GT SEMI
    ;

// Enhanced scan statements with proper comparison operators and arrow syntax
scanStatement
    : SCAN scanObject WHILE scanTerminator TO nextAddr
    | SCAN scanObject WHILE scanTerminator ARROW nextAddr
    | SCAN scanObject WHILE comparisonOp scanTerminator ARROW nextAddr
    | SCAN scanObject UNTIL scanTerminator TO nextAddr
    | SCAN scanObject UNTIL scanTerminator ARROW nextAddr
    ;

comparisonOp: NEQ | SIMPLE_EQ | LT | GT | LE | GE;

rscanStatement
    : RSCAN scanObject WHILE scanTerminator TO nextAddr
    | RSCAN scanObject WHILE scanTerminator ARROW nextAddr
    | RSCAN scanObject UNTIL scanTerminator TO nextAddr
    | RSCAN scanObject UNTIL scanTerminator ARROW nextAddr
    ;

// ----------------------
// Enhanced File I/O Operations
// ----------------------
fileOperationStatement
    : openStatement
    | closeStatement
    | readStatement
    | writeStatement
    | readxStatement
    | writexStatement
    | fileInfoStatement
    | awaitioStatement
    | setmodeStatement
    | unlockfileStatement
    | lockfileStatement
    ;

openStatement
    : OPEN fileExpr (FOR openMode)? (ACCESS accessMode)? (SHARE shareMode)? (NOWAIT)?
    ;

closeStatement
    : CLOSE fileExpr (NOWAIT)?
    ;

readStatement
    : READ fileExpr INTO variableExpr (maxLength)? (actualLength)? (NOWAIT)?
    ;

writeStatement
    : WRITE expression TO fileExpr (actualLength)? (NOWAIT)?
    ;

readxStatement
    : READX fileExpr LPAREN parameterList RPAREN (INTO variableExpr)? (maxLength)? (actualLength)? (NOWAIT)?
    ;

writexStatement
    : WRITEX fileExpr LPAREN parameterList RPAREN (FROM expression)? (actualLength)? (NOWAIT)?
    ;

fileInfoStatement
    : FILEINFO fileExpr LPAREN parameterList RPAREN
    ;

awaitioStatement
    : AWAITIO fileExpr (conditionCode)?
    ;

setmodeStatement
    : SETMODE fileExpr LPAREN parameterList RPAREN
    ;

unlockfileStatement
    : UNLOCKFILE fileExpr
    ;

lockfileStatement
    : LOCKFILE fileExpr
    ;

openMode: INPUT | OUTPUT | SHARED | EDIT | APPEND | UPDATE;
accessMode: RANDOM | SEQUENTIAL | KEY_SEQUENTIAL | BROWSE;
shareMode: SHARE | EXCLUSIVE | SHARED;
fileExpr: expression;
maxLength: COMMA expression;
actualLength: COMMA variableExpr;
conditionCode: COMMA variableExpr;

// ----------------------
// Enhanced Process Control Statements
// ----------------------
processControlStatement
    : processCreateStatement
    | processStopStatement
    | processActivateStatement
    | processDebugStatement
    | processMonitorStatement
    | processWaitStatement
    | processSignalStatement
    ;

processCreateStatement
    : PROCESS_CREATE LPAREN processCreateParams RPAREN
    ;

processStopStatement
    : PROCESS_STOP LPAREN processStopParams RPAREN
    ;

processActivateStatement
    : PROCESS_ACTIVATE LPAREN processActivateParams RPAREN
    ;

processDebugStatement
    : PROCESS_DEBUG LPAREN processDebugParams RPAREN
    ;

processMonitorStatement
    : PROCESS_MONITOR LPAREN processMonitorParams RPAREN
    ;

processWaitStatement
    : PROCESS_WAIT LPAREN processWaitParams RPAREN
    ;

processSignalStatement
    : PROCESS_SIGNAL LPAREN processSignalParams RPAREN
    ;

processCreateParams: parameterList;
processStopParams: parameterList;
processActivateParams: parameterList;
processDebugParams: parameterList;
processMonitorParams: parameterList;
processWaitParams: parameterList;
processSignalParams: parameterList;

// ----------------------
// Enhanced Trap Handling
// ----------------------
trapHandlingStatement
    : trapSetStatement
    | trapClearStatement
    | trapEnableStatement
    | trapDisableStatement
    | trapCallStatement
    ;

trapSetStatement
    : TRAP_SET LPAREN trapNumber COMMA trapHandler RPAREN
    ;

trapClearStatement
    : TRAP_CLEAR LPAREN trapNumber RPAREN
    ;

trapEnableStatement
    : TRAP_ENABLE LPAREN trapMask? RPAREN
    ;

trapDisableStatement
    : TRAP_DISABLE LPAREN trapMask? RPAREN
    ;

trapCallStatement
    : TRAP_CALL LPAREN trapNumber COMMA parameterList? RPAREN
    ;

trapNumber: expression;
trapHandler: IDENTIFIER;
trapMask: expression;

scanObject: variableExpr;
scanTerminator: expression | STRING_LITERAL | CHAR_LITERAL;
nextAddr: expression;
bitPosition: expression;

// Local declaration statements
localDeclarationStatement
    : constSection
    | typeSection
    | varSection
    | labelDeclaration
    | simpleVariableDeclaration
    | talPointerDeclaration
    | structureDeclaration
    ;

// Enhanced control flow statements with inline comment support and error recovery
ifStatement
    : IF expression THEN statementSequence (ELSE statementSequence)? ENDIF SEMI
    | IF expression THEN statement (ELSE statement)?  // Single statement form
    ;

caseStatement
    : CASE expression OF BEGIN caseArm* otherwiseArm? END SEMI
    ;

otherwiseArm: OTHERWISE COLON statement;
statementSequence: statement*;
caseArm: caseLabelList COLON statement;
caseLabelList: caseLabel (COMMA caseLabel)*;
caseLabel: INT_LITERAL | CHAR_LITERAL | STRING_LITERAL | INTEGER_VALUE | IDENTIFIER;

// Enhanced while statements
whileStatement
    : WHILE expression DO BEGIN statement* END SEMI?
    | WHILE expression DO statement
    ;

doUntilStatement: DO statementSequence UNTIL expression;

// Enhanced for statements
forStatement
    : FOR IDENTIFIER ASSIGN initialValue direction limitValue (BY stepValue)? DO statementSequence ENDFOR SEMI?
    ;
initialValue: expression;
direction: TO | DOWNTO;
limitValue: expression;
stepValue: expression;

// Enhanced procedure calls with system functions
callStatement
    : CALL? (procedureNameCall | systemProcedureCall) (LPAREN callParameters? RPAREN)?
    | CALL? qualifiedName LPAREN argList? RPAREN
    | variableExpr ASSIGN functionCall
    | CALL DOLLAR IDENTIFIER LPAREN argList? RPAREN
    | CALL guardianSystemCall
    | CALL enscribeSystemCall
    ;

guardianSystemCall
    : GUARDIAN_PROCEDURE_CALL LPAREN callParameters? RPAREN
    ;

enscribeSystemCall
    : ENSCRIBE_PROCEDURE_CALL LPAREN callParameters? RPAREN
    ;

systemProcedureCall
    : INITIALIZER
    | PROCESS_CREATE
    | PROCESS_STOP  
    | PROCESS_ACTIVATE
    | PROCESS_DEBUG
    | PROCESS_GETINFO
    | PROCESS_GETINFOLIST
    | FILE_OPEN
    | FILE_CLOSE
    | FILE_READ
    | FILE_WRITE
    | MEMORY_ALLOCATE
    | MEMORY_DEALLOCATE
    | MESSAGE_SEND
    | MESSAGE_RECEIVE
    | TIME_GET
    | TIMER_START
    | TIMER_CANCEL
    ;

procedureNameCall: IDENTIFIER;

callParameters: callParameter (COMMA callParameter)*;
callParameter: expression | IDENTIFIER | MUL;
argList: expression (COMMA expression)*;

// Other Statements
gotoStatement: GOTO IDENTIFIER;

// Enhanced return statements with bit field support
returnStatement
    : RETURN (expression (COMMA expression)?)? SEMI
    ;

// Enhanced assert statements with levels
assertStatement: ASSERT assertLevel? expression;
assertLevel: (INT_LITERAL | INTEGER_VALUE) COLON;

useStatement: USE identList;
dropStatement: DROP identList;
stackStatement: STACK expressionList;
storeStatement: STORE variableList;

codeStatement: CODE LPAREN machineCode RPAREN;

machineCode: (machineMnemonic (machineOperand (COMMA machineOperand)*)? SEMI?)+;
machineMnemonic: IDENTIFIER | CON | FULL;
machineOperand: (INT_LITERAL | INTEGER_VALUE | TAL_LIT_BINARY | TAL_LIT_OCTAL | TAL_LIT_HEX) | IDENTIFIER | ADDRESS IDENTIFIER | PERIOD IDENTIFIER | STRING_LITERAL;

labeledStatement: IDENTIFIER COLON statement;

// ----------------------
// Enhanced Compiler Directives
// ----------------------
directiveLine
    : QUESTION_MARK sourceDirective SEMI?
    | QUESTION_MARK listingDirective SEMI
    | QUESTION_MARK pageDirective SEMI
    | QUESTION_MARK sectionDirective SEMI
    | QUESTION_MARK ifDirective
    | QUESTION_MARK compilerOptionDirective SEMI
    | QUESTION_MARK precompiledHeaderImport
    | QUESTION_MARK SYMBOLS SEMI
    | QUESTION_MARK IDENTIFIER SEMI
    | QUESTION_MARK directiveArgument (COMMA directiveArgument)* (LPAREN directiveArgumentList RPAREN)? SEMI
    | QUESTION_MARK creDirective SEMI
    | QUESTION_MARK heapDirective SEMI
    | QUESTION_MARK envDirective SEMI
    | QUESTION_MARK searchDirective SEMI
    | QUESTION_MARK largestackDirective SEMI
    | QUESTION_MARK toggleDirective SEMI
    | QUESTION_MARK errorDirective SEMI
    | QUESTION_MARK warningDirective SEMI
    ;

errorDirective
    : ERROR STRING_LITERAL
    | ERROR IDENTIFIER
    ;

warningDirective
    : WARNING STRING_LITERAL
    | WARNING IDENTIFIER
    ;

sourceDirective
    : SOURCE ((ASSIGN | COMMA)? STRING_LITERAL | (ASSIGN | COMMA)? IDENTIFIER (LPAREN directiveSourceParams RPAREN)? | INCLUDE STRING_LITERAL)
    | SOURCE ASSIGN directiveSourceExpression
    ;

directiveSourceExpression
    : IDENTIFIER (LPAREN directiveSourceParams? RPAREN)?
    ;

directiveSourceParams
    : directiveSourceParam (COMMA directiveSourceParam)*
    ;

directiveSourceParam
    : IDENTIFIER
    | QUESTION_MARK IDENTIFIER
    ;

listingDirective: LIST | NOLIST;
pageDirective:
    PAGE STRING_LITERAL?                           # basicPage
    | PAGE PAGESKIP                                # pageSkip
    | PAGE EJECT                                   # pageEject
    | PAGE HEADER STRING_LITERAL                   # pageHeader
    | PAGE FOOTER STRING_LITERAL                   # pageFooter
    | PAGE LINES INTEGER_VALUE                     # pageLines
    | PAGE SIZE INTEGER_VALUE COMMA INTEGER_VALUE  # pageSize
    ;
sectionDirective: SECTION IDENTIFIER?;
ifDirective: IF directiveExpression SEMI | IFNOT directiveExpression SEMI | ENDIF SEMI;
directiveExpression: IDENTIFIER (ASSIGN | NEQ) (STRING_LITERAL | INT_LITERAL | INTEGER_VALUE | TAL_LIT_BINARY | TAL_LIT_OCTAL | TAL_LIT_HEX | IDENTIFIER) | IDENTIFIER;
compilerOptionDirective: COMPACT | CHECK | INSPECT | SYMBOLS | NOLMAP | HIGHPIN | HIGHREQUESTERS | CROSSREF | GMAP | INNERLIST | NOCODE | NOMAP | LMAP;

creDirective: CRE | NOCRE;
heapDirective: HEAP expression;
envDirective: ENV IDENTIFIER;
searchDirective: SEARCH LPAREN directiveArgumentList RPAREN;
largestackDirective: LARGESTACK expression;
toggleDirective: SETTOG IDENTIFIER | RESETTOG IDENTIFIER | DEFINETOG IDENTIFIER | TOG IDENTIFIER;

precompiledHeaderImport: PCH STRING_LITERAL SEMI;
directiveArgument: IDENTIFIER | INT_LITERAL | INTEGER_VALUE | STRING_LITERAL | TAL_LIT_BINARY | TAL_LIT_OCTAL | TAL_LIT_HEX;
directiveArgumentList: directiveArgument (COMMA directiveArgument)*;

// ----------------------
// Enhanced Expressions
// ----------------------
expression: conditionalExpr;

conditionalExpr: logicalOrExpr (QUESTION_MARK logicalOrExpr COLON logicalOrExpr)?;
logicalOrExpr: logicalAndExpr (OR logicalAndExpr)*;
logicalAndExpr: bitwiseOrExpr (AND bitwiseOrExpr)*;
bitwiseOrExpr: bitwiseXorExpr (BOR bitwiseXorExpr)*;
bitwiseXorExpr: bitwiseAndExpr (BXOR bitwiseAndExpr)*;
bitwiseAndExpr: equalityExpr (BAND equalityExpr)*;
equalityExpr: relationalExpr ((EQ | NEQ | SIMPLE_EQ) relationalExpr)*;
relationalExpr: shiftExpr ((LT | GT | LE | GE) shiftExpr)*;
shiftExpr: additiveExpr ((SHL | SHR) additiveExpr)*;
additiveExpr: multiplicativeExpr ((PLUS | MINUS) multiplicativeExpr)*;
multiplicativeExpr: unaryExpr ((MUL | DIV | MOD) unaryExpr)*;
unaryExpr: (PLUS | MINUS | NOT | BNOT | ADDRESS)? primaryExpr;

primaryExpr
    : constantExpr
    | variableExpr
    | functionCall
    | standardFunction
    | systemFunction
    | LPAREN expression RPAREN
    | sizeof
    | typeof
    | addressof
    ;

sizeof
    : SIZEOF LPAREN (expression | typeSpec) RPAREN
    ;

typeof
    : TYPEOF LPAREN expression RPAREN
    ;

addressof
    : ADDRESSOF LPAREN variableExpr RPAREN
    ;

// ----------------------
// Enhanced Standard Functions (TAL Built-ins)
// ----------------------
standardFunction
    : lenFunction
    | typeFunction
    | occursFunction
    | offsetFunction
    | highFunction
    | dblFunction
    | fixFunction
    | lfixFunction
    | floatFunction
    | udblFunction
    | emodFunction
    | imodFunction
    | minFunction
    | maxFunction
    | absFunction
    | sgnFunction
    | rotateFunction
    | shiftFunction
    | testbitFunction
    | setbitFunction
    | clearbitFunction
    | flipbitFunction
    | addressFunction
    | optionalFunction
    | paramFunction
    | timeFunction
    | scaleFunction
    | overflowFunction
    | carryFunction
    | conditionFunction
    ;

lenFunction: LEN_FUNC LPAREN expression RPAREN;
typeFunction: TYPE_FUNC LPAREN expression RPAREN;
occursFunction: OCCURS_FUNC LPAREN expression RPAREN;
offsetFunction: OFFSET_FUNC LPAREN expression RPAREN;
highFunction: HIGH_FUNC LPAREN expression RPAREN;
dblFunction: DBL_FUNC LPAREN expression RPAREN;
fixFunction: FIX_FUNC LPAREN expression RPAREN;
lfixFunction: LFIX_FUNC LPAREN expression RPAREN;
floatFunction: FLOAT_FUNC LPAREN expression RPAREN;
udblFunction: UDBL_FUNC LPAREN expression RPAREN;
absFunction: ABS_FUNC LPAREN expression RPAREN;
sgnFunction: SGN_FUNC LPAREN expression RPAREN;
addressFunction: ADDRESS_FUNC LPAREN variableExpr RPAREN;
optionalFunction: OPTIONAL_FUNC LPAREN expression RPAREN;
paramFunction: PARAM_FUNC LPAREN expression RPAREN;
emodFunction: EMOD_FUNC LPAREN expression COMMA expression RPAREN;
imodFunction: IMOD_FUNC LPAREN expression COMMA expression RPAREN;
minFunction: MIN_FUNC LPAREN expression COMMA expression RPAREN;
maxFunction: MAX_FUNC LPAREN expression COMMA expression RPAREN;
rotateFunction: ROTATE_FUNC LPAREN expression COMMA expression RPAREN;
shiftFunction: SHIFT_FUNC LPAREN expression COMMA expression RPAREN;
testbitFunction: TESTBIT_FUNC LPAREN expression COMMA expression RPAREN;
setbitFunction: SETBIT_FUNC LPAREN expression COMMA expression RPAREN;
clearbitFunction: CLEARBIT_FUNC LPAREN expression COMMA expression RPAREN;
flipbitFunction: FLIPBIT_FUNC LPAREN expression COMMA expression RPAREN;
scaleFunction: SCALE_FUNC LPAREN expression COMMA expression RPAREN;
overflowFunction: OVERFLOW_FUNC LPAREN expression RPAREN;
carryFunction: CARRY_FUNC LPAREN expression RPAREN;
conditionFunction: CONDITION_FUNC LPAREN expression RPAREN;

// ----------------------
// Enhanced System Functions
// ----------------------
systemFunction
    : asciiFunction
    | numericFunction
    | bitFunction
    | conversionFunction
    | stringFunction
    | mathFunction
    | guardianFunction
    | enscribeFunction
    ;

guardianFunction
    : GUARDIAN_FUNC LPAREN parameterList RPAREN
    | CONTROL_FUNC LPAREN parameterList RPAREN
    | FILENAME_FUNC LPAREN expression RPAREN
    ;

enscribeFunction
    : KEYPOSITION_FUNC LPAREN parameterList RPAREN
    | POSITION_FUNC LPAREN parameterList RPAREN
    | SETMODE_FUNC LPAREN parameterList RPAREN
    ;

mathFunction
    : SIN_FUNC LPAREN expression RPAREN
    | COS_FUNC LPAREN expression RPAREN
    | TAN_FUNC LPAREN expression RPAREN
    | SQRT_FUNC LPAREN expression RPAREN
    | LOG_FUNC LPAREN expression RPAREN
    | EXP_FUNC LPAREN expression RPAREN
    | ATAN_FUNC LPAREN expression RPAREN
    | ATAN2_FUNC LPAREN expression COMMA expression RPAREN
    ;

asciiFunction
    : ASCII_FUNC LPAREN expression RPAREN
    | NUMERIC_FUNC LPAREN expression RPAREN
    | ALPHA_FUNC LPAREN expression RPAREN
    | ALPHANUM_FUNC LPAREN expression RPAREN
    | UPSHIFT_FUNC LPAREN expression RPAREN
    | DOWNSHIFT_FUNC LPAREN expression RPAREN
    ;

numericFunction
    : SCALE_FUNC LPAREN expression COMMA expression RPAREN
    | OVERFLOW_FUNC LPAREN expression RPAREN
    | CARRY_FUNC LPAREN expression RPAREN
    | CONDITION_FUNC LPAREN expression RPAREN
    ;

bitFunction
    : BITLSHIFT_FUNC LPAREN expression COMMA expression RPAREN
    | BITRSHIFT_FUNC LPAREN expression COMMA expression RPAREN
    | BITAND_FUNC LPAREN expression COMMA expression RPAREN
    | BITOR_FUNC LPAREN expression COMMA expression RPAREN
    | BITXOR_FUNC LPAREN expression COMMA expression RPAREN
    | BITNOT_FUNC LPAREN expression RPAREN
    ;

conversionFunction
    : BINARY_FUNC LPAREN expression (COMMA expression)? RPAREN
    | OCTAL_FUNC LPAREN expression (COMMA expression)? RPAREN
    | HEX_FUNC LPAREN expression (COMMA expression)? RPAREN
    | DECIMAL_FUNC LPAREN expression (COMMA expression)? RPAREN
    ;

stringFunction
    : MOVL_FUNC LPAREN expression COMMA expression COMMA expression RPAREN
    | MOVR_FUNC LPAREN expression COMMA expression COMMA expression RPAREN
    | FILL_FUNC LPAREN expression COMMA expression COMMA expression RPAREN
    | CMPL_FUNC LPAREN expression COMMA expression COMMA expression RPAREN
    | CMPR_FUNC LPAREN expression COMMA expression COMMA expression RPAREN
    | SCANL_FUNC LPAREN expression COMMA expression COMMA expression RPAREN
    | SCANR_FUNC LPAREN expression COMMA expression COMMA expression RPAREN
    ;

timeFunction
    : TIME_FUNC LPAREN RPAREN
    | JULIANTIMESTAMP_FUNC LPAREN expression RPAREN
    | TIMESTAMP_FUNC LPAREN expression RPAREN
    | CONTIME_FUNC LPAREN expression RPAREN
    | INTERPRETTIME_FUNC LPAREN expression RPAREN
    | DAYOFWEEK_FUNC LPAREN expression RPAREN
    ;

functionCall
    : IDENTIFIER LPAREN parameterList? RPAREN
    ;

constantExpr
    : INT_LITERAL | FIXED_LITERAL | REAL_LITERAL | STRING_LITERAL
    | IDENTIFIER
    | INTEGER_VALUE
    | TAL_LIT_BINARY
    | TAL_LIT_OCTAL
    | TAL_LIT_HEX
    | NIL
    | TRUE
    | FALSE
    ;

// Enhanced variable expressions with proper bit field handling
variableExpr
    : IDENTIFIER (arrayRef | memberAccess | bitField | functionArgs)*
    | standardIndirectSymbol IDENTIFIER (arrayRef | memberAccess | bitField | functionArgs)*
    | ADDRESS IDENTIFIER (arrayRef | memberAccess | bitField | functionArgs)*
    | systemGlobalAccess
    | guardianFileName
    | extendedIndirectAccess
    ;

extendedIndirectAccess
    : EXTINDIRECT IDENTIFIER (arrayRef | memberAccess | bitField | functionArgs)*
    | DOT_EXT IDENTIFIER (arrayRef | memberAccess | bitField | functionArgs)*
    | DOT_SG IDENTIFIER (arrayRef | memberAccess | bitField | functionArgs)*
    ;

arrayRef: LBRACK expression RBRACK;
memberAccess: PERIOD IDENTIFIER | DOT_EXT IDENTIFIER | DOT_SG IDENTIFIER;

// Fixed bit field handling with proper tokens
bitField: BITFIELD_OPEN expression (COLON expression)? GT;

functionArgs: LPAREN parameterList? RPAREN;

guardianFileName: IDENTIFIER (PERIOD IDENTIFIER)*;
systemGlobalAccess: IDENTIFIER;
standardIndirectSymbol: PERIOD;

parameterList: parameter (COMMA parameter)* | MUL;
parameter: expression | IDENTIFIER | MUL;

qualifiedName
    : IDENTIFIER (PERIOD IDENTIFIER)*
    ;

literal
    : INT_LITERAL
    | CHAR_LITERAL
    | STRING_LITERAL
    | FIXED_LITERAL
    | REAL_LITERAL
    | TAL_LIT_BINARY
    | TAL_LIT_OCTAL
    | TAL_LIT_HEX
    | TRUE
    | FALSE
    | NIL
    ;

// REMOVED the problematic error recovery rule entirely
// error: {notifyErrorListeners("syntax error"); recover(_input, null);} ;

// ----------------------
// ENHANCED LEXER RULES  
// ----------------------

// Fixed bit field tokens
BITFIELD_OPEN        : '.<';

// Fixed extended addressing tokens  
DOT_EXT              : '.EXT' | '.ext';
DOT_SG               : '.SG' | '.sg';

// TAL-specific preprocessed tokens
SYSFUNCSTRING        : '__TAL_SYS_FUNC_STRING__';
SYSFUNCPARAM         : '__TAL_SYS_FUNC_PARAM__';
SYSFUNCDISPLAY       : '__TAL_SYS_FUNC_DISPLAY__';
SYSFUNCWRITE         : '__TAL_SYS_FUNC_WRITE__';
SYSFUNCREAD          : '__TAL_SYS_FUNC_READ__';
SYSFUNCUPSHIFT       : '__TAL_SYS_FUNC_UPSHIFT__';
SYSFUNCDOWNSHIFT     : '__TAL_SYS_FUNC_DOWNSHIFT__';

STRINGASSIGN         : '__TAL_OP_STRING_ASSIGN__';

// Fixed TAL string move operators
STRINGMOVE           : '\'' ':=' '\'';
QUOTED_STRINGMOVE    : '\'\':=\'\'';

EXTINDIRECT          : '__TAL_OP_EXT_INDIRECT__';
SGINDIRECT           : '__TAL_OP_SG_INDIRECT__';

// Arrow operator for scan statements
ARROW                : '->';

MOVEREVASSIGN        : '__TAL_OP_MOVE_REV_ASSIGN_OP__';

// Enhanced TAL numeric literals with type prefixes
TAL_LIT_BINARY       : '%B' [01]+ | '%' [01]+;
TAL_LIT_OCTAL        : '%O' [0-7]+ | '%' [0-7]+;
TAL_LIT_HEX          : '%H' DIGIT_HEX+ | '%' DIGIT_HEX+;

// Enhanced Standard Functions - TAL Built-ins
LEN_FUNC             : '$LEN';
TYPE_FUNC            : '$TYPE';
OCCURS_FUNC          : '$OCCURS';
OFFSET_FUNC          : '$OFFSET';
HIGH_FUNC            : '$HIGH';
DBL_FUNC             : '$DBL';
FIX_FUNC             : '$FIX';
LFIX_FUNC            : '$LFIX';
FLOAT_FUNC           : '$FLOAT';
UDBL_FUNC            : '$UDBL';
EMOD_FUNC            : '$EMOD';
IMOD_FUNC            : '$IMOD';
MIN_FUNC             : '$MIN';
MAX_FUNC             : '$MAX';
ABS_FUNC             : '$ABS';
SGN_FUNC             : '$SGN';
ROTATE_FUNC          : '$ROTATE';
SHIFT_FUNC           : '$SHIFT';
TESTBIT_FUNC         : '$TESTBIT';
SETBIT_FUNC          : '$SETBIT';
CLEARBIT_FUNC        : '$CLEARBIT';
FLIPBIT_FUNC         : '$FLIPBIT';
ADDRESS_FUNC         : '$ADDRESS';
OPTIONAL_FUNC        : '$OPTIONAL';
PARAM_FUNC           : '$PARAM';
SCALE_FUNC           : '$SCALE';
OVERFLOW_FUNC        : '$OVERFLOW';
CARRY_FUNC           : '$CARRY';
CONDITION_FUNC       : '$CONDITION';

// Enhanced System Functions
ASCII_FUNC           : '$ASCII';
NUMERIC_FUNC         : '$NUMERIC';
ALPHA_FUNC           : '$ALPHA';
ALPHANUM_FUNC        : '$ALPHANUM';
UPSHIFT_FUNC         : '$UPSHIFT';
DOWNSHIFT_FUNC       : '$DOWNSHIFT';
BITLSHIFT_FUNC       : '$BITLSHIFT';
BITRSHIFT_FUNC       : '$BITRSHIFT';
BITAND_FUNC          : '$BITAND';
BITOR_FUNC           : '$BITOR';
BITXOR_FUNC          : '$BITXOR';
BITNOT_FUNC          : '$BITNOT';
BINARY_FUNC          : '$BINARY';
OCTAL_FUNC           : '$OCTAL';
HEX_FUNC             : '$HEX';
DECIMAL_FUNC         : '$DECIMAL';
MOVL_FUNC            : '$MOVL';
MOVR_FUNC            : '$MOVR';
FILL_FUNC            : '$FILL';
CMPL_FUNC            : '$CMPL';
CMPR_FUNC            : '$CMPR';
SCANL_FUNC           : '$SCANL';
SCANR_FUNC           : '$SCANR';
TIME_FUNC            : '$TIME';
JULIANTIMESTAMP_FUNC : '$JULIANTIMESTAMP';
TIMESTAMP_FUNC       : '$TIMESTAMP';
CONTIME_FUNC         : '$CONTIME';
INTERPRETTIME_FUNC   : '$INTERPRETTIME';
DAYOFWEEK_FUNC       : '$DAYOFWEEK';

// Enhanced Math Functions
SIN_FUNC             : '$SIN';
COS_FUNC             : '$COS';
TAN_FUNC             : '$TAN';
SQRT_FUNC            : '$SQRT';
LOG_FUNC             : '$LOG';
EXP_FUNC             : '$EXP';
ATAN_FUNC            : '$ATAN';
ATAN2_FUNC           : '$ATAN2';

// Guardian System Functions
GUARDIAN_FUNC        : '$GUARDIAN';
CONTROL_FUNC         : '$CONTROL';
FILENAME_FUNC        : '$FILENAME';

// Enscribe System Functions  
KEYPOSITION_FUNC     : '$KEYPOSITION';
POSITION_FUNC        : '$POSITION';
SETMODE_FUNC         : '$SETMODE';

// Enhanced System Procedures
INITIALIZER          : 'INITIALIZER';
PROCESS_CREATE       : 'PROCESS_CREATE';
PROCESS_STOP         : 'PROCESS_STOP';
PROCESS_ACTIVATE     : 'PROCESS_ACTIVATE';
PROCESS_DEBUG        : 'PROCESS_DEBUG';
PROCESS_GETINFO      : 'PROCESS_GETINFO';
PROCESS_GETINFOLIST  : 'PROCESS_GETINFOLIST';
PROCESS_MONITOR      : 'PROCESS_MONITOR';
PROCESS_WAIT         : 'PROCESS_WAIT';
PROCESS_SIGNAL       : 'PROCESS_SIGNAL';
FILE_OPEN            : 'FILE_OPEN';
FILE_CLOSE           : 'FILE_CLOSE';
FILE_READ            : 'FILE_READ';
FILE_WRITE           : 'FILE_WRITE';
MEMORY_ALLOCATE      : 'MEMORY_ALLOCATE';
MEMORY_DEALLOCATE    : 'MEMORY_DEALLOCATE';
MESSAGE_SEND         : 'MESSAGE_SEND';
MESSAGE_RECEIVE      : 'MESSAGE_RECEIVE';
TIMER_START          : 'TIMER_START';
TIMER_CANCEL         : 'TIMER_CANCEL';
TIME_GET             : 'TIME_GET';

// Guardian and Enscribe procedure calls
GUARDIAN_PROCEDURE_CALL : 'GUARDIAN_PROCEDURE_CALL';
ENSCRIBE_PROCEDURE_CALL : 'ENSCRIBE_PROCEDURE_CALL';

// Enhanced Trap Handling
TRAP                 : 'TRAP';
TRAP_SET             : 'TRAP_SET';
TRAP_CLEAR           : 'TRAP_CLEAR';
TRAP_ENABLE          : 'TRAP_ENABLE';
TRAP_DISABLE         : 'TRAP_DISABLE';
TRAP_CALL            : 'TRAP_CALL';

// Enhanced File I/O Keywords
INPUT                : 'INPUT';
OUTPUT               : 'OUTPUT';
SHARED               : 'SHARED';
EDIT                 : 'EDIT';
APPEND               : 'APPEND';
UPDATE               : 'UPDATE';
ACCESS               : 'ACCESS';
SHARE                : 'SHARE';
EXCLUSIVE            : 'EXCLUSIVE';
RANDOM               : 'RANDOM';
SEQUENTIAL           : 'SEQUENTIAL';
KEY_SEQUENTIAL       : 'KEY-SEQUENTIAL';
BROWSE               : 'BROWSE';
NOWAIT               : 'NOWAIT';
INTO                 : 'INTO';
FROM                 : 'FROM';
CONTROL              : 'CONTROL';

// SQL Keywords
EXEC                 : 'EXEC';
SQL                  : 'SQL';
SELECT               : 'SELECT';
INSERT               : 'INSERT';
DELETE               : 'DELETE';
COMMIT               : 'COMMIT';
ROLLBACK             : 'ROLLBACK';
FETCH                : 'FETCH';
CURSOR               : 'CURSOR';
DECLARE              : 'DECLARE';
VALUES               : 'VALUES';
SET                  : 'SET';
WHERE                : 'WHERE';
ORDER                : 'ORDER';
GROUP                : 'GROUP';
HAVING               : 'HAVING';
UNION                : 'UNION';
DISTINCT             : 'DISTINCT';
ALL                  : 'ALL';
AS                   : 'AS';
ASC                  : 'ASC';
DESC                 : 'DESC';
IN                   : 'IN';
BETWEEN              : 'BETWEEN';
LIKE                 : 'LIKE';
IS                   : 'IS';
NULL                 : 'NULL';
EXISTS               : 'EXISTS';
ANY                  : 'ANY';
SOME                 : 'SOME';
WORK                 : 'WORK';

// File System Keywords
GUARDIAN             : 'GUARDIAN';
ENSCRIBE             : 'ENSCRIBE';
RECORD_SIZE          : 'RECORD-SIZE';
BLOCK_SIZE           : 'BLOCK-SIZE';
FILE_TYPE            : 'FILE-TYPE';
ACCESS_METHOD        : 'ACCESS-METHOD';
ORGANIZATION         : 'ORGANIZATION';

// Additional built-in functions
SIZEOF               : 'SIZEOF';
TYPEOF               : 'TYPEOF';
ADDRESSOF            : 'ADDRESSOF';

// Call statement helpers
MOVL_CALL            : 'MOVL';
MOVR_CALL            : 'MOVR';

// Case-sensitive keywords for procedures
PROC_UPPER           : 'PROC';
PROC_LOWER           : 'proc';
SUBPROC_UPPER        : 'SUBPROC';
SUBPROC_LOWER        : 'subproc';
SUBPROC_SPECIFIC     : 'SUBPROC_SPECIFIC';

// Enhanced File I/O keywords  
READ                 : 'READ' | 'read';  
WRITE                : 'WRITE' | 'write';
OPEN                 : 'OPEN' | 'open';
CLOSE                : 'CLOSE' | 'close';
AWAITIO              : 'AWAITIO' | 'awaitio';
READX                : 'READX' | 'readx';
WRITEX               : 'WRITEX' | 'writex';
FILEINFO             : 'FILEINFO' | 'fileinfo';
UNLOCKFILE           : 'UNLOCKFILE' | 'unlockfile';
LOCKFILE             : 'LOCKFILE' | 'lockfile';
POSITION             : 'POSITION' | 'position';
KEYPOSITION          : 'KEYPOSITION' | 'keyposition';
SETMODE              : 'SETMODE' | 'setmode';

// Other keywords that can be used as identifiers in certain contexts
BEGINNING            : 'beginning';
STOP                 : 'stop';

// Enhanced Keywords - TAL specific (order is important - specific before general)
ASSERT               : 'ASSERT';
BLOCK                : 'BLOCK';
BY                   : 'BY';
CALLABLE             : 'CALLABLE';
CHECK                : 'CHECK';
CODE                 : 'CODE';
COMPACT              : 'COMPACT';
COBOL85              : 'COBOL85';
CON                  : 'CON';
DEFINE               : 'DEFINE';
DOWNTO               : 'DOWNTO';
DROP                 : 'DROP';
ENDFOR               : 'ENDFOR';
ENDIF                : 'ENDIF';
ENTRY                : 'ENTRY';
EXTADDR              : 'EXTADDR';
EXTENSIBLE           : 'EXTENSIBLE';
FILLER               : 'FILLER';
FIXED                : 'FIXED';
FORTRAN              : 'FORTRAN';
FULL                 : 'FULL';
GOTO                 : 'GOTO';
HIGHPIN              : 'HIGHPIN';
HIGHREQUESTERS       : 'HIGHREQUESTERS';
IFNOT                : 'IFNOT';
INCLUDE              : 'INCLUDE';
IMPORT               : 'IMPORT';
INSPECT              : 'INSPECT';
INTERRUPT            : 'INTERRUPT';
INT32                : 'INT(32)';
INT64                : 'INT(64)';
LABEL                : 'LABEL';
LANGUAGE             : 'LANGUAGE';
LIST                 : 'LIST';
LITERAL              : 'LITERAL';
MAIN                 : 'MAIN';
NAME                 : 'NAME';
NIL                  : 'NIL';
NOLIST               : 'NOLIST';
NOLMAP               : 'NOLMAP';
OTHERWISE            : 'OTHERWISE';
PAGE                 : 'PAGE';
PAGESKIP             : 'SKIP';
EJECT                : 'EJECT';
HEADER               : 'HEADER';
FOOTER               : 'FOOTER';
LINES                : 'LINES';
SIZE                 : 'SIZE';
PASCAL               : 'PASCAL';
PCH                  : 'PCH';
PRAGMA               : 'PRAGMA';
PRIVATE              : 'PRIVATE';
PRIV                 : 'PRIV';
REAL                 : 'REAL';
REAL32               : 'REAL(32)';
REAL64               : 'REAL(64)';
RESIDENT             : 'RESIDENT';
RSCAN                : 'RSCAN';
SCAN                 : 'SCAN';
SECTION              : 'SECTION';
SGADDR               : 'SGADDR';
SOURCE               : 'SOURCE' | 'source';
STACK                : 'STACK';
STORE                : 'STORE';
STRUCT               : 'STRUCT';
SYMBOLS              : 'SYMBOLS';
TIMESTAMP            : 'TIMESTAMP';
TIME                 : 'TIME';
UNSPECIFIED          : 'UNSPECIFIED';
UNSIGNED             : 'UNSIGNED';
USE                  : 'USE';
MOVE                 : 'MOVE';
BITDEPOSIT           : 'BITDEPOSIT';
TEMPLATE             : 'TEMPLATE';
BOOLEAN              : 'BOOLEAN';
REENTRANT            : 'REENTRANT';
SAVEABEND            : 'SAVEABEND';
UNTIL                : 'UNTIL';

// Enhanced Compiler Directives
CRE                  : 'CRE';
NOCRE                : 'NOCRE';
HEAP                 : 'HEAP';
ENV                  : 'ENV';
SEARCH               : 'SEARCH';
LARGESTACK           : 'LARGESTACK';
SETTOG               : 'SETTOG';
RESETTOG             : 'RESETTOG';
DEFINETOG            : 'DEFINETOG';
TOG                  : 'TOG';
CROSSREF             : 'CROSSREF';
GMAP                 : 'GMAP';
INNERLIST            : 'INNERLIST';
NOCODE               : 'NOCODE';
NOMAP                : 'NOMAP';
LMAP                 : 'LMAP';
ERROR                : 'ERROR';
WARNING              : 'WARNING';

// Keywords - Common language constructs
PROCEDURE            : 'PROCEDURE';
RETURNS              : 'RETURNS';
OPTIONS              : 'OPTIONS';
EXTERNAL             : 'EXTERNAL';
CONST                : 'CONST';
TYPE                 : 'TYPE';
VAR                  : 'VAR';
RECORD               : 'RECORD';
BEGIN                : 'BEGIN';
END                  : 'END';
IF                   : 'IF';
THEN                 : 'THEN';
ELSE                 : 'ELSE';
CASE                 : 'CASE';
OF                   : 'OF';
WHILE                : 'WHILE';
DO                   : 'DO';
FOR                  : 'FOR';
TO                   : 'TO';
RETURN               : 'RETURN';
CALL                 : 'CALL';
REF                  : 'REF';
STRING               : 'STRING';
FORWARD              : 'FORWARD';
VARIABLE             : 'VARIABLE';

// Enhanced data types
INT                  : 'INT';
UINT                 : 'UINT';
SHORT                : 'SHORT';
USHORT               : 'USHORT';
LONG                 : 'LONG';
ULONG                : 'ULONG';
BOOL                 : 'BOOL';
CHAR                 : 'CHAR';
BYTE                 : 'BYTE';

// Enhanced language specifiers
C                    : 'C';
TAL                  : 'TAL';

// Boolean literals
TRUE                 : 'TRUE';
FALSE                : 'FALSE';

// Control tokens
PCONTROL             : '\'' 'P' '\'';
SGCONTROL            : '\'' 'SG' '\'';
GCONTROL             : '\'' 'G' '\'';
LCONTROL             : '\'' 'L' '\'';
SCONTROL             : '\'' 'S' '\'';

// Fixed operators - Token precedence matters!
ASSIGN               : ':=';
PLUS                 : '+';
MINUS                : '-';
MUL                  : '*';
DIV                  : '/';
MOD                  : 'MOD' | '%';

// Comparison operators - Fixed precedence
NEQ                  : '<>' | '!=' | 'NEQ';
EQ                   : '==';
SIMPLE_EQ            : '=';  // Must come after NEQ and EQ

LT                   : '<';
LE                   : '<=';
GT                   : '>';  // This handles the bit field close
GE                   : '>=';

SHL                  : 'SHL' | '<<';
SHR                  : 'SHR' | '>>';

AND                  : 'AND' | '&&';
OR                   : 'OR' | '||';
NOT                  : 'NOT' | '!';
BAND                 : '&';
BOR                  : '|';
BXOR                 : '^';
BNOT                 : '~';

// Delimiters
DOLLAR               : '$';
LPAREN               : '(';
RPAREN               : ')';
LBRACK               : '[';
RBRACK               : ']';
LBRACE               : '{';
RBRACE               : '}';
COMMA                : ',';
COLON                : ':';
SEMI                 : ';';
PERIOD               : '.';
HASH                 : '#';
QUESTION_MARK        : '?';  // Fixed from DIRECTIVE
ADDRESS              : '@';

// Enhanced literals with better precedence
INTEGER_VALUE        : DIGIT+;
INT_LITERAL          : ('%' [0-7]+)
                     | ('%' [Xx] DIGIT_HEX+)
                     | ('%' [Bb] DIGIT_BIN+)
                     | (DIGIT+ 'D')
                     | '0' | [1-9][0-9]* | '0'[xX][0-9a-fA-F]+
                     ;
FIXED_LITERAL        : DIGIT+ (PERIOD DIGIT+)? 'F' | '%' [Xx] DIGIT_HEX+ 'F';
REAL_LITERAL         : (DIGIT+ PERIOD DIGIT* | PERIOD DIGIT+ | DIGIT+) ([Ee] [+-]? DIGIT+)? 'L'?
                     | (DIGIT+ PERIOD DIGIT* | PERIOD DIGIT+ | DIGIT+) ([Ee] [+-]? DIGIT+)
                     ;
CHAR_LITERAL         : '\'' (~['\\\r\n] | '\\' . | '\'\'')* '\'';
STRING_LITERAL       : '"' (~["\\\r\n] | '\\' . | '""')* '"';

// Enhanced identifiers - Must come AFTER all keywords
IDENTIFIER           : LETTER (LETTER | DIGIT | '^')*;

// Fragment rules
fragment DIGIT       : [0-9];
fragment DIGIT_HEX   : [0-9a-fA-F];
fragment DIGIT_BIN   : [01];
fragment LETTER      : [a-zA-Z_];

// Enhanced comment handling with proper channels
TAL_BLOCK_COMMENT    : '!*' .*? '*!' -> channel(HIDDEN);
TAL_INLINE_COMMENT   : '!' ~[\r\n]* -> channel(HIDDEN);

// Line comments (C++ style, if supported)
LINE_COMMENT         : '//' ~[\r\n]* -> skip;

// Documentation comments (preserved on separate channel for tools)
DOC_COMMENT          : '/**' .*? '*/' -> channel(HIDDEN);

// Standard comments (skip these completely)
C_COMMENT            : '/*' .*? '*/' -> skip;

// Whitespace
WS                   : [ \t\r\n\f]+ -> skip;

// Enhanced error handling - must be last
UNEXPECTED_CHAR      : . ;

