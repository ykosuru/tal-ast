#!/bin/bash
# Quick Context Search - Convenience Wrapper
# Makes searching even easier!

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
INDEX="both"
MAX=5
FORMAT="text"

# Show help
show_help() {
    echo "Quick Context Search - Convenience Wrapper"
    echo ""
    echo "Usage: ./search.sh [options] \"query\""
    echo ""
    echo "Options:"
    echo "  -u, --universal     Search universal index only"
    echo "  -h, --hybrid        Search hybrid index only"
    echo "  -c, --code          Show only code files"
    echo "  -p, --pdf           Show only PDFs"
    echo "  -n NUM              Number of results (default: 5)"
    echo "  --html              Output as HTML"
    echo "  --json              Output as JSON"
    echo "  --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  ./search.sh \"payment validation\""
    echo "  ./search.sh --code \"OFAC screening\""
    echo "  ./search.sh --hybrid --html \"payment flow\""
    echo "  ./search.sh -n 10 \"ISO 20022\""
    exit 0
}

# Parse arguments
QUERY=""
FILE_TYPE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--universal)
            INDEX="universal"
            shift
            ;;
        -h|--hybrid)
            INDEX="hybrid"
            shift
            ;;
        -c|--code)
            FILE_TYPE="code"
            shift
            ;;
        -p|--pdf)
            FILE_TYPE="pdf"
            shift
            ;;
        -n)
            MAX="$2"
            shift 2
            ;;
        --html)
            FORMAT="html"
            shift
            ;;
        --json)
            FORMAT="json"
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            QUERY="$1"
            shift
            ;;
    esac
done

# Check if query provided
if [ -z "$QUERY" ]; then
    echo "Error: No query provided"
    echo "Usage: ./search.sh \"your query\""
    echo "Use --help for more options"
    exit 1
fi

# Build command
CMD="python3 quick_context_extractor_cli.py \"$QUERY\" --index $INDEX --max $MAX --format $FORMAT"

if [ -n "$FILE_TYPE" ]; then
    CMD="$CMD --file-type $FILE_TYPE"
fi

# Show what we're doing
echo -e "${BLUE}Searching for:${NC} $QUERY"
echo -e "${BLUE}Index:${NC} $INDEX"
echo -e "${BLUE}Max results:${NC} $MAX"
if [ -n "$FILE_TYPE" ]; then
    echo -e "${BLUE}File type:${NC} $FILE_TYPE"
fi
echo -e "${BLUE}Format:${NC} $FORMAT"
echo ""

# Run command
eval $CMD

# Show success message
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Search complete!${NC}"
    
    if [ "$FORMAT" = "html" ]; then
        OUTPUT_FILE="extracted_context.html"
        echo -e "${GREEN}Open in browser:${NC} $OUTPUT_FILE"
        # Try to open in browser
        if command -v xdg-open &> /dev/null; then
            xdg-open "$OUTPUT_FILE" 2>/dev/null
        elif command -v open &> /dev/null; then
            open "$OUTPUT_FILE" 2>/dev/null
        fi
    elif [ "$FORMAT" = "json" ]; then
        OUTPUT_FILE="extracted_context.json"
        echo -e "${GREEN}JSON saved:${NC} $OUTPUT_FILE"
    else
        OUTPUT_FILE="extracted_context.txt"
        echo -e "${GREEN}Results saved:${NC} $OUTPUT_FILE"
    fi
fi
