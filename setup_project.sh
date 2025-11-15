#!/bin/bash
# setup_project.sh - Automated project setup and verification

echo "======================================================================"
echo "Twitter Sentiment Analysis - Project Setup"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if folder exists
check_folder() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 (missing)"
        return 1
    fi
}

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${YELLOW}○${NC} $1 (will be created)"
        return 1
    fi
}

echo "Step 1: Checking Folder Structure"
echo "----------------------------------------------------------------------"

# Check main folders
check_folder "data"
check_folder "data/raw"
check_folder "data/processed"
check_folder "models"
check_folder "results"
check_folder "sentiment_analysis"
check_folder "venvtweet"

echo ""
echo "Step 2: Checking Data Files"
echo "----------------------------------------------------------------------"

check_file "data/raw/upset.csv"
check_file "data/unsmile_words.csv"
check_file ".env"

echo ""
echo "Step 3: Checking Scripts"
echo "----------------------------------------------------------------------"

check_file "sentiment_analysis/data_preprocessing.py"
check_file "sentiment_analysis/feature_extraction.py"
check_file "sentiment_analysis/train_multiclass_model.py"
check_file "sentiment_analysis/predict_multiclass.py"
check_file "sentiment_analysis/create_hybrid_dataset.py"

echo ""
echo "Step 4: Python Environment Check"
echo "----------------------------------------------------------------------"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✓${NC} Virtual environment is activated"
    echo "  Path: $VIRTUAL_ENV"
else
    echo -e "${YELLOW}○${NC} Virtual environment is NOT activated"
    echo "  Run: source venvtweet/Scripts/activate"
fi

# Check Python packages
echo ""
echo "Checking required packages..."

packages=("pandas" "numpy" "sklearn" "nltk" "matplotlib" "seaborn" "tweepy")

for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $package"
    else
        echo -e "${RED}✗${NC} $package (not installed)"
    fi
done

echo ""
echo "======================================================================"
echo "Setup Summary"
echo "======================================================================"
echo ""

# Count what's ready
if [ -f "data/raw/upset.csv" ] && [ -f "sentiment_analysis/create_hybrid_dataset.py" ]; then
    echo -e "${GREEN}Ready to create dataset!${NC}"
    echo "  → Run: cd sentiment_analysis && python create_hybrid_dataset.py"
    echo ""
fi

if [ -f "data/processed/multi_sentiment_dataset.csv" ]; then
    echo -e "${GREEN}Dataset exists!${NC}"
    echo "  → Run: cd sentiment_analysis && python train_multiclass_model.py"
    echo ""
fi

if [ -f "models/multiclass_model_logistic_regression.pkl" ]; then
    echo -e "${GREEN}Models trained!${NC}"
    echo "  → Run: cd sentiment_analysis && python predict_multiclass.py"
    echo ""
fi

echo "======================================================================"
echo "Next Steps:"
echo "======================================================================"
echo ""
echo "1. If virtual env not activated:"
echo "   source venvtweet/Scripts/activate"
echo ""
echo "2. Install missing packages:"
echo "   pip install pandas numpy scikit-learn nltk matplotlib seaborn tweepy python-dotenv"
echo ""
echo "3. Create dataset:"
echo "   cd sentiment_analysis && python create_hybrid_dataset.py"
echo ""
echo "4. Train models:"
echo "   python train_multiclass_model.py"
echo ""
echo "5. Make predictions:"
echo "   python predict_multiclass.py"
echo ""
echo "======================================================================"