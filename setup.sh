#!/bin/bash

# Heart Disease Prediction MLOps Setup Script

set -e

echo "=========================================="
echo "Heart Disease Prediction MLOps Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

echo -e "${GREEN}✓ Virtual environment created${NC}"

echo -e "${YELLOW}Step 2: Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"

echo -e "${YELLOW}Step 3: Downloading dataset...${NC}"
python src/data_processing/download_data.py

echo -e "${GREEN}✓ Dataset downloaded${NC}"

echo -e "${YELLOW}Step 4: Running tests...${NC}"
pytest tests/ -v

echo -e "${GREEN}✓ Tests passed${NC}"

echo -e "${YELLOW}Step 5: Training models...${NC}"
python src/model/train.py

echo -e "${GREEN}✓ Models trained${NC}"

echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start the API: python -m uvicorn src.api.app:app --reload"
echo "2. View MLflow UI: mlflow ui"
echo "3. Build Docker: make docker-build"
echo "4. View documentation: cat README.md"
echo ""
