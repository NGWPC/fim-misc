#!/bin/bash
# Test Setup Script
# Verifies that the tool is correctly configured

set -e

echo "======================================================================"
echo "Flood Animation Tool - Setup Verification"
echo "======================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1"
        return 0
    else
        echo -e "${RED}✗${NC} Missing: $1"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Found: $1/"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} Missing: $1/ (will be created)"
        mkdir -p "$1"
        return 0
    fi
}

# Check Docker
echo "1. Checking Docker..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker installed: $(docker --version)"
    if docker ps &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docker daemon running"
    else
        echo -e "${RED}✗${NC} Docker daemon not running"
        echo "  Start Docker Desktop or run: sudo systemctl start docker"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Docker not found"
    echo "  Install from: https://docs.docker.com/get-docker/"
    exit 1
fi
echo ""

# Check Docker Compose
echo "2. Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose installed: $(docker-compose --version)"
else
    echo -e "${RED}✗${NC} Docker Compose not found"
    exit 1
fi
echo ""

# Check required files
echo "3. Checking required files..."
check_file "config.yaml"
check_file "Dockerfile"
check_file "docker-compose.yml"
check_file "requirements.txt"
check_file "run_workflow.py"
check_file "generate_flow_files.py"
check_file "generate_batch_fims.py"
check_file "generate_animation.py"
echo ""

# Check .env
echo "4. Checking environment..."
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} Found: .env"
    if grep -q "AWS_ACCESS_KEY_ID=your_access_key_here" .env; then
        echo -e "${YELLOW}⚠${NC} .env contains template values"
        echo "  Edit .env with your AWS credentials if using private S3"
    fi
else
    echo -e "${YELLOW}⚠${NC} Missing: .env"
    echo "  Run: cp .env.example .env"
    echo "  Then edit .env with your credentials"
fi
echo ""

# Check directories
echo "5. Checking directory structure..."
check_dir "data"
check_dir "data/input"
check_dir "data/output"
check_dir "data/cache"
echo ""

# Check config
echo "6. Checking config.yaml..."
if command -v python3 &> /dev/null; then
    python3 -c "
import yaml
try:
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    print('\033[0;32m✓\033[0m Config is valid YAML')
    print(f'  Collection: {config[\"collection\"][\"id\"]}')
    print(f'  Event: {config[\"event\"][\"start_date\"]} to {config[\"event\"][\"end_date\"]}')
except Exception as e:
    print(f'\033[0;31m✗\033[0m Config error: {e}')
    exit(1)
" || echo -e "${YELLOW}⚠${NC} Could not validate config (Python 3 with PyYAML required)"
else
    echo -e "${YELLOW}⚠${NC} Python 3 not found, skipping config validation"
fi
echo ""

# Check disk space
echo "7. Checking disk space..."
AVAILABLE=$(df -h . | awk 'NR==2 {print $4}')
echo "  Available: $AVAILABLE"
echo "  Required: ~20GB minimum"
echo ""

# Summary
echo "======================================================================"
echo "Setup Verification Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Review config.yaml"
echo "  2. Edit .env if using private S3 buckets"
echo "  3. Run: make build"
echo "  4. Run: make run-workflow"
echo ""
echo "Quick commands:"
echo "  make help             - Show all commands"
echo "  make shell            - Open interactive shell"
echo "  make generate-flows   - Generate flow files"
echo ""
echo "Documentation:"
echo "  README.md             - Full documentation"
echo ""
