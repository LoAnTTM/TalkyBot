# TalkyBot Automatic Setup Script
# This script creates conda environment and installs dependencies

set -e  # Exit on any error

echo "🤖 TalkyBot Setup Script"
echo "========================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda is not installed!${NC}"
    echo "Please install Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✅ Conda found!${NC}"

# Environment name
ENV_NAME="TalkyBot"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}⚠️  Environment '${ENV_NAME}' already exists!${NC}"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🗑️  Removing existing environment...${NC}"
        conda env remove -n ${ENV_NAME} -y
    else
        echo -e "${YELLOW}Skipping environment creation...${NC}"
        conda activate ${ENV_NAME}
        echo -e "${GREEN}✅ Activated existing environment: ${ENV_NAME}${NC}"
        exit 0
    fi
fi

# Create conda environment
echo -e "${BLUE}🔧 Creating conda environment: ${ENV_NAME} with Python 3.10...${NC}"
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
echo -e "${BLUE}🔄 Activating environment...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found!${NC}"
    echo "Please make sure you're running this script from the TalkyBot directory."
    exit 1
fi

# Install basic dependencies
echo -e "${BLUE}📦 Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt

# Install additional packages manually
echo -e "${BLUE}📦 Installing additional packages...${NC}"

echo -e "${YELLOW}Installing TensorFlow for macOS...${NC}"
pip install tensorflow-macos

echo -e "${YELLOW}Installing OpenWakeWord...${NC}"
pip install openwakeword==0.6.0

echo -e "${YELLOW}Installing TTS...${NC}"
pip install TTS

echo -e "${YELLOW}Installing ONNX Runtime...${NC}"
pip install onnxruntime

# Create models directory
echo -e "${BLUE}📁 Creating models directory...${NC}"
mkdir -p models/openwakeword

echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: conda activate ${ENV_NAME}"
echo "2. Test wake word detection: cd components && python wakeword.py"
echo "3. Say 'Alexa' to test the wake word detection"
echo ""
echo -e "${BLUE}💡 To activate the environment in the future:${NC}"
echo "   conda activate ${ENV_NAME}"
