#!/bin/bash

# Script to set up the Medical Speech-to-Text Evaluation project environment

echo "Setting up Medical Speech-to-Text Evaluation environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install essential tools
echo "Upgrading pip and installing essential tools..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check for spaCy models and install if needed
echo "Installing spaCy models..."
python -m spacy download en_core_web_lg
python -m spacy download fr_core_news_lg
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz

# Create professional directory structure
echo "Creating professional directory structure..."

# Source code directory
mkdir -p src/pipeline
mkdir -p src/evaluation
mkdir -p src/visualization
mkdir -p src/utils
touch src/__init__.py
touch src/pipeline/__init__.py
touch src/evaluation/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py

# Data directories
mkdir -p data/raw/audio
mkdir -p data/raw/transcripts
mkdir -p data/processed/english
mkdir -p data/processed/french
mkdir -p data/processed/evaluation

# Results directories
mkdir -p results/figures
mkdir -p results/tables
mkdir -p results/reports

# Documentation
mkdir -p docs/images
mkdir -p docs/api

# Test directory
mkdir -p tests
touch tests/__init__.py

# Scripts directory
mkdir -p scripts

# Medical data directories
echo "Creating medical data directories..."
mkdir -p data-med/cardiology/en-CA/json data-med/cardiology/en-CA/audio data-med/cardiology/en-CA/transcripts data-med/cardiology/en-CA/soap
mkdir -p data-med/cardiology/fr-CA/json data-med/cardiology/fr-CA/audio data-med/cardiology/fr-CA/transcripts data-med/cardiology/fr-CA/soap
mkdir -p data-med/gp/en-CA/json data-med/gp/en-CA/audio data-med/gp/en-CA/transcripts data-med/gp/en-CA/soap 
mkdir -p data-med/gp/fr-CA/json data-med/gp/fr-CA/audio data-med/gp/fr-CA/transcripts data-med/gp/fr-CA/soap

# Create evaluation directories
echo "Creating evaluation directories..."
mkdir -p evaluation/comparative evaluation/figures
mkdir -p evaluation/Without-noise-Azure-for-English-Nova-2-for-French
mkdir -p evaluation/Semi-noise-Azure-for-French-Nova-3-medical-for-English
mkdir -p evaluation/Semi-noise-Azure-for-English-Nova-2-for-French
mkdir -p evaluation/Noisy-Azure-for-French-Nova-3-medical-for-English
mkdir -p evaluation/Noisy-Azure-for-English-Nova-2-for-French

# Create symbolic links for backward compatibility
echo "Creating symbolic links for backward compatibility..."
ln -sf ../results/figures custom_viz

# Download sample data if available
echo "Downloading sample data..."
if command -v curl &> /dev/null; then
    echo "Sample data download skipped in setup. Run the sample data script separately."
fi

# Create sample config files
echo "Creating configuration files..."
cat > .env.example << EOF
# API Keys - Replace with your actual keys
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=canadacentral

# Configuration Options
DEFAULT_ENGLISH_MODEL=nova-3-medical
DEFAULT_FRENCH_MODEL=nova-2
DEFAULT_NOISE_LEVEL=moderate
NUM_CONVERSATIONS=5
EOF

# Create basic setup.py
cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="medical-stt-evaluation",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Basic requirements will be installed from requirements.txt
    ],
    author="Bitlab Team",
    author_email="k.khader@bitlab.co",
    description="A framework for evaluating speech-to-text models in medical contexts",
    keywords="medical, speech-to-text, evaluation, healthcare",
    url="https://github.com/bitlab/medical-stt-evaluation",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ]
)
EOF

# Check for GPU support
echo "Checking for GPU support..."
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "GPU support detected. You can enable GPU acceleration for faster processing."
else
    echo "No GPU support detected. Models will run on CPU which may be slower."
fi

echo "Setup complete! To get started:"
echo "1. Copy .env.example to .env and add your API keys"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the pipeline: python run_pipeline.py --specialty cardiology --language en-CA"
echo ""
echo "For more information, see the README.md file." 