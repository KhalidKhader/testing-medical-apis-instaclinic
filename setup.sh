#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install setuptools first
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
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

# Create a sample .env file
echo "Creating sample .env file..."
cat > .env.sample << EOF
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

echo "Setup complete! Now run:"
echo "1. source venv/bin/activate"
echo "2. Add your API keys to .env file:"
echo "   OPENAI_API_KEY=your_key"
echo "   DEEPGRAM_API_KEY=your_key"
echo "   AZURE_SPEECH_KEY=your_key"
echo "   AZURE_SPEECH_REGION=canadacentral"
echo "3. Run the complete pipeline:"
echo "   python run_pipeline.py --num 3 --specialty all --noise moderate" 