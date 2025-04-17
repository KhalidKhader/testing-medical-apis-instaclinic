# Medical Speech-to-Text Evaluation Project

## Overview

This project evaluates and compares various speech-to-text models specifically for medical transcription, focusing on performance across different languages, medical specialties, and audio conditions.

## Key Models Evaluated

- **OpenAI Whisper v3 Large**: High-accuracy model with excellent medical term recognition
- **OpenAI Whisper v3 Turbo**: Balanced speed-accuracy model with strong performance
- **Deepgram Nova-3-medical**: Specialized for English medical content
- **Deepgram Nova-2**: Strong performer for French content
- **Azure Speech Services**: Baseline comparison model

## Diarization Systems
- **NVIDIA NeMo**: Advanced clustering-based speaker diarization
- **PyAnnote**: Neural speaker diarization system

## Repository Structure

```
├── docs/                   # Documentation files
├── med-data/               # Medical transcription data
│   ├── cardiology/         # Cardiology specialty audio and transcripts
│   └── gp/                 # General Practice specialty audio and transcripts
├── results/                # Evaluation results
│   ├── figures/            # Generated charts and visualizations
│   └── tables/             # CSV result tables
├── report.md               # Comprehensive analysis report
├── requirements.txt        # Python dependencies
├── setup.sh                # Environment setup script
├── transcribe_medical_whisper_v3_turbo.py  # Whisper v3 Turbo transcription
└── transcribe_with_whisper_pyannote.py     # Whisper with PyAnnote
```

## Key Features

- Multi-language support (English/French)
- Medical terminology recognition evaluation
- Speaker diarization performance analysis
- Noise resilience assessment
- Performance metrics for different medical specialties

## Setup and Usage

### Environment Setup

```bash
# Install requirements
./setup.sh
```

### Running Transcriptions

#### Single Audio File Transcription
```bash
# Using Whisper v3 Turbo
python transcribe_medical_whisper_v3_turbo.py path/to/audio/file.wav

# Using Whisper with PyAnnote
python transcribe_with_whisper_pyannote.py path/to/audio/file.wav
```

#### Process All Files in Directory
```bash
# Process all files in default directory
python transcribe_medical_whisper_v3_turbo.py --process-all
```

### Output Format

The transcription output is saved as a JSON file with the following structure:

```json
{
  "metadata": {
    "filename": "example.wav",
    "language": "en-CA",
    "model": "whisper-v3-turbo",
    "duration_seconds": 120.5,
    "processing_time_seconds": 35.2,
    "segment_count": 15,
    "character_count": 2500,
    "turn_count": 10
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.5,
      "text": "Hello, how are you feeling today?",
      "speaker": "DOCTOR"
    },
    // Additional segments...
  ],
  "turns": [
    {
      "speaker": "DOCTOR",
      "text": "Hello, how are you feeling today?"
    },
    {
      "speaker": "PATIENT",
      "text": "I've been having some chest pain lately."
    },
    // Additional turns...
  ]
}
```

## Key Findings

- **English Medical Terminology**: Best performance with Whisper large-v3 + NVIDIA NeMo (92.2% F1 score)
- **French Medical Terminology**: Best performance with Deepgram Nova-2 (75.6% F1 score)
- **Speaker Identification**: Best performance with Whisper v3 Turbo + NVIDIA NeMo for French (84.3% accuracy)
- **Processing Speed**: Whisper v3 Turbo processes 2.5x faster than v3 Large with minimal accuracy reduction

## Requirements

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers
- NVIDIA NeMo (optional, for advanced diarization)
- PyAnnote (optional, for alternative diarization)
- Additional requirements specified in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.
