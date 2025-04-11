# Data Generation Pipeline

This directory contains the core components of the medical speech-to-text data generation pipeline.

## Components

- **run_pipeline.py**: Main entry point that orchestrates the full pipeline execution
- **generate_medical_conversations.py**: Generates synthetic medical conversations using OpenAI
- **convert_to_speech.py**: Converts text conversations to speech audio files
- **convert_to_speech_noisy.py**: Converts text to speech with added background noise
- **transcribe_conversations.py**: Transcribes audio using specified STT model
- **transcribe_conversations_opposite.py**: Transcribes using alternative STT model configurations
- **soap_generation.py**: Generates SOAP notes from transcribed conversations

## Usage

From the project root directory, run:

```bash
# Run the complete pipeline
python run_pipeline.py --num 3 --specialty all --noise moderate

# Run specific pipeline components
python -m src.pipeline.generate_medical_conversations --num 5 --specialty cardiology
python -m src.pipeline.convert_to_speech --specialty cardiology --language en-CA
```

## Pipeline Flow

1. **Generate Conversations**: Create synthetic medical dialogues in English and French
2. **Convert to Speech**: Generate audio files from the text conversations
3. **Add Noise Variants**: Create copies with different noise levels
4. **Transcribe**: Process audio through different STT models
5. **Evaluate**: Compare transcription quality across models and conditions
6. **Generate SOAP Notes**: Create structured clinical notes from transcripts 