# Medical Conversation Generation and Evaluation Pipeline

This project provides a comprehensive pipeline for generating, processing, and evaluating medical conversations in Canadian English and Canadian French. The system creates realistic doctor-patient dialogues for both Cardiology and General Practitioner (GP) specialties, converts them to speech, transcribes them, and evaluates the accuracy of the transcriptions.

## Features

- Generate realistic medical conversations using GPT models (consultation + follow-up pairs)
- Convert conversations to speech with speaker differentiation
- Add realistic background noise to simulate real-world conditions
- Transcribe audio files using specialist medical speech recognition
  - Nova-3-medical model from Deepgram for English
  - Azure Speech Services for French
- Evaluate transcription accuracy with focus on medical terminology
- Process both Canadian English and Canadian French content

## Directory Structure

After running the pipeline, the data will be organized as follows:

```
data-med/
├── cardiology/
│   ├── en-CA/
│   │   ├── json/         # Original conversation data
│   │   ├── soap/         # SOAP notes in markdown format
│   │   ├── audio/        # Audio files with doctor/patient voices
│   │   └── transcripts/  # Transcriptions and diarization results
│   └── fr-CA/
│       ├── json/
│       ├── soap/
│       ├── audio/
│       └── transcripts/
├── gp/
│   ├── en-CA/
│   │   ├── json/
│   │   ├── soap/
│   │   ├── audio/
│   │   └── transcripts/
│   └── fr-CA/
│       ├── json/
│       ├── soap/
│       ├── audio/
│       └── transcripts/
└── evaluation/           # Evaluation results and visualizations
    ├── cardiology/
    └── gp/
```

## Requirements

1. Python 3.8 or higher
2. API keys for:
   - OpenAI (for generating medical conversations)
   - Deepgram (for Nova-3-medical transcription)
   - Azure Speech Services (for French transcription)
3. Required Python packages (install with `pip install -r requirements.txt`)

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   DEEPGRAM_API_KEY=your_deepgram_key
   AZURE_SPEECH_KEY=your_azure_speech_key
   AZURE_SPEECH_REGION=eastus
   ```

## Usage

### Running the Complete Pipeline

To run the entire pipeline (generation, speech conversion, transcription, and evaluation):

```bash
python run_pipeline.py --num 3 --specialty all
```

Options:
- `--num`: Number of conversation pairs to generate per specialty and language (default: 3)
- `--specialty`: Medical specialty to process (`cardiology`, `gp`, or `all`) (default: `all`)

### Running Individual Components

You can also run each component separately:

#### 1. Generate Conversations

```bash
python generate_medical_conversations.py --num 5 --specialty cardiology
```

Options:
- `--num`: Number of conversation pairs to generate (default: 5)
- `--specialty`: Medical specialty (`cardiology`, `gp`, or `all`) (default: `all`)
- `--model`: GPT model to use (default: `gpt-4`)

#### 2. Convert to Speech

```bash
python convert_to_speech.py --specialty cardiology --lang en-CA
```

Options:
- `--specialty`: Medical specialty to process (default: `all`)
- `--lang`: Language to process (`en-CA`, `fr-CA`, or `all`) (default: `all`)

#### 3. Transcribe Audio

```bash
python transcribe_conversations.py --specialty gp --lang fr-CA
```

Options:
- `--specialty`: Medical specialty to process (default: `all`)
- `--lang`: Language to transcribe (`en-CA`, `fr-CA`, or `all`) (default: `all`)

#### 4. Evaluate Transcriptions

```bash
python evaluate_transcriptions.py --specialty all --lang all
```

Options:
- `--specialty`: Medical specialty to evaluate (default: `all`)
- `--lang`: Language to evaluate (`en-CA`, `fr-CA`, or `all`) (default: `all`)

## Evaluation Metrics

The evaluation produces the following metrics:

- **Word Error Rate (WER)**: Measures the edit distance between the original and transcribed text
- **Similarity Score**: Text similarity using sequence matching
- **BLEU Score**: Standard metric for evaluating machine translation quality
- **Medical Term Accuracy**: Specifically measures how well medical terminology is recognized
- **Speaker Accuracy**: How well the speaker roles (doctor/patient) are identified

## Evaluation Results

The pipeline has been tested with medical conversations in both English and French for cardiology specialties. Here's a summary of our findings:

### English Transcription (Deepgram Nova 3 Medical)
- **Word Error Rate (WER)**: 0.13 (Lower is better)
- **Text Similarity**: 0.68
- **BLEU Score**: 0.68
- **Medical Term Accuracy**: 0.71
- **Speaker Accuracy**: 0.90

### French Transcription (Azure Speech Services)
- **Word Error Rate (WER)**: 0.62 (Lower is better)
- **Text Similarity**: 0.43
- **BLEU Score**: 0.43
- **Medical Term Accuracy**: 0.41
- **Speaker Accuracy**: 1.00

### Key Insights

1. **Transcription Accuracy**: Deepgram Nova 3 Medical demonstrates significantly higher transcription accuracy for English medical conversations with a WER of 0.13 compared to Azure's 0.62 for French.

2. **Medical Terminology Recognition**: Nova 3 Medical excels at recognizing medical terminology in English (71% accuracy) compared to Azure's performance in French (41% accuracy).

3. **Speaker Diarization**: Azure Speech Services achieves perfect speaker identification in French conversations, while Nova 3 Medical maintains excellent but slightly lower accuracy (90%).

4. **Consistency**: Deepgram's results show more consistency across samples, while Azure's performance varies more widely depending on the conversation complexity.

These results highlight the current state-of-the-art in medical transcription technology and demonstrate the effectiveness of specialized models like Nova 3 Medical for healthcare applications.

## Visualizations

The evaluation script generates visualizations in the `data-med/evaluation/` directory:

- Word Error Rate distributions
- Medical Term Accuracy distributions
- Speaker Accuracy vs Medical Term Accuracy plots

## Example Output

### JSON Conversation Format

```json
{
  "metadata": {
    "case_id": "cardiology_1",
    "conversation_id": "cardiology_1-consult",
    "language": "en-CA",
    "specialty": "cardiology",
    "encounter_type": "consultation",
    "condition": "atrial fibrillation",
    "patient_age": 67,
    "patient_gender": "male",
    "timestamp": "2023-08-15T14:32:17.123456"
  },
  "conversation": [
    {
      "speaker": "doctor",
      "text": "Good morning, Mr. Johnson. I'm Dr. Smith, a cardiologist. How are you feeling today?"
    },
    {
      "speaker": "patient",
      "text": "Not great, doctor. I've been having these episodes where my heart feels like it's racing and fluttering."
    },
    ...
  ]
}
```

### SOAP Note Format

```markdown
**Subjective**:
67-year-old male presents with complaints of intermittent palpitations, described as "racing and fluttering" for the past 3 months. Episodes last 15-30 minutes and are associated with shortness of breath, dizziness, and fatigue. No syncope. Patient reports symptoms worsen with caffeine and stress.

**Objective**:
Vitals: BP 142/88, HR 92 and irregular, RR 18, T 36.7°C, O2 98% on RA
Heart: Irregular rhythm, no murmurs, gallops, or rubs
ECG: Atrial fibrillation with rapid ventricular response, rate 110
...

**Assessment**:
1. Atrial fibrillation, new onset, with rapid ventricular response
2. Hypertension, poorly controlled
...

**Plan**:
1. Start rate control with metoprolol 25mg BID
2. Initiate anticoagulation with apixaban 5mg BID
...

**Keywords / Tags**:
- Diagnosis: Atrial fibrillation, Hypertension
- Treatment: Metoprolol, Apixaban
- Symptoms: Palpitations, Dyspnea, Dizziness, Fatigue
- Conditions: Atrial fibrillation, Hypertension
- Allergies: None
```

## Limitations

- Speech recognition accuracy varies based on background noise levels and speaker characteristics
- Medical terminology recognition remains challenging for speech recognition systems
- Diarization accuracy (speaker identification) may be limited in conversations with frequent speaker turns
- Language support is currently limited to Canadian English and Canadian French

## License

This project is licensed under the MIT License - see the LICENSE file for details. 