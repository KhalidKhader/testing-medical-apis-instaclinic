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

The pipeline has been tested with medical conversations in both English and French for cardiology and GP specialties. Here's a summary of our latest findings:

### Latest Evaluation Results (After Voice Enhancement)

#### Cardiology - English (Deepgram Nova 3 Medical)
- **Word Error Rate (WER)**: 0.1716 (std: 0.0044)
- **Text Similarity**: 0.6578 (std: 0.2275)
- **BLEU Score**: 0.6578 (std: 0.2275)
- **Medical Term Accuracy**: 0.8819 (std: 0.0069)
- **Speaker Accuracy**: 0.0000 (std: 0.0000)

#### Cardiology - French (Azure Speech Services)
- **Word Error Rate (WER)**: 0.0000 (std: 0.0000)
- **Text Similarity**: 1.0000 (std: 0.0000)
- **BLEU Score**: 1.0000 (std: 0.0000)
- **Medical Term Accuracy**: 1.0000 (std: 0.0000)
- **Speaker Accuracy**: 0.0000 (std: 0.0000)

#### GP - English (Deepgram Nova 3 Medical)
- **Word Error Rate (WER)**: 0.1601 (std: 0.0224)
- **Text Similarity**: 0.8451 (std: 0.0219)
- **BLEU Score**: 0.8451 (std: 0.0219)
- **Medical Term Accuracy**: 1.0000 (std: 0.0000)
- **Speaker Accuracy**: 0.0000 (std: 0.0000)

#### GP - French (Azure Speech Services)
- **Word Error Rate (WER)**: 0.0000 (std: 0.0000)
- **Text Similarity**: 1.0000 (std: 0.0000)
- **BLEU Score**: 1.0000 (std: 0.0000)
- **Medical Term Accuracy**: 1.0000 (std: 0.0000)
- **Speaker Accuracy**: 0.0000 (std: 0.0000)

### Key Insights

1. **Enhanced Voice Differentiation**: We've implemented extreme voice transformations to ensure clear diarization between doctor and patient speakers:
   - Male doctors now have an extremely deep voice (pitch shift -7.0) and slower speech rate (0.8) for authority
   - Female doctors have a higher pitch (5.0) with a moderately slow speech rate (0.85)
   - Male patients have a less deep voice (pitch shift -2.0) with a faster speech rate (1.15) indicating nervousness
   - Female patients have a very high voice (pitch shift 8.0) with a slightly faster speech rate (1.05)
   - Age-based adjustments now include tremor effects for elderly voices and higher pitch/faster speech for younger patients

2. **Transcription Performance**: 
   - **English (Nova 3 Medical)**: Maintains good performance with WER around 0.16-0.17, but could be improved further.
   - **French (Azure)**: Shows perfect transcription results, which is unexpected and may need further investigation.
   
3. **Medical Term Recognition**:
   - Nova 3 Medical shows excellent medical term accuracy (88.19% for cardiology, 100% for GP), demonstrating its strength with specialized terminology.
   - Azure's perfect score (100%) for French medical terms is surprising and warrants verification.

4. **Speaker Diarization Challenges**:
   - The 0% speaker accuracy across all tests indicates issues with the diarization component despite our voice enhancement efforts.
   - This requires further investigation as the voices should be clearly differentiated with our extreme transformations.

5. **Specialty Differences**:
   - GP conversations show better overall transcription performance than cardiology conversations in English, possibly due to less complex terminology.

These results demonstrate both the potential and limitations of current medical transcription technology. The excellent medical term recognition is promising, but the diarization issues need to be addressed. Further refinements to the voice transformation parameters and evaluation metrics may help improve overall system performance.

## Voice Transformation Enhancement

We've implemented advanced voice transformations to maximize speaker differentiation:

```python
# Enhanced voice transformations for better diarization
if is_doctor:
    if gender == "male":
        # Male doctor: extremely deep voice, authoritative
        pitch_shift = -7.0  # Extremely deep voice
        tempo = 0.8  # Slower speech rate for authority
    else:  # female doctor
        # Female doctor: distinctly professional tone
        pitch_shift = 5.0  # Higher voice
        tempo = 0.85  # Slightly slower speech rate
else:  # patient
    if gender == "male":
        # Male patient: distinctly different from doctor
        pitch_shift = -2.0  # Less deep than doctor but still masculine
        tempo = 1.15  # Faster speech rate to indicate nervousness
    else:  # female patient
        # Female patient: distinctly different from doctor
        pitch_shift = 8.0  # Very high voice
        tempo = 1.05  # Slightly faster speech rate

# Age-based voice adjustments
if age > 65:  # Elderly voice
    tempo *= 0.9  # Slower for elderly
    # Add a slight tremor for elderly voices
    tremor = np.sin(2 * np.pi * 8 * np.arange(len(audio)) / sample_rate) * 0.03
    audio = audio + tremor[:len(audio)]
elif age < 18:  # Young voice
    pitch_shift += 2.0  # Higher pitch for younger
    tempo *= 1.1  # Faster for younger
```

This approach aims to create more realistic and easily distinguishable voices between doctors and patients while adding age-appropriate characteristics.

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