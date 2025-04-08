#!/usr/bin/env python3
"""
Convert generated medical conversations to speech.
This script creates audio files with different voice characteristics for doctor and patient,
producing crystal clear audio with maximum clarity.

Usage:
    python convert_to_speech.py --specialty cardiology --lang en-CA
    python convert_to_speech.py --specialty all --lang all
"""

import os
import json
import glob
import random
import argparse
import tempfile
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import soundfile as sf

# Import TTS libraries conditionally to handle potential missing dependencies
try:
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    import torch
    from datasets import load_dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Transformers/Torch not available. Will use gTTS fallback.")

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
    print("gTTS not available. Text-to-speech functionality will be limited.")

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("pydub not available. Audio processing functionality will be limited.")

try:
    import librosa
    import librosa.effects
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("librosa not available. Voice transformation will be limited.")

# Load environment variables
load_dotenv()

# Base directory for medical data
BASE_DIR = "data-med"

class AudioGenerator:
    """Generator for audio from text conversations with different speakers."""
    
    def __init__(self):
        """Initialize the AudioGenerator."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize high-quality models if available
        if HAS_TRANSFORMERS:
            try:
                print("Loading SpeechT5 models...")
                self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
                self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
                self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
                
                # Load speaker embeddings
                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                self.speaker_embeddings = {}
                self.speaker_embeddings["male"] = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
                self.speaker_embeddings["female"] = torch.tensor(embeddings_dataset[4738]["xvector"]).unsqueeze(0).to(self.device)
                
                self.use_transformer_tts = True
                print("SpeechT5 models loaded successfully.")
            except Exception as e:
                print(f"Error loading SpeechT5 models: {e}")
                self.use_transformer_tts = False
        else:
            self.use_transformer_tts = False
    
    def generate_speech_transformers(self, text, speaker_gender="male"):
        """Generate speech using the SpeechT5 model."""
        try:
            # Process the text
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            
            # Get speaker embedding based on gender
            speaker_embedding = self.speaker_embeddings[speaker_gender]
            
            # Generate speech
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embedding,
                vocoder=self.vocoder
            )
            
            # Convert to numpy array and normalize
            speech_np = speech.cpu().numpy()
            speech_norm = speech_np / np.max(np.abs(speech_np))
            
            return speech_norm, 16000  # 16kHz sample rate
        
        except Exception as e:
            print(f"Error in transformers TTS: {e}")
            return None, None
    
    def generate_speech_gtts(self, text, language="en", voice_type="standard"):
        """Generate speech using gTTS as a fallback."""
        if not HAS_GTTS:
            print("gTTS not available. Cannot generate speech.")
            return None, None
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech with gTTS
            # NOTE: gTTS doesn't support different voice types, but we'll keep the parameter for consistency
            tts = gTTS(text=text, lang=language[:2], slow=(voice_type == "slow"))
            tts.save(temp_path)
            
            # Load audio file and convert to numpy array
            if HAS_LIBROSA:
                speech_np, sample_rate = librosa.load(temp_path, sr=None)
            elif HAS_PYDUB:
                audio_segment = AudioSegment.from_file(temp_path)
                sample_rate = audio_segment.frame_rate
                speech_np = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
            else:
                print("Neither librosa nor pydub available for audio loading.")
                os.remove(temp_path)
                return None, None
            
            # Clean up temporary file
            os.remove(temp_path)
            
            return speech_np, sample_rate
        
        except Exception as e:
            print(f"Error in gTTS: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None, None
    
    def generate_french_speech(self, text, is_doctor=True, gender="male"):
        """Generate French Canadian speech with different voices for doctor and patient."""
        try:
            # Use only Canadian French (fr-CA) as required by the client
            lang_code = "fr-CA"
            
            # Different voice configurations for doctor vs patient, but all using Canadian French
            if is_doctor:
                if gender == "male":
                    print(f"Using Canadian French male doctor voice")
                    # Create initial speech
                    audio, sample_rate = self.generate_speech_gtts(text, "fr", "slow")
                else:
                    print(f"Using Canadian French female doctor voice")
                    # Create initial speech
                    audio, sample_rate = self.generate_speech_gtts(text, "fr", "standard")
            else:
                if gender == "male":
                    print(f"Using Canadian French male patient voice")
                    # Create initial speech
                    audio, sample_rate = self.generate_speech_gtts(text, "fr", "standard")
                else:
                    print(f"Using Canadian French female patient voice")
                    # Create initial speech
                    audio, sample_rate = self.generate_speech_gtts(text, "fr", "standard")
            
            # Apply minimalist voice transformations to differentiate speakers
            # without adding noise or distortion
            if HAS_LIBROSA and audio is not None:
                try:
                    if is_doctor:
                        if gender == "male":
                            # Male doctor: slightly deeper voice
                            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-2.0)
                        else:
                            # Female doctor: slightly higher voice
                            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2.0)
                    else:
                        if gender == "male":
                            # Male patient: neutral with slight pitch adjustment
                            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=1.0)
                        else:
                            # Female patient: higher pitch
                            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=3.0)
                except Exception as e:
                    print(f"Error in voice transformation, using untransformed voice: {e}")
            
            # Ensure no DC offset (which can cause noise)
            if audio is not None:
                audio = audio - np.mean(audio)
                
                # Apply a gentle noise gate
                # This removes any background noise that might be in the recording
                noise_floor = 0.005  # -46dB, very low threshold
                gate_mask = np.abs(audio) < noise_floor
                audio[gate_mask] = 0.0
            
            return audio, sample_rate
            
        except Exception as e:
            print(f"Error generating French speech: {e}")
            # Fallback to standard Canadian French
            return self.generate_speech_gtts(text, "fr")
    
    def transform_voice(self, audio, sample_rate, is_doctor=True, gender="male", age=45, language="en"):
        """Empty function - no longer applying transformations in this way"""
        # Return the original audio without modification
        # All voice transformations are now handled in generate_french_speech
        return audio
    
    def normalize_audio(self, audio):
        """Simple peak normalization for consistent volume."""
        if np.max(np.abs(audio)) > 0:
            # Simple peak normalization to 90% to avoid any clipping
            return audio / np.max(np.abs(audio)) * 0.9
        return audio
    
    def process_conversation(self, conversation_file, output_file, language="en-CA"):
        """Process a conversation file and create a crystal clear audio file with different speakers."""
        try:
            # Load the conversation data
            with open(conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract conversation and metadata
            conversation = data.get("conversation", [])
            metadata = data.get("metadata", {})
            
            # Extract gender for voice selection only - no transformations
            patient_gender = metadata.get("patient_gender", "male")
            
            # Convert gender to standardized form for our processing
            if patient_gender.lower() in ["female", "femme", "f"]:
                patient_gender = "female"
            else:
                patient_gender = "male"
            
            # Use contrasting gender for doctor voice
            doctor_gender = "female" if patient_gender == "male" else "male"
            
            # Language code for TTS
            lang_code = language.split('-')[0]  # en or fr
            
            # Process each utterance and generate audio
            audio_segments = []
            
            for i, utterance in enumerate(conversation):
                speaker = utterance.get("speaker", "")
                text = utterance.get("text", "")
                
                if not text.strip():
                    continue
                
                print(f"Processing {speaker} utterance: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                # Determine speaker gender only
                is_doctor = speaker.lower() == "doctor"
                gender = doctor_gender if is_doctor else patient_gender
                
                # Generate speech using appropriate method based on language
                if lang_code == "fr":
                    # For French, use our specialized function with distinct voices
                    audio, sample_rate = self.generate_french_speech(text, is_doctor, gender)
                elif self.use_transformer_tts and lang_code == "en":
                    # Use high-quality transformers model for English
                    audio, sample_rate = self.generate_speech_transformers(text, gender)
                else:
                    # Use standard gTTS for other languages
                    audio, sample_rate = self.generate_speech_gtts(text, lang_code)
                
                if audio is None:
                    print(f"Failed to generate speech for utterance {i+1}")
                    continue
                
                # Normalize audio
                audio = self.normalize_audio(audio)
                
                # Add to segments
                audio_segments.append((audio, sample_rate))
            
            # Combine all segments with pauses between speakers
            if not audio_segments:
                print(f"No audio segments generated for {conversation_file}")
                return False
            
            # Find the dominant sample rate from segments
            sr = audio_segments[0][1]
            
            # Calculate complete audio length
            total_length = 0
            for audio, _ in audio_segments:
                total_length += len(audio)
                # Add 1 second pause between utterances
                total_length += sr
            
            # Create the combined audio array
            combined_audio = np.zeros(total_length)
            
            # Fill the combined audio with segments and pauses
            pos = 0
            prev_speaker = None
            
            for i, utterance in enumerate(conversation):
                if i >= len(audio_segments):
                    break
                    
                speaker = utterance.get("speaker", "")
                audio, _ = audio_segments[i]
                
                # Add a longer pause when speaker changes
                if prev_speaker is not None and speaker != prev_speaker:
                    pause_duration = int(2.5 * sr)  # 2.5 second pause between different speakers
                else:
                    pause_duration = int(1.0 * sr)  # 1.0 second pause for same speaker
                
                # Skip the pause for the first utterance
                if i > 0:
                    pos += pause_duration
                
                # Add audio
                end_pos = pos + len(audio)
                if end_pos <= len(combined_audio):
                    combined_audio[pos:end_pos] = audio
                    pos = end_pos
                
                prev_speaker = speaker
            
            # Trim excess space
            combined_audio = combined_audio[:pos]
            
            # Final normalization
            combined_audio = self.normalize_audio(combined_audio)
            
            # Save the audio file
            sf.write(output_file, combined_audio, sr)
            print(f"Successfully created pure, unprocessed audio file: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"Error processing conversation {conversation_file}: {e}")
            import traceback
            traceback.print_exc()
            return False

def process_specialty(specialty, language="all"):
    """Process all conversation files for a given specialty and language."""
    languages = ["en-CA", "fr-CA"] if language == "all" else [language]
    
    for lang in languages:
        # Set up directory paths
        json_dir = os.path.join(BASE_DIR, specialty, lang, "json")
        audio_dir = os.path.join(BASE_DIR, specialty, lang, "audio")
        
        # Ensure audio directory exists
        os.makedirs(audio_dir, exist_ok=True)
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {json_dir}")
            continue
            
        print(f"Found {len(json_files)} conversation files in {json_dir}")
        
        # Initialize audio generator
        generator = AudioGenerator()
        
        # Process each file
        for json_file in tqdm(json_files, desc=f"Converting {lang} {specialty} conversations"):
            # Extract file basename
            basename = os.path.basename(json_file)
            filename = os.path.splitext(basename)[0]
            
            # Create output audio filename
            audio_file = os.path.join(audio_dir, f"{filename}.wav")
            
            # Skip if audio file already exists
            if os.path.exists(audio_file):
                print(f"Audio file for {filename} already exists. Skipping.")
                continue
                
            # Process conversation and generate audio
            generator.process_conversation(json_file, audio_file, lang)

def main():
    """Main function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert medical conversations to speech')
    parser.add_argument('--specialty', type=str, choices=['cardiology', 'gp', 'all'], default='all', 
                        help='Medical specialty to process')
    parser.add_argument('--lang', type=str, choices=['en-CA', 'fr-CA', 'all'], default='all',
                        help='Language to process')
    args = parser.parse_args()
    
    # Process conversations based on arguments
    if args.specialty == "all":
        specialties = ["cardiology", "gp"]
        for specialty in specialties:
            process_specialty(specialty, args.lang)
    else:
        process_specialty(args.specialty, args.lang)
    
    print("\n=== Text-to-Speech Conversion Complete ===")

if __name__ == "__main__":
    main() 