#!/usr/bin/env python3
"""
Convert generated medical conversations to speech.
This script creates audio files with different voice characteristics for doctor and patient,
and adds random background noise to simulate real-world conditions.

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
    print("librosa not available. Advanced audio processing will be limited.")

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False
    print("noisereduce not available. Noise processing will be limited.")

# Load environment variables
load_dotenv()

# Base directory for medical data
BASE_DIR = "data-med"

# Define background noise types and probabilities
BACKGROUND_NOISES = {
    "office": 0.4,      # Office background noise (typing, distant voices)
    "hospital": 0.3,    # Hospital sounds (beeping, distant announcements)
    "quiet": 0.2,       # Almost silent room (very light background noise)
    "home": 0.1         # Home environment (occasional household sounds)
}

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
    
    def generate_speech_gtts(self, text, language="en"):
        """Generate speech using gTTS as a fallback."""
        if not HAS_GTTS:
            print("gTTS not available. Cannot generate speech.")
            return None, None
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech with gTTS
            tts = gTTS(text=text, lang=language[:2], slow=False)
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
    
    def transform_voice(self, audio, sample_rate, is_doctor=True, gender="male", age=45):
        """Transform voice characteristics to better match speaker."""
        if not HAS_LIBROSA:
            return audio
        
        try:
            # Determine voice transformation parameters based on speaker characteristics
            if is_doctor:
                if gender == "male":
                    # Male doctor: deeper voice, authoritative
                    pitch_shift = -2.0 if age > 50 else -1.0
                    tempo = 0.95 if age > 50 else 0.98
                else:  # female doctor
                    # Female doctor: professional tone
                    pitch_shift = -1.0 if age > 50 else 0.0
                    tempo = 0.97 if age > 50 else 1.0
            else:  # patient
                if gender == "male":
                    # Male patient: varies by age
                    pitch_shift = -3.0 if age > 65 else (-1.5 if age > 40 else 0.5)
                    tempo = 0.9 if age > 65 else (0.95 if age > 40 else 1.05)
                else:  # female patient
                    # Female patient: varies by age
                    pitch_shift = -1.0 if age > 65 else (0.5 if age > 40 else 2.0)
                    tempo = 0.9 if age > 65 else (0.98 if age > 40 else 1.05)
            
            # Apply pitch shifting
            y_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_shift)
            
            # Apply time stretching for speech rate
            y_stretched = librosa.effects.time_stretch(y_shifted, rate=tempo)
            
            return y_stretched
        
        except Exception as e:
            print(f"Error in voice transformation: {e}")
            return audio
    
    def generate_background_noise(self, length, sample_rate):
        """Generate realistic background noise for medical conversations."""
        # Choose noise type based on probability
        noise_type = random.choices(
            list(BACKGROUND_NOISES.keys()),
            weights=list(BACKGROUND_NOISES.values()),
            k=1
        )[0]
        
        # Generate different types of background noise
        if noise_type == "office":
            # Office noise: light hum with occasional typing
            noise = np.random.normal(0, 0.005, size=length)
            # Add occasional typing sounds
            for _ in range(int(length / sample_rate * 2)):  # 2 typing bursts per second
                if random.random() > 0.7:  # Only 30% of possible typing sounds
                    start = random.randint(0, length - int(0.2 * sample_rate))
                    end = min(start + int(0.2 * sample_rate), length)
                    noise[start:end] += np.random.normal(0, 0.02, size=end-start)
        
        elif noise_type == "hospital":
            # Hospital noise: equipment beeps and background voices
            noise = np.random.normal(0, 0.01, size=length)
            # Add beeping sounds
            beep_interval = sample_rate * 2  # Every 2 seconds
            for i in range(0, length, beep_interval):
                if i + int(0.1 * sample_rate) < length:
                    noise[i:i+int(0.1*sample_rate)] += 0.1 * np.sin(2*np.pi*1000*np.arange(int(0.1*sample_rate))/sample_rate)
        
        elif noise_type == "home":
            # Home environment: occasional background sounds
            noise = np.random.normal(0, 0.003, size=length)
            # Add random household sounds
            for _ in range(3):  # Add 3 random household sounds
                if random.random() > 0.5:
                    start = random.randint(0, length - int(0.5 * sample_rate))
                    end = min(start + int(0.5 * sample_rate), length)
                    noise[start:end] += np.random.normal(0, 0.03, size=end-start)
        
        else:  # "quiet" or default
            # Very quiet room with minimal background noise
            noise = np.random.normal(0, 0.002, size=length)
        
        # Ensure noise level is appropriate (not too loud)
        noise_level = random.uniform(0.01, 0.05)  # Random noise level between 1-5%
        noise = noise_level * noise / np.max(np.abs(noise))
        
        return noise
    
    def apply_audio_enhancements(self, audio, sample_rate):
        """Apply enhancements to make the audio more realistic."""
        if not HAS_LIBROSA:
            return audio
        
        try:
            # Apply a subtle room reverb effect
            if random.random() > 0.3:  # 70% chance of reverb
                reverb_level = random.uniform(0.1, 0.3)
                # Create a simple room impulse response
                ir_len = int(sample_rate * 0.5)  # 500ms reverb
                ir = np.zeros(ir_len)
                for i in range(5):  # Add a few reflections
                    pos = int(sample_rate * random.uniform(0.01, 0.3))
                    if pos < ir_len:
                        ir[pos] = reverb_level * (0.9 ** i)
                
                # Apply reverb
                audio_reverb = np.convolve(audio, ir)[:len(audio)]
                audio = audio + audio_reverb * reverb_level
            
            # Add subtle compression
            if random.random() > 0.5:  # 50% chance of compression
                # Simple compression: reduce the dynamic range
                threshold = 0.5
                ratio = 4.0
                makeup_gain = 1.2
                
                # Apply compression
                mask = np.abs(audio) > threshold
                audio[mask] = threshold + (np.abs(audio[mask]) - threshold) / ratio * np.sign(audio[mask])
                audio = audio * makeup_gain
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            return audio
        
        except Exception as e:
            print(f"Error in audio enhancements: {e}")
            return audio
    
    def process_conversation(self, conversation_file, output_file, language="en-CA"):
        """Process a conversation file and create an audio file with different speakers."""
        try:
            # Load the conversation data
            with open(conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract conversation and metadata
            conversation = data.get("conversation", [])
            metadata = data.get("metadata", {})
            
            # Extract age and gender for voice characteristics
            patient_age = metadata.get("patient_age", 45)
            patient_gender = metadata.get("patient_gender", "male")
            
            # Convert gender to standardized form for our processing
            if patient_gender.lower() in ["female", "femme", "f"]:
                patient_gender = "female"
            else:
                patient_gender = "male"
            
            # Doctor characteristics - randomly assigned but consistent within conversation
            doctor_gender = random.choice(["male", "female"])
            doctor_age = random.randint(35, 65)
            
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
                
                # Determine speaker characteristics
                is_doctor = speaker.lower() == "doctor"
                gender = doctor_gender if is_doctor else patient_gender
                age = doctor_age if is_doctor else patient_age
                
                # Generate speech using appropriate method
                if self.use_transformer_tts and lang_code == "en":
                    # Use high-quality transformers model for English
                    audio, sample_rate = self.generate_speech_transformers(text, gender)
                else:
                    # Use gTTS for all other cases
                    audio, sample_rate = self.generate_speech_gtts(text, lang_code)
                
                if audio is None:
                    print(f"Failed to generate speech for utterance {i+1}")
                    continue
                
                # Transform voice to better match speaker characteristics
                audio = self.transform_voice(audio, sample_rate, is_doctor, gender, age)
                
                # Apply enhancements for realism
                audio = self.apply_audio_enhancements(audio, sample_rate)
                
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
                    pause_duration = int(1.5 * sr)  # 1.5 second pause
                else:
                    pause_duration = int(0.7 * sr)  # 0.7 second pause
                
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
            
            # Generate background noise for the entire conversation
            background_noise = self.generate_background_noise(len(combined_audio), sr)
            
            # Mix conversation with background noise
            mixed_audio = combined_audio + background_noise
            
            # Apply optional noise reduction to make it more realistic 
            # (sometimes real recordings have noise reduction applied)
            if HAS_NOISEREDUCE and random.random() > 0.5:  # 50% chance of noise reduction
                try:
                    # Apply light noise reduction
                    mixed_audio = nr.reduce_noise(
                        y=mixed_audio, 
                        sr=sr,
                        prop_decrease=random.uniform(0.3, 0.6),  # Random amount of noise reduction
                        stationary=False  # Non-stationary noise
                    )
                except Exception as e:
                    print(f"Error applying noise reduction: {e}")
            
            # Normalize final audio
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
            
            # Save the audio file
            sf.write(output_file, mixed_audio, sr)
            print(f"Successfully created audio file: {output_file}")
            
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