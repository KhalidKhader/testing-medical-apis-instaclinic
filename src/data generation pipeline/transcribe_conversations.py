#!/usr/bin/env python3
"""
Transcribe medical conversation audio files using:
- Deepgram Nova-3 for English
- Azure Speech Services for French (canadacentral region)

Usage:
    python transcribe_conversations.py --specialty cardiology --lang en-CA
    python transcribe_conversations.py --specialty all --lang all
    python transcribe_conversations.py --audio test_data/test_conversation.wav --lang en-CA
"""

import os
import json
import glob
import argparse
import requests
import traceback
from dotenv import load_dotenv
from tqdm import tqdm
import re
import tempfile

# Load environment variables
load_dotenv()

# Base directory for medical data
BASE_DIR = "data-med"

# API keys from environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "canadacentral")

class AudioTranscriber:
    """Transcribe audio files using appropriate service based on language."""
    
    def __init__(self):
        """Initialize the transcriber."""
        # Check Deepgram API key for English
        if DEEPGRAM_API_KEY:
            self.deepgram_headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav"
            }
            print("Deepgram client initialized for English transcription.")
        else:
            self.deepgram_headers = None
            print("DEEPGRAM_API_KEY not found in environment. Set it in your .env file.")
        
        # Check Azure Speech key for French
        if AZURE_SPEECH_KEY:
            print("Azure Speech API key found. French transcription will be available.")
        else:
            print("AZURE_SPEECH_KEY not found in environment. Set it in your .env file.")
    
    def transcribe_with_deepgram_nova(self, audio_path):
        """Transcribe English audio using Deepgram Nova-3 model with direct HTTP."""
        if not self.deepgram_headers:
            print("Deepgram headers not available. Cannot transcribe English audio.")
            return None, None
        
        try:
            print(f"Transcribing {audio_path} with Deepgram Nova-3 model for English...")
            
            # Check audio file format and convert if necessary
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path)
                
                # Deepgram works best with 16kHz, mono WAV
                if sample_rate != 16000 or len(audio_data.shape) > 1:
                    print(f"Converting audio file to optimal format for Deepgram (16kHz, mono)...")
                    import librosa
                    
                    # Load and resample audio
                    y, _ = librosa.load(audio_path, sr=16000, mono=True)
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Save as 16-bit PCM WAV
                    sf.write(temp_path, y, 16000, 'PCM_16')
                    
                    # Update audio path to the temporary file
                    audio_path = temp_path
                    print(f"Audio converted and saved to temporary file: {temp_path}")
            except ImportError:
                print("Audio format conversion libraries not available, using original file.")
            except Exception as e:
                print(f"Error checking/converting audio format: {e}")
            
            # Use nova-3 model for English
            url = "https://api.deepgram.com/v1/listen?model=nova-3&diarize=true&language=en&punctuate=true&utterances=true"
            print(f"Sending request to Deepgram API with model=nova-3, language=en, diarize=true, utterances=true")
            
            with open(audio_path, "rb") as audio_file:
                response = requests.post(url, headers=self.deepgram_headers, data=audio_file)
            
            # Clean up temporary file if it was created
            if 'temp_path' in locals():
                try:
                    os.remove(temp_path)
                    print(f"Temporary file removed: {temp_path}")
                except:
                    pass
                
            if response.status_code != 200:
                print(f"Error from Deepgram API: {response.status_code} - {response.text}")
                return None, None
                
            # Parse the response
            result = response.json()
            
            # Save the full raw response for debugging
            debug_path = os.path.join(os.path.dirname(audio_path), f"{os.path.basename(audio_path)}_dg_response.json")
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Extract the full transcript
            transcript = ""
            if "results" in result and "channels" in result["results"] and len(result["results"]["channels"]) > 0:
                if "alternatives" in result["results"]["channels"][0] and len(result["results"]["channels"][0]["alternatives"]) > 0:
                    transcript = result["results"]["channels"][0]["alternatives"][0].get("transcript", "")
                    print(f"Extracted transcript: {len(transcript)} characters")
            
            if not transcript:
                print("Warning: No transcript found in Deepgram response")
                return None, None
            
            # Process utterances with speaker information
            diarized_transcript = []
            
            if "utterances" in result.get("results", {}):
                utterances = result["results"]["utterances"]
                print(f"Found {len(utterances)} utterances with speaker information")
                
                # Sort utterances by start time
                utterances.sort(key=lambda x: x.get("start", 0))
                
                # Process utterances with speaker information
                for utterance in utterances:
                    text = utterance.get("transcript", "").strip()
                    if not text:
                        continue
                        
                    speaker = utterance.get("speaker", "0")
                    # Maps even numbers to doctor, odd to patient
                    speaker_role = "doctor" if int(speaker) % 2 == 0 else "patient"
                    
                    # Add to conversation - merge with previous if same speaker
                    if diarized_transcript and diarized_transcript[-1]["speaker"] == speaker_role:
                        # Same speaker continuing
                        diarized_transcript[-1]["text"] += " " + text
                    else:
                        # New speaker
                        diarized_transcript.append({
                            "speaker": speaker_role,
                            "text": text
                        })
            
            # If no utterances, use word-level diarization
            if not diarized_transcript and "channels" in result.get("results", {}) and "words" in result["results"]["channels"][0]["alternatives"][0]:
                words = result["results"]["channels"][0]["alternatives"][0]["words"]
                print(f"Using word-level diarization with {len(words)} words")
                
                current_speaker = None
                current_text = []
                
                for word in words:
                    if "speaker" in word:
                        speaker = word["speaker"]
                        text = word.get("word", "")
                        
                        # If new speaker, add previous text to conversation
                        if speaker != current_speaker and current_speaker is not None and current_text:
                            speaker_role = "doctor" if int(current_speaker) % 2 == 0 else "patient"
                            diarized_transcript.append({
                                "speaker": speaker_role,
                                "text": " ".join(current_text)
                            })
                            current_text = []
                        
                        # Update current speaker and add word
                        current_speaker = speaker
                        current_text.append(text)
                
                # Add final segment
                if current_text and current_speaker is not None:
                    speaker_role = "doctor" if int(current_speaker) % 2 == 0 else "patient"
                    diarized_transcript.append({
                        "speaker": speaker_role,
                        "text": " ".join(current_text)
                    })
            
            # Fallback to alternating conversation if no diarization available
            if not diarized_transcript and transcript:
                print("Creating alternating conversation as fallback")
                
                # Split text into sentences
                sentences = re.split(r'(?<=[.!?])\s+', transcript)
                sentences = [s for s in sentences if s.strip()]
                
                is_doctor = True  # Start with doctor
                
                for sentence in sentences:
                    diarized_transcript.append({
                        "speaker": "doctor" if is_doctor else "patient",
                        "text": sentence.strip()
                    })
                    is_doctor = not is_doctor
            
            print(f"Extracted transcript with {len(diarized_transcript)} segments")
            return transcript, diarized_transcript
                
        except Exception as e:
            print(f"Error transcribing with Deepgram: {str(e)}")
            traceback.print_exc()
            
            # Clean up temporary file if it was created
            if 'temp_path' in locals():
                try:
                    os.remove(temp_path)
                except:
                    pass
                
            return None, None
    
    def transcribe_with_azure_simple(self, audio_path):
        """Transcribe French audio using Azure Speech Services REST API."""
        if not AZURE_SPEECH_KEY:
            print("Azure Speech key not available. Cannot transcribe French audio.")
            return None, None
        
        try:
            print(f"Transcribing {audio_path} with Azure Speech Services REST API...")
            
            # Process audio to ensure it meets Azure requirements
            try:
                import soundfile as sf
                import librosa
                audio_data, sample_rate = sf.read(audio_path)
                
                # Azure requires PCM WAV format, 16kHz, 16-bit, mono
                if sample_rate != 16000 or len(audio_data.shape) > 1:
                    print(f"Converting audio file to required format (16kHz, mono)...")
                    
                    # Load and resample audio
                    y, _ = librosa.load(audio_path, sr=16000, mono=True)
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Save as 16-bit PCM WAV
                    sf.write(temp_path, y, 16000, 'PCM_16')
                    
                    # Update audio path to the temporary file
                    audio_path = temp_path
                    print(f"Audio converted and saved to temporary file: {temp_path}")
            except ImportError:
                print("Audio format conversion libraries not available, using original file.")
            except Exception as e:
                print(f"Error checking/converting audio format: {e}")
            
            # Use canadacentral region for Azure - it's more reliable for French
            region = "canadacentral"
            print(f"Using Azure Speech Services region: {region}")
            
            # Set up the API request for French recognition
            url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=fr-CA"
            
            headers = {
                "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000"
            }
            
            # Send the API request
            with open(audio_path, "rb") as audio_file:
                response = requests.post(url, headers=headers, data=audio_file, timeout=60)
            
            # Clean up temporary file if it was created
            if 'temp_path' in locals():
                try:
                    os.remove(temp_path)
                    print(f"Temporary file removed: {temp_path}")
                except:
                    pass
            
            # If we got a result, process it
            if response.status_code == 200:
                result = response.json()
                
                # Save the full raw response for debugging
                debug_path = os.path.join(os.path.dirname(audio_path), f"{os.path.basename(audio_path)}_azure_response.json")
                with open(debug_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Extract the transcription
                if 'DisplayText' in result:
                    transcript = result['DisplayText']
                    print(f"Transcription successful: {len(transcript)} characters")
                    
                    # Since Azure REST API doesn't directly provide speaker diarization,
                    # we'll implement our own based on linguistic patterns
                    sentences = re.split(r'(?<=[.!?])\s+', transcript)
                    sentences = [s for s in sentences if s.strip()]
                    
                    # Create alternating diarization for French
                    diarized_transcript = []
                    is_doctor = True  # Start with doctor
                    
                    for sentence in sentences:
                        if sentence.strip():
                            diarized_transcript.append({
                                "speaker": "doctor" if is_doctor else "patient",
                                "text": sentence.strip()
                            })
                            is_doctor = not is_doctor
                    
                    print(f"Created {len(diarized_transcript)} segments with alternating speaker identification")
                    return transcript, diarized_transcript
                else:
                    print(f"Azure speech recognition failed: No transcription in response")
                    print(f"Response content: {result}")
            else:
                print(f"Azure API request failed with status code: {response.status_code}")
                print(f"Response content: {response.text}")
            
        except Exception as e:
            print(f"Exception during Azure API request: {str(e)}")
            traceback.print_exc()
        
        # Clean up temporary file if it was created
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
                print(f"Temporary file removed: {temp_path}")
            except:
                pass
            
        return None, None
    
    def transcribe_audio(self, audio_path, language):
        """
        Transcribe audio using the appropriate service based on language.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (en-CA or fr-CA)
            
        Returns:
            Tuple of (full_transcript, diarized_transcript)
        """
        if language.startswith("en"):
            # For English, use Deepgram Nova-3
            print(f"Using Deepgram Nova-3 for English transcription")
            return self.transcribe_with_deepgram_nova(audio_path)
        elif language.startswith("fr"):
            # For French, use Azure Speech Services
            print(f"Using Azure Speech Services REST API for French transcription")
            return self.transcribe_with_azure_simple(audio_path)
        else:
            print(f"Unsupported language: {language}")
            return None, None
    
    def transcribe_and_save(self, audio_path, output_dir, language):
        """
        Transcribe an audio file and save the results.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save transcription results
            language: Language code (en-CA or fr-CA)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract base filename
            basename = os.path.basename(audio_path)
            filename = os.path.splitext(basename)[0]
            
            # Create output file paths
            transcript_path = os.path.join(output_dir, f"{filename}_transcript.txt")
            json_path = os.path.join(output_dir, f"{filename}.json")
            
            # Skip if both output files already exist
            if os.path.exists(transcript_path) and os.path.exists(json_path):
                print(f"Transcription files for {filename} already exist. Skipping.")
                return True
            
            # Transcribe the audio
            transcript, diarized_transcript = self.transcribe_audio(audio_path, language)
            
            if not transcript:
                print(f"Failed to transcribe {audio_path}")
                return False
            
            # Save the full transcript
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            # Save the diarized transcript if available
            if diarized_transcript:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({"conversation": diarized_transcript}, f, indent=2, ensure_ascii=False)
            
            print(f"Transcription saved for {filename}")
            return True
            
        except Exception as e:
            print(f"Error in transcribe_and_save for {audio_path}: {str(e)}")
            traceback.print_exc()
            return False

def process_specialty(specialty, language="all"):
    """Process all audio files for a given specialty and language."""
    languages = ["en-CA", "fr-CA"] if language == "all" else [language]
    
    # Initialize transcriber
    transcriber = AudioTranscriber()
    
    for lang in languages:
        # Skip English if Deepgram key is not available
        if lang == "en-CA" and not DEEPGRAM_API_KEY:
            print("Skipping English transcription as DEEPGRAM_API_KEY is not available.")
            continue
            
        # Skip French if Azure key is not available
        if lang == "fr-CA" and not AZURE_SPEECH_KEY:
            print("Skipping French transcription as AZURE_SPEECH_KEY is not available.")
            continue
        
        # Set up directory paths
        audio_dir = os.path.join(BASE_DIR, specialty, lang, "audio")
        transcripts_dir = os.path.join(BASE_DIR, specialty, lang, "transcripts")
        
        # Ensure transcripts directory exists
        os.makedirs(transcripts_dir, exist_ok=True)
        
        # Find all WAV files
        audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
        
        if not audio_files:
            print(f"No WAV files found in {audio_dir}")
            continue
            
        print(f"Found {len(audio_files)} audio files in {audio_dir}")
        
        # Process each file
        for audio_file in tqdm(audio_files, desc=f"Transcribing {lang} {specialty} conversations"):
            transcriber.transcribe_and_save(audio_file, transcripts_dir, lang)

def transcribe_specific_audio(audio_file_path, language="en-CA"):
    """
    Transcribe a specific audio file using the appropriate service.
    
    Args:
        audio_file_path: Path to the audio file
        language: Language code (en-CA or fr-CA)
    
    Returns:
        Path to the saved transcript file
    """
    # Check for required API keys
    if language == "en-CA" and not DEEPGRAM_API_KEY:
        print("Deepgram API key not found. Set DEEPGRAM_API_KEY in your .env file.")
        return False
    
    if language == "fr-CA" and not AZURE_SPEECH_KEY:
        print("Azure Speech key not found. Set AZURE_SPEECH_KEY in your .env file.")
        return False
    
    # Create output directory in the same location as the audio file
    output_dir = os.path.dirname(audio_file_path)
    basename = os.path.basename(audio_file_path)
    filename = os.path.splitext(basename)[0]
    
    # Create output file paths
    transcript_path = os.path.join(output_dir, f"{filename}_transcript.txt")
    json_path = os.path.join(output_dir, f"{filename}.json")
    
    # Initialize transcriber
    transcriber = AudioTranscriber()
    
    # Transcribe the audio
    transcript, diarized_transcript = transcriber.transcribe_audio(audio_file_path, language)
    
    if not transcript:
        print(f"Failed to transcribe {audio_file_path}")
        return None
    
    # Save the full transcript
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    # Save the diarized transcript if available
    if diarized_transcript:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({"conversation": diarized_transcript}, f, indent=2, ensure_ascii=False)
    
    print(f"Transcription saved to {transcript_path} and {json_path}")
    return transcript_path

def main():
    """Parse command line arguments and run the transcription process."""
    parser = argparse.ArgumentParser(description='Transcribe medical conversations with Deepgram (English) and Azure (French)')
    
    # Add arguments
    parser.add_argument('--specialty', type=str, choices=['cardiology', 'gp', 'all'], default='all',
                      help='Medical specialty to process (default: all)')
    parser.add_argument('--lang', type=str, choices=['en-CA', 'fr-CA', 'all'], default='all',
                      help='Language to process (default: all)')
    parser.add_argument('--audio', type=str, help='Process a specific audio file instead of directories')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.audio:
        # Process a specific audio file
        print(f"Processing single audio file: {args.audio}")
        if not os.path.exists(args.audio):
            print(f"Error: Audio file not found: {args.audio}")
            return
            
        # Use the explicit language argument if provided (not 'all')
        if args.lang and args.lang != 'all':
            language = args.lang
        # Otherwise determine language from filename
        elif 'fr-CA' in args.audio:
            language = 'fr-CA'
        elif 'en-CA' in args.audio:
            language = 'en-CA'
        else:
            # Default to English if not specified
            language = 'en-CA'
            
        print(f"Using language: {language}")
        transcribe_specific_audio(args.audio, language)
    else:
        # Process directories based on specialty and language
        specialties = ['cardiology', 'gp'] if args.specialty == 'all' else [args.specialty]
        
        for specialty in specialties:
            print(f"Processing {specialty} specialty...")
            process_specialty(specialty, args.lang)

if __name__ == "__main__":
    main() 