#!/usr/bin/env python3
"""
Transcribe medical conversation audio files using:
- Azure Speech Services for English (canadacentral region)
- Deepgram Nova for French

Usage:
    python test.py --specialty cardiology --lang en-CA
    python test.py --specialty all --lang all
    python test.py --audio test_data/test_conversation.wav --lang en-CA
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
import threading
import time

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
        # Check Azure Speech key for English
        if AZURE_SPEECH_KEY:
            print("Azure Speech API key found. English transcription will be available.")
            try:
                import azure.cognitiveservices.speech as speechsdk
                self.speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
                print(f"Azure Speech SDK initialized with region: {AZURE_SPEECH_REGION}")
            except ImportError:
                print("Azure Speech SDK not installed. Please install with: pip install azure-cognitiveservices-speech")
                self.speech_config = None
        else:
            print("AZURE_SPEECH_KEY not found in environment. Set it in your .env file.")
            self.speech_config = None
        
        # Check Deepgram API key for French
        if DEEPGRAM_API_KEY:
            print("Deepgram API key found. French transcription will be available.")
        else:
            print("DEEPGRAM_API_KEY not found in environment. Set it in your .env file.")
    
    def transcribe_with_azure(self, audio_path):
        """Transcribe English audio using Azure Speech Services with diarization."""
        if not AZURE_SPEECH_KEY:
            print("Azure Speech key not available. Cannot transcribe English audio.")
            return None, None
        
        try:
            print(f"Transcribing {audio_path} with Azure Speech Services and diarization...")
            
            # Verify audio format and convert if needed
            import soundfile as sf
            try:
                audio_data, sample_rate = sf.read(audio_path)
                duration = len(audio_data) / sample_rate
                print(f"Audio duration: {duration:.2f} seconds")
                
                # Convert audio to 16kHz mono WAV if needed
                temp_file_created = False
                if sample_rate != 16000 or len(audio_data.shape) > 1:
                    print("Converting audio to 16kHz mono for Azure...")
                    from scipy import signal
                    if len(audio_data.shape) > 1:  # If stereo, convert to mono
                        audio_data = audio_data.mean(axis=1)
                    if sample_rate != 16000:
                        audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                    
                    # Save temporary file
                    temp_path = f"{audio_path}_temp.wav"
                    sf.write(temp_path, audio_data, 16000)
                    audio_path = temp_path
                    temp_file_created = True
            except Exception as e:
                print(f"Error reading audio file: {e}")
                traceback.print_exc()
                return None, None
            
            # Import Azure Speech SDK
            import azure.cognitiveservices.speech as speechsdk
            
            # Configure speech settings
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            speech_config.speech_recognition_language = "en-US"
            
            # Enable diarization for conversation transcription
            speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, "true")
            
            # Create audio configuration from file
            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
            
            # Create conversation transcriber
            transcriber = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Storage for results
            transcript = ""
            segments = []
            speaker_mapping = {}  # Maps speaker IDs to roles
            
            # Set up completion signal
            done_event = threading.Event()
            
            # Handle transcribed results
            def handle_transcribed(evt):
                nonlocal transcript
                result = evt.result
                text = result.text
                speaker_id = result.speaker_id
                
                # Speaker ID will be like "Guest-1", "Guest-2", etc.
                print(f"TRANSCRIBED: Text=\"{text}\" Speaker ID={speaker_id}")
                
                # Assign roles (first speaker is doctor, others are patients)
                if speaker_id not in speaker_mapping:
                    if not speaker_mapping:
                        speaker_mapping[speaker_id] = "doctor"
                        print(f"Assigning speaker {speaker_id} as doctor")
                    else:
                        speaker_mapping[speaker_id] = "patient"
                        print(f"Assigning speaker {speaker_id} as patient")
                
                role = speaker_mapping.get(speaker_id, "patient")
                
                # Add to full transcript
                transcript += text + " "
                
                # Add as new segment or merge with previous if same speaker
                if segments and segments[-1]["speaker"] == role:
                    # Same speaker, merge with previous segment
                    segments[-1]["text"] += " " + text
                else:
                    # New speaker segment
                    segments.append({
                        "speaker": role,
                        "text": text
                    })
            
            # Handle session ending
            def handle_session_stopped(evt):
                print(f"SESSION STOPPED: {evt}")
                done_event.set()
            
            # Handle cancellation
            def handle_canceled(evt):
                print(f"CANCELED: Reason={evt.reason}")
                if evt.reason == speechsdk.CancellationReason.Error:
                    print(f"CANCELED: ErrorCode={evt.error_code}")
                    print(f"CANCELED: ErrorDetails={evt.error_details}")
                done_event.set()
            
            # Connect callbacks
            transcriber.transcribed.connect(handle_transcribed)
            transcriber.session_stopped.connect(handle_session_stopped)
            transcriber.canceled.connect(handle_canceled)
            
            # Start transcription
            print("Starting transcription with diarization...")
            transcriber.start_transcribing_async()
            
            # Wait for completion or timeout
            timeout = min(duration * 1.5, 300)  # At most 5 minutes or 1.5x audio duration
            done_event.wait(timeout)
            
            # Stop transcription
            transcriber.stop_transcribing_async()
            
            # Clean up temp file if created
            if temp_file_created and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Temporary file removed: {temp_path}")
            
            if not segments:
                print("No transcription segments were generated")
                return None, None
            
            print(f"Transcription complete: {len(transcript)} characters in {len(segments)} segments")
            print(f"Speaker mapping: {speaker_mapping}")
            
            return transcript, segments
            
        except Exception as e:
            print(f"Error during Azure transcription: {e}")
            traceback.print_exc()
            return None, None
    
    def transcribe_with_deepgram(self, audio_path):
        """Transcribe French audio using Deepgram Nova with diarization."""
        if not DEEPGRAM_API_KEY:
            print("Deepgram API key not available. Cannot transcribe French audio.")
            return None, None
        
        try:
            print(f"Transcribing {audio_path} with Deepgram Nova model for French...")
            
            # Check audio file format and convert if necessary
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path)
                
                # Deepgram works best with 16kHz mono WAV
                temp_file_created = False
                if sample_rate != 16000 or len(audio_data.shape) > 1:
                    print(f"Converting audio to 16kHz mono format for Deepgram...")
                    temp_path = os.path.join(os.path.dirname(audio_path), f"temp_{os.path.basename(audio_path)}")
                    temp_file_created = True
                    
                    # Convert to mono if needed
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    # Resample to 16kHz
                    if sample_rate != 16000:
                        from scipy import signal
                        audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                    
                    # Save temporary file
                    sf.write(temp_path, audio_data, 16000, 'PCM_16')
                    audio_path_to_use = temp_path
                else:
                    audio_path_to_use = audio_path
            except Exception as e:
                print(f"Warning: Error checking audio format, using original file: {e}")
                audio_path_to_use = audio_path
                temp_file_created = False
            
            with open(audio_path_to_use, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            print(f"Audio file size: {len(audio_data)/1024/1024:.2f} MB")
            
            # Nova model with diarization for French
            url = "https://api.deepgram.com/v1/listen?model=nova-2&diarize=true&language=fr&punctuate=true&utterances=true"
            
            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav"
            }
            
            response = requests.post(url, headers=headers, data=audio_data)
            
            # Clean up temp file if created
            if temp_file_created and 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Temporary file removed: {temp_path}")
            
            if response.status_code != 200:
                print(f"Error: Deepgram API returned status code {response.status_code}")
                print(f"Response: {response.text}")
                return None, None
            
            # Process the response
            response_json = response.json()
            
            # Save raw response for debugging
            debug_dir = os.path.dirname(audio_path)
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f"{os.path.basename(audio_path)}_deepgram_response.json")
            
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(response_json, f, indent=2, ensure_ascii=False)
            
            # Extract transcript
            transcript = ""
            try:
                results = response_json['results']
                transcript = results['channels'][0]['alternatives'][0]['transcript']
                print(f"Successfully extracted transcript ({len(transcript)} characters)")
            except (KeyError, IndexError) as e:
                print(f"Error extracting transcript: {e}")
                return None, None
            
            # Extract diarized segments
            diarized_transcript = []
            
            # Get utterances with speaker information
            if 'utterances' in results:
                utterances = results['utterances']
                print(f"Found {len(utterances)} utterances with speaker information")
                
                for utterance in utterances:
                    text = utterance.get('transcript', '').strip()
                    speaker = utterance.get('speaker', 0)
                    
                    # Map speakers to roles (0=doctor, others=patient)
                    role = "doctor" if speaker == 0 else "patient"
                    
                    # Add to segments or merge with previous
                    if diarized_transcript and diarized_transcript[-1]["speaker"] == role:
                        diarized_transcript[-1]["text"] += " " + text
                    else:
                        diarized_transcript.append({
                            "speaker": role,
                            "text": text
                        })
            
            # If no utterances, fall back to word-level diarization
            elif 'words' in results['channels'][0]['alternatives'][0]:
                words = results['channels'][0]['alternatives'][0]['words']
                print(f"Using word-level diarization with {len(words)} words")
                
                current_speaker = None
                current_text = []
                
                for word in words:
                    if 'speaker' in word:
                        speaker = word['speaker']
                        text = word.get('word', '')
                        
                        # If speaker changes, add previous segment and start new one
                        if speaker != current_speaker and current_speaker is not None and current_text:
                            role = "doctor" if current_speaker == 0 else "patient"
                            diarized_transcript.append({
                                "speaker": role,
                                "text": " ".join(current_text)
                            })
                            current_text = []
                        
                        # Update current speaker and add word
                        current_speaker = speaker
                        current_text.append(text)
                
                # Add final segment
                if current_text and current_speaker is not None:
                    role = "doctor" if current_speaker == 0 else "patient"
                    diarized_transcript.append({
                        "speaker": role,
                        "text": " ".join(current_text)
                    })
            
            if not diarized_transcript:
                print("No diarization information found in Deepgram response")
                return None, None
            
            print(f"Created {len(diarized_transcript)} diarized segments")
            return transcript, diarized_transcript
            
        except Exception as e:
            print(f"Error during Deepgram transcription: {e}")
            traceback.print_exc()
            return None, None
    
    def transcribe_audio(self, audio_path, language=None):
        """Transcribe audio file using the appropriate service based on language."""
        # Determine language from file path if not provided
        if language is None:
            if "fr-CA" in audio_path:
                language = "fr-CA"
            elif "en-CA" in audio_path:
                language = "en-CA"
            else:
                language = "en-CA"  # Default to English
        
        # Route to appropriate service based on language
        if language.startswith("fr"):
            print(f"Using Deepgram Nova for French transcription of {audio_path}")
            return self.transcribe_with_deepgram(audio_path)
        else:
            print(f"Using Azure Speech Services for English transcription of {audio_path}")
            return self.transcribe_with_azure(audio_path)
    
    def transcribe_and_save(self, audio_path, language=None):
        """Transcribe audio and save the result to a JSON file."""
        print(f"Transcribing {audio_path}...")
        
        transcript, segments = self.transcribe_audio(audio_path, language)
        
        if transcript and segments:
            # Determine output path - create in transcript subdirectory
            audio_dir = os.path.dirname(audio_path)
            if "/audio/" in audio_dir:
                # Replace audio with transcripts in the path
                transcript_dir = audio_dir.replace("/audio/", "/transcripts/")
            else:
                # Create transcripts dir at same level as audio dir
                specialty_lang_dir = os.path.dirname(audio_dir)
                transcript_dir = os.path.join(specialty_lang_dir, "transcripts")
            
            os.makedirs(transcript_dir, exist_ok=True)
            
            # Create output filename - same as audio but with .json extension
            basename = os.path.basename(audio_path)
            output_file = os.path.join(transcript_dir, f"{os.path.splitext(basename)[0]}.json")
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            
            print(f"Transcript saved to {output_file}")
            return output_file
        else:
            print(f"Failed to transcribe {audio_path}")
            return None

def process_specialty(specialty, language="all"):
    """Process all audio files for a given specialty and language."""
    languages = ["en-CA", "fr-CA"] if language == "all" else [language]
    
    # Initialize transcriber
    transcriber = AudioTranscriber()
    
    for lang in languages:
        # Skip English if Azure key is not available
        if lang == "en-CA" and not AZURE_SPEECH_KEY:
            print("Skipping English transcription as AZURE_SPEECH_KEY is not available.")
            continue
            
        # Skip French if Deepgram key is not available
        if lang == "fr-CA" and not DEEPGRAM_API_KEY:
            print("Skipping French transcription as DEEPGRAM_API_KEY is not available.")
            continue
        
        # Set up directory paths
        audio_dir = os.path.join(BASE_DIR, specialty, lang, "audio")
        
        # Ensure audio directory exists
        if not os.path.exists(audio_dir):
            print(f"Audio directory does not exist: {audio_dir}")
            continue
        
        # Find all WAV files
        audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
        
        if not audio_files:
            print(f"No WAV files found in {audio_dir}")
            continue
            
        print(f"Found {len(audio_files)} audio files in {audio_dir}")
        
        # Process each file
        for audio_file in tqdm(audio_files, desc=f"Transcribing {lang} {specialty} conversations"):
            transcriber.transcribe_and_save(audio_file, lang)

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
    if language == "en-CA" and not AZURE_SPEECH_KEY:
        print("Azure Speech key not found. Set AZURE_SPEECH_KEY in your .env file.")
        return False
    
    if language == "fr-CA" and not DEEPGRAM_API_KEY:
        print("Deepgram API key not found. Set DEEPGRAM_API_KEY in your .env file.")
        return False
    
    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return False
    
    # Initialize transcriber
    transcriber = AudioTranscriber()
    
    # Transcribe the audio
    return transcriber.transcribe_and_save(audio_file_path, language)

def main():
    """Parse command line arguments and run the transcription process."""
    parser = argparse.ArgumentParser(description='Transcribe medical conversations with Azure (English) and Deepgram (French)')
    
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
        
        # Determine language from argument or filename
        if args.lang and args.lang != 'all':
            language = args.lang
        elif 'fr-CA' in args.audio:
            language = 'fr-CA'
        elif 'en-CA' in args.audio:
            language = 'en-CA'
        else:
            language = 'en-CA'  # Default to English
            
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
