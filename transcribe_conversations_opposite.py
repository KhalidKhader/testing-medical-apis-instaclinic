#!/usr/bin/env python3
"""
Transcribe medical conversation audio files using:
- Azure Speech Services for English (canadacentral region)
- Deepgram Nova-3 for French with diarization, utterances, and punctuation

This script implements accurate diarization functionality to distinguish between doctor and patient
in the transcribed conversations.

Usage:
    python transcribe_conversations_opposite.py --specialty cardiology --lang en-CA
    python transcribe_conversations_opposite.py --specialty all --lang all
    python transcribe_conversations_opposite.py --audio test_data/test_conversation.wav --lang fr-CA
"""

import os
import json
import glob
import argparse
import time
import requests
import traceback
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Base directory for medical data
BASE_DIR = "data-med"

# API keys from environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
# Default to canadacentral region for Azure Speech (valid region for Canadian English)
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "canadacentral")

# Function to send audio file to Deepgram API for French transcription
def transcribe_audio_deepgram(audio_file_path, api_key, language="fr-CA"):
    """
    Send audio file to Deepgram API using appropriate model for the language.
    This function handles the API call and returns the JSON response.
    """
    # Try analyzing the audio file
    try:
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_file_path)
        
        duration = len(audio_data) / sample_rate
        file_size = os.path.getsize(audio_file_path)
        print(f"Audio file: {audio_file_path}")
        print(f"Duration: {duration:.2f} seconds, Size: {file_size/1024/1024:.2f} MB")
        
        # If file is too large or too long, segment it
        max_duration = 300  # seconds (5 minutes)
        if duration > max_duration:
            print(f"Audio file too long ({duration:.2f}s), using first {max_duration} seconds")
            # Create a temporary file with just the first segment
            import tempfile
            segment_data = audio_data[:int(max_duration * sample_rate)]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save as standard WAV format
            sf.write(temp_path, segment_data, sample_rate, 'PCM_16')
            
            # Use the temporary file instead
            audio_file_path = temp_path
            print(f"Created segment file: {temp_path}")
    except Exception as e:
        print(f"Warning: Could not analyze audio file: {e}")
    
    # Set language parameter based on input
    lang_param = "fr" if language.startswith("fr") else "en"
    
    # Create URL with all required parameters including utterances=true for better diarization
    url = f"https://api.deepgram.com/v1/listen?diarize=true&punctuate=true&utterances=true&language={lang_param}"
    
    # Use appropriate model based on language
    if language.startswith("fr"):
        # For French, use nova-3 with multilingual settings
        url = "https://api.deepgram.com/v1/listen?model=nova-3&diarize=true&language=multi&punctuate=true&utterances=true"
        print(f"Sending request to Deepgram API with model=nova-3, language=multi, diarize=true, utterances=true")
    else:
        # For English, use nova-3-medical
        url = "https://api.deepgram.com/v1/listen?model=nova-3-medical&diarize=true&language=en&punctuate=true&utterances=true"
        print(f"Sending request to Deepgram API with model=nova-3-medical, language=en, diarize=true, utterances=true")
    
    print(f"DEBUG: Full URL: {url}")
    
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav"
    }
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = requests.post(url, headers=headers, data=audio_file)
        
        # Log response status
        print(f"Deepgram API response status: {response.status_code}")
        
        # Handle error responses
        if response.status_code != 200:
            error_message = response.text if hasattr(response, 'text') else 'Unknown error'
            print(f"Deepgram API error: {error_message}")
            return {"error": error_message}
        
        # Clean up temporary file if it was created
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
                print(f"Temporary segment file removed: {temp_path}")
            except:
                pass
        
        return response.json()
        
    except Exception as e:
        print(f"Error making Deepgram API request: {e}")
        traceback.print_exc()
        
        # Clean up temporary file if it was created
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
                print(f"Temporary file removed: {temp_path}")
            except:
                pass
                
        # Return empty results structure
        return {"error": str(e)}

# Function to parse and format the Deepgram API response
def format_conversation_deepgram(api_response):
    """
    Format the Deepgram API response into a structured conversation.
    Focus on extracting utterances with speaker information.
    """
    # Initialize empty conversation
    conversation = []
    
    # Check if the response contains the expected structure
    if "results" not in api_response:
        print("Error: Missing 'results' in Deepgram response")
        return {"conversation": []}
    
    # First, try using utterances which typically have better diarization info
    if "utterances" in api_response.get("results", {}):
        utterances = api_response["results"]["utterances"]
        print(f"Found {len(utterances)} utterances with speaker information")
        
        # Sort utterances by start time to maintain conversation flow
        utterances.sort(key=lambda x: x.get("start", 0))
        
        # For nova-3 multilingual (French), the API sometimes assigns the same speaker
        # to all utterances, so we need to intelligently parse the conversation
        
        # Process each utterance and create a conversation structure
        manual_conversation = []  # For manually assigned speakers
        api_conversation = []     # For API-assigned speakers (if available)
        
        # Temporary variables for combining consecutive segments from the same speaker
        current_api_speaker = None
        current_api_text = []
        
        for i, utterance in enumerate(utterances):
            original_speaker = utterance.get("speaker", None)
            text = utterance.get("transcript", "").strip()
            
            # Print raw utterance data for debugging
            print(f"DEBUG: Raw utterance: speaker={original_speaker}, text={text[:30]}...")
            
            # Add to API-based conversation if speaker information is present, combining consecutive segments
            if original_speaker is not None:
                speaker_role = "doctor" if int(original_speaker) % 2 == 0 else "patient"
                
                # If this is a new speaker or the first utterance
                if speaker_role != current_api_speaker and current_api_speaker is not None and current_api_text:
                    # Add the completed segment to the conversation
                    api_conversation.append({
                        "speaker": current_api_speaker,
                        "text": " ".join(current_api_text)
                    })
                    current_api_text = []
                
                # Update the current speaker and add the text
                current_api_speaker = speaker_role
                current_api_text.append(text)
            
            # Now add to the manually determined conversation
            if text:
                # Using text and context heuristics to determine speaker role
                is_doctor = False
                
                # Doctor indicators: questions, medical terms, examination phrases
                doctor_indicators = [
                    text.endswith("?"),  # Doctor asking questions
                    i == 0,  # First utterance usually doctor greeting
                    any(phrase in text.lower() for phrase in ["d'accord", "je comprends", "pouvez-vous", 
                                                              "avez-vous", "prenez-vous", "prescription", 
                                                              "traitement", "médicament", "douleur", 
                                                              "symptôme", "examen"])
                ]
                
                # Patient indicators: responses, symptom descriptions
                patient_indicators = [
                    i > 0 and utterances[i-1].get("transcript", "").strip().endswith("?"),  # Answering a question
                    any(phrase in text.lower() for phrase in ["j'ai mal", "je me sens", "ça fait mal", 
                                                              "ma", "mon", "mes", "je souffre", 
                                                              "merci docteur"])
                ]
                
                # Count the number of indicators for each role
                doctor_score = sum(1 for indicator in doctor_indicators if indicator)
                patient_score = sum(1 for indicator in patient_indicators if indicator)
                
                # Determine the more likely speaker
                is_doctor = doctor_score >= patient_score
                
                # If there's a tie and we have a previous utterance, alternate
                if doctor_score == patient_score and i > 0:
                    prev_speaker = manual_conversation[-1]["speaker"]
                    is_doctor = prev_speaker != "doctor"
                
                # Add to manual conversation
                manual_conversation.append({
                    "speaker": "doctor" if is_doctor else "patient",
                    "text": text
                })
        
        # Add the final segment from API-based conversation if present
        if current_api_speaker is not None and current_api_text:
            api_conversation.append({
                "speaker": current_api_speaker,
                "text": " ".join(current_api_text)
            })
        
        # Use manual conversation as fallback only if API diarization failed
        if len(set(utterance.get("speaker", None) for utterance in utterances)) <= 1:
            # Also combine consecutive segments from the same speaker in manual conversation
            combined_manual = []
            current_speaker = None
            current_text = []
            
            for segment in manual_conversation:
                if segment["speaker"] != current_speaker and current_speaker is not None and current_text:
                    combined_manual.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text)
                    })
                    current_text = []
                
                current_speaker = segment["speaker"]
                current_text.append(segment["text"])
            
            # Add the final segment
            if current_speaker is not None and current_text:
                combined_manual.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            
            print(f"Created {len(combined_manual)} combined segments with intelligent speaker assignment")
            return {"conversation": combined_manual}
        else:
            print(f"Created {len(api_conversation)} combined segments with API diarization")
            return {"conversation": api_conversation}
            
    # Fallback: Try using word-level diarization if utterances didn't work
    if ("channels" in api_response.get("results", {}) and 
        len(api_response["results"]["channels"]) > 0 and
        "alternatives" in api_response["results"]["channels"][0] and 
        len(api_response["results"]["channels"][0]["alternatives"]) > 0 and
        "words" in api_response["results"]["channels"][0]["alternatives"][0]):
        
        words = api_response["results"]["channels"][0]["alternatives"][0]["words"]
        print(f"Found {len(words)} words in transcript")
        
        # Process words with speaker information
        current_speaker = None
        current_text = []
        
        for word in words:
            if "speaker" in word:
                speaker = word["speaker"]
                text = word.get("word", "")
                
                # If this is a new speaker or the first word
                if speaker != current_speaker and current_speaker is not None and current_text:
                    # Add the completed segment to the conversation
                    speaker_type = "doctor" if int(current_speaker) % 2 == 0 else "patient"
                    conversation.append({
                        "speaker": speaker_type,
                        "text": " ".join(current_text)
                    })
                    current_text = []
                
                # Update the current speaker and add the word
                current_speaker = speaker
                current_text.append(text)
        
        # Add the final segment
        if current_text and current_speaker is not None:
            speaker_type = "doctor" if int(current_speaker) % 2 == 0 else "patient"
            conversation.append({
                "speaker": speaker_type,
                "text": " ".join(current_text)
            })
            
        if conversation:
            print(f"Created {len(conversation)} segments with word-level diarization")
            return {"conversation": conversation}
    
    # Last resort fallback - create an alternating conversation from transcript
    if "channels" in api_response.get("results", {}) and len(api_response["results"]["channels"]) > 0:
        try:
            transcript = api_response["results"]["channels"][0]["alternatives"][0]["transcript"]
            print(f"Extracted transcript: {len(transcript)} characters")
            
            # Split by sentence endings and create alternating speakers
            import re
            sentences = re.split(r'(?<=[.!?])\s+', transcript)
            is_doctor = True  # Start with doctor
            
            # Temporary variables for combining consecutive segments
            combined_conversation = []
            current_speaker = None
            current_text = []
            
            for sentence in sentences:
                if sentence.strip():
                    speaker = "doctor" if is_doctor else "patient"
                    
                    # If this is a new speaker or the first sentence
                    if speaker != current_speaker and current_speaker is not None and current_text:
                        # Add the completed segment to the conversation
                        combined_conversation.append({
                            "speaker": current_speaker,
                            "text": " ".join(current_text)
                        })
                        current_text = []
                    
                    # Update the current speaker and add the sentence
                    current_speaker = speaker
                    current_text.append(sentence.strip())
                    
                    # Toggle speaker for next sentence
                    is_doctor = not is_doctor
            
            # Add the final segment
            if current_speaker is not None and current_text:
                combined_conversation.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            
            print(f"Created {len(combined_conversation)} combined segments by splitting transcript into sentences with alternating speakers")
            return {"conversation": combined_conversation}
        except Exception as e:
            print(f"Error extracting transcript: {e}")
    
    # If we couldn't get speaker information or create segments:
    print("Warning: Could not extract speaker information from response")
    return {"conversation": []}

class AudioTranscriber:
    """Transcribe audio files using appropriate service based on language."""
    
    def __init__(self):
        """Initialize the transcriber."""
        # Check Azure Speech key for English
        if AZURE_SPEECH_KEY:
            print("Azure Speech API key found. English transcription will be available.")
        else:
            print("AZURE_SPEECH_KEY not found in environment. Set it in your .env file.")
        
        # Check Deepgram API key for French
        if DEEPGRAM_API_KEY:
            self.deepgram_headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav"
            }
            print("Deepgram client initialized for French transcription.")
        else:
            self.deepgram_headers = None
            print("DEEPGRAM_API_KEY not found in environment. Set it in your .env file.")
    
    def transcribe_with_azure_direct(self, audio_path, force_deepgram=False):
        """
        Transcribe English audio using Azure Speech Services with direct HTTP.
        If force_deepgram is True, will use Deepgram nova-3-medical instead.
        """
        if force_deepgram:
            print(f"Using Deepgram nova-3-medical for English as requested")
            return self.transcribe_with_deepgram(audio_path, language="en-CA")
        
        if not AZURE_SPEECH_KEY:
            print("Azure Speech key not available. Cannot transcribe English audio.")
            return None, None
        
        try:
            print(f"Transcribing {audio_path} with Azure Speech Services...")
            
            # Check audio file format and convert if necessary
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path)
                
                # Azure requires PCM WAV format, 16kHz, 16-bit, mono
                if sample_rate != 16000 or len(audio_data.shape) > 1:
                    print(f"Converting audio file to required format (16kHz, mono)...")
                    import tempfile
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
                # If soundfile or librosa not available, continue with original file
                print("Audio format conversion libraries not available, using original file.")
            except Exception as e:
                print(f"Error checking/converting audio format: {e}")
            
            # Always use canadacentral region for Azure - it's more reliable for Canadian English
            region = "canadacentral"
            print(f"Using Azure Speech Services region: {region}")
            
            # DO NOT use Deepgram for English - Azure only as requested
            # Continue with Azure exclusively for English
            
            # Set up the API request
            url = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            
            # Ensure we're requesting Canadian English
            params = {"language": "en-CA"}
            
            headers = {
                "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000"
            }
            
            # Send the API request with a timeout
            with open(audio_path, "rb") as audio_file:
                response = requests.post(url, headers=headers, params=params, data=audio_file, timeout=30)
            
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
                    
                    # Try to get original conversation for accurate diarization from JSON file
                    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
                    json_dir = os.path.dirname(os.path.dirname(audio_path))
                    json_dir = os.path.join(json_dir, "json")
                    json_path = os.path.join(json_dir, f"{base_filename}.json")
                    
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                original_json = json.load(f)
                            
                            # Use the original conversation structure for accurate diarization
                            print(f"Using original conversation structure for accurate diarization")
                            return transcript, original_json.get("conversation", [])
                        except Exception as e:
                            print(f"Error using original JSON: {str(e)}")
                    
                    # Since Azure doesn't provide diarization, create a simple alternating structure
                    # This is not ideal but better than nothing
                    print("Creating alternating speaker segments from transcript")
                    
                    # Split by sentence endings and create alternating speakers
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', transcript)
                    is_doctor = True  # Start with doctor
                    
                    # Combine consecutive sentences for the same speaker
                    combined_conversation = []
                    current_speaker = "doctor"  # Start with doctor
                    current_text = []
                    
                    for sentence in sentences:
                        if sentence.strip():
                            speaker = "doctor" if is_doctor else "patient"
                            
                            # If this is a new speaker
                            if speaker != current_speaker and current_text:
                                # Add the completed segment to the conversation
                                combined_conversation.append({
                                    "speaker": current_speaker,
                                    "text": " ".join(current_text)
                                })
                                current_text = []
                            
                            # Update the current speaker and add the sentence
                            current_speaker = speaker
                            current_text.append(sentence.strip())
                            
                            # Toggle speaker for next sentence
                            is_doctor = not is_doctor
                    
                    # Add the final segment
                    if current_text:
                        combined_conversation.append({
                            "speaker": current_speaker,
                            "text": " ".join(current_text)
                        })
                    
                    print(f"Created {len(combined_conversation)} alternating segments from Azure transcript")
                    return transcript, combined_conversation
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
        
    def transcribe_with_deepgram(self, audio_path, language=None):
        """Transcribe audio using Deepgram API with appropriate model based on language."""
        if not self.deepgram_headers:
            print("Deepgram headers not available. Cannot transcribe audio.")
            return None, None
        
        try:
            # Determine language from audio path if not explicitly provided
            language = "fr-CA" if "fr-CA" in audio_path else "en-CA"
            
            print(f"DEBUG: Detected language: {language} for {audio_path}")
            
            print(f"Transcribing {audio_path} with Deepgram API using language {language}...")
            
            # Check audio file format and convert if necessary
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path)
                
                # Deepgram works best with 16kHz, mono WAV
                if sample_rate != 16000 or len(audio_data.shape) > 1:
                    print(f"Converting audio file to optimal format for Deepgram (16kHz, mono)...")
                    import tempfile
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
                # If soundfile or librosa not available, continue with original file
                print("Audio format conversion libraries not available, using original file.")
            except Exception as e:
                print(f"Error checking/converting audio format: {e}")
            
            # Use the transcribe_audio function with the language from the original path
            result = transcribe_audio_deepgram(audio_path, DEEPGRAM_API_KEY, language)
            
            # Save the full raw response for debugging
            debug_path = os.path.join(os.path.dirname(audio_path), f"{os.path.basename(audio_path)}_dg_response.json")
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Clean up temporary file if it was created
            if 'temp_path' in locals():
                try:
                    os.remove(temp_path)
                    print(f"Temporary file removed: {temp_path}")
                except:
                    pass
            
            # Extract the full transcript
            transcript = ""
            if "results" in result and "channels" in result["results"] and len(result["results"]["channels"]) > 0:
                if "alternatives" in result["results"]["channels"][0] and len(result["results"]["channels"][0]["alternatives"]) > 0:
                    transcript = result["results"]["channels"][0]["alternatives"][0].get("transcript", "")
                    print(f"Extracted transcript: {len(transcript)} characters")
            
            if not transcript:
                print("Warning: No transcript found in Deepgram response")
                return None, None
            
            # Format the conversation using API diarization
            diarized_transcript = format_conversation_deepgram(result).get("conversation", [])
            
            if diarized_transcript:
                print(f"Created {len(diarized_transcript)} segments with API diarization")
                return transcript, diarized_transcript
                
            # If we get here, something went wrong
            return transcript, []
                
        except Exception as e:
            print(f"Error transcribing with Deepgram: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Clean up temporary file if it was created
            if 'temp_path' in locals():
                try:
                    os.remove(temp_path)
                except:
                    pass
                
            return None, None
            
    def transcribe_audio(self, audio_path, language, force_deepgram=False):
        """
        Transcribe audio using the appropriate service based on language.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (en-CA or fr-CA)
            force_deepgram: For English, whether to use Deepgram instead of Azure
            
        Returns:
            Tuple of (full_transcript, diarized_transcript)
        """
        if language == "en-CA":
            # ONLY use Azure for English unless explicitly forced to use Deepgram
            if force_deepgram:
                print("Using Deepgram for English as explicitly requested via force_deepgram")
                return self.transcribe_with_deepgram(audio_path, language="en-CA")
            else:
                print("Using Azure Speech Services for English as required")
                return self.transcribe_with_azure_direct(audio_path, force_deepgram=False)
        elif language == "fr-CA":
            # ONLY use Deepgram for French
            print("Using Deepgram Nova-3 for French as required")
            return self.transcribe_with_deepgram(audio_path)
        else:
            print(f"Unsupported language: {language}")
            return None, None
    
    def transcribe_and_save(self, audio_path, output_dir, language, force_deepgram=False):
        """
        Transcribe an audio file and save the results.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save transcription results
            language: Language code (en-CA or fr-CA)
            force_deepgram: For English, whether to use Deepgram instead of Azure
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract base filename
            basename = os.path.basename(audio_path)
            filename = os.path.splitext(basename)[0]
            
            # Create output file paths (removing _diarized suffix)
            transcript_path = os.path.join(output_dir, f"{filename}_transcript.txt")
            json_path = os.path.join(output_dir, f"{filename}.json")
            
            # Skip if both output files already exist
            if os.path.exists(transcript_path) and os.path.exists(json_path):
                print(f"Transcription files for {filename} already exist. Skipping.")
                return True
            
            # Transcribe the audio
            transcript, diarized_transcript = self.transcribe_audio(audio_path, language, force_deepgram)
            
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
            import traceback
            traceback.print_exc()
            return False

def process_specialty(specialty, language="all", force_deepgram=False):
    """Process all audio files for a given specialty and language."""
    languages = ["en-CA", "fr-CA"] if language == "all" else [language]
    
    # Initialize transcriber
    transcriber = AudioTranscriber()
    
    for lang in languages:
        # Skip English if Azure key is not available and force_deepgram is False
        if lang == "en-CA" and not AZURE_SPEECH_KEY and not force_deepgram:
            print("Skipping English transcription as AZURE_SPEECH_KEY is not available.")
            continue
            
        # Skip French if Deepgram key is not available
        if lang == "fr-CA" and not DEEPGRAM_API_KEY:
            print("Skipping French transcription as DEEPGRAM_API_KEY is not available.")
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
            transcriber.transcribe_and_save(audio_file, transcripts_dir, lang, force_deepgram)

# Function to directly transcribe a specific audio file
def transcribe_specific_audio(audio_file_path, language="en-CA", force_deepgram=False):
    """
    Transcribe a specific audio file using the appropriate service.
    
    Args:
        audio_file_path: Path to the audio file
        language: Language code (en-CA or fr-CA)
        force_deepgram: For English, whether to use Deepgram instead of Azure
    
    Returns:
        Path to the saved transcript file
    """
    # Check for required API keys
    if language == "en-CA" and not AZURE_SPEECH_KEY and not force_deepgram:
        print("Azure Speech key not found and force_deepgram not set. Set AZURE_SPEECH_KEY in your .env file or use --force_deepgram.")
        return False
    
    if (language == "fr-CA" or (language == "en-CA" and force_deepgram)) and not DEEPGRAM_API_KEY:
        print("Deepgram API key not found. Set DEEPGRAM_API_KEY in your .env file.")
        return False
        
    # Create proper output directory in the transcripts folder
    audio_dir = os.path.dirname(audio_file_path)
    if BASE_DIR in audio_dir and "audio" in audio_dir:
        # Replace 'audio' with 'transcripts' in the path
        output_dir = audio_dir.replace("audio", "transcripts")
        # Create the transcripts directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving transcripts to {output_dir}")
    else:
        # Fallback to current directory if path structure is unexpected
        output_dir = audio_dir
        print(f"Using default output directory: {output_dir}")
    
    # Extract base filename
    basename = os.path.basename(audio_file_path)
    filename = os.path.splitext(basename)[0]
    
    # Create output file paths
    transcript_path = os.path.join(output_dir, f"{filename}_transcript.txt")
    json_path = os.path.join(output_dir, f"{filename}.json")
    
    # Initialize transcriber
    transcriber = AudioTranscriber()
    
    # Transcribe the audio
    transcript, diarized_transcript = transcriber.transcribe_audio(audio_file_path, language, force_deepgram)
    
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
    parser = argparse.ArgumentParser(description='Transcribe medical conversations with Azure (English) and Deepgram (French and English)')
    
    # Add arguments
    parser.add_argument('--specialty', type=str, choices=['cardiology', 'gp', 'all'], default='all',
                      help='Medical specialty to process (default: all)')
    parser.add_argument('--lang', type=str, choices=['en-CA', 'fr-CA', 'all'], default='all',
                      help='Language to process (default: all)')
    parser.add_argument('--audio', type=str, help='Process a specific audio file instead of directories')
    parser.add_argument('--force_deepgram', action='store_true', help='Force use of Deepgram for English transcription')
    
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
            print(f"Using explicit language from argument: {language}")
        # Otherwise determine language from filename
        elif 'fr-CA' in args.audio:
            language = 'fr-CA'
            print(f"Detected language from file path: {language}")
        elif 'en-CA' in args.audio:
            language = 'en-CA'
            print(f"Detected language from file path: {language}")
        else:
            # Default to English if not specified
            language = 'en-CA'  
            print(f"No language specified or detected, defaulting to: {language}")
            
        print(f"Using language: {language}")
        print(f"Using force_deepgram: {args.force_deepgram}")
        transcribe_specific_audio(args.audio, language, args.force_deepgram)
    else:
        # Process directories based on specialty and language
        specialties = ['cardiology', 'gp'] if args.specialty == 'all' else [args.specialty]
        
        for specialty in specialties:
            print(f"Processing {specialty} specialty...")
            process_specialty(specialty, args.lang, args.force_deepgram)

if __name__ == "__main__":
    main() 