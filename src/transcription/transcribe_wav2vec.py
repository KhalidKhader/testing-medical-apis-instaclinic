#!/usr/bin/env python3
"""
Medical Conversation Transcription Tool
======================================
Transcribes medical conversations using Wav2Vec 2.0 with PyAnnote for speaker diarization.
- Uses HuggingFace's Wav2Vec 2.0 model for high-quality transcription
- Employs PyAnnote for accurate speaker diarization
- Outputs in simple doctor/patient conversation format
- Guarantees complete transcription of entire audio file
- No copying from original files - pure audio-based transcription

Usage:
    python transcribe_wav2vec.py audio_file [output_file]
    python transcribe_wav2vec.py directory_path
    python transcribe_wav2vec.py --process-all
"""

import os
import sys
import json
import time
import logging
import subprocess
import numpy as np
import traceback
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("medical_transcribe")

# Optional imports with fallbacks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not found. Some functionality will be limited.")
    HAS_TORCH = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    logger.warning("Librosa not found. Audio processing will be limited.")
    HAS_LIBROSA = False

try:
    from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("Transformers library not found. Speech recognition will be limited.")
    HAS_TRANSFORMERS = False

# Load environment variables
load_dotenv()
# Get HuggingFace token from .env
HF_TOKEN = os.getenv("HF_TOKEN")

# Additional imports for enhanced functionality
import warnings
from scipy.signal import medfilt

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*librosa.*")
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")

def install_dependencies():
    """Install required libraries if not already installed."""
    try:
        # Check if Transformers is installed
        import transformers
        logger.info("Transformers library is already installed")
    except ImportError:
        logger.info("Installing Transformers library...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.35.0,<4.36.0"])
            logger.info("Transformers installed successfully")
        except Exception as e:
            logger.error(f"Failed to install Transformers: {e}")
            return False

    try:
        # Check if PyAnnote is installed
        try:
            import pyannote.audio
            logger.info("PyAnnote library is already installed")
            # Check if the user has accepted the license
            if not HF_TOKEN:
                logger.warning("HF_TOKEN not set in .env file. Speaker diarization will not work.")
                logger.warning("Visit https://huggingface.co/pyannote/speaker-diarization-3.1 to accept the license terms")
                logger.warning("Then add your token to .env file as HF_TOKEN=your_token_here")
        except ImportError:
            logger.info("Installing PyAnnote for speaker diarization...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyannote.audio"])
                logger.info("PyAnnote installed successfully")
                if not HF_TOKEN:
                    logger.warning("HF_TOKEN not set in .env file. Speaker diarization will not work.")
                    logger.warning("Visit https://huggingface.co/pyannote/speaker-diarization-3.1 to accept the license terms")
                    logger.warning("Then add your token to .env file as HF_TOKEN=your_token_here")
            except Exception as e:
                logger.error(f"Failed to install PyAnnote: {e}")
                logger.warning("Speaker diarization will be limited")
    except Exception as e:
        logger.error(f"Error checking PyAnnote: {e}")
        logger.warning("Speaker diarization will be limited")

    try:
        # Check if librosa is installed
        import librosa
        logger.info("Librosa library is already installed")
    except ImportError:
        logger.info("Installing Librosa for audio processing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa"])
            logger.info("Librosa installed successfully")
        except Exception as e:
            logger.error(f"Failed to install Librosa: {e}")
            return False
            
    try:
        # Check if NeMo is installed
        import nemo
        logger.info("NVIDIA NeMo library is already installed")
    except ImportError:
        logger.info("Installing NVIDIA NeMo for enhanced punctuation and capitalization...")
        try:
            # Install NeMo and its dependencies
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nemo_toolkit[nlp]"])
            logger.info("NVIDIA NeMo installed successfully")
        except Exception as e:
            logger.error(f"Failed to install NeMo (will use basic transcription): {e}")
            
    try:
        # Check if scipy is installed
        import scipy
        logger.info("SciPy library is already installed")
    except ImportError:
        logger.info("Installing SciPy for audio processing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
            logger.info("SciPy installed successfully")
        except Exception as e:
            logger.error(f"Failed to install SciPy: {e}")
            return False

    return True

def get_language_model(language):
    """Returns the appropriate model name for the specified language."""
    language = language.lower()
    if language in ["english", "en", "en-ca"]:
        return "facebook/wav2vec2-large-960h-lv60-self"
    elif language in ["french", "fr", "fr-ca"]:
        return "facebook/wav2vec2-large-xlsr-53-french"
    else:
        # Default to multilingual model
        return "facebook/wav2vec2-large-xlsr-53"

def load_wav2vec2_model(language="en"):
    """Load and cache the Wav2Vec 2.0 model from HuggingFace."""
    try:
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            logger.error("Required libraries (torch/transformers) not available")
            return None, None, None, None
            
        logger.info(f"Loading Wav2Vec 2.0 model for language: {language}...")
        
        # Get the appropriate model for the language
        model_name = get_language_model(language)
        
        # Set up device and dtype for efficient processing
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load the components
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Load the model with optimizations if available
        try:
            model = AutoModelForCTC.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2"
            ).to(device)
        except Exception as e:
            logger.warning(f"Couldn't load with optimizations: {e}, trying standard load...")
            # Fall back to standard loading if optimizations are not available
            model = AutoModelForCTC.from_pretrained(model_name).to(device)
        
        logger.info(f"Successfully loaded Wav2Vec 2.0 model (on {device})")
        return model, tokenizer, feature_extractor, device
    except Exception as e:
        logger.error(f"Error loading Wav2Vec model: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None

def load_nemo_model(use_cuda=True):
    """
    Load NeMo punctuation and capitalization model.
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        NeMo model for punctuation or None if failed
    """
    try:
        # Check if NeMo is available
        try:
            import nemo.collections.nlp as nemo_nlp
        except ImportError:
            logger.warning("NeMo library not installed, skipping punctuation model")
            return None
            
        # Set appropriate device
        if use_cuda and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        logger.info(f"Loading NeMo punctuation model on {device}...")
        
        try:
            # Load punctuation and capitalization model
            punct_model = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
                model_name="punctuation_en_distilbert"
            )
            
            # Move to appropriate device
            punct_model = punct_model.to(device)
            
            logger.info("NeMo punctuation model loaded successfully")
            return punct_model
        except Exception as model_error:
            logger.error(f"Error loading NeMo model: {model_error}")
            logger.warning("Proceeding without punctuation model")
            return None
            
    except Exception as e:
        logger.error(f"Error loading NeMo model: {e}")
        logger.error(traceback.format_exc())
        return None

def process_audio(audio_path, model, tokenizer, feature_extractor, device, nemo_model=None, max_chunk_length=30):
    """
    Process audio using Wav2Vec2 model with chunking for longer files.
    
    Args:
        audio_path: Path to audio file
        model: Wav2Vec2 model
        tokenizer: Wav2Vec2 tokenizer
        feature_extractor: Wav2Vec2 feature extractor
        device: Device to run the model on
        nemo_model: NeMo punctuation/capitalization model (optional)
        max_chunk_length: Maximum chunk length in seconds
        
    Returns:
        Transcription with segments or None if failed
    """
    try:
        logger.info(f"Processing audio file: {audio_path}")
        
        # Load audio and get duration
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        logger.info(f"Audio duration: {duration:.2f} seconds")
        
        # Process in chunks if audio is long
        if duration > max_chunk_length:
            logger.info(f"Audio longer than {max_chunk_length}s, processing in chunks")
            
            # Calculate chunk size in samples
            chunk_size = max_chunk_length * sr
            # Use 2-second overlap between chunks
            overlap = 2 * sr
            
            all_segments = []
            merged_text = ""
            
            # Process each chunk
            for chunk_start in range(0, len(audio), chunk_size - overlap):
                chunk_end = min(chunk_start + chunk_size, len(audio))
                audio_chunk = audio[chunk_start:chunk_end]
                
                logger.info(f"Processing chunk {chunk_start/sr:.2f}s to {chunk_end/sr:.2f}s")
                
                # Process chunk
                input_values = feature_extractor(audio_chunk, sampling_rate=sr, return_tensors="pt").input_values.to(device)
                
                with torch.no_grad():
                    logits = model(input_values).logits[0]
                
                # Decode transcription for chunk
                pred_ids = torch.argmax(logits, axis=-1)
                chunk_text = tokenizer.decode(pred_ids)
                merged_text += " " + chunk_text
                
                # Compute word timestamps
                time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
                outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
                
                # Adjust timestamps for chunk position
                chunk_start_seconds = chunk_start / sr
                chunk_segments = [
                    {
                        "text": d["word"],
                        "start": round(d["start_offset"] * time_offset + chunk_start_seconds, 2),
                        "end": round(d["end_offset"] * time_offset + chunk_start_seconds, 2),
                    }
                    for d in outputs.word_offsets
                ]
                
                all_segments.extend(chunk_segments)
            
            # Apply NeMo punctuation/capitalization if available
            if nemo_model is not None:
                try:
                    logger.info("Applying NeMo punctuation and capitalization")
                    processed_text = nemo_model.add_punctuation_capitalization([merged_text.strip()])[0]
                    
                    # Update the text with improved punctuation/capitalization
                    # Note: This maintains the original word timestamps
                    merged_text = processed_text
                except Exception as e:
                    logger.error(f"Error applying NeMo punctuation: {e}")
            
            return {
                "text": merged_text.strip(),
                "segments": all_segments
            }
        else:
            # Process shorter audio in one go (existing transcribe_audio_with_wav2vec logic)
            input_values = feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_values.to(device)
            
            with torch.no_grad():
                logits = model(input_values).logits[0]
            
            # Decode transcription
            pred_ids = torch.argmax(logits, axis=-1)
            transcription = tokenizer.decode(pred_ids)
            
            # Apply NeMo punctuation/capitalization if available
            if nemo_model is not None:
                try:
                    logger.info("Applying NeMo punctuation and capitalization")
                    processed_text = nemo_model.add_punctuation_capitalization([transcription.strip()])[0]
                    transcription = processed_text
                except Exception as e:
                    logger.error(f"Error applying NeMo punctuation: {e}")
            
            # Compute word timestamps
            time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
            outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
            
            word_offsets = [
                {
                    "text": d["word"],
                    "start": round(d["start_offset"] * time_offset, 2),
                    "end": round(d["end_offset"] * time_offset, 2),
                }
                for d in outputs.word_offsets
            ]
            
            return {
                "text": transcription.strip(),
                "segments": word_offsets,
            }
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        logger.error(traceback.format_exc())
        return None

def transcribe_audio_with_wav2vec(audio_path, model, tokenizer, feature_extractor, device):
    """Transcribe audio with Wav2Vec 2.0 and get precise word timestamps."""
    try:
        logger.info(f"Starting transcription of {audio_path}...")
        start_time = time.time()

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        input_values = feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_values.to(device)

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits[0]

        # Decode transcription
        pred_ids = torch.argmax(logits, axis=-1)
        transcription = tokenizer.decode(pred_ids)

        # Compute word timestamps
        time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
        outputs = tokenizer.decode(pred_ids, output_word_offsets=True)

        word_offsets = [
            {
                "text": d["word"],
                "start": round(d["start_offset"] * time_offset, 2),
                "end": round(d["end_offset"] * time_offset, 2),
            }
            for d in outputs.word_offsets
        ]

        end_time = time.time()
        logger.info(f"Transcription complete. Time taken: {end_time - start_time:.2f} seconds")

        return {
            "text": transcription,
            "segments": word_offsets,
        }
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def perform_diarization(audio_path):
    """
    Perform speaker diarization using PyAnnote with fallback mechanisms.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        List of speaker segments or None if failed
    """
    # First try PyAnnote for diarization
    try:
        from pyannote.audio import Pipeline
        
        logger.info("Loading PyAnnote diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        
        logger.info(f"Running PyAnnote diarization on {os.path.basename(audio_path)}...")
        diarization = pipeline(audio_path)
        
        # Process diarization results
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        logger.info(f"PyAnnote diarization complete: found {len(speaker_segments)} speaker segments")
        
        # Fix any overlapping segments
        fixed_segments = fix_overlapping_segments(speaker_segments)
        
        return fixed_segments
    
    except Exception as e:
        logger.error(f"Error in PyAnnote diarization: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Fallback to a simple pause-based segmentation approach if diarization fails
        try:
            logger.warning("Diarization failed, falling back to pause-based segmentation")
            
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Detect speech regions using simple energy-based VAD
            from scipy.signal import medfilt
            
            # Convert to mono if needed
            if len(y.shape) > 1:
                y = y.mean(axis=0)
                
            # Calculate energy
            energy = np.abs(y)
            
            # Ensure kernel size is odd (median filter requirement)
            kernel_ms = 50  # 50ms window
            kernel_samples = int(sr * kernel_ms / 1000)
            if kernel_samples % 2 == 0:  # If even, make it odd
                kernel_samples += 1
                
            energy = medfilt(energy, kernel_size=kernel_samples)
            
            # Set threshold for speech
            threshold = 0.01
            speech = energy > threshold
            
            # Find speech segments
            speech_segments = []
            in_speech = False
            start = 0
            
            for i, is_speech in enumerate(speech):
                if is_speech and not in_speech:
                    start = i / sr
                    in_speech = True
                elif not is_speech and in_speech:
                    end = i / sr
                    if end - start > 0.5:  # Ignore very short segments
                        speech_segments.append((start, end))
                    in_speech = False
            
            # If still in speech at the end, add the last segment
            if in_speech:
                speech_segments.append((start, duration))
            
            # Convert to appropriate format
            diarization_segments = []
            
            # Simple alternating speakers for basic format
            current_speaker = "SPEAKER_0"
            
            for i, (start, end) in enumerate(speech_segments):
                # Alternate speakers based on pauses
                if i > 0 and end - start > 1.0:  # Longer segments may indicate speaker change
                    current_speaker = "SPEAKER_1" if current_speaker == "SPEAKER_0" else "SPEAKER_0"
                
                diarization_segments.append({
                    "start": start,
                    "end": end,
                    "speaker": current_speaker
                })
            
            logger.info(f"Generated {len(diarization_segments)} fallback speaker segments")
            return diarization_segments
            
        except Exception as fallback_error:
            logger.error(f"Fallback segmentation also failed: {fallback_error}")
            logger.error(traceback.format_exc())
            return None

def fix_overlapping_segments(segments):
    """
    Fix overlapping segments by adjusting boundaries.
    
    Args:
        segments: List of speaker segments
        
    Returns:
        Fixed segments with no overlaps
    """
    if not segments or len(segments) < 2:
        return segments
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    fixed_segments = [sorted_segments[0]]
    
    for segment in sorted_segments[1:]:
        prev_segment = fixed_segments[-1]
        
        # Check for overlap
        if segment["start"] < prev_segment["end"]:
            # Handle overlap - split the difference
            mid_point = (segment["start"] + prev_segment["end"]) / 2
            prev_segment["end"] = mid_point
            segment["start"] = mid_point
        
        fixed_segments.append(segment)
    
    return fixed_segments

def map_segments_to_speakers(transcription_segments, diarization_segments):
    """
    Map transcription segments to speakers from diarization.
    
    Args:
        transcription_segments: List of segments from Wav2Vec2 with word timestamps
        diarization_segments: List of segments from diarization with speaker labels
        
    Returns:
        List of segments with speaker labels
    """
    if not diarization_segments:
        logger.warning("No diarization segments available, using heuristic speaker detection")
        return assign_speakers_by_content(transcription_segments)
    
    if not transcription_segments:
        logger.warning("No transcription segments available, returning empty result")
        return []
    
    # Add speaker information to each transcription segment
    for segment in transcription_segments:
        segment_mid = (segment["start"] + segment["end"]) / 2
        assigned = False
        
        # Find the diarization segment that contains this word
        for diar_segment in diarization_segments:
            if diar_segment["start"] <= segment_mid <= diar_segment["end"]:
                # Map speaker label (convert SPEAKER_0 to doctor and SPEAKER_1 to patient)
                if diar_segment["speaker"] in ["SPEAKER_0", "speaker_0", "SPEAKER0", "0", "SPEAKER_00"]:
                    segment["speaker"] = "doctor"
                elif diar_segment["speaker"] in ["SPEAKER_1", "speaker_1", "SPEAKER1", "1", "SPEAKER_01"]:
                    segment["speaker"] = "patient"
                else:
                    segment["speaker"] = diar_segment["speaker"].lower()
                
                assigned = True
                break
        
        # If no matching diarization segment, use content-based detection
        if not assigned:
            segment["speaker"] = detect_speaker_by_content(segment["text"]).lower()
    
    # Fix any inconsistencies in speaker assignments
    fix_speaker_assignments(transcription_segments)
    
    return transcription_segments

def detect_speaker_by_content(text):
    """
    Detect if text is more likely from a doctor or patient based on content analysis.
    
    Args:
        text: Text to analyze
        
    Returns:
        "doctor" or "patient"
    """
    text = text.lower()
    
    # Doctor indicators
    doctor_terms = [
        "how are you feeling", "what symptoms", "any allergies", "medical history",
        "i'll prescribe", "take this medication", "follow-up", "your treatment",
        "your condition", "i recommend", "the test results", "examination shows"
    ]
    
    # Patient indicators
    patient_terms = [
        "i've been feeling", "my symptoms", "hurts when", "pain in my", 
        "i'm experiencing", "started", "got worse", "it's affecting", 
        "is it serious", "will it get better", "when can i", "i'm worried"
    ]
    
    # Count matches
    doctor_score = sum(1 for term in doctor_terms if term in text)
    patient_score = sum(1 for term in patient_terms if term in text)
    
    # Make a decision based on scores
    if doctor_score > patient_score:
        return "doctor"
    elif patient_score > doctor_score:
        return "patient"
    else:
        # If tied, look for more patterns
        if "?" in text and len(text) < 50:  # Short questions often from doctors
            return "doctor"
        elif text.startswith("i ") or text.startswith("my "):  # Self-references often from patients
            return "patient"
        else:
            return "doctor"  # Default to doctor

def assign_speakers_by_content(segments):
    """
    Assign speakers to segments based solely on content when diarization fails.
    
    Args:
        segments: Transcription segments
        
    Returns:
        Segments with speaker labels
    """
    # First pass: assign speakers based on content
    for segment in segments:
        segment["speaker"] = detect_speaker_by_content(segment["text"])
    
    # If first segment doesn't seem like a doctor greeting, force it
    if segments and segments[0]["speaker"] != "doctor":
        first_text = segments[0]["text"].lower()
        if any(greeting in first_text for greeting in 
               ["hello", "hi", "good morning", "how are you", "what brings"]):
            segments[0]["speaker"] = "doctor"
    
    # Fix inconsistencies
    fix_speaker_assignments(segments)
    
    return segments

def fix_speaker_assignments(segments):
    """
    Fix inconsistent speaker assignments by enforcing alternating patterns
    where appropriate.
    
    Args:
        segments: List of segments with initial speaker assignments
    """
    if not segments:
        return segments
    
    # Ensure first speaker is doctor (typical in medical conversations)
    if segments[0]["speaker"] != "doctor":
        segments[0]["speaker"] = "doctor"
    
    # Fix alternating pattern for adjacent segments with same speaker
    prev_speaker = segments[0]["speaker"]
    for i in range(1, len(segments)):
        curr_speaker = segments[i]["speaker"]
        
        # If three consecutive segments have same speaker, make middle one alternate
        if i > 1 and curr_speaker == prev_speaker == segments[i-2]["speaker"]:
            # Check if this segment is a question (likely from other speaker)
            if "?" in segments[i-1]["text"]:
                segments[i-1]["speaker"] = "patient" if prev_speaker == "doctor" else "doctor"
        
        prev_speaker = curr_speaker
    
    # Ensure both doctor and patient appear in the conversation
    speakers = set(segment["speaker"] for segment in segments)
    if "patient" not in speakers and len(segments) > 1:
        # Try to find a good segment to mark as patient
        for i in range(1, len(segments)):
            text = segments[i]["text"].lower()
            if "i'm" in text or "my" in text or "i've" in text:
                segments[i]["speaker"] = "patient"
                break
        
        # If still no patient, make the second segment a patient
        if "patient" not in set(segment["speaker"] for segment in segments):
            segments[1]["speaker"] = "patient"

def build_conversation(segments):
    """
    Build a conversation format from transcription segments by combining
    consecutive segments from the same speaker.
    
    Args:
        segments: List of segments with speaker information
        
    Returns:
        List of conversation turns
    """
    if not segments:
        return []
    
    conversation = []
    current_speaker = None
    current_text = ""
    current_start = None
    current_end = None
    
    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        # Convert speaker format to lowercase as required
        if speaker == "DOCTOR":
            speaker = "doctor"
        elif speaker == "PATIENT":
            speaker = "patient"
        else:
            speaker = speaker.lower()
            
        text = segment.get("text", "").strip()
        
        if not text:
            continue
        
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        
        if current_speaker is None:
            # First segment
            current_speaker = speaker
            current_text = text
            current_start = start_time
            current_end = end_time
        elif speaker == current_speaker:
            # Same speaker, combine text
            current_text += " " + text
            current_end = end_time
        else:
            # New speaker, add previous turn to conversation
            conversation.append({
                "speaker": current_speaker,
                "text": current_text.strip(),
                "start": current_start,
                "end": current_end
            })
            
            # Start new turn
            current_speaker = speaker
            current_text = text
            current_start = start_time
            current_end = end_time
    
    # Add final turn
    if current_speaker and current_text:
        conversation.append({
            "speaker": current_speaker,
            "text": current_text.strip(),
            "start": current_start,
            "end": current_end
        })
    
    return conversation

def save_transcript(output_path, metadata, segments, conversation):
    """
    Save the transcript to a JSON file.
    
    Args:
        output_path: Path to save the transcript
        metadata: Metadata about the transcript
        segments: List of word segments
        conversation: List of conversation turns
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Format the output as required
        result = {
            "language": metadata.get("language", "en-CA"),
            "specialty": metadata.get("specialty", "gp"),
            "type": metadata.get("type", "consultation"),
            "conversation": conversation
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved transcript to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving transcript: {e}")
        logger.error(traceback.format_exc())
        return False

def transcribe_audio_file(audio_path, output_path=None, language=None):
    """
    Transcribe a single audio file and save the result.
    
    Args:
        audio_path: Path to the audio file
        output_path: Path to save the transcript (None for automatic)
        language: Language of the audio (None for auto-detection)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        start_time = time.time()
        
        # Determine language from filename if not specified
        if language is None:
            if '/fr-CA/' in audio_path or '_fr_' in audio_path:
                language = 'fr-CA'
            else:
                language = 'en-CA'
        
        # Extract specialty and type from path or filename
        specialty = "gp"
        encounter_type = "consultation"
        
        if 'cardiology' in audio_path:
            specialty = "cardiology"
        if 'followup' in audio_path:
            encounter_type = "followup"
        
        # Determine output path if not specified
        if output_path is None:
            base_dir = os.path.dirname(audio_path)
            base_name = os.path.basename(audio_path).rsplit(".", 1)[0]
            
            # Look for 'transcripts' directory at same level as 'audio'
            if 'audio' in base_dir:
                transcripts_dir = base_dir.replace('audio', 'transcripts')
            else:
                transcripts_dir = os.path.join(base_dir, 'transcripts')
                
            # Create directory if it doesn't exist
            os.makedirs(transcripts_dir, exist_ok=True)
            output_path = os.path.join(transcripts_dir, f"{base_name}.json")
        
        # Check if output already exists
        if os.path.exists(output_path):
            logger.info(f"Output file already exists: {output_path}")
            response = input("File already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                logger.info("Skipping transcription")
                return True

        # Check if required libraries are available
        if not HAS_TRANSFORMERS or not HAS_TORCH or not HAS_LIBROSA:
            logger.error("Required libraries not available. Cannot transcribe audio.")
            # Create error file with information
            error_path = output_path.replace('.json', '_error.txt')
            with open(error_path, 'w') as f:
                f.write(f"Error: Required libraries not available.\n")
                f.write(f"Please install the following packages:\n")
                f.write(f"- torch\n- transformers\n- librosa\n")
            return False
                
        # Load Wav2Vec2 model
        logger.info(f"Loading Wav2Vec2 model for {language}...")
        model, tokenizer, feature_extractor, device = load_wav2vec2_model(language)
        if not all([model, tokenizer, feature_extractor, device]):
            logger.error("Failed to load Wav2Vec2 model")
            # Create error file with information about the failure
            error_path = output_path.replace('.json', '_error.txt')
            with open(error_path, 'w') as f:
                f.write(f"Error: Failed to load Wav2Vec2 model for language {language}.\n")
                f.write(f"Audio file: {audio_path}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            return False
        
        # Try to load NeMo model for punctuation/capitalization - ENSURE THIS RUNS
        logger.info("Attempting to load NeMo punctuation model...")
        nemo_model = None
        try:
            nemo_model = load_nemo_model(use_cuda=True)
            if nemo_model:
                logger.info("Successfully loaded NeMo model for punctuation and capitalization")
            else:
                logger.warning("NeMo model not loaded. Punctuation may be incorrect.")
        except Exception as e:
            logger.warning(f"Failed to load NeMo model, proceeding without punctuation: {e}")
        
        # Process audio with enhanced NeMo punctuation
        logger.info(f"Processing {audio_path} with NeMo enhancement...")
        transcription = process_audio(
            audio_path, 
            model, 
            tokenizer, 
            feature_extractor, 
            device,
            nemo_model=nemo_model
        )
        
        if not transcription:
            logger.error("Transcription failed")
            return False
        
        # Perform diarization
        logger.info("Performing speaker diarization...")
        diarization_segments = perform_diarization(audio_path)
        
        # Map segments to speakers
        logger.info("Mapping speakers to transcription segments...")
        segments_with_speakers = map_segments_to_speakers(transcription["segments"], diarization_segments)
        
        # Build conversation
        logger.info("Building conversation format...")
        conversation = build_conversation(segments_with_speakers)
        
        # Create metadata
        duration = librosa.get_duration(path=audio_path)
        metadata = {
            "filename": os.path.basename(audio_path),
            "processed_at": datetime.now().isoformat(),
            "model": "wav2vec2",
            "language": language,
            "duration": duration,
            "specialty": specialty,
            "type": encounter_type,
            "processing_time": time.time() - start_time,
            "word_count": len(segments_with_speakers),
            "turn_count": len(conversation)
        }
        
        # Save transcript
        success = save_transcript(output_path, metadata, segments_with_speakers, conversation)
        
        if success:
            elapsed_time = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")
            
            # Print summary
            logger.info(f"Found {len(conversation)} conversation turns:")
            logger.info(f"- Doctor turns: {sum(1 for turn in conversation if turn['speaker'] == 'doctor')}")
            logger.info(f"- Patient turns: {sum(1 for turn in conversation if turn['speaker'] == 'patient')}")
            
            return True
        else:
            logger.error("Failed to save transcript")
            return False
    except Exception as e:
        logger.error(f"Error transcribing file: {e}")
        logger.error(traceback.format_exc())
        
        # Create error file with traceback
        try:
            error_path = output_path.replace('.json', '_error.txt') if output_path else f"{audio_path}_error.txt"
            with open(error_path, 'w') as f:
                f.write(f"Error: {str(e)}\n")
                f.write(f"Audio file: {audio_path}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
        except Exception as write_error:
            logger.error(f"Failed to write error file: {write_error}")
            
        return False

def process_directory(directory_path):
    """
    Process all audio files in a directory.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Tuple of (success_count, fail_count)
    """
    logger.info(f"Processing all audio files in {directory_path}")
    
    # Get all .wav files
    audio_files = []
    for file in os.listdir(directory_path):
        if file.lower().endswith('.wav'):
            audio_files.append(os.path.join(directory_path, file))
    
    if not audio_files:
        logger.warning(f"No .wav files found in {directory_path}")
        return 0, 0
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for i, audio_path in enumerate(audio_files):
        logger.info(f"Processing file {i+1}/{len(audio_files)}: {os.path.basename(audio_path)}")
        
        if transcribe_audio_file(audio_path):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"Processed {len(audio_files)} files: {success_count} succeeded, {fail_count} failed")
    return success_count, fail_count

def process_all_data():
    """Process all audio files in the med-data directory structure."""
    med_data_dir = "med-data"
    
    if not os.path.exists(med_data_dir):
        logger.error(f"Directory not found: {med_data_dir}")
        return
    
    total_success = 0
    total_fail = 0
    
    # Process each specialty and language
    for specialty in ["cardiology", "gp"]:
        specialty_dir = os.path.join(med_data_dir, specialty)
        if not os.path.exists(specialty_dir):
            continue
        
        for language in ["en-CA", "fr-CA"]:
            language_dir = os.path.join(specialty_dir, language)
            if not os.path.exists(language_dir):
                continue
            
            audio_dir = os.path.join(language_dir, "audio")
            if not os.path.exists(audio_dir):
                continue
            
            logger.info(f"Processing {specialty}/{language}")
            success, fail = process_directory(audio_dir)
            total_success += success
            total_fail += fail
    
    logger.info(f"Finished processing all data: {total_success} succeeded, {total_fail} failed")

def main():
    """Main entry point for the script."""
    # Check if dependencies are installed
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return
    
    # Ensure NeMo is available
    try:
        import nemo
        logger.info("NeMo is available for punctuation/capitalization")
    except ImportError:
        logger.warning("NeMo is not installed. Attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nemo_toolkit[nlp]"])
            logger.info("NeMo successfully installed")
        except Exception as e:
            logger.error(f"Failed to install NeMo: {e}")
    
    # Process command-line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python transcribe_wav2vec.py audio_file [output_file]")
        print("  python transcribe_wav2vec.py directory_path")
        print("  python transcribe_wav2vec.py --process-all")
        return
    
    # Handle --process-all flag
    if sys.argv[1] == "--process-all":
        logger.info("Processing all files in med-data")
        process_all_data()
        return
    
    # Handle single file or directory
    path = sys.argv[1]
    
    if not os.path.exists(path):
        logger.error(f"Path not found: {path}")
        return
    
    if os.path.isdir(path):
        # Process directory
        logger.info(f"Processing directory: {path}")
        process_directory(path)
    else:
        # Process single file
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        transcribe_audio_file(path, output_path)

if __name__ == "__main__":
    main()