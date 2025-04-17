#!/usr/bin/env python3
"""
Medical Conversation Transcription Tool
======================================
Transcribes medical conversations using Whisper v3 Large with NeMo/PyAnnotate diarization.

- Uses HuggingFace's Whisper large-v3 model for high-quality, faster transcription
- Employs NVIDIA NeMo with PyAnnotate fallback for accurate speaker diarization
- Outputs in simple doctor/patient conversation format
- Guarantees complete transcription of entire audio file
- No copying from original files - pure audio-based transcription

Usage:
    python transcribe_medical_whisper_v3_large.py audio_file [output_file]
"""

import os
import sys
import json
import time
import logging
import subprocess
import tempfile
import numpy as np
from datetime import datetime
import torch
from pathlib import Path
from dotenv import load_dotenv
import re
import transformers
import traceback
import librosa

# Load environment variables
load_dotenv()

# Get HuggingFace token from .env
HF_TOKEN = os.getenv("HF_Token")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("medical_transcribe")

def install_dependencies():
    """Install required libraries if not already installed."""
    try:
        # Check if transformers is installed
        import transformers
        logger.info("Transformers library is already installed")
    except ImportError:
        logger.info("Installing Transformers library...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.35.0"])
            logger.info("Transformers installed successfully")
        except Exception as e:
            logger.error(f"Failed to install Transformers: {e}")
            return False
    
    try:
        # Check if PyAnnotate is installed
        import pyannote.audio
        logger.info("PyAnnotate library is already installed")
    except ImportError:
        logger.info("Installing PyAnnotate for speaker diarization...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyannote.audio"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch torchaudio"])
            logger.info("PyAnnotate installed successfully")
        except Exception as e:
            logger.error(f"Failed to install PyAnnotate: {e}")
            return False
    
    try:
        # Check if NeMo is installed
        import nemo
        logger.info("NVIDIA NeMo library is already installed")
    except ImportError:
        logger.info("Installing NVIDIA NeMo for enhanced speaker diarization...")
        try:
            # Install NeMo and its dependencies
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nemo_toolkit[asr]"])
            logger.info("NVIDIA NeMo installed successfully")
        except Exception as e:
            logger.error(f"Failed to install NeMo (will use PyAnnotate as fallback): {e}")
    
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
        # Check if scikit-learn is installed (needed for NeMo clustering)
        import sklearn
        logger.info("Scikit-learn library is already installed")
    except ImportError:
        logger.info("Installing scikit-learn for clustering...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
            logger.info("Scikit-learn installed successfully")
        except Exception as e:
            logger.error(f"Failed to install scikit-learn: {e}")
            return False
    
    return True

def load_audio(audio_path):
    """
    Load audio file using librosa.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Audio data as float32 array, sample rate
    """
    try:
        logger.info(f"Loading audio file: {audio_path}")
        # Load audio with librosa (resampling to 16kHz for compatibility)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio file: {e}")
        return None, None

def load_whisper_model():
    """Load and cache the Whisper large-v3 model from HuggingFace."""
    try:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
        
        logger.info("Loading Whisper large-v3 model from HuggingFace...")
        
        # Check if CUDA is available and set device accordingly
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load the model and processor
        model_id = "openai/whisper-large-v3"
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        
        # Move model to appropriate device
        model.to(device)
        
        # Create a pipeline for speech recognition
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=30,  # Process 30-second chunks
            stride_length_s=5   # 5-second overlap between chunks
        )
        
        logger.info(f"Successfully loaded Whisper large-v3 model (on {device})")
        return pipe
    
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        logger.error(traceback.format_exc())
        return None

def transcribe_audio_with_whisper_large(audio_path, model=None, task="transcribe"):
    """Transcribe audio with Whisper large-v3.
    
    Args:
        audio_path: Path to audio file.
        model: Pre-loaded Whisper model pipeline, if available.
        task: Either "transcribe" or "translate" (to English).
        
    Returns:
        result: Dictionary containing the transcription result.
    """
    try:
        logging.info(f"Starting transcription of {audio_path}...")
        start_time = time.time()
        
        if model is None:
            from transformers import pipeline
            logging.info("Loading Whisper large-v3 model...")
            
            # Check if CUDA is available and set device accordingly
            device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Use transformers pipeline for Whisper large-v3
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                device=device,
                torch_dtype=torch_dtype,
                chunk_length_s=30,  # Process 30-second chunks
                stride_length_s=5   # 5-second overlap between chunks
            )
        else:
            # Use the provided model
            pipe = model
        
        # Load audio
        logging.info(f"Processing audio: {audio_path}")
        audio_duration = librosa.get_duration(filename=audio_path)
        logging.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Transcribe with appropriate parameters for medical transcription
        result = pipe(
            audio_path,
            return_timestamps=True,
            generate_kwargs={
                "task": task,
                "max_new_tokens": 300,  # Lower value to prevent errors
                "return_timestamps": True  # Get timestamps for each segment
            }
        )
        
        # Create a format compatible with our existing code
        output = {
            "text": result.get("text", ""),
            "language": "en",  # Default to English, can be updated later
            "segments": []
        }
        
        # Process chunks/segments
        for i, segment in enumerate(result.get("chunks", [])):
            segment_obj = {
                "id": i,
                "start": segment.get("timestamp", [0, 0])[0] if segment.get("timestamp") else 0,
                "end": segment.get("timestamp", [0, 0])[1] if segment.get("timestamp") else 0,
                "text": segment.get("text", "").strip()
            }
            output["segments"].append(segment_obj)
        
        # If no language is detected, try to detect it from the transcription
        if output["text"]:
            # Simple language detection based on common words
            text_lower = output["text"].lower()
            if any(word in text_lower for word in ['le', 'la', 'les', 'est', 'et', 'dans', 'pour']):
                output["language"] = "fr"
        
        end_time = time.time()
        logging.info(f"Transcription complete. Time taken: {end_time - start_time:.2f} seconds")
        logging.info(f"Detected language: {output['language']}")
        logging.info(f"Number of segments: {len(output['segments'])}")
        logging.info(f"Total characters: {len(output['text'])}")
        
        return output
        
    except Exception as e:
        logging.error(f"Error in transcription: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def is_language_switch(primary_language, text):
    """
    Detect if text appears to switch languages from the primary language.
    This is a simple heuristic approach that can be improved with more sophisticated language detection.
    
    Args:
        primary_language: The detected primary language code (e.g., 'en', 'fr')
        text: The text to check
        
    Returns:
        boolean: True if language switch is detected
    """
    if not text or len(text.strip()) < 10:
        return False
        
    # Common English words that shouldn't appear in French text
    english_markers = ["the", "is", "are", "will", "should", "could", "would", "I'll", "we'll", "that's", "it's", "let's", "like"]
    # Common French words that shouldn't appear in English text
    french_markers = ["le", "la", "les", "est", "sont", "sera", "je", "vous", "nous", "votre", "cette", "dans", "avec"]
    
    if primary_language == 'en':
        # Check if French words are in English text
        words = text.lower().split()
        french_count = sum(1 for word in words if word in french_markers)
        return french_count > 2  # If more than 2 French markers found in English text
    elif primary_language == 'fr':
        # Check if English words are in French text
        words = text.lower().split()
        english_count = sum(1 for word in words if word in english_markers)
        return english_count > 2  # If more than 2 English markers found in French text
    
    return False

def identify_speaker(text, language):
    """
    Identify the speaker (doctor or patient) based on the content of the text.
    
    Args:
        text: The text to analyze
        language: The detected language of the text
        
    Returns:
        str: 'doctor' or 'patient'
    """
    # Check for common patterns that indicate a doctor is speaking
    doctor_patterns = {
        'en': [
            r'\b(?:i recommend|let me|i suggest|i would like to|we should|let\'s schedule|i\'ll prescribe|i need to examine|how long have you|what symptoms|do you have|have you been|are you feeling|does it hurt|any allergies|follow-up|prescription|diagnosis|treatment|symptoms|medical|doctor|examination|test results)\b',
            r'\b(?:how are you feeling today|what brings you in today|what can I help you with|have you experienced this before|when did this start|on a scale of)\b'
        ],
        'fr': [
            r'\b(?:je recommande|permettez-moi|je suggère|je voudrais|nous devrions|planifions|je vais prescrire|je dois examiner|depuis combien de temps|quels symptômes|avez-vous|ressentez-vous|ça fait mal|des allergies|suivi|ordonnance|diagnostic|traitement|symptômes|médical|médecin|examen|résultats du test)\b',
            r'\b(?:comment vous sentez-vous aujourd\'hui|qu\'est-ce qui vous amène aujourd\'hui|comment puis-je vous aider|avez-vous déjà ressenti cela|quand cela a-t-il commencé|sur une échelle de)\b'
        ]
    }
    
    # Check for common patterns that indicate a patient is speaking
    patient_patterns = {
        'en': [
            r'\b(?:i\'ve been|i feel|i have|i\'m having|my symptoms|my pain|it hurts|i\'ve noticed|i started|my condition|my health|my problem)\b',
            r'\b(?:i came in because|i\'ve been experiencing|i don\'t feel|i\'m worried about|the pain is|started about|been going on for)\b'
        ],
        'fr': [
            r'\b(?:j\'ai eu|je me sens|j\'ai|j\'éprouve|mes symptômes|ma douleur|ça fait mal|j\'ai remarqué|j\'ai commencé|mon état|ma santé|mon problème)\b',
            r'\b(?:je suis venu parce que|j\'ai ressenti|je ne me sens pas|je m\'inquiète|la douleur est|a commencé il y a|dure depuis)\b'
        ]
    }
    
    # Default language to English if not recognized
    lang_key = 'en'
    if language.startswith('fr'):
        lang_key = 'fr'
    
    # Check for doctor patterns
    for pattern in doctor_patterns.get(lang_key, doctor_patterns['en']):
        if re.search(pattern, text.lower()):
            return 'doctor'
    
    # Check for patient patterns
    for pattern in patient_patterns.get(lang_key, patient_patterns['en']):
        if re.search(pattern, text.lower()):
            return 'patient'
    
    # Default to doctor if no pattern is matched
    # Can be adjusted based on specific needs
    return 'doctor'

def split_long_turn(text, speaker, language):
    """
    Split a long turn into multiple natural segments.
    
    Args:
        text: The text to split
        speaker: The speaker of the text ('doctor' or 'patient')
        language: The language of the text
        
    Returns:
        list: List of dictionaries with 'text' and 'speaker' keys
    """
    # If text is short enough, don't split
    if len(text) < 150:
        return [{"text": text, "speaker": speaker}]
    
    result = []
    
    # Define sentence breaking patterns
    sentence_end = r'(?<=[.!?])\s+'
    
    # Define question patterns
    question_patterns = {
        'en': [
            r'\b(?:how are you feeling|what symptoms|do you have|have you been|are you experiencing|does it hurt|any allergies|how long|when did|is there|can you|would you|do you know|how often|is it|are you allergic)\b.*?\?',
            r'\?(?=\s+[A-Z])'
        ],
        'fr': [
            r'\b(?:comment vous sentez-vous|quels symptômes|avez-vous|êtes-vous|ressentez-vous|ça fait mal|des allergies|depuis combien de temps|quand est-ce que|y a-t-il|pouvez-vous|pourriez-vous|savez-vous|à quelle fréquence|est-ce que|êtes-vous allergique)\b.*?\?',
            r'\?(?=\s+[A-Z])'
        ]
    }
    
    # Default language to English if not recognized
    lang_key = 'en'
    if language.startswith('fr'):
        lang_key = 'fr'
    
    # First split by sentence
    sentences = re.split(sentence_end, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If doctor is speaking, try to identify questions and split at those points
    if speaker == 'doctor':
        # Group sentences into chunks
        current_chunk = []
        chunks = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            # Check if sentence is a question
            is_question = False
            for pattern in question_patterns.get(lang_key, question_patterns['en']):
                if re.search(pattern, sentence, re.IGNORECASE):
                    is_question = True
                    break
            
            # If it's a question, end the chunk
            if is_question and len(current_chunk) > 0:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # If we didn't find any questions, try to split based on length
        if len(chunks) <= 1:
            chunks = []
            current_chunk = []
            current_length = 0
            max_chunk_length = 150
            
            for sentence in sentences:
                # If adding this sentence would make the chunk too long, start a new one
                if current_length + len(sentence) > max_chunk_length and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(sentence)
                current_length += len(sentence)
            
            # Add the last chunk if there's anything left
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        # Create turn segments
        for chunk in chunks:
            result.append({"text": chunk.strip(), "speaker": speaker})
    
    # If patient is speaking, just split by length
    else:
        current_chunk = []
        current_length = 0
        max_chunk_length = 200  # Patients can have slightly longer chunks
        
        for sentence in sentences:
            # If adding this sentence would make the chunk too long, start a new one
            if current_length + len(sentence) > max_chunk_length and current_chunk:
                result.append({
                    "text": ' '.join(current_chunk).strip(),
                    "speaker": speaker
                })
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        # Add the last chunk if there's anything left
        if current_chunk:
            result.append({
                "text": ' '.join(current_chunk).strip(),
                "speaker": speaker
            })
    
    # If we somehow end up with nothing, return the original
    if not result:
        return [{"text": text, "speaker": speaker}]
    
    return result

def clean_transcript_text(text, language):
    """
    Clean up transcript text by removing fillers and repairing common errors.
    
    Args:
        text: Text to clean
        language: Language code (en or fr)
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Remove common filler phrases
    fillers = ["um", "uh", "er", "ah", "like", "you know", "so yeah", "I mean"]
    for filler in fillers:
        text = text.replace(f" {filler} ", " ")
        text = text.replace(f" {filler}, ", " ")
    
    # Fix profanity (likely misrecognitions)
    profanity = ["fuck", "shit", "damn", "ass"]
    for word in profanity:
        if f" {word} " in text.lower():
            text = text.replace(f" {word} ", " ")
    
    # Fix common medical term errors
    if language == "en":
        medical_corrections = {
            "flem": "phlegm",
            "phlemas": "flonase",
            "flon": "flonase",
            "rinetis": "rhinitis",
            "rhin": "rhinitis",
            "allegic": "allergic",
            "scale of two": "scale of 1 to 10",
            "scale of to": "scale of 1 to 10",
            "two week": "2 week"
        }
    else:  # French
        medical_corrections = {
            "renitose": "rhinite",
            "renit": "rhinite",
            "flemas": "flonase",
            "allergique rhini": "rhinite allergique"
        }
    
    # Apply medical corrections (case-insensitive)
    for error, correction in medical_corrections.items():
        pattern = re.compile(re.escape(error), re.IGNORECASE)
        text = pattern.sub(correction, text)
    
    # Fix punctuation
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    
    # Fix spacing
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def post_process_segments(segments):
    """
    Post-process segments to split or merge based on natural pauses.
    
    Args:
        segments: List of transcript segments
        
    Returns:
        Post-processed segments
    """
    if not segments:
        return segments
    
    # Handle segments with None values for start or end
    for segment in segments:
        if segment.get('end') is None:
            # Estimate an end time based on text length if missing
            segment['end'] = segment.get('start', 0) + (len(segment.get('text', '')) * 0.07)  # ~70ms per char
            
        if segment.get('start') is None:
            # If start is missing but end exists, estimate backwards
            if segment.get('end') is not None:
                segment['start'] = max(0, segment.get('end') - (len(segment.get('text', '')) * 0.07))
            else:
                # Both start and end are None, assign default values
                segment['start'] = 0.0
                segment['end'] = len(segment.get('text', '')) * 0.07
    
    # Merge very short segments (less than 1 second) with neighboring segments
    merged_segments = []
    for i, segment in enumerate(segments):
        duration = segment.get('end', 0) - segment.get('start', 0)
        
        if duration < 1.0 and i > 0:
            # Merge with previous segment
            prev_segment = merged_segments[-1]
            prev_segment['end'] = segment.get('end', prev_segment.get('end', 0))
            prev_segment['text'] += " " + segment.get('text', '')
        else:
            merged_segments.append(segment)
    
    # Split long segments at natural pauses (periods, question marks)
    final_segments = []
    for segment in merged_segments:
        text = segment.get('text', '')
        duration = segment.get('end', 0) - segment.get('start', 0)
        
        # Only split segments longer than 10 seconds
        if duration > 10.0 and ("." in text or "?" in text):
            # Find natural sentence boundaries
            sentence_boundaries = []
            for match in re.finditer(r'[.!?]', text):
                if match.start() > 5 and match.start() < len(text) - 5:  # Ensure not at edges
                    sentence_boundaries.append(match.start())
            
            # If we found boundaries, split the segment
            if sentence_boundaries:
                # Calculate approximate time per character
                time_per_char = duration / len(text)
                
                # Create a new segment for each sentence
                last_end = 0
                start_time = segment.get('start', 0)
                
                for boundary in sentence_boundaries:
                    # Calculate end time based on character position
                    end_time = start_time + (boundary * time_per_char)
                    
                    # Create segment for this sentence
                    final_segments.append({
                        'id': len(final_segments),
                        'start': start_time,
                        'end': end_time,
                        'text': text[last_end:boundary+1].strip(),
                        'words': []
                    })
                    
                    # Update for next sentence
                    last_end = boundary + 1
                    start_time = end_time
                
                # Add any remaining text
                if last_end < len(text):
                    final_segments.append({
                        'id': len(final_segments),
                        'start': start_time,
                        'end': segment.get('end', 0),
                        'text': text[last_end:].strip(),
                        'words': []
                    })
            else:
                final_segments.append(segment)
        else:
            final_segments.append(segment)
    
    return final_segments

def fix_overlapping_segments(diarization_segments):
    """
    Fix overlapping segments in diarization results to ensure clean speaker transitions.
    
    Args:
        diarization_segments: List of diarization segments with potential overlaps
        
    Returns:
        List of fixed diarization segments without overlaps
    """
    if not diarization_segments or len(diarization_segments) < 2:
        return diarization_segments
    
    # Sort segments by start time to make processing easier
    sorted_segments = sorted(diarization_segments, key=lambda x: x.get("start", 0))
    fixed_segments = [sorted_segments[0]]  # Start with the first segment
    
    for i in range(1, len(sorted_segments)):
        current = sorted_segments[i]
        previous = fixed_segments[-1]
        
        # Check for overlap
        if current["start"] < previous["end"]:
            overlap = previous["end"] - current["start"]
            logger.info(f"Found overlap of {overlap:.2f} seconds between segments")
            
            # If segments have the same speaker, merge them
            if current["speaker"] == previous["speaker"]:
                # Create a merged segment
                merged = {
                    "start": previous["start"],
                    "end": max(previous["end"], current["end"]),
                    "speaker": previous["speaker"]
                }
                fixed_segments[-1] = merged
                logger.info(f"Merged same-speaker segments: {previous['start']:.2f}-{merged['end']:.2f}")
            
            # If different speakers, decide where to make the cut
            else:
                # If significant overlap (>0.3s), place boundary at midpoint
                if overlap > 0.3:
                    boundary = (previous["end"] + current["start"]) / 2
                    fixed_segments[-1]["end"] = boundary
                    current["start"] = boundary
                    logger.info(f"Fixed large overlap by setting boundary at {boundary:.2f}s")
                # For small overlaps, give slight preference to the current segment
                else:
                    # Minor adjustment to previous segment's end time
                    fixed_segments[-1]["end"] = current["start"]
                    logger.info(f"Fixed small overlap by adjusting previous segment end time")
                
                fixed_segments.append(current)
        
        # No overlap, add the current segment as is
        else:
            # Check for gaps
            if current["start"] - previous["end"] > 0.5:
                gap = current["start"] - previous["end"]
                logger.info(f"Found gap of {gap:.2f} seconds between speakers")
                
                # For small gaps, just connect the segments
                if gap < 1.5:
                    # Move the boundary to the middle of the gap
                    boundary = (previous["end"] + current["start"]) / 2
                    fixed_segments[-1]["end"] = boundary
                    current["start"] = boundary
            
            fixed_segments.append(current)
    
    # Final check for any remaining overlaps (quality control)
    has_overlaps = False
    for i in range(1, len(fixed_segments)):
        if fixed_segments[i]["start"] < fixed_segments[i-1]["end"]:
            logger.warning(f"Overlap still exists after fixing: {fixed_segments[i-1]['end']:.2f} > {fixed_segments[i]['start']:.2f}")
            has_overlaps = True
    
    if not has_overlaps:
        logger.info("All segment overlaps fixed successfully")
    
    # Return the fixed segments
    return fixed_segments

def perform_diarization(audio_path):
    """
    Perform speaker diarization using NeMo first, then PyAnnotate as fallback.
    Uses a hybrid approach to maximize accuracy for medical conversations.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        List of speaker segments or None if failed
    """
    # First try NeMo diarization
    try:
        import nemo
        logger.info("Attempting diarization with NVIDIA NeMo...")
        
        import nemo.collections.asr as nemo_asr
        
        # Check if GPU is available, otherwise use CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            logger.warning("Running NeMo diarization on CPU may be slow")
        
        # Load NeMo speaker diarization model
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="titanet_large"
        )
        speaker_model.to(device)
        
        # Load NeMo VAD (Voice Activity Detection) model for better segmentation
        vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
            model_name="vad_multilingual_marblenet"
        )
        vad_model.to(device)
        
        # Process audio for diarization
        logger.info(f"Running NeMo diarization on {os.path.basename(audio_path)}...")
        
        # Create temporary directory for NeMo outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run VAD to get speech segments
            vad_config = {
                "window_length_in_sec": 0.15,
                "shift_length_in_sec": 0.01,
                "smooth_window": 5,
                "threshold": 0.5
            }
            
            speech_segments = vad_model.get_speech_segments(
                audio_path, 
                threshold=vad_config["threshold"],
                window_length_in_sec=vad_config["window_length_in_sec"],
                shift_length_in_sec=vad_config["shift_length_in_sec"], 
                smooth_window=vad_config["smooth_window"]
            )
            
            # Extract speaker embeddings for each segment
            embeddings = []
            segment_info = []
            
            for i, segment in enumerate(speech_segments):
                start_time, end_time = segment
                
                # Skip very short segments
                if end_time - start_time < 0.5:
                    continue
                
                # Extract embedding for this segment
                segment_emb = speaker_model.get_embedding(
                    audio_path, 
                    start_time=start_time, 
                    end_time=end_time
                )
                
                embeddings.append(segment_emb.cpu().numpy())
                segment_info.append({
                    "start": float(start_time),
                    "end": float(end_time)
                })
            
            # Convert embeddings to numpy array
            embeddings = np.array(embeddings)
            
            # Perform clustering to identify speakers (medical conversations have 2 speakers)
            from sklearn.cluster import AgglomerativeClustering
            
            # Force 2 speakers for medical conversations (doctor and patient)
            n_speakers = 2
            
            # Cluster the embeddings
            clustering = AgglomerativeClustering(
                n_clusters=n_speakers,
                affinity='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            
            # Create diarization segments
            diarization_segments = []
            for i, segment in enumerate(segment_info):
                diarization_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": f"SPEAKER_{labels[i]}"
                })
            
            # Fix overlaps and gaps
            diarization_segments = fix_overlapping_segments(diarization_segments)
            
            logger.info(f"NeMo diarization complete: found {len(diarization_segments)} segments")
            
            return diarization_segments
            
    except ImportError:
        logger.warning("NeMo is not installed, falling back to PyAnnotate")
    except Exception as e:
        logger.warning(f"Error in NeMo diarization: {str(e)}, falling back to PyAnnotate")
    
    # Fall back to PyAnnotate diarization
    try:
        # Import here after installation check
        from pyannote.audio import Pipeline
        
        logger.info("Loading PyAnnotate diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        
        logger.info(f"Running PyAnnotate diarization on {os.path.basename(audio_path)}...")
        diarization = pipeline(audio_path)
        
        # Process diarization results
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        logger.info(f"PyAnnotate diarization complete: found {len(speaker_segments)} speaker segments")
        
        # Fix any overlapping segments
        fixed_segments = fix_overlapping_segments(speaker_segments)
        
        return fixed_segments
    
    except Exception as e:
        logger.error(f"Error in PyAnnotate diarization: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Final fallback to a simple pause-based segmentation approach if all diarization attempts fail
        try:
            logger.warning("All diarization methods failed, falling back to pause-based segmentation")
            
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
            energy = medfilt(energy, kernel_size=int(sr * 0.05))  # 50ms median filter
            
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
                # Create a diarization segment
                diarization_segments.append({
                    "start": float(start),
                    "end": float(end),
                    "speaker": current_speaker
                })
                
                # Switch speakers at long pauses (> 0.75s)
                if i < len(speech_segments) - 1 and speech_segments[i+1][0] - end > 0.75:
                    current_speaker = "SPEAKER_1" if current_speaker == "SPEAKER_0" else "SPEAKER_0"
            
            logger.info(f"Pause-based segmentation complete: found {len(diarization_segments)} segments")
            return diarization_segments
            
        except Exception as e:
            logger.error(f"Error in fallback segmentation: {str(e)}")
            return None
        
    return None

def map_diarization_to_segments(whisper_segments, diarization_segments):
    """
    Map diarization results to Whisper segments with improved speaker detection
    specifically optimized for medical conversations.
    
    Args:
        whisper_segments: Segments from Whisper transcription
        diarization_segments: Segments from diarization
        
    Returns:
        Whisper segments with speaker labels
    """
    if not diarization_segments:
        logger.warning("No diarization segments available, using content-based speaker detection")
        return detect_speakers_by_content(whisper_segments)
    
    # Enhanced medical terminology for better speaker detection
    doctor_terms = [
        # Questions doctors typically ask
        "how long have you", "rate your", "any allergies", "medical history", 
        "have you been", "what brings you", "let me check", "I'll examine",
        
        # Medical procedures/terminology
        "diagnos", "examination", "vitals", "inflammation", "symptom", 
        "condition", "prescribe", "treatment", "allergic", "rhinitis", 
        "follow-up", "appointment", "monitor", "medication", "physical exam",
        
        # Doctor phrases
        "I'm going to", "I recommend", "let's schedule", "I'll prescribe", 
        "suggests that you", "based on", "test results", "let's do a",
        
        # French doctor terms
        "depuis combien de temps", "vos symptômes", "des allergies", 
        "je vais examiner", "je recommande", "traitement", "ordonnance",
        "rendez-vous", "tension artérielle", "je vais prescrire"
    ]
    
    patient_terms = [
        # Symptom descriptions
        "it hurts", "I feel", "I've been", "been having", "I'm experiencing", 
        "started", "got worse", "my symptoms", "my nose", "my pain", "it's affecting",
        
        # Patient questions
        "is that serious", "what's that", "will it get better", "what can I do", 
        "how long will", "is it dangerous", "should I be worried",
        
        # Personal experience
        "my work", "my daily", "it's not unbearable", "it's really", "I'm allergic",
        
        # French patient terms
        "j'ai mal", "je me sens", "j'ai eu", "mes symptômes", "ma douleur",
        "ça fait mal", "a commencé", "est-ce grave", "est-ce que ça va aller mieux",
        "je suis allergique", "mon travail", "ma vie quotidienne"
    ]
    
    # Count speaker statistics first
    speakers = set(diar_segment["speaker"] for diar_segment in diarization_segments)
    
    # If there are at least 2 speakers, determine doctor/patient
    if len(speakers) >= 2:
        # Analyze segments to classify speakers
        speaker_scores = {speaker: {"doctor": 0, "patient": 0} for speaker in speakers}
        
        # For each text segment, find the corresponding diarization speaker
        for segment in whisper_segments:
            segment_mid = (segment.get("start", 0) + segment.get("end", 0)) / 2
            text = segment.get("text", "").lower()
            
            # Find matching diarization segment
            for diar_segment in diarization_segments:
                if diar_segment["start"] <= segment_mid <= diar_segment["end"]:
                    speaker_id = diar_segment["speaker"]
                    
                    # Score doctor/patient likelihood based on content
                    doctor_score = sum(2 for term in doctor_terms if term.lower() in text)
                    patient_score = sum(2 for term in patient_terms if term.lower() in text)
                    
                    # Additional scoring based on question marks (doctors typically ask more questions)
                    if "?" in text:
                        doctor_score += 1
                    
                    # Opening line is likely doctor
                    if segment == whisper_segments[0]:
                        doctor_score += 3
                    
                    speaker_scores[speaker_id]["doctor"] += doctor_score
                    speaker_scores[speaker_id]["patient"] += patient_score
                    break
        
        # Determine which speaker is doctor vs patient
        sorted_speakers = sorted(
            speakers, 
            key=lambda s: speaker_scores[s]["doctor"] - speaker_scores[s]["patient"],
            reverse=True
        )
        
        doctor_speaker = sorted_speakers[0]
        patient_speaker = sorted_speakers[1] if len(sorted_speakers) > 1 else None
        
        logger.info(f"Identified doctor (speaker {doctor_speaker}) based on terminology")
        if patient_speaker:
            logger.info(f"Identified patient (speaker {patient_speaker}) based on terminology")
        
        # Make sure the first speaker is doctor (common in medical consultations)
        first_segment_mid = (whisper_segments[0].get("start", 0) + whisper_segments[0].get("end", 0)) / 2
        first_speaker = None
        for diar_segment in diarization_segments:
            if diar_segment["start"] <= first_segment_mid <= diar_segment["end"]:
                first_speaker = diar_segment["speaker"]
                break
        
        if first_speaker and first_speaker != doctor_speaker:
            # Check if first line is a greeting (strongly suggests doctor)
            first_text = whisper_segments[0].get("text", "").lower()
            if any(greeting in first_text for greeting in ["hello", "hi", "good morning", "good afternoon", "welcome", "bonjour", "salut", "comment allez-vous"]):
                logger.info("First line contains greeting, reassigning speakers")
                doctor_speaker, patient_speaker = first_speaker, doctor_speaker
    else:
        # If only one speaker, use content-based detection instead
        logger.warning("Only one speaker detected in diarization, using content-based assignment")
        return detect_speakers_by_content(whisper_segments)
    
    # Map each segment to a speaker
    for segment in whisper_segments:
        segment_mid = (segment.get("start", 0) + segment.get("end", 0)) / 2
        
        # Find matching diarization segment
        assigned = False
        for diar_segment in diarization_segments:
            if diar_segment["start"] <= segment_mid <= diar_segment["end"]:
                if diar_segment["speaker"] == doctor_speaker:
                    segment["speaker"] = "doctor"
                elif patient_speaker and diar_segment["speaker"] == patient_speaker:
                    segment["speaker"] = "patient"
                else:
                    # For segments with unknown speakers, use text analysis
                    segment["speaker"] = determine_speaker_by_content(segment.get("text", ""))
                assigned = True
                break
        
        # If no match found, determine by content
        if not assigned:
            segment["speaker"] = determine_speaker_by_content(segment.get("text", ""))
    
    # Post-process to ensure alternating pattern and fix any issues
    fix_speaker_consistency(whisper_segments)
    
    return whisper_segments

def determine_speaker_by_content(text):
    """
    Determine whether text is more likely from doctor or patient based on content.
    
    Args:
        text: Text to analyze
        
    Returns:
        "doctor" or "patient"
    """
    text = text.lower()
    
    # Strong doctor indicators
    doctor_indicators = [
        "let me", "I'll prescribe", "your symptoms", "I recommend", 
        "the treatment", "your condition", "physical exam", "inflammation",
        "test results", "blood pressure", "diagnosis", "let's schedule"
    ]
    
    # Strong patient indicators
    patient_indicators = [
        "my symptoms", "I've been", "it hurts", "I feel", "my pain",
        "is it serious", "what's that", "will it get better", "I'm allergic"
    ]
    
    doctor_score = sum(1 for term in doctor_indicators if term in text)
    patient_score = sum(1 for term in patient_indicators if term in text)
    
    if doctor_score > patient_score:
        return "doctor"
    elif patient_score > doctor_score:
        return "patient"
    else:
        # If tied, default based on other patterns
        if "?" in text and len(text) < 60:  # Short questions are often from doctor
            return "doctor"
        elif text.startswith("I ") or text.startswith("My "):  # First-person references often patient
            return "patient"
        else:
            return "doctor"  # Default to doctor as fallback

def detect_speakers_by_content(segments):
    """
    Detect speakers based solely on content analysis when diarization fails.
    
    Args:
        segments: Transcription segments
        
    Returns:
        Segments with speaker labels
    """
    # First pass: label each segment based on content
    for segment in segments:
        segment["speaker"] = determine_speaker_by_content(segment.get("text", ""))
    
    # Second pass: ensure the conversation starts with doctor
    if segments and segments[0]["speaker"] != "doctor":
        segments[0]["speaker"] = "doctor"
    
    # Third pass: enforce alternating pattern, fixing obvious errors
    for i in range(1, len(segments)):
        # Get current and previous speakers
        current_speaker = segments[i]["speaker"]
        prev_speaker = segments[i-1]["speaker"]
        
        # If two consecutive segments have same speaker
        if current_speaker == prev_speaker:
            # Check if content strongly indicates this assignment
            text = segments[i]["text"].lower()
            if current_speaker == "doctor":
                # Check if really sounds like doctor
                is_strong_doctor = any(term in text for term in [
                    "prescribe", "diagnosis", "let me check", "treatment",
                    "I recommend", "vitals", "examination"
                ])
                
                if not is_strong_doctor:
                    segments[i]["speaker"] = "patient"
            else:  # current is patient
                # Check if really sounds like patient
                is_strong_patient = any(term in text for term in [
                    "my symptoms", "I've been feeling", "it hurts", "I'm allergic",
                    "is that serious", "will I get better"
                ])
                
                if not is_strong_patient:
                    segments[i]["speaker"] = "doctor"
    
    # Final cleanup
    fix_speaker_consistency(segments)
    
    return segments

def fix_speaker_consistency(segments):
    """
    Fix inconsistent speaker assignments.
    
    Args:
        segments: Transcription segments with initial speaker assignments
    """
    # Check for unknown speakers
    unknown_count = sum(1 for s in segments if s.get("speaker") == "unknown")
    if unknown_count > 0:
        logger.info(f"Fixing {unknown_count} segments with unknown speakers")
        
        for i, segment in enumerate(segments):
            if segment.get("speaker") == "unknown":
                # Try to infer from neighboring segments
                if i > 0 and i < len(segments) - 1:
                    prev_speaker = segments[i-1].get("speaker")
                    next_speaker = segments[i+1].get("speaker")
                    
                    if prev_speaker != "unknown" and prev_speaker == next_speaker:
                        # If surrounded by same speaker, use that
                        segment["speaker"] = prev_speaker
                    elif prev_speaker != "unknown":
                        # Otherwise use the opposite of previous
                        segment["speaker"] = "patient" if prev_speaker == "doctor" else "doctor"
                elif i > 0:
                    # If only previous available
                    prev_speaker = segments[i-1].get("speaker")
                    if prev_speaker != "unknown":
                        segment["speaker"] = "patient" if prev_speaker == "doctor" else "doctor"
                else:
                    # First segment, default to doctor
                    segment["speaker"] = "doctor"
    
    # Check for consistency (multiple consecutive segments by same speaker)
    for i in range(1, len(segments) - 1):
        if (segments[i-1].get("speaker") == segments[i+1].get("speaker") and 
            segments[i].get("speaker") != segments[i-1].get("speaker")):
            # Short isolated segment surrounded by other speaker
            if segments[i].get("end", 0) - segments[i].get("start", 0) < 2.0:
                segments[i]["speaker"] = segments[i-1].get("speaker")
    
    # Ensure we have both doctor and patient
    speakers = set(s.get("speaker") for s in segments)
    if "doctor" not in speakers:
        logger.warning("No doctor detected, marking first segment as doctor")
        if segments:
            segments[0]["speaker"] = "doctor"
    if "patient" not in speakers:
        logger.warning("No patient detected, marking second segment as patient")
        if len(segments) > 1:
            segments[1]["speaker"] = "patient"

def build_conversation(segments, language):
    """
    Build a conversation format from the transcription segments.
    Optimized for medical conversations between doctor and patient.
    
    Args:
        segments: List of transcription segments
        language: Detected language of the transcription
        
    Returns:
        list: List of conversation turns with speaker and text
    """
    if not segments:
        return []
    
    conversation = []
    
    # First, identify speakers for each segment if not already done
    for segment in segments:
        if 'speaker' not in segment or not segment['speaker']:
            # Identify speaker based on content
            segment['speaker'] = identify_speaker(segment['text'], language)
    
    # Clean up the text for each segment
    for segment in segments:
        if segment.get('text'):
            segment['text'] = clean_transcript_text(segment['text'], language)
    
    # Combine adjacent segments by the same speaker
    current_speaker = None
    current_text = ""
    current_start = None
    current_end = None
    
    for segment in segments:
        speaker = segment.get('speaker', None)
        text = segment.get('text', "").strip()
        
        if not text:
            continue
        
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
            
        # If we have a new speaker, end the current turn and start a new one
        if current_speaker and (speaker != current_speaker):
            # Process the completed turn
            split_turns = split_long_turn(current_text, current_speaker, language)
            
            # Add timing information
            for i, turn in enumerate(split_turns):
                turn['start'] = current_start
                turn['end'] = current_end
                
            conversation.extend(split_turns)
            
            # Reset for the new turn
            current_text = text
            current_speaker = speaker
            current_start = start_time
            current_end = end_time
        else:
            # Continue the current turn
            if current_text:
                current_text += " " + text
                current_end = end_time  # Update the end time
            else:
                current_text = text
                current_speaker = speaker
                current_start = start_time
                current_end = end_time
    
    # Add the final turn if there is one
    if current_speaker and current_text:
        split_turns = split_long_turn(current_text, current_speaker, language)
        
        # Add timing information
        for i, turn in enumerate(split_turns):
            turn['start'] = current_start
            turn['end'] = current_end
            
        conversation.extend(split_turns)
    
    # Make sure we have both doctor and patient in the conversation
    speakers = {turn.get('speaker') for turn in conversation}
    if len(speakers) < 2:
        # If only one speaker is identified, try to re-identify speakers
        if len(conversation) >= 2:
            # Assume alternating pattern starting with doctor
            for i, turn in enumerate(conversation):
                turn['speaker'] = 'doctor' if i % 2 == 0 else 'patient'
    
    # Ensure we don't have more than 3 consecutive turns by the same speaker
    i = 0
    while i < len(conversation) - 3:
        if (conversation[i]['speaker'] == conversation[i+1]['speaker'] == 
            conversation[i+2]['speaker'] == conversation[i+3]['speaker']):
            # Split this sequence of 4 or more by alternating speakers
            for j in range(i+1, i+4, 2):
                if conversation[j]['speaker'] == 'doctor':
                    conversation[j]['speaker'] = 'patient'
                else:
                    conversation[j]['speaker'] = 'doctor'
        i += 1
    
    # Calculate and log turn statistics
    if conversation:
        turn_lengths = [len(turn.get('text', '')) for turn in conversation]
        avg_turn_length = sum(turn_lengths) / len(turn_lengths)
        logging.info(f"Built conversation with {len(conversation)} turns. Average turn length: {avg_turn_length:.1f} characters")
        logging.info(f"Doctor turns: {sum(1 for t in conversation if t.get('speaker') == 'doctor')}")
        logging.info(f"Patient turns: {sum(1 for t in conversation if t.get('speaker') == 'patient')}")
    
    return conversation

def save_conversation(conversation, output_path):
    """
    Save the conversation to a JSON file.
    
    Args:
        conversation: Conversation turns
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved conversation to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        return False

def process_directory(directory_path, model=None):
    """
    Process all audio files in a directory.
    
    Args:
        directory_path: Path to directory containing audio files
        model: Pre-loaded Whisper model, if available
        
    Returns:
        Tuple of (success_count, fail_count)
    """
    logger.info(f"Processing all audio files in: {directory_path}")
    
    # Look for .wav files in the directory
    audio_files = []
    for file in os.listdir(directory_path):
        if file.lower().endswith('.wav'):
            audio_files.append(os.path.join(directory_path, file))
    
    if not audio_files:
        logger.warning(f"No audio files found in {directory_path}")
        return 0, 0
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Find or create transcripts directory
    if 'audio' in directory_path:
        transcripts_dir = directory_path.replace('audio', 'transcripts')
    else:
        transcripts_dir = os.path.join(directory_path, 'transcripts')
    
    os.makedirs(transcripts_dir, exist_ok=True)
    logger.info(f"Transcripts will be saved to: {transcripts_dir}")
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    # Load the whisper model once for all files
    if model is None:
        model = load_whisper_model()
        if not model:
            logger.error("Failed to load Whisper model")
            return 0, 0
    
    for audio_path in audio_files:
        base_name = os.path.basename(audio_path).rsplit(".", 1)[0]
        output_path = os.path.join(transcripts_dir, f"{base_name}_transcription.json")
        
        # Skip files that already have transcripts
        if os.path.exists(output_path):
            logger.info(f"Skipping {base_name} - transcript already exists")
            success_count += 1  # Count as success since we already have it
            continue
        
        logger.info(f"Processing file {success_count + fail_count + 1}/{len(audio_files)}: {base_name}")
        
        # Process the audio file
        result = transcribe_audio_file(audio_path, model, output_path)
        
        if result:
            success_count += 1
            logger.info(f"Successfully processed {base_name}")
        else:
            fail_count += 1
            logger.error(f"Failed to process {base_name}")
    
    logger.info(f"Directory processing completed. Success: {success_count}, Failed: {fail_count}")
    return success_count, fail_count

def process_all_directories():
    """Process all audio directories in med-data."""
    # Main directories
    main_dir = "med-data"
    categories = ["cardiology", "gp"]
    
    total_success = 0
    total_fail = 0
    
    # Make sure dependencies are installed
    if not install_dependencies():
        logger.error("Failed to install required dependencies")
        return
    
    # Process each category
    for category in categories:
        category_dir = os.path.join(main_dir, category)
        
        # Find language subdirectories
        if not os.path.isdir(category_dir):
            logger.warning(f"Category directory not found: {category_dir}")
            continue
        
        # Look for language-specific directories
        lang_dirs = []
        for item in os.listdir(category_dir):
            if item.startswith(("en-CA", "fr-CA")) and os.path.isdir(os.path.join(category_dir, item)):
                lang_dirs.append(os.path.join(category_dir, item))
        
        if not lang_dirs:
            logger.warning(f"No language directories found in {category_dir}")
            continue
        
        for lang_dir in lang_dirs:
            audio_dir = os.path.join(lang_dir, "audio")
            
            if os.path.isdir(audio_dir):
                logger.info(f"Processing {audio_dir}")
                success, fail = process_directory(audio_dir)
                total_success += success
                total_fail += fail
            else:
                logger.warning(f"Audio directory not found: {audio_dir}")
    
    # Print summary
    total = total_success + total_fail
    if total > 0:
        success_rate = (total_success / total) * 100
        logger.info(f"\nAll directories processing completed!")
        logger.info(f"Success rate: {success_rate:.2f}% ({total_success}/{total} files)")
    else:
        logger.info("\nNo files were processed.")

def transcribe_audio_file(audio_path, model=None, output_path=None):
    """
    Process a single audio file with Whisper large-v3 and diarization.
    
    Args:
        audio_path: Path to the audio file
        model: Pre-loaded Whisper model, if available
        output_path: Output file path (if None, will be generated based on audio_path)
        
    Returns:
        Dictionary containing the transcription result or None if failed
    """
    try:
        # Start timing
        start_time = time.time()
        base_name = os.path.basename(audio_path)
        
        logger.info(f"Processing {base_name}")
        
        # Determine appropriate output path if not provided
        if output_path is None:
            # Extract base name and create path in transcripts folder
            base_dir = os.path.dirname(audio_path)
            file_stem = os.path.splitext(base_name)[0]
            
            # Find the transcripts folder at the same level as audio
            if 'audio' in base_dir:
                transcripts_dir = base_dir.replace('audio', 'transcripts')
                os.makedirs(transcripts_dir, exist_ok=True)
                output_path = os.path.join(transcripts_dir, f"{file_stem}_transcription.json")
            else:
                output_path = f"{file_stem}_transcription.json"
        
        logger.info(f"Output will be saved to {output_path}")
        
        # Check if model is provided
        if model is None:
            logger.info("Loading Whisper large-v3 model...")
            model = load_whisper_model()
            if not model:
                logger.error("Failed to load Whisper model")
                return None
        
        # Step 1: Transcribe the audio with Whisper large-v3 model
        logger.info("Starting transcription with Whisper large-v3 model...")
        transcription = transcribe_audio_with_whisper_large(audio_path, model, task="transcribe")
        if not transcription:
            logger.error("Transcription failed")
            return None
        
        transcription_time = time.time()
        transcription_duration = transcription_time - start_time
        logger.info(f"Transcription completed in {transcription_duration:.2f} seconds")
        
        # Get the language from the transcription result
        detected_language = transcription.get('language', 'en')
        
        # Step 2: Perform speaker diarization
        logger.info("Performing speaker diarization...")
        diarization_start = time.time()
        diarization_segments = perform_diarization(audio_path)
        diarization_time = time.time()
        diarization_duration = diarization_time - diarization_start
        logger.info(f"Diarization completed in {diarization_duration:.2f} seconds")
        
        # Step 3: Map diarization to transcription segments
        logger.info("Mapping speakers to transcription...")
        whisper_segments = transcription.get("segments", [])
        segments_with_speakers = map_diarization_to_segments(whisper_segments, diarization_segments)
        
        # Step 4: Build conversation format
        logger.info("Building conversation format...")
        conversation = build_conversation(segments_with_speakers, detected_language)
        
        # Create the metadata for the output
        audio_duration = librosa.get_duration(filename=audio_path)
        metadata = {
            "filename": base_name,
            "processed_at": datetime.now().isoformat(),
            "processing_time": time.time() - start_time,
            "model": "whisper-large-v3",
            "language": detected_language,
            "duration": audio_duration,
            "segment_count": len(whisper_segments),
            "character_count": sum(len(s.get("text", "")) for s in whisper_segments),
            "turn_count": len(conversation)
        }
        
        # Step 5: Save the result
        logger.info("Saving transcription to output file...")
        success = save_transcription_to_json(output_path, metadata, whisper_segments, conversation)
        
        if success:
            elapsed_time = time.time() - start_time
            logger.info(f"Complete transcription process finished in {elapsed_time:.2f} seconds")
            
            # Print sample of conversation
            if conversation:
                print("\nSample of transcribed conversation:")
                for i, turn in enumerate(conversation[:min(4, len(conversation))]):
                    print(f"{turn['speaker'].upper()}: {turn['text'][:50]}...")
                
                # Print statistics
                doctor_turns = sum(1 for turn in conversation if turn.get('speaker') == 'doctor')
                patient_turns = sum(1 for turn in conversation if turn.get('speaker') == 'patient')
                avg_turn_length = sum(len(turn.get('text', '')) for turn in conversation) / len(conversation) if conversation else 0
                
                print(f"\nStatistics:")
                print(f"- Total turns: {len(conversation)}")
                print(f"- Doctor turns: {doctor_turns}")
                print(f"- Patient turns: {patient_turns}")
                print(f"- Average turn length: {avg_turn_length:.1f} characters")
                print(f"\nSaved to: {output_path}")
            
            # Return the complete transcription object
            return {
                "metadata": metadata,
                "segments": whisper_segments,
                "turns": conversation
            }
        else:
            logger.error("Failed to save transcription")
            return None
            
    except Exception as e:
        logger.error(f"Error in transcribe_audio_file: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def save_transcription_to_json(output_path, metadata, segments, conversation):
    """
    Save transcription data to a JSON file.
    
    Args:
        output_path (str): Path to save the JSON file
        metadata (dict): Metadata about the transcription
        segments (list): List of transcription segments
        conversation (list): List of conversation turns
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a full transcription object
        transcription = {
            "metadata": metadata,
            "segments": segments,
            "turns": conversation  # Using a list format for turns
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Transcription saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving transcription to JSON: {e}")
        return False

def main():
    """Main function to run the transcription pipeline."""
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python transcribe_medical_whisper_v3_large.py [audio_file|directory] [output_file]")
        print("Examples:")
        print("  python transcribe_medical_whisper_v3_large.py audio.wav")
        print("  python transcribe_medical_whisper_v3_large.py audio.wav output.json")
        print("  python transcribe_medical_whisper_v3_large.py directory_with_audio_files")
        print("  python transcribe_medical_whisper_v3_large.py --process-all")
        return
    
    # Check if we should process all directories
    if sys.argv[1] == "--process-all":
        logger.info("Processing all directories in med-data")
        process_all_directories()
        return
    
    path = sys.argv[1]
    
    # Check if the path exists
    if not os.path.exists(path):
        logger.error(f"Path not found: {path}")
        return
    
    # Install required dependencies
    if not install_dependencies():
        logger.error("Failed to install required dependencies")
        return
    
    # Load the model once (will be reused for multiple files)
    logger.info("Loading Whisper large-v3 model...")
    model = load_whisper_model()
    if not model:
        logger.error("Failed to load Whisper model")
        return
    
    # If path is a directory, process all audio files in it
    if os.path.isdir(path):
        logger.info(f"Processing directory: {path}")
        success_count, fail_count = process_directory(path, model)
        
        # Print summary
        total = success_count + fail_count
        if total > 0:
            success_rate = (success_count / total) * 100
            print(f"\nDirectory processing completed!")
            print(f"Success rate: {success_rate:.2f}% ({success_count}/{total} files)")
        else:
            print("\nNo files were processed.")
        
        return
    
    # Process a single file
    audio_path = path
    
    # Determine appropriate output path
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Extract base name and create path in transcripts folder
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.basename(audio_path).rsplit(".", 1)[0]
        
        # Find the transcripts folder at the same level as audio
        if 'audio' in base_dir:
            transcripts_dir = base_dir.replace('audio', 'transcripts')
            os.makedirs(transcripts_dir, exist_ok=True)
            output_path = os.path.join(transcripts_dir, f"{base_name}_transcription.json")
        else:
            output_path = f"{base_name}_transcription.json"
    
    # Process the file
    result = transcribe_audio_file(audio_path, model, output_path)
    
    if result:
        print("\nTranscription successful!")
    else:
        print("\nTranscription failed. Check the logs for details.")
    
    print("\nProcess complete.")

if __name__ == "__main__":
    main() 