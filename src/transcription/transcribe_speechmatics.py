#!/usr/bin/env python3

import os
import json
import time
import argparse
import logging
import glob
from pathlib import Path
from dotenv import load_dotenv
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("speechmatics_transcription.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("speechmatics")


# Load environment variables
load_dotenv()

# Get SPEECHMATICS_API_KEY token from .env
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe audio files using Speechmatics API with diarization")
    parser.add_argument("--audio-path", type=str, help="Path to audio file or directory containing audio files")
    parser.add_argument("--language", type=str, choices=["en", "fr"], default="en", 
                        help="Language code: 'en' for English or 'fr' for French")
    parser.add_argument("--process-all", action="store_true", help="Process all files in the specified directory")
    parser.add_argument("--process-english", action="store_true", help="Process all English files in Cardiology and GP directories")
    parser.add_argument("--process-french", action="store_true", help="Process all French files in Cardiology and GP directories")
    parser.add_argument("--sensitivity", type=float, default=0.6, 
                        help="Speaker diarization sensitivity (0.0-1.0)")
    parser.add_argument("--output-dir", type=str, help="Output directory for transcriptions")
    return parser.parse_args()

def convert_segments_to_turns(segments):
    """Convert the segments from Speechmatics into a format compatible with our other scripts."""
    turns = []
    current_speaker = None
    current_text = ""
    
    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "")
        
        # Start a new turn if the speaker changes
        if speaker != current_speaker:
            if current_speaker and current_text:
                turns.append({
                    "speaker": current_speaker,
                    "text": current_text.strip()
                })
            current_speaker = speaker
            current_text = text
        else:
            # Continue the current turn
            current_text += " " + text
    
    # Add the last turn
    if current_speaker and current_text:
        turns.append({
            "speaker": current_speaker,
            "text": current_text.strip()
        })
    
    return turns

def process_transcript(transcript_json, audio_path, language, sensitivity, output_path=None):
    """Process the transcript JSON to extract relevant information."""
    try:
        # If transcript is already a dict, use it as is
        if isinstance(transcript_json, dict):
            speechmatics_result = transcript_json
        else:
            # Parse JSON if it's a string
            speechmatics_result = json.loads(transcript_json)
        
        # Extract filename and duration from the job information
        filename = os.path.basename(audio_path)
        duration = speechmatics_result.get("job", {}).get("duration", 0)
        
        # Calculate character count
        char_count = 0
        segments = []
        speaker_map = {}  # Map Speechmatics speaker IDs to our format
        segment_id = 0
        
        # Process the results into segments
        results = speechmatics_result.get("results", [])
        
        # Group words by speaker
        current_speaker = None
        current_text = ""
        current_start = 0
        current_end = 0
        
        # Map speaker IDs (S1, S2, etc.) to roles (DOCTOR, PATIENT)
        for result in results:
            if result.get("type") == "word":
                speaker = result.get("alternatives", [{}])[0].get("speaker")
                if speaker and speaker not in speaker_map and speaker != "UU":
                    if speaker == "S1":
                        speaker_map[speaker] = "DOCTOR"
                    elif speaker == "S2":
                        speaker_map[speaker] = "PATIENT"
                    else:
                        speaker_map[speaker] = f"SPEAKER_{speaker[1:]}"
        
        # Process the results to build segments
        for result in results:
            if result.get("type") in ["word", "punctuation"]:
                if "alternatives" in result and len(result["alternatives"]) > 0:
                    char_count += len(result["alternatives"][0].get("content", ""))
                
                    speaker_id = result["alternatives"][0].get("speaker", "UU")
                    start_time = result.get("start_time", 0)
                    end_time = result.get("end_time", 0)
                    content = result["alternatives"][0].get("content", "")
                    
                    # Map the speaker ID to our standard format
                    speaker = speaker_map.get(speaker_id, "UNKNOWN")
                    
                    # Detect speaker changes
                    if speaker_id != current_speaker:
                        # Save the previous segment if it exists
                        if current_text:
                            segments.append({
                                "id": segment_id,
                                "start": current_start,
                                "end": current_end,
                                "text": current_text.strip(),
                                "speaker": speaker_map.get(current_speaker, "UNKNOWN")
                            })
                            segment_id += 1
                        
                        # Start a new segment
                        current_speaker = speaker_id
                        current_text = content
                        current_start = start_time
                        current_end = end_time
                    else:
                        # Continue the current segment
                        current_text += " " + content if content and not content.startswith((".", ",", "?", "!")) else content
                        current_end = end_time
        
        # Add the last segment
        if current_text:
            segments.append({
                "id": segment_id,
                "start": current_start,
                "end": current_end,
                "text": current_text.strip(),
                "speaker": speaker_map.get(current_speaker, "UNKNOWN")
            })
        
        # Convert segments to turns (grouped by speaker)
        turns = []
        for segment in segments:
            turns.append({
                "speaker": segment["speaker"],
                "text": segment["text"]
            })
        
        # Create metadata
        metadata = {
            "filename": filename,
            "language": language,
            "model": "speechmatics",
            "duration_seconds": duration,
            "processing_time_seconds": speechmatics_result.get("processing_time", 0),
            "segment_count": len(segments),
            "character_count": char_count,
            "turn_count": len(turns)
        }
        
        # Create the output data
        output_data = {
            "metadata": metadata,
            "segments": segments,
            "turns": turns
        }
        
        # Determine output path if not provided
        if output_path is None:
            base_dir = os.path.dirname(audio_path)
            file_stem = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Find the transcripts folder at the same level as audio
            if 'audio' in base_dir:
                transcripts_dir = base_dir.replace('audio', 'transcripts')
                os.makedirs(transcripts_dir, exist_ok=True)
                output_path = os.path.join(transcripts_dir, f"{file_stem}_speechmatics.json")
            else:
                output_path = f"{file_stem}_speechmatics.json"
        
        # Save output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed transcript saved to {output_path}")
        logger.info(f"Transcript stats: {len(segments)} segments, {char_count} characters, {len(turns)} turns")
        
        return output_data
    
    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def transcribe_audio_file(audio_path, language="en", sensitivity=0.6, output_path=None):
    """Transcribe a single audio file using Speechmatics Python client."""
    try:
        logger.info(f"Processing {audio_path}")
        
        # Create connection settings using the API key from test.py
        settings = ConnectionSettings(
            url="https://asr.api.speechmatics.com/v2",
            auth_token=SPEECHMATICS_API_KEY,
        )
        
        # Define transcription parameters with diarization (matching test.py configuration)
        conf = {
            "type": "transcription",
            "transcription_config": {
                "language": language,
                "diarization": "speaker",
                "speaker_diarization_config": {
                    "speaker_sensitivity": sensitivity
                }
            }
        }
        
        # Transcribe using the Speechmatics client
        with BatchClient(settings) as client:
            try:
                start_time = time.time()
                
                # Submit the job
                logger.info(f"Submitting job for {audio_path}")
                job_id = client.submit_job(
                    audio=audio_path,
                    transcription_config=conf,
                )
                logger.info(f"Job {job_id} submitted successfully, waiting for transcript")
                
                # Wait for the transcript with detailed JSON output for processing
                transcript = client.wait_for_completion(job_id, transcription_format='json-v2')
                
                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time
                logger.info(f"Job completed in {processing_time:.2f} seconds")
                
                # Add processing time to the data
                if isinstance(transcript, dict):
                    transcript["processing_time"] = processing_time
                
                result = process_transcript(transcript, audio_path, language, sensitivity, output_path)
                return result
                
            except HTTPStatusError as e:
                if e.response.status_code == 401:
                    logger.error("Invalid API key - Check your API key!")
                elif e.response.status_code == 400:
                    logger.error(f"Bad request: {e.response.json().get('detail', 'unknown error')}")
                else:
                    logger.error(f"HTTP error: {e}")
                return None
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
    
    except Exception as e:
        logger.error(f"Error setting up transcription: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_directory(directory_path, language="en", sensitivity=0.6, output_dir=None):
    """Process all audio files in a directory."""
    if not os.path.isdir(directory_path):
        logger.error(f"Not a directory: {directory_path}")
        return
    
    logger.info(f"Processing all audio files in {directory_path}")
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    processed_count = 0
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_path = os.path.join(root, file)
                
                # Determine output path
                if output_dir:
                    file_stem = os.path.splitext(file)[0]
                    output_path = os.path.join(output_dir, f"{file_stem}_speechmatics.json")
                else:
                    output_path = None  # Will be determined in transcribe_audio_file
                
                logger.info(f"Processing file: {audio_path}")
                result = transcribe_audio_file(
                    audio_path, 
                    language=language,
                    sensitivity=sensitivity,
                    output_path=output_path
                )
                
                if result:
                    processed_count += 1
    
    logger.info(f"Completed processing {processed_count} audio files in {directory_path}")

def process_all_files(language, sensitivity=0.6, output_dir=None):
    """Process all files in cardiology and GP directories for the specified language."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    med_data_dir = os.path.join(base_dir, "med-data")
    
    # Process all language variants
    lang_prefixes = ["en-CA-whisper-v3-large", "en-CA-whisper-v3-turbo", "fr-CA-whisper-v3-large", "fr-CA-whisper-v3-turbo"]
    
    # Filter prefixes based on selected language
    if language == "en":
        lang_prefixes = [prefix for prefix in lang_prefixes if prefix.startswith("en")]
    elif language == "fr":
        lang_prefixes = [prefix for prefix in lang_prefixes if prefix.startswith("fr")]
    
    # Process both specialties for the specified language prefixes
    specialties = ["cardiology", "gp"]
    for specialty in specialties:
        for prefix in lang_prefixes:
            audio_dir = os.path.join(med_data_dir, specialty, prefix, "audio")
            if os.path.exists(audio_dir):
                logger.info(f"Processing {language} {specialty} files with prefix {prefix}")
                process_directory(audio_dir, language=language, sensitivity=sensitivity, output_dir=output_dir)
            else:
                logger.warning(f"Directory not found: {audio_dir}")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Ensure sensitivity is in range
    sensitivity = max(0.0, min(1.0, args.sensitivity))
    
    # Process based on arguments
    if args.process_english:
        process_all_files("en", sensitivity=sensitivity, output_dir=args.output_dir)
    
    elif args.process_french:
        process_all_files("fr", sensitivity=sensitivity, output_dir=args.output_dir)
    
    elif args.process_all and args.audio_path:
        if os.path.isdir(args.audio_path):
            process_directory(args.audio_path, language=args.language, 
                             sensitivity=sensitivity, output_dir=args.output_dir)
        else:
            logger.error(f"Not a directory: {args.audio_path}")
    
    elif args.audio_path:
        # Process a single file
        if os.path.isfile(args.audio_path):
            output_path = None
            if args.output_dir:
                file_stem = os.path.splitext(os.path.basename(args.audio_path))[0]
                output_path = os.path.join(args.output_dir, f"{file_stem}_speechmatics.json")
                
            transcribe_audio_file(args.audio_path, language=args.language, 
                                 sensitivity=sensitivity, output_path=output_path)
        else:
            logger.error(f"File not found: {args.audio_path}")
    
    else:
        logger.error("No action specified. Use --process-english, --process-french, --process-all, or specify an audio path.")

if __name__ == "__main__":
    main() 