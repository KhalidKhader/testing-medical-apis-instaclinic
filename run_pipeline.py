#!/usr/bin/env python3
"""
Run the complete medical conversation generation and evaluation pipeline.

This script will:
1. Generate medical conversations for cardiology and GP specialties in English and French
2. Convert the conversations to speech with different voices
3. Transcribe the speech using appropriate services
4. Evaluate the transcription accuracy

Usage:
    python run_pipeline.py --num 3 --specialty cardiology
    python run_pipeline.py --num 5 --specialty all
"""

import os
import subprocess
import argparse
import sys
import glob
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory for medical data
BASE_DIR = "data-med"

def check_environment():
    """Check if the required environment variables are set."""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for generating conversations",
        "DEEPGRAM_API_KEY": "Deepgram API key for English transcription",
        "AZURE_SPEECH_KEY": "Azure Speech key for French transcription",
        "AZURE_SPEECH_REGION": "Azure Speech region (e.g., canadacentral)"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        print("The following environment variables are missing:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in a .env file or your environment before running the pipeline.")
        print("Note: For Azure Speech Services, we recommend using 'canadacentral' as the region.")
        return False
    
    # Verify Azure region is valid
    azure_region = os.getenv("AZURE_SPEECH_REGION", "").lower()
    if azure_region and azure_region not in ["canadacentral", "eastus", "westus", "westus2", "eastus2", "northeurope", "westeurope"]:
        print(f"Warning: '{azure_region}' might not be a valid Azure Speech region.")
        print("Recommended regions: canadacentral, eastus, westus, westus2, eastus2, northeurope, westeurope")
    
    return True

def run_script(script_name, args=None):
    """Run a Python script with the given arguments."""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"Warning: {script_name} exited with code {result.returncode}")
    
    return result.returncode

def run_pipeline(num_cases=3, specialty="all"):
    """Run the complete pipeline."""
    print("=" * 80)
    print("MEDICAL CONVERSATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Generate conversations
    print("\n\n" + "=" * 30 + " STEP 1: GENERATING CONVERSATIONS " + "=" * 30)
    generate_args = ["--num", str(num_cases), "--specialty", specialty]
    run_script("generate_medical_conversations.py", generate_args)
    
    # Step 2: Convert to speech
    print("\n\n" + "=" * 30 + " STEP 2: CONVERTING TO SPEECH " + "=" * 30)
    speech_args = ["--specialty", specialty, "--lang", "all"]
    run_script("convert_to_speech.py", speech_args)
    
    # Step 3: Transcribe conversations
    print("\n\n" + "=" * 30 + " STEP 3: TRANSCRIBING AUDIO " + "=" * 30)
    transcribe_args = ["--specialty", specialty, "--lang", "all"]
    run_script("transcribe_conversations.py", transcribe_args)
    
    # Step 4: Evaluate transcriptions
    print("\n\n" + "=" * 30 + " STEP 4: EVALUATING RESULTS " + "=" * 30)
    evaluate_args = ["--specialty", specialty, "--lang", "all"]
    run_script("evaluate_transcriptions.py", evaluate_args)
    
    # Step 5: Clean up unnecessary intermediate files
    print("\n\n" + "=" * 30 + " STEP 5: CLEANING UP WORKSPACE " + "=" * 30)
    cleanup_files(specialty)
    
    print("\n\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nResults are available in the data-med directory:")
    print("  - Original conversations: data-med/[specialty]/[language]/json")
    print("  - SOAP notes: data-med/[specialty]/[language]/soap")
    print("  - Audio files: data-med/[specialty]/[language]/audio")
    print("  - Transcriptions: data-med/[specialty]/[language]/transcripts")
    print("  - Evaluation results: data-med/evaluation/[specialty]")
    print("\nKey evaluations to check:")
    print("  - Comparison of transcription accuracy between:")
    print("    * Deepgram Nova 3 Medical (English)")
    print("    * Azure Speech Services (French)")
    print("  - Evaluation metrics include WER, medical term accuracy, and speaker accuracy")
    print("  - CSV results: data-med/evaluation/[specialty]/[language]_[specialty]_results.csv")
    print("  - Visualization plots: data-med/evaluation/[specialty]/")

def cleanup_files(specialty="all"):
    """Clean up unnecessary files."""
    # Determine specialties to process
    specialties = ["cardiology", "gp"] if specialty == "all" else [specialty]
    
    for spec in specialties:
        # Process each language
        for lang in ["en-CA", "fr-CA"]:
            # Clean up debug responses and other temporary files
            audio_dir = os.path.join(BASE_DIR, spec, lang, "audio")
            
            # Find and remove temporary/debug files
            if os.path.exists(audio_dir):
                for file in os.listdir(audio_dir):
                    if file.endswith("_dg_response.json") or file.endswith("_azure_response.json"):
                        try:
                            os.remove(os.path.join(audio_dir, file))
                            print(f"Removed {file}")
                        except:
                            pass
    
    # Remove any __pycache__ directories
    for root, dirs, files in os.walk(".", topdown=False):
        for name in dirs:
            if name == "__pycache__":
                try:
                    import shutil
                    shutil.rmtree(os.path.join(root, name))
                    print(f"Removed {os.path.join(root, name)}")
                except:
                    pass

def main():
    """Main function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the medical conversation pipeline')
    parser.add_argument('--num', type=int, default=3, help='Number of conversation pairs to generate per specialty and language')
    parser.add_argument('--specialty', type=str, choices=['cardiology', 'gp', 'all'], default='all', help='Medical specialty to process')
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run the pipeline
    run_pipeline(args.num, args.specialty)

if __name__ == "__main__":
    main() 