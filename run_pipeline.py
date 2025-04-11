#!/usr/bin/env python3
"""
Medical Speech-to-Text Pipeline Runner

This script orchestrates the full pipeline for generating, transcribing,
and evaluating medical conversations across different models and languages.
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("medical_stt_pipeline")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the medical speech-to-text evaluation pipeline")
    
    parser.add_argument("--specialty", type=str, choices=["cardiology", "gp", "all"], default="all",
                        help="Medical specialty to process (default: all)")
    
    parser.add_argument("--language", type=str, choices=["en-CA", "fr-CA", "all"], default="all",
                        help="Language to process (default: all)")
    
    parser.add_argument("--noise", type=str, choices=["none", "moderate", "high", "all"], default="all",
                        help="Noise level to apply (default: all)")
    
    parser.add_argument("--num", type=int, default=5,
                        help="Number of conversations to generate (default: 5)")
    
    parser.add_argument("--models", type=str, nargs="+", 
                        choices=["azure", "nova-3-medical", "nova-2", "all"], default=["all"],
                        help="Models to evaluate (default: all)")
    
    parser.add_argument("--steps", type=str, nargs="+",
                        choices=["generate", "convert", "transcribe", "evaluate", "visualize", "all"],
                        default=["all"],
                        help="Pipeline steps to run (default: all)")
    
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for output files (default: results)")
    
    return parser.parse_args()

def main():
    """Run the complete pipeline."""
    args = parse_args()
    
    logger.info("Starting Medical STT Evaluation Pipeline")
    logger.info(f"Arguments: {args}")
    
    # Import here to avoid circular imports
    from src.pipeline import generate_medical_conversations
    from src.pipeline import convert_to_speech
    from src.pipeline import convert_to_speech_noisy
    from src.pipeline import transcribe_conversations
    from src.pipeline import transcribe_conversations_opposite
    from src.pipeline import soap_generation
    from src.evaluation import evaluator
    from src.visualization import plots
    
    # Expand "all" options
    if "all" in args.steps or args.steps == ["all"]:
        steps = ["generate", "convert", "transcribe", "evaluate", "visualize"]
    else:
        steps = args.steps
        
    specialties = ["cardiology", "gp"] if args.specialty == "all" else [args.specialty]
    languages = ["en-CA", "fr-CA"] if args.language == "all" else [args.language]
    noise_levels = ["none", "moderate", "high"] if args.noise == "all" else [args.noise]
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "reports"), exist_ok=True)
    
    # Step 1: Generate conversations
    if "generate" in steps:
        logger.info("Step 1: Generating medical conversations")
        for specialty in specialties:
            for language in languages:
                generate_medical_conversations.generate(
                    specialty=specialty,
                    language=language,
                    num_conversations=args.num,
                    output_dir=os.path.join("data", "raw", "transcripts")
                )
    
    # Step 2: Convert to speech
    if "convert" in steps:
        logger.info("Step 2: Converting text to speech")
        for noise_level in noise_levels:
            if noise_level == "none":
                convert_to_speech.convert(
                    input_dir=os.path.join("data", "raw", "transcripts"),
                    output_dir=os.path.join("data", "processed", "audio", "clean"),
                    specialties=specialties,
                    languages=languages
                )
            elif noise_level == "moderate":
                convert_to_speech_noisy.convert(
                    input_dir=os.path.join("data", "raw", "transcripts"),
                    output_dir=os.path.join("data", "processed", "audio", "semi-noise"),
                    noise_level="moderate",
                    specialties=specialties,
                    languages=languages
                )
            elif noise_level == "high":
                convert_to_speech_noisy.convert(
                    input_dir=os.path.join("data", "raw", "transcripts"),
                    output_dir=os.path.join("data", "processed", "audio", "high-noise"),
                    noise_level="high",
                    specialties=specialties,
                    languages=languages
                )
    
    # Step 3: Transcribe audio
    if "transcribe" in steps:
        logger.info("Step 3: Transcribing audio")
        models = []
        if "all" in args.models or args.models == ["all"]:
            models = ["azure", "nova-3-medical", "nova-2"]
        else:
            models = args.models
            
        for model in models:
            for noise_level in noise_levels:
                noise_dir = "clean" if noise_level == "none" else f"{noise_level}-noise"
                transcribe_conversations.transcribe(
                    model=model,
                    input_dir=os.path.join("data", "processed", "audio", noise_dir),
                    output_dir=os.path.join("data", "processed", "transcripts", model),
                    specialties=specialties,
                    languages=languages
                )
                
        # Generate SOAP notes
        logger.info("Generating SOAP notes from transcriptions")
        for model in models:
            soap_generation.generate(
                input_dir=os.path.join("data", "processed", "transcripts", model),
                output_dir=os.path.join("data", "processed", "soap", model),
                specialties=specialties,
                languages=languages
            )
    
    # Step 4: Evaluate transcriptions
    if "evaluate" in steps:
        logger.info("Step 4: Evaluating transcription quality")
        evaluator.evaluate(
            reference_dir=os.path.join("data", "raw", "transcripts"),
            transcription_dir=os.path.join("data", "processed", "transcripts"),
            soap_dir=os.path.join("data", "processed", "soap"),
            output_dir=os.path.join(args.output_dir, "tables"),
            specialties=specialties,
            languages=languages
        )
    
    # Step 5: Generate visualizations
    if "visualize" in steps:
        logger.info("Step 5: Generating visualizations")
        plots.generate_all_plots(
            data_file=os.path.join(args.output_dir, "tables", "evaluation_results.csv"),
            output_dir=os.path.join(args.output_dir, "figures")
        )
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main() 