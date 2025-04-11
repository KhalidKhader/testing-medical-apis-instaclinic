import os
import glob
import subprocess
import argparse
from pathlib import Path

def process_folder(input_dir, output_dir, file_type):
    """Process all JSON files in a folder to generate SOAP notes."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    print(f"Found {len(json_files)} files in {input_dir}")
    
    # Process each file
    for json_file in json_files:
        base_name = os.path.basename(json_file)
        output_file = os.path.join(output_dir, base_name.replace('.json', '_soap.txt'))
        
        # Skip if output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {base_name} - output file already exists")
            continue
        
        print(f"Processing {base_name}...")
        
        # Build command to run process_single_file.py
        cmd = [
            "python", "process_single_file.py", 
            json_file, 
            output_file
        ]
        
        if file_type == "transcript":
            cmd.append("--type")
            cmd.append("transcript")
        
        # Run the command
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {base_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {base_name}: {e}")

def process_dataset(base_dir, noise_condition, specialty):
    """Process a specific dataset with both JSON and transcript files."""
    # Construct paths
    dataset_path = os.path.join(base_dir, noise_condition, specialty)
    
    # Handle different directory structures based on the noise condition
    if "French-Nova-3-medical" in noise_condition:
        # The structure for Noisy-Azure for French-Nova-3-medical- for English is different
        en_json_dir = os.path.join(dataset_path, "en-CA - Nova - 3 - medical", "json")
        en_transcripts_dir = os.path.join(dataset_path, "en-CA - Nova - 3 - medical", "transcripts")
        
        # Output directories
        en_json_output_dir = os.path.join(dataset_path, "en-CA - Nova - 3 - medical", "soap", "original from json")
        en_transcript_output_dir = os.path.join(dataset_path, "en-CA - Nova - 3 - medical", "soap", "transcribed")
        
        # French directories
        fr_json_dir = os.path.join(dataset_path, "fr-CA - Azure", "json")
        fr_transcripts_dir = os.path.join(dataset_path, "fr-CA - Azure", "transcripts")
        
        # Output directories
        fr_json_output_dir = os.path.join(dataset_path, "fr-CA - Azure", "soap", "original from json")
        fr_transcript_output_dir = os.path.join(dataset_path, "fr-CA - Azure", "soap", "transcribed")
    else:
        # The original structure for other datasets
        en_json_dir = os.path.join(dataset_path, "en-CA - Azure", "json")
        en_transcripts_dir = os.path.join(dataset_path, "en-CA - Azure", "transcripts")
        
        # Output directories
        en_json_output_dir = os.path.join(dataset_path, "en-CA - Azure", "soap", "original from json")
        en_transcript_output_dir = os.path.join(dataset_path, "en-CA - Azure", "soap", "transcribed")
        
        # French directories
        fr_json_dir = os.path.join(dataset_path, "fr-CA - Nova - 2", "json")
        fr_transcripts_dir = os.path.join(dataset_path, "fr-CA - Nova - 2", "transcripts")
        
        # Output directories
        fr_json_output_dir = os.path.join(dataset_path, "fr-CA - Nova - 2", "soap", "original from json")
        fr_transcript_output_dir = os.path.join(dataset_path, "fr-CA - Nova - 2", "soap", "transcribed")
    
    # Process files if directories exist
    if os.path.exists(en_json_dir):
        print(f"\nProcessing English JSON files from {en_json_dir}...")
        process_folder(en_json_dir, en_json_output_dir, "json")
    
    if os.path.exists(en_transcripts_dir):
        print(f"\nProcessing English transcript files from {en_transcripts_dir}...")
        process_folder(en_transcripts_dir, en_transcript_output_dir, "transcript")
    
    # Process French files
    if os.path.exists(fr_json_dir):
        print(f"\nProcessing French JSON files from {fr_json_dir}...")
        process_folder(fr_json_dir, fr_json_output_dir, "json")
    
    if os.path.exists(fr_transcripts_dir):
        print(f"\nProcessing French transcript files from {fr_transcripts_dir}...")
        process_folder(fr_transcripts_dir, fr_transcript_output_dir, "transcript")

def main():
    parser = argparse.ArgumentParser(description='Process folders of JSON and transcript files to generate SOAP notes.')
    parser.add_argument('--base_dir', default='all-data', help='Base directory containing the datasets')
    parser.add_argument('--noise_condition', help='Specific noise condition to process (process all if not specified)')
    parser.add_argument('--specialty', help='Specific specialty to process (process all if not specified)')
    args = parser.parse_args()
    
    base_dir = os.path.join(os.getcwd(), args.base_dir)
    
    # Available noise conditions and specialties
    noise_conditions = [
        "Noisy-Azure for English-Nova-2- for French",
        "Noisy-Azure for French-Nova-3-medical- for English",
        "Semi-noise - Azure for English-Nova-2 for French",
        "Semi-noise - Azure for French-Nova-3-medical- for English",
        "Without-noise-Azure for English-Nova-2- for French",
        "Without-noise-Azure for French-Nova-3-medical- for English"
    ]
    
    specialties = ["cardiology", "gp"]
    
    # If specific noise_condition and specialty are provided, process only those
    if args.noise_condition and args.specialty:
        if args.noise_condition in noise_conditions and args.specialty in specialties:
            print(f"Processing {args.noise_condition} - {args.specialty}")
            process_dataset(base_dir, args.noise_condition, args.specialty)
        else:
            print(f"Invalid noise condition or specialty")
            return
    
    # If only noise_condition is provided, process all specialties for that condition
    elif args.noise_condition:
        if args.noise_condition in noise_conditions:
            print(f"Processing {args.noise_condition} - all specialties")
            for specialty in specialties:
                process_dataset(base_dir, args.noise_condition, specialty)
        else:
            print(f"Invalid noise condition")
            return
    
    # If only specialty is provided, process that specialty across all noise conditions
    elif args.specialty:
        if args.specialty in specialties:
            print(f"Processing all noise conditions - {args.specialty}")
            for noise_condition in noise_conditions:
                process_dataset(base_dir, noise_condition, args.specialty)
        else:
            print(f"Invalid specialty")
            return
    
    # If neither is provided, process everything
    else:
        print("Processing all noise conditions and specialties")
        for noise_condition in noise_conditions:
            for specialty in specialties:
                process_dataset(base_dir, noise_condition, specialty)

if __name__ == "__main__":
    main() 