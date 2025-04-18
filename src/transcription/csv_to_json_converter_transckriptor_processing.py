import csv
import json
import os
import glob
from pathlib import Path

def csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to JSON format
    """
    # Extract metadata from file path
    file_name = os.path.basename(csv_file_path)
    base_name = os.path.splitext(file_name)[0]  # Remove extension
    
    # Extract metadata about the file
    file_parts = base_name.split('_')
    if len(file_parts) >= 3:
        specialty = file_parts[0]
        case_number = file_parts[1]
        encounter_type = file_parts[2]  # consultation or followup
    else:
        specialty = "unknown"
        case_number = "unknown"
        encounter_type = "unknown"
    
    # Read CSV file
    data = {
        "metadata": {
            "specialty": specialty,
            "case_number": case_number,
            "encounter_type": encounter_type,
            "full_name": base_name,
            "original_file": file_name
        },
        "transcript": []
    }
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as csv_file:  # Using utf-8-sig to handle BOM
            csv_reader = csv.DictReader(csv_file)
            
            # Process each row
            for row in csv_reader:
                # Check for speaker name column (handling potential BOM character)
                speaker_key = None
                if 'SpeakerName' in row:
                    speaker_key = 'SpeakerName'
                elif '\ufeffSpeakerName' in row:
                    speaker_key = '\ufeffSpeakerName'
                
                # Check if required columns exist
                if speaker_key and 'Timestamp' in row and 'Text' in row:
                    utterance = {
                        "speaker": row[speaker_key],
                        "timestamp": row['Timestamp'],
                        "text": row['Text']
                    }
                    data["transcript"].append(utterance)
                else:
                    print(f"Warning: Missing expected columns in {csv_file_path}")
                    print(f"Row data: {row}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        
        # Write JSON file
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)
        
        print(f"Converted {csv_file_path} -> {json_file_path}")
        return True
    
    except Exception as e:
        print(f"Error processing {csv_file_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, language_code):
    """
    Process all CSV files in a directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return 0
    
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    
    # Process each file
    success_count = 0
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        base_name = os.path.splitext(file_name)[0]
        json_file = os.path.join(output_dir, f"{base_name}.json")
        
        if csv_to_json(csv_file, json_file):
            success_count += 1
    
    print(f"Successfully converted {success_count} out of {len(csv_files)} files from {input_dir}")
    return success_count

def main():
    # Base directories
    transkriptor_dir = "/Users/khalidkhader/Desktop/Transkriptor"
    output_base_dir = "converted_json"
    
    # Create the output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Define input and output directories with language codes
    directories = [
        {
            "input": os.path.join(transkriptor_dir, "Carddiology", "cardiolgy -en"),
            "output": os.path.join(output_base_dir, "cardiology-en"),
            "language": "en-CA"
        },
        {
            "input": os.path.join(transkriptor_dir, "Carddiology", "Cardiology - FR"),
            "output": os.path.join(output_base_dir, "cardiology-fr"),
            "language": "fr-CA"
        },
        {
            "input": os.path.join(transkriptor_dir, "GP", "GP - En"),
            "output": os.path.join(output_base_dir, "gp-en"),
            "language": "en-CA"
        },
        {
            "input": os.path.join(transkriptor_dir, "GP", "GP - FR"),
            "output": os.path.join(output_base_dir, "gp-fr"),
            "language": "fr-CA"
        }
    ]
    
    # Process each directory
    total_files = 0
    for directory in directories:
        print("\n" + "="*80)
        print(f"Processing {directory['input']} (Language: {directory['language']})")
        print("="*80)
        
        # Add language to output path
        success_count = process_directory(
            directory["input"], 
            directory["output"], 
            directory["language"]
        )
        total_files += success_count
    
    print("\n" + "="*80)
    print(f"Summary: Converted a total of {total_files} CSV files to JSON format")
    print("="*80)

if __name__ == "__main__":
    main() 