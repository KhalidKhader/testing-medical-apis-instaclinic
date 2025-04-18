import json
import requests
from dotenv import load_dotenv
import os
import time
from pathlib import Path
import glob

def process_audio_file(file_path, api_key, language_code):
    """
    Process a single audio file using Transkriptor API
    """
    file_name = os.path.basename(file_path)
    print(f"\n{'='*80}")
    print(f"Processing file: {file_name}")
    print(f"{'='*80}")

    # Step 1: Obtain the Upload URL
    url = "https://api.tor.app/developer/transcription/local_file/get_upload_url"
    
    # Set up the headers, including the API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    # Request body with the file name
    body = json.dumps({"file_name": file_name})
    
    # Request to get the upload URL
    response = requests.post(url, headers=headers, data=body)
    if response.status_code == 200:
        response_json = response.json()
        upload_url = response_json["upload_url"]
        public_url = response_json["public_url"]
        print("Upload URL obtained")
    else:
        print("Failed to get upload URL:", response.status_code, response.text)
        return None
    
    # Step 2: Upload the Local File
    with open(file_path, "rb") as file_data:
        print(f"Uploading file {file_name}...")
        upload_response = requests.put(upload_url, data=file_data)
        if upload_response.status_code == 200:
            print("File uploaded successfully")
        else:
            print("File upload failed:", upload_response.status_code, upload_response.text)
            return None
    
    # Step 3: Initiate Transcription for the Uploaded File
    initiate_url = "https://api.tor.app/developer/transcription/local_file/initiate_transcription"
    
    # Configure transcription parameters
    config = json.dumps({
        "url": public_url,
        "language": language_code,  # Language code (en-CA for English Canadian)
        "service": "Standard",
    })
    
    # Send request to initiate transcription
    print(f"Initiating transcription with language: {language_code}...")
    transcription_response = requests.post(initiate_url, headers=headers, data=config)
    if transcription_response.status_code == 202:
        transcription_json = transcription_response.json()
        print(transcription_json["message"])
        order_id = transcription_json["order_id"]
        print("Order ID:", order_id)
        return order_id
    else:
        print(
            "Failed to initiate transcription:",
            transcription_response.status_code,
            transcription_response.text,
        )
        return None

def check_transcription_status(order_id, api_key):
    """
    Check the status of a transcription job
    """
    url = f"https://api.tor.app/developer/files/{order_id}"
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        response_json = response.json()
        status = response_json.get("status", "unknown")
        # Map "Completed" status to "ready" for our script logic
        if status == "Completed":
            return "ready"
        return status
    else:
        print(f"Failed to check status for order {order_id}: {response.status_code}")
        return "error"

def get_transcription_content(order_id, api_key):
    """
    Get the content of a completed transcription
    """
    url = f"https://api.tor.app/developer/files/{order_id}/content"
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get content for order {order_id}: {response.status_code}, {response.text}")
        return None

def save_transcription(order_id, content, output_dir, original_filename):
    """
    Save the transcription content to a file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename based on original audio filename
    base_name = os.path.splitext(original_filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}_transcript.json")
    
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
    
    # Add metadata to the content for reference
    content["order_id"] = order_id
    content["original_file"] = original_filename
    content["metadata"] = {
        "specialty": specialty,
        "case_number": case_number,
        "encounter_type": encounter_type,
        "full_name": base_name,
        "full_filename": original_filename
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(content, f, indent=2)
    
    print(f"Transcription saved to {output_path}")
    return output_path

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("TRANSKRIPTOR_API_KEY")
    if not api_key:
        print("Error: TRANSKRIPTOR_API_KEY not found in .env file")
        return
    
    # Set paths for English GP audio files
    audio_dir = "med-data/gp/en-CA/audio"
    output_dir = "results/transkriptor/gp-en-CA"
    language_code = "en-CA"  # English Canadian
    
    # Get list of all .wav files in the directory
    files = glob.glob(f"{audio_dir}/*.wav")
    
    if not files:
        print(f"No .wav files found in {audio_dir}")
        return
    
    print(f"Found {len(files)} English GP audio files to process")
    
    # Store order IDs for all files to check status later
    orders = {}
    
    # Process each file
    for file_path in files:
        order_id = process_audio_file(file_path, api_key, language_code)
        if order_id:
            orders[order_id] = os.path.basename(file_path)
    
    print(f"\n{'='*80}")
    print(f"Submitted {len(orders)} files for transcription")
    print(f"{'='*80}")
    
    # Wait for all transcriptions to complete
    pending_orders = orders.copy()
    completed_orders = {}
    max_attempts = 30  # Maximum number of status check attempts
    wait_time = 15  # Time to wait between status checks (seconds)
    
    print(f"\nWaiting for transcriptions to complete...")
    print(f"Will check status every {wait_time} seconds")
    
    attempt = 0
    while pending_orders and attempt < max_attempts:
        attempt += 1
        print(f"\nStatus check attempt {attempt}/{max_attempts}")
        
        for order_id, filename in list(pending_orders.items()):
            status = check_transcription_status(order_id, api_key)
            print(f"Order {order_id} ({filename}): {status}")
            
            if status == "ready":
                print(f"Retrieving content for {filename}...")
                content = get_transcription_content(order_id, api_key)
                if content:
                    save_path = save_transcription(order_id, content, output_dir, filename)
                    completed_orders[order_id] = save_path
                    del pending_orders[order_id]
                else:
                    print(f"Failed to retrieve content for {filename}, will retry later")
            elif status == "error":
                print(f"Error in transcription for {filename}")
                del pending_orders[order_id]
        
        if pending_orders:
            print(f"\nWaiting for {len(pending_orders)} pending transcriptions...")
            print(f"Sleeping for {wait_time} seconds...")
            time.sleep(wait_time)
    
    # Summary
    print(f"\n{'='*80}")
    print("TRANSCRIPTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {len(orders)}")
    print(f"Successfully completed: {len(completed_orders)}")
    print(f"Failed or pending: {len(pending_orders)}")
    
    if pending_orders:
        print("\nTranscriptions still pending:")
        for order_id, filename in pending_orders.items():
            print(f"  - {filename} (Order ID: {order_id})")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 