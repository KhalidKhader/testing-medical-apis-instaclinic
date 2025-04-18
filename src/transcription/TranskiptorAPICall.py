import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: Obtain the Upload URL
url = "https://api.tor.app/developer/transcription/local_file/get_upload_url"

# API key from environment variables
api_key = os.getenv("TRANSKRIPTOR_API_KEY")

# Set up the headers, including the API key
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json",
}

# Use an existing file and its name in the request
file_path = "cardiology_1_consultation.wav"
file_name = os.path.basename(file_path)

# Request body with the file name
body = json.dumps({"file_name": file_name})

print(f"Using file: {file_path}")
print(f"Using API key: {api_key[:5]}..." if api_key else "API key not found!")

# Request to get the upload URL
response = requests.post(url, headers=headers, data=body)
if response.status_code == 200:
    response_json = response.json()
    upload_url = response_json["upload_url"]
    public_url = response_json[
        "public_url"
    ]  # URL to pass in initiate transcription step
    print("Upload URL obtained:", upload_url)
    print("Public URL obtained:", public_url)
else:
    print("Failed to get upload URL:", response.status_code, response.text)
    exit()

# Step 2: Upload the Local File
with open(file_path, "rb") as file_data:
    print(f"Uploading file {file_path}...")
    upload_response = requests.put(upload_url, data=file_data)
    if upload_response.status_code == 200:
        print("File uploaded successfully")
    else:
        print("File upload failed:", upload_response.status_code, upload_response.text)
        exit()

# Step 3: Initiate Transcription for the Uploaded File
initiate_url = (
    "https://api.tor.app/developer/transcription/local_file/initiate_transcription"
)

# Configure transcription parameters
config = json.dumps(
    {
        "url": public_url,  # Passing public_url to initiate transcription
        "language": "en-CA",  # Canadian English
        "service": "Standard",
        # "folder_id": "your_folder_id",  # Optional folder_id
        # "triggering_word": "example",  # Optional triggering_word
    }
)

# Send request to initiate transcription
print("Initiating transcription...")
transcription_response = requests.post(initiate_url, headers=headers, data=config)
if transcription_response.status_code == 202:
    transcription_json = transcription_response.json()
    print(transcription_json["message"])
    print("Order ID:", transcription_json["order_id"])
    order_id = transcription_json["order_id"]
    # Set up the headers, including the API key
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    url = f"https://api.tor.app/developer/files/{order_id}/content"


    response = requests.get(url, headers=headers)

    response_json = response.json()

    # print(response_json)
else:
    print(
        "Failed to initiate transcription:",
        transcription_response.status_code,
        transcription_response.text,
    )


