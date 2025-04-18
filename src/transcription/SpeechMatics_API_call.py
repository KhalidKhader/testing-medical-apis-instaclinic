

from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get SPEECHMATICS_API_KEY token from .env
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")

PATH_TO_FILE = "med-data/gp/en-CA-whisper-v3-large/audio/gp_10_followup.wav"
LANGUAGE = "en"

settings = ConnectionSettings(
    url="https://asr.api.speechmatics.com/v2",
    auth_token=API_KEY,
)

# Define transcription parameters
conf = {
    "type": "transcription",
    "transcription_config": {
        "language": LANGUAGE, 
        "diarization": "speaker",
        "speaker_diarization_config": {
        "speaker_sensitivity": 0.6
        }
    }
}

# Open the client using a context manager
with BatchClient(settings) as client:
    try:
        job_id = client.submit_job(
            audio=PATH_TO_FILE,
            transcription_config=conf,
        )
        print(f'job {job_id} submitted successfully, waiting for transcript')

        # Note that in production, you should set up notifications instead of polling.
        # Notifications are described here: https://docs.speechmatics.com/features-other/notifications
        transcript = client.wait_for_completion(job_id, transcription_format='txt')
        # To see the full output, try setting transcription_format='json-v2'.
        print(transcript)
    except HTTPStatusError as e:
        if e.response.status_code == 401:
            print('Invalid API key - Check your API_KEY at the top of the code!')
        elif e.response.status_code == 400:
            print(e.response.json()['detail'])
        else:
            raise e
