#!/usr/bin/env python3
"""
Generate medical conversations and SOAP notes for Cardiology and GP specialties using OpenAI's GPT models.
This script creates realistic medical conversations between doctors and patients,
along with corresponding SOAP notes for consultations and follow-ups.

Usage:
    python generate_medical_conversations.py --num 5 --specialty cardiology 
    python generate_medical_conversations.py --num 5 --specialty gp
    python generate_medical_conversations.py --num 5 --specialty all
"""

import os
import json
import random
import time
import argparse
import openai
from openai import AzureOpenAI
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://notegen-openai-eastca.openai.azure.com")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# # Set up OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")
# if not openai.api_key:
#     print("Warning: OPENAI_API_KEY not found in environment. Set it in your .env file.")

# Define base directories for output
BASE_DIR = "data-med"

# Medical specialties
SPECIALTIES = ["cardiology", "gp"]

# Medical conditions by specialty
MEDICAL_CONDITIONS = {
    "cardiology": {
        "en": [
            "hypertension", "atrial fibrillation", "heart failure", "coronary artery disease", 
            "aortic stenosis", "mitral valve prolapse", "myocardial infarction", 
            "ventricular tachycardia", "pericarditis", "endocarditis"
        ],
        "fr": [
            "hypertension", "fibrillation auriculaire", "insuffisance cardiaque", "maladie coronarienne", 
            "sténose aortique", "prolapsus de la valve mitrale", "infarctus du myocarde", 
            "tachycardie ventriculaire", "péricardite", "endocardite"
        ]
    },
    "gp": {
        "en": [
            "type 2 diabetes", "asthma", "osteoarthritis", "GERD", "urinary tract infection", 
            "upper respiratory infection", "allergic rhinitis", "depression", 
            "hypothyroidism", "migraine"
        ],
        "fr": [
            "diabète de type 2", "asthme", "arthrose", "RGO", "infection des voies urinaires", 
            "infection des voies respiratoires supérieures", "rhinite allergique", "dépression", 
            "hypothyroïdie", "migraine"
        ]
    }
}

# Treatments by specialty
TREATMENTS = {
    "cardiology": {
        "en": [
            "lisinopril", "metoprolol", "warfarin", "apixaban", "atorvastatin", 
            "furosemide", "spironolactone", "amiodarone", "digoxin", "clopidogrel"
        ],
        "fr": [
            "lisinopril", "métoprolol", "warfarine", "apixaban", "atorvastatine", 
            "furosémide", "spironolactone", "amiodarone", "digoxine", "clopidogrel"
        ]
    },
    "gp": {
        "en": [
            "metformin", "albuterol", "ibuprofen", "omeprazole", "ciprofloxacin", 
            "amoxicillin", "cetirizine", "sertraline", "levothyroxine", "sumatriptan"
        ],
        "fr": [
            "metformine", "salbutamol", "ibuprofène", "oméprazole", "ciprofloxacine", 
            "amoxicilline", "cétirizine", "sertraline", "lévothyroxine", "sumatriptan"
        ]
    }
}

# SOAP note prompt templates for initial consultations
CONSULT_PROMPTS = {
    "cardiology": {
        "en": """Generate a realistic cardiology consultation between a doctor and a patient, along with a SOAP note.

Patient profile: {age} year old {gender} with {condition}.

Format the response as follows:
1. First, provide a conversation transcript showing exactly what the doctor and patient said to each other. Each line should start with either "Doctor:" or "Patient:" to clearly indicate the speaker.
2. Then, after the conversation, provide a detailed SOAP note for this encounter.
3. Finally, include a "Keywords / Tags" section that categorizes key aspects of the encounter.

The conversation should be detailed but realistic, covering:
- Introduction and rapport building (use "Good morning/afternoon" for the initial greeting, not "How have you been since your last visit" as this is a first consultation)
- Patient's chief complaint and history
- Review of cardiovascular symptoms
- Discussion of any diagnostic tests (ECG, echocardiogram, stress test, etc.)
- Physical examination findings
- Assessment and explanation of the cardiac condition
- Treatment plan discussion (medications, lifestyle changes, procedures if needed)
- Follow-up instructions

The SOAP note should include:
- Subjective: Patient's reported symptoms, history, and concerns
- Objective: Cardiovascular examination findings, vital signs, and any test results
- Assessment: Detailed cardiac assessment and differential diagnoses
- Plan: Treatment plan, cardiac medications, procedures, and follow-up recommendations

The Keywords / Tags section should include:
- Diagnosis: Primary and secondary cardiac diagnoses
- Treatment: Cardiac medications and interventions
- Symptoms: Key cardiac symptoms reported
- Conditions: Underlying or related cardiac conditions
- Allergies: Any medication allergies

Be specific to cardiology practice in Canada, using Canadian medical terminology and medication names.""",
        
        "fr": """Générez une consultation cardiologique réaliste entre un médecin et un patient, accompagnée d'une note SOAP.

Profil du patient : {gender} de {age} ans avec {condition}.

Formatez la réponse comme suit :
1. D'abord, fournissez une transcription de la conversation montrant exactement ce que le médecin et le patient se sont dit. Chaque ligne doit commencer par "Médecin:" ou "Patient:" pour indiquer clairement qui parle.
2. Ensuite, après la conversation, fournissez une note SOAP détaillée pour cette rencontre.
3. Enfin, incluez une section "Mots-clés / Étiquettes" qui catégorise les aspects clés de la rencontre.

La conversation doit être détaillée mais réaliste, couvrant:
- Introduction et établissement de la relation (utilisez "Bonjour" pour la salutation initiale, pas "Comment allez-vous depuis notre dernière visite" car c'est une première consultation)
- Plainte principale et antécédents du patient
- Examen des symptômes cardiovasculaires
- Discussion des tests diagnostiques (ECG, échocardiogramme, test d'effort, etc.)
- Résultats de l'examen physique
- Évaluation et explication de la condition cardiaque
- Discussion du plan de traitement (médicaments, changements de style de vie, procédures si nécessaires)
- Instructions de suivi

La note SOAP doit inclure:
- Subjectif: Symptômes rapportés par le patient, antécédents et préoccupations
- Objectif: Résultats de l'examen cardiovasculaire, signes vitaux et résultats des tests
- Évaluation: Évaluation cardiaque détaillée et diagnostics différentiels
- Plan: Plan de traitement, médicaments cardiaques, procédures et recommandations de suivi

La section Mots-clés / Étiquettes doit inclure:
- Diagnostic: Diagnostics cardiaques primaires et secondaires
- Traitement: Médicaments et interventions cardiaques
- Symptômes: Principaux symptômes cardiaques rapportés
- Conditions: Conditions cardiaques sous-jacentes ou connexes
- Allergies: Toute allergie aux médicaments

Soyez spécifique à la pratique cardiologique au Canada, en utilisant la terminologie médicale et les noms de médicaments canadiens."""
    },
    
    "gp": {
        "en": """Generate a realistic General Practitioner (GP) consultation between a doctor and a patient, along with a SOAP note.

Patient profile: {age} year old {gender} with {condition}.

Format the response as follows:
1. First, provide a conversation transcript showing exactly what the doctor and patient said to each other. Each line should start with either "Doctor:" or "Patient:" to clearly indicate the speaker.
2. Then, after the conversation, provide a detailed SOAP note for this encounter.
3. Finally, include a "Keywords / Tags" section that categorizes key aspects of the encounter.

The conversation should be detailed but realistic, covering:
- Introduction and rapport building (use "Good morning/afternoon, what brings you in today?" for the initial greeting, not "How have you been since your last visit" as this is a first consultation)
- Patient's chief complaint and history
- Review of relevant symptoms
- Physical examination discussion
- Assessment and explanation to the patient
- Treatment plan discussion
- Preventive care recommendations
- Follow-up instructions

The SOAP note should include:
- Subjective: Patient's reported symptoms, history, and concerns
- Objective: Physical examination findings, vital signs, and any test results
- Assessment: Medical assessment and differential diagnoses
- Plan: Treatment plan, prescriptions, referrals, and follow-up recommendations

The Keywords / Tags section should include:
- Diagnosis: List of diagnoses and conditions identified
- Treatment: Medications, procedures, and interventions
- Symptoms: Key symptoms reported by the patient
- Conditions: Underlying or related medical conditions
- Allergies: Any allergies mentioned

Be specific to primary care practice in Canada, using Canadian medical terminology and medication names.""",
        
        "fr": """Générez une consultation réaliste de médecine générale entre un médecin et un patient, accompagnée d'une note SOAP.

Profil du patient : {gender} de {age} ans avec {condition}.

Formatez la réponse comme suit :
1. D'abord, fournissez une transcription de la conversation montrant exactement ce que le médecin et le patient se sont dit. Chaque ligne doit commencer par "Médecin:" ou "Patient:" pour indiquer clairement qui parle.
2. Ensuite, après la conversation, fournissez une note SOAP détaillée pour cette rencontre.
3. Enfin, incluez une section "Mots-clés / Étiquettes" qui catégorise les aspects clés de la rencontre.

La conversation doit être détaillée mais réaliste, couvrant:
- Introduction et établissement de la relation (utilisez "Bonjour, qu'est-ce qui vous amène aujourd'hui?" pour la salutation initiale, pas "Comment allez-vous depuis notre dernière visite" car c'est une première consultation)
- Plainte principale et antécédents du patient
- Examen des symptômes pertinents
- Discussion de l'examen physique
- Évaluation et explication au patient
- Discussion du plan de traitement
- Recommandations de soins préventifs
- Instructions de suivi

La note SOAP doit inclure:
- Subjectif: Symptômes rapportés par le patient, antécédents et préoccupations
- Objectif: Résultats de l'examen physique, signes vitaux et résultats des tests
- Évaluation: Évaluation médicale et diagnostics différentiels
- Plan: Plan de traitement, prescriptions, références et recommandations de suivi

La section Mots-clés / Étiquettes doit inclure:
- Diagnostic: Liste des diagnostics et conditions identifiés
- Traitement: Médicaments, procédures et interventions
- Symptômes: Principaux symptômes rapportés par le patient
- Conditions: Conditions médicales sous-jacentes ou connexes
- Allergies: Toutes allergies mentionnées

Soyez spécifique à la pratique de médecine générale au Canada, en utilisant la terminologie médicale et les noms de médicaments canadiens."""
    }
}

# SOAP note prompt templates for follow-up visits
FOLLOWUP_PROMPTS = {
    "cardiology": {
        "en": """Generate a realistic cardiology follow-up conversation between a doctor and a patient, along with a follow-up SOAP note.

Patient profile: {age} year old {gender} with {condition}.

Format the response as follows:
1. First, provide a conversation transcript showing exactly what the doctor and patient said to each other. Each line should start with either "Doctor:" or "Patient:" to clearly indicate the speaker.
2. Then, after the conversation, provide a detailed SOAP note for this follow-up encounter.
3. Finally, include a "Keywords / Tags" section that categorizes key aspects of the encounter.

The conversation should be detailed but realistic, covering:
- Greeting with reference to their previous visit (e.g., "Good morning, Mrs. Smith. How have you been feeling since our last visit?")
- Discussion of how the patient has been doing since the last visit
- Review of any changes in symptoms
- Discussion of adherence to treatment plan
- Any side effects from medications
- Physical examination findings
- Review of any new test results
- Assessment of the effectiveness of the current treatment
- Any adjustments to the treatment plan
- Follow-up instructions

The SOAP note should include:
- Subjective: Patient's reported symptoms, progress, medication adherence, and any side effects
- Objective: Updated cardiovascular examination findings, vital signs, and any new test results
- Assessment: Updated cardiac assessment and progress evaluation
- Plan: Any changes to treatment plan, medications, and follow-up recommendations

The Keywords / Tags section should include:
- Diagnosis: Updated cardiac diagnoses
- Treatment: Current and any new cardiac medications or interventions
- Symptoms: Any changes in symptoms since the last visit
- Conditions: Updated status of cardiac conditions
- Allergies: Any medication allergies

Be specific to cardiology practice in Canada, using Canadian medical terminology and medication names.""",
        
        "fr": """Générez une conversation de suivi cardiologique réaliste entre un médecin et un patient, accompagnée d'une note SOAP de suivi.

Profil du patient : {gender} de {age} ans avec {condition}.

Formatez la réponse comme suit :
1. D'abord, fournissez une transcription de la conversation montrant exactement ce que le médecin et le patient se sont dit. Chaque ligne doit commencer par "Médecin:" ou "Patient:" pour indiquer clairement qui parle.
2. Ensuite, après la conversation, fournissez une note SOAP détaillée pour cette rencontre de suivi.
3. Enfin, incluez une section "Mots-clés / Étiquettes" qui catégorise les aspects clés de la rencontre.

La conversation doit être détaillée mais réaliste, couvrant:
- Salutation avec référence à la visite précédente (par exemple, "Bonjour, Madame. Comment vous sentez-vous depuis notre dernière rencontre?")
- Discussion sur l'état du patient depuis la dernière visite
- Examen de tout changement dans les symptômes
- Discussion sur l'adhérence au plan de traitement
- Tout effet secondaire des médicaments
- Résultats de l'examen physique
- Examen de nouveaux résultats de tests
- Évaluation de l'efficacité du traitement actuel
- Tout ajustement au plan de traitement
- Instructions de suivi

La note SOAP doit inclure:
- Subjectif: Symptômes rapportés par le patient, progrès, adhérence aux médicaments et tout effet secondaire
- Objectif: Mise à jour des résultats de l'examen cardiovasculaire, des signes vitaux et de nouveaux résultats de tests
- Évaluation: Mise à jour de l'évaluation cardiaque et évaluation des progrès
- Plan: Tout changement au plan de traitement, aux médicaments et recommandations de suivi

La section Mots-clés / Étiquettes doit inclure:
- Diagnostic: Diagnostics cardiaques mis à jour
- Traitement: Médicaments cardiaques actuels et nouveaux ou interventions
- Symptômes: Tout changement dans les symptômes depuis la dernière visite
- Conditions: Statut mis à jour des conditions cardiaques
- Allergies: Toute allergie aux médicaments

Soyez spécifique à la pratique cardiologique au Canada, en utilisant la terminologie médicale et les noms de médicaments canadiens."""
    },
    
    "gp": {
        "en": """Generate a realistic General Practitioner (GP) follow-up conversation between a doctor and a patient, along with a follow-up SOAP note.

Patient profile: {age} year old {gender} with {condition}.

Format the response as follows:
1. First, provide a conversation transcript showing exactly what the doctor and patient said to each other. Each line should start with either "Doctor:" or "Patient:" to clearly indicate the speaker.
2. Then, after the conversation, provide a detailed SOAP note for this follow-up encounter.
3. Finally, include a "Keywords / Tags" section that categorizes key aspects of the encounter.

The conversation should be detailed but realistic, covering:
- Greeting with reference to their previous visit (e.g., "Good morning, how have you been feeling since our last visit?")
- Discussion of how the patient has been doing since the last visit
- Review of any changes in symptoms
- Discussion of adherence to treatment plan
- Any side effects from medications
- Physical examination findings
- Review of any new test results
- Assessment of the effectiveness of the current treatment
- Any adjustments to the treatment plan
- Follow-up instructions

The SOAP note should include:
- Subjective: Patient's reported symptoms, progress, medication adherence, and any side effects
- Objective: Updated physical examination findings, vital signs, and any new test results
- Assessment: Updated medical assessment and progress evaluation
- Plan: Any changes to treatment plan, medications, and follow-up recommendations

The Keywords / Tags section should include:
- Diagnosis: Updated diagnoses
- Treatment: Current and any new medications or interventions
- Symptoms: Any changes in symptoms since the last visit
- Conditions: Updated status of medical conditions
- Allergies: Any medication allergies

Be specific to primary care practice in Canada, using Canadian medical terminology and medication names.""",
        
        "fr": """Générez une conversation de suivi réaliste de médecine générale entre un médecin et un patient, accompagnée d'une note SOAP de suivi.

Profil du patient : {gender} de {age} ans avec {condition}.

Formatez la réponse comme suit :
1. D'abord, fournissez une transcription de la conversation montrant exactement ce que le médecin et le patient se sont dit. Chaque ligne doit commencer par "Médecin:" ou "Patient:" pour indiquer clairement qui parle.
2. Ensuite, après la conversation, fournissez une note SOAP détaillée pour cette rencontre de suivi.
3. Enfin, incluez une section "Mots-clés / Étiquettes" qui catégorise les aspects clés de la rencontre.

La conversation doit être détaillée mais réaliste, couvrant:
- Salutation avec référence à la visite précédente (par exemple, "Bonjour, comment allez-vous depuis notre dernière visite?")
- Discussion sur l'état du patient depuis la dernière visite
- Examen de tout changement dans les symptômes
- Discussion sur l'adhérence au plan de traitement
- Tout effet secondaire des médicaments
- Résultats de l'examen physique
- Examen de nouveaux résultats de tests
- Évaluation de l'efficacité du traitement actuel
- Tout ajustement au plan de traitement
- Instructions de suivi

La note SOAP doit inclure:
- Subjectif: Symptômes rapportés par le patient, progrès, adhérence aux médicaments et tout effet secondaire
- Objectif: Mise à jour des résultats de l'examen physique, des signes vitaux et de nouveaux résultats de tests
- Évaluation: Mise à jour de l'évaluation médicale et évaluation des progrès
- Plan: Tout changement au plan de traitement, aux médicaments et recommandations de suivi

La section Mots-clés / Étiquettes doit inclure:
- Diagnostic: Diagnostics mis à jour
- Traitement: Médicaments actuels et nouveaux ou interventions
- Symptômes: Tout changement dans les symptômes depuis la dernière visite
- Conditions: Statut mis à jour des conditions médicales
- Allergies: Toute allergie aux médicaments

Soyez spécifique à la pratique de médecine générale au Canada, en utilisant la terminologie médicale et les noms de médicaments canadiens."""
    }
}

def setup_directories():
    """Create the necessary directories for data storage."""
    specialties = ["cardiology", "gp"]
    languages = ["en-CA", "fr-CA"]
    subdirs = ["json", "soap", "audio", "transcripts"]
    
    for specialty in specialties:
        for lang in languages:
            for subdir in subdirs:
                os.makedirs(os.path.join(BASE_DIR, specialty, lang, subdir), exist_ok=True)
    
    print("Directory structure created successfully.")

def generate_random_params(specialty, lang_code):
    """Generate random parameters for a medical case."""
    # Select a random condition for this specialty and language
    condition = random.choice(MEDICAL_CONDITIONS[specialty][lang_code])
    
    # Generate patient parameters
    patient_age = random.randint(30, 80)
    patient_gender = random.choice(["male", "female"])
    if lang_code == "fr":
        patient_gender = "homme" if patient_gender == "male" else "femme"
    
    # Generate treatment
    treatment = random.choice(TREATMENTS[specialty][lang_code])
    
    # Generate time since last visit (for follow-ups)
    weeks_ago = random.randint(4, 12)
    
    return {
        "condition": condition,
        "age": patient_age,
        "gender": patient_gender,
        "treatment": treatment,
        "weeks_ago": weeks_ago
    }

def generate_conversation_with_gpt(prompt, model="gpt-4", temperature=0.7, max_retries=3):
    """Generate conversation using GPT model with retry logic."""
    for attempt in range(max_retries):
        try:
            # response = openai.ChatCompletion.create(
            #     model=model,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=temperature,
            #     max_tokens=4000
            # )
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error on attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 * (attempt + 1))  # Exponential backoff
            
    return None

def process_french_conversations(conversations, raw_response):
    """Process French conversations to separate doctor/patient exchanges."""
    # Check for doctor/patient prefixes in conversation
    if "Médecin :" in raw_response or "Médecin:" in raw_response or "Docteur :" in raw_response or "Docteur:" in raw_response:
        # Extract the conversation exchanges
        try:
            # Try to split by Médecin/Patient prefixes
            doctor_patient_pairs = []
            
            # Clean up and standardize the prefixes
            processed_text = raw_response
            # Replace common prefixes with standardized format
            for prefix in ["Médecin ", "Médecin:", "Docteur ", "Docteur:"]:
                processed_text = processed_text.replace(prefix, "DOCTOR_START")
            
            for prefix in ["Patient ", "Patient:"]:
                processed_text = processed_text.replace(prefix, "PATIENT_START")
            
            # Split by standardized markers
            parts = processed_text.split("DOCTOR_START")
            for part in parts[1:]:  # Skip the first part which is empty
                if "PATIENT_START" in part:
                    doctor_text, patient_part = part.split("PATIENT_START", 1)
                    # Find the end of the patient text (next doctor or end of text)
                    if "DOCTOR_START" in patient_part:
                        patient_text = patient_part.split("DOCTOR_START")[0]
                    else:
                        patient_text = patient_part
                    
                    # Clean up the texts and remove any prefix residue
                    doctor_text = doctor_text.strip().replace(":", "").strip()
                    patient_text = patient_text.strip().replace(":", "").strip()
                    
                    if doctor_text and patient_text:
                        doctor_patient_pairs.append(("doctor", doctor_text))
                        doctor_patient_pairs.append(("patient", patient_text))
            
            # If we successfully parsed doctor/patient pairs, use them
            if doctor_patient_pairs:
                conversations = []
                for speaker, text in doctor_patient_pairs:
                    conversations.append({"speaker": speaker, "text": text})
                return conversations
            
        except Exception as e:
            print(f"Error parsing doctor/patient prefixes: {e}")
            print("Attempting fallback parsing...")
    
    # If we couldn't parse, return the original
    return conversations

def generate_soap_note_markdown(soap_note_text, filename):
    """Generate a SOAP note in markdown format."""
    # Ensure the soap note text has proper markdown formatting
    if "SOAP Note:" in soap_note_text:
        soap_note_text = soap_note_text.split("SOAP Note:", 1)[1].strip()
    
    # Make sure sections are properly formatted with markdown headers
    sections = ["Subjective:", "Objective:", "Assessment:", "Plan:"]
    for section in sections:
        if section in soap_note_text:
            soap_note_text = soap_note_text.replace(section, f"## {section}")
    
    # Add markdown for Keywords / Tags section
    if "Keywords / Tags:" in soap_note_text:
        soap_note_text = soap_note_text.replace("Keywords / Tags:", "## Keywords / Tags:")
    
    # Add title
    soap_note_markdown = f"# SOAP Note\n\n{soap_note_text}"
    
    # Save to file with .md extension
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(soap_note_markdown)
        print(f"Saved markdown SOAP note to {filename}")
        return True
    except Exception as e:
        print(f"Error saving markdown SOAP note: {e}")
        return False

def generate_soap_from_transcript(conversation_json, soap_dir, specialty, language, case_num, encounter_type):
    """Generate a SOAP note from the transcription."""
    # Create directories for original and transcribed SOAPs
    original_soap_dir = os.path.join(soap_dir, "original")
    transcribed_soap_dir = os.path.join(soap_dir, "transcribed")
    os.makedirs(original_soap_dir, exist_ok=True)
    os.makedirs(transcribed_soap_dir, exist_ok=True)
    
    # Extract the conversation text
    conversation_text = ""
    for item in conversation_json.get("conversation", []):
        speaker = item.get("speaker", "").lower()
        text = item.get("text", "")
        if speaker and text:
            # Format as Doctor: or Patient: for the prompt
            speaker_prefix = "Doctor:" if speaker == "doctor" else "Patient:"
            conversation_text += f"{speaker_prefix} {text}\n"
    
    # Prepare the SOAP note prompt
    if language == "en":
        soap_prompt = f"""Based on the following medical conversation transcript, generate a comprehensive SOAP note:

{conversation_text}

Format the SOAP note with the following sections:
- Subjective: Patient's reported symptoms, history, and concerns
- Objective: Physical examination findings, vital signs, and any test results
- Assessment: Medical assessment and differential diagnoses
- Plan: Treatment plan, prescriptions, referrals, and follow-up recommendations

Also include a "Keywords / Tags" section with:
- Diagnosis: List of diagnoses and conditions identified
- Treatment: Medications, procedures, and interventions
- Symptoms: Key symptoms reported by the patient
- Conditions: Underlying or related medical conditions
- Allergies: Any allergies mentioned

Use Canadian medical terminology and medication names."""
    else:  # French
        soap_prompt = f"""Basé sur la transcription de conversation médicale suivante, générez une note SOAP complète:

{conversation_text}

Formatez la note SOAP avec les sections suivantes:
- Subjectif: Symptômes rapportés par le patient, antécédents et préoccupations
- Objectif: Résultats de l'examen physique, signes vitaux et résultats des tests
- Évaluation: Évaluation médicale et diagnostics différentiels
- Plan: Plan de traitement, prescriptions, références et recommandations de suivi

Incluez également une section "Mots-clés / Étiquettes" avec:
- Diagnostic: Liste des diagnostics et conditions identifiés
- Traitement: Médicaments, procédures et interventions
- Symptômes: Principaux symptômes rapportés par le patient
- Conditions: Conditions médicales sous-jacentes ou connexes
- Allergies: Toutes allergies mentionnées

Utilisez la terminologie médicale et les noms de médicaments canadiens."""
    
    # Generate the SOAP note using OpenAI
    try:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a medical professional who creates detailed SOAP notes from conversation transcripts."},
                {"role": "user", "content": soap_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        soap_text = response.choices[0].message.content.strip()
        
        # Save the SOAP note in markdown format
        soap_filename = f"{specialty}_{case_num}_{encounter_type}"
        transcribed_soap_path = os.path.join(transcribed_soap_dir, f"{soap_filename}.md")
        generate_soap_note_markdown(soap_text, transcribed_soap_path)
        
        return True
    except Exception as e:
        print(f"Error generating SOAP from transcript: {e}")
        return False

def handle_consultation_response(response_text, specialty, language, case_num):
    """Parse and save consultation response."""
    # First, find the conversation part by looking for Doctor/Patient prefixes
    conversation_parts = []
    soap_part = ""
    
    try:
        # Try to extract conversation and SOAP parts
        if language == "en":
            # For English, look for Doctor: and Patient: prefixes
            in_conversation = True
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line is part of the conversation
                if line.startswith("Doctor:") or line.startswith("Patient:"):
                    in_conversation = True
                    conversation_parts.append(line)
                elif in_conversation and (line.startswith("SOAP") or line.startswith("1.") or line.startswith("2.")):
                    # End of conversation, start of SOAP
                    in_conversation = False
                    soap_part = line + "\n"
                elif not in_conversation:
                    # Continue adding to SOAP part
                    soap_part += line + "\n"
        else:  # French
            # For French, look for Médecin: and Patient: prefixes
            in_conversation = True
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line is part of the conversation
                if line.startswith("Médecin:") or line.startswith("Médecin :") or line.startswith("Patient:") or line.startswith("Patient :") or line.startswith("Docteur:") or line.startswith("Docteur :"):
                    in_conversation = True
                    conversation_parts.append(line)
                elif in_conversation and (line.startswith("Note SOAP") or line.startswith("1.") or line.startswith("2.")):
                    # End of conversation, start of SOAP
                    in_conversation = False
                    soap_part = line + "\n"
                elif not in_conversation:
                    # Continue adding to SOAP part
                    soap_part += line + "\n"
    
        # Process the conversation parts into JSON structure
        conversation = []
        current_speaker = None
        current_text = []
        
        for part in conversation_parts:
            if part.startswith("Doctor:") or part.startswith("Médecin:") or part.startswith("Médecin :") or part.startswith("Docteur:") or part.startswith("Docteur :"):
                # Save previous speaker's text
                if current_speaker and current_text:
                    conversation.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text).strip()
                    })
                
                # Start new doctor text
                current_speaker = "doctor"
                if ":" in part:
                    current_text = [part.split(":", 1)[1].strip()]
                else:
                    current_text = []
            elif part.startswith("Patient:") or part.startswith("Patient :"):
                # Save previous speaker's text
                if current_speaker and current_text:
                    conversation.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text).strip()
                    })
                
                # Start new patient text
                current_speaker = "patient"
                if ":" in part:
                    current_text = [part.split(":", 1)[1].strip()]
                else:
                    current_text = []
            else:
                # Continue with current speaker
                current_text.append(part)
        
        # Add the last speaker's text
        if current_speaker and current_text:
            conversation.append({
                "speaker": current_speaker,
                "text": " ".join(current_text).strip()
            })
        
        # For French conversations, apply additional processing
        if language == "fr":
            conversation = process_french_conversations(conversation, response_text)
        
        # Create the final JSON structure
        consultation_json = {
            "language": f"{language}-CA",
            "specialty": specialty,
            "type": "consultation",
            "condition": MEDICAL_CONDITIONS[specialty][language][random.randint(0, len(MEDICAL_CONDITIONS[specialty][language])-1)],
            "patient": {
                "age": random.randint(18, 85),
                "gender": random.choice(["homme", "femme"]) if language == "fr" else random.choice(["male", "female"])
            },
            "conversation": conversation
        }
        
        # Create the directory structure
        json_dir = os.path.join(BASE_DIR, specialty, f"{language}-CA", "json")
        soap_dir = os.path.join(BASE_DIR, specialty, f"{language}-CA", "soap")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(soap_dir, exist_ok=True)
        
        # Save the conversation to JSON
        json_path = os.path.join(json_dir, f"{specialty}_{case_num}_consultation.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(consultation_json, f, indent=2, ensure_ascii=False)
        
        # Save the SOAP note in markdown format
        soap_path = os.path.join(soap_dir, f"{specialty}_{case_num}_consultation.md")
        generate_soap_note_markdown(soap_part, soap_path)
        
        # Also save in original/transcribed subdirectories
        original_soap_dir = os.path.join(soap_dir, "original")
        os.makedirs(original_soap_dir, exist_ok=True)
        original_soap_path = os.path.join(original_soap_dir, f"{specialty}_{case_num}_consultation.md")
        generate_soap_note_markdown(soap_part, original_soap_path)
        
        # Generate SOAP from the transcription itself (for simulation)
        generate_soap_from_transcript(consultation_json, soap_dir, specialty, language, case_num, "consultation")
        
        return True
    except Exception as e:
        print(f"Error in handle_consultation_response: {e}")
        import traceback
        traceback.print_exc()
        return False

def handle_followup_response(response_text, specialty, language, case_num):
    """Parse and save follow-up response."""
    # First, find the conversation part by looking for Doctor/Patient prefixes
    conversation_parts = []
    soap_part = ""
    
    try:
        # Try to extract conversation and SOAP parts
        if language == "en":
            # For English, look for Doctor: and Patient: prefixes
            in_conversation = True
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line is part of the conversation
                if line.startswith("Doctor:") or line.startswith("Patient:"):
                    in_conversation = True
                    conversation_parts.append(line)
                elif in_conversation and (line.startswith("SOAP") or line.startswith("1.") or line.startswith("2.")):
                    # End of conversation, start of SOAP
                    in_conversation = False
                    soap_part = line + "\n"
                elif not in_conversation:
                    # Continue adding to SOAP part
                    soap_part += line + "\n"
        else:  # French
            # For French, look for Médecin: and Patient: prefixes
            in_conversation = True
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line is part of the conversation
                if line.startswith("Médecin:") or line.startswith("Médecin :") or line.startswith("Patient:") or line.startswith("Patient :") or line.startswith("Docteur:") or line.startswith("Docteur :"):
                    in_conversation = True
                    conversation_parts.append(line)
                elif in_conversation and (line.startswith("Note SOAP") or line.startswith("1.") or line.startswith("2.")):
                    # End of conversation, start of SOAP
                    in_conversation = False
                    soap_part = line + "\n"
                elif not in_conversation:
                    # Continue adding to SOAP part
                    soap_part += line + "\n"
    
        # Process the conversation parts into JSON structure
        conversation = []
        current_speaker = None
        current_text = []
        
        for part in conversation_parts:
            if part.startswith("Doctor:") or part.startswith("Médecin:") or part.startswith("Médecin :") or part.startswith("Docteur:") or part.startswith("Docteur :"):
                # Save previous speaker's text
                if current_speaker and current_text:
                    conversation.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text).strip()
                    })
                
                # Start new doctor text
                current_speaker = "doctor"
                if ":" in part:
                    current_text = [part.split(":", 1)[1].strip()]
                else:
                    current_text = []
            elif part.startswith("Patient:") or part.startswith("Patient :"):
                # Save previous speaker's text
                if current_speaker and current_text:
                    conversation.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text).strip()
                    })
                
                # Start new patient text
                current_speaker = "patient"
                if ":" in part:
                    current_text = [part.split(":", 1)[1].strip()]
                else:
                    current_text = []
            else:
                # Continue with current speaker
                current_text.append(part)
        
        # Add the last speaker's text
        if current_speaker and current_text:
            conversation.append({
                "speaker": current_speaker,
                "text": " ".join(current_text).strip()
            })
        
        # For French conversations, apply additional processing
        if language == "fr":
            conversation = process_french_conversations(conversation, response_text)
        
        # Create the final JSON structure
        followup_json = {
            "language": f"{language}-CA",
            "specialty": specialty,
            "type": "followup",
            "condition": MEDICAL_CONDITIONS[specialty][language][random.randint(0, len(MEDICAL_CONDITIONS[specialty][language])-1)],
            "patient": {
                "age": random.randint(18, 85),
                "gender": random.choice(["homme", "femme"]) if language == "fr" else random.choice(["male", "female"])
            },
            "conversation": conversation
        }
        
        # Create the directory structure
        json_dir = os.path.join(BASE_DIR, specialty, f"{language}-CA", "json")
        soap_dir = os.path.join(BASE_DIR, specialty, f"{language}-CA", "soap")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(soap_dir, exist_ok=True)
        
        # Save the conversation to JSON
        json_path = os.path.join(json_dir, f"{specialty}_{case_num}_followup.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(followup_json, f, indent=2, ensure_ascii=False)
        
        # Save the SOAP note in markdown format
        soap_path = os.path.join(soap_dir, f"{specialty}_{case_num}_followup.md")
        generate_soap_note_markdown(soap_part, soap_path)
        
        # Also save in original/transcribed subdirectories
        original_soap_dir = os.path.join(soap_dir, "original")
        os.makedirs(original_soap_dir, exist_ok=True)
        original_soap_path = os.path.join(original_soap_dir, f"{specialty}_{case_num}_followup.md")
        generate_soap_note_markdown(soap_part, original_soap_path)
        
        # Generate SOAP from the transcription itself (for simulation)
        generate_soap_from_transcript(followup_json, soap_dir, specialty, language, case_num, "followup")
        
        return True
    except Exception as e:
        print(f"Error in handle_followup_response: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_consultation(specialty, language, case_id):
    """Generate a consultation conversation for a given specialty and language."""
    lang_code = language.split('-')[0]  # en or fr
    params = generate_random_params(specialty, lang_code)
    
    # Choose template based on specialty
    template = CONSULT_PROMPTS[specialty][lang_code]
    
    # Generate the consultation
    response_text = generate_conversation_with_gpt(template, params)
    if not response_text:
        return None, None, None
    
    structured_conversation, soap_note = process_gpt_response(response_text, lang_code)
    
    if not structured_conversation:
        print(f"Failed to process {language} consultation")
        return None, None, None
    
    # Create a proper conversation structure with metadata
    conversation_data = {
        "metadata": {
            "case_id": case_id,
            "conversation_id": f"{case_id}-consult",
            "language": language,
            "specialty": specialty,
            "encounter_type": "consultation",
            "condition": params["condition"],
            "patient_age": params["age"],
            "patient_gender": params["gender"],
            "timestamp": datetime.now().isoformat()
        },
        "conversation": structured_conversation
    }
    
    return conversation_data, soap_note, params

def generate_follow_up_with_gpt(consultation_text, soap_note, follow_up_prompt, params, model="gpt-4"):
    """Generate a follow-up conversation based on the initial consultation."""
    
    # Extract only consultation part for context
    conversation_only = consultation_text
    
    # Format the prompt
    formatted_prompt = follow_up_prompt.format(**params)
    
    # Assemble the prompt for the GPT model
    prompt = f"""
I need to generate a realistic medical follow-up visit based on this previous consultation.

Previous consultation:
{conversation_only}

Please generate the follow-up visit using this format:
{formatted_prompt}
"""

    try:
        # response = openai.ChatCompletion.create(
        #     model=model,
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a medical professional who creates realistic doctor-patient conversations for training purposes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating follow-up with GPT: {str(e)}")
        time.sleep(5)  # Backoff if rate limited
        return None

def generate_medical_case(specialty, lang_code, case_id, model="gpt-4"):
    """Generate a medical case with conversation and SOAP note.
    
    Args:
        specialty: Medical specialty (e.g., 'cardiology', 'gp')
        lang_code: Language code ('en' or 'fr')
        case_id: Case identifier (integer)
        model: GPT model to use
        
    Returns:
        True if successful, False otherwise
    """
    # Set up paths for files
    lang_suffix = f"{lang_code}-CA"
    case_id_str = f"{specialty}_{case_id}"
    
    # Set up file paths
    consultation_json = os.path.join(BASE_DIR, specialty, lang_suffix, "json", f"{case_id_str}_consultation.json")
    consultation_soap = os.path.join(BASE_DIR, specialty, lang_suffix, "soap", f"{case_id_str}_consultation.md")
    followup_json = os.path.join(BASE_DIR, specialty, lang_suffix, "json", f"{case_id_str}_followup.json")
    followup_soap = os.path.join(BASE_DIR, specialty, lang_suffix, "soap", f"{case_id_str}_followup.md")
    
    # Check if files already exist
    if os.path.exists(consultation_json) and os.path.exists(followup_json):
        print(f"Case {case_id_str} already exists, skipping")
        return True
    
    # Generate random patient data
    patient_age = random.randint(25, 85)
    patient_gender = random.choice(["homme", "femme"]) if lang_code == "fr" else random.choice(["male", "female"])
    condition = random.choice(MEDICAL_CONDITIONS[specialty][lang_code])
    treatment = random.choice(TREATMENTS[specialty][lang_code])
    weeks_ago = random.randint(4, 12)
    
    # Parameters for the prompts
    params = {
        "condition": condition,
        "age": patient_age,
        "gender": patient_gender,
        "treatment": treatment,
        "weeks_ago": weeks_ago
    }
    
    # Generate consultation
    try:
        print(f"Generating {lang_suffix} {specialty} case {case_id_str}...")
        
        # Get consultation prompt
        consultation_prompt = CONSULT_PROMPTS[specialty][lang_code]
        
        # Generate initial consultation
        consultation_content = generate_conversation_with_gpt(
            consultation_prompt.format(**params),
            model=model,
            temperature=0.7
        )
        
        if not consultation_content:
            print(f"Failed to generate consultation for case {case_id_str}")
            return False
        
        # Handle the consultation response
        success = handle_consultation_response(consultation_content, specialty, lang_code, case_id)
        
        if not success:
            print(f"Failed to handle consultation content for case {case_id_str}")
            return False
        
        # Get follow-up prompt
        followup_prompt = FOLLOWUP_PROMPTS[specialty][lang_code]
        
        # Generate follow-up
        followup_content = generate_follow_up_with_gpt(
            consultation_content,
            "",  # We no longer need the SOAP note as input
            followup_prompt,
            params,
            model=model
        )
        
        if not followup_content:
            print(f"Failed to generate follow-up for case {case_id_str}")
            return False
        
        # Handle the follow-up response
        success = handle_followup_response(followup_content, specialty, lang_code, case_id)
        
        if not success:
            print(f"Failed to handle follow-up content for case {case_id_str}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error generating case {case_id_str}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_cases(specialty, num_cases, model="gpt-4"):
    """Generate a specified number of medical cases for a given specialty."""
    # Create directory structure for this specialty
    setup_directory_for_specialty(specialty)
    
    # Generate English cases
    for i in tqdm(range(1, num_cases + 1), desc=f"English {specialty} cases"):
        generate_medical_case(specialty, "en", i, model=model)
        time.sleep(1)  # To avoid rate limiting
    
    # Generate French cases
    for i in tqdm(range(1, num_cases + 1), desc=f"French {specialty} cases"):
        generate_medical_case(specialty, "fr", i, model=model)
        time.sleep(1)  # To avoid rate limiting

def setup_directory_for_specialty(specialty):
    """Create directory structure for a specific specialty."""
    # Create directories for English
    en_dir = os.path.join(BASE_DIR, specialty, "en-CA")
    en_json_dir = os.path.join(en_dir, "json")
    en_soap_dir = os.path.join(en_dir, "soap") 
    en_audio_dir = os.path.join(en_dir, "audio")
    en_transcripts_dir = os.path.join(en_dir, "transcripts")
    
    # Create directories for French
    fr_dir = os.path.join(BASE_DIR, specialty, "fr-CA")
    fr_json_dir = os.path.join(fr_dir, "json")
    fr_soap_dir = os.path.join(fr_dir, "soap")
    fr_audio_dir = os.path.join(fr_dir, "audio")
    fr_transcripts_dir = os.path.join(fr_dir, "transcripts")
    
    # Create all directories
    for directory in [
        en_json_dir, en_soap_dir, en_audio_dir, en_transcripts_dir,
        fr_json_dir, fr_soap_dir, fr_audio_dir, fr_transcripts_dir
    ]:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate medical conversations')
    parser.add_argument('--num', type=int, default=5, help='Number of cases to generate per specialty and language')
    parser.add_argument('--specialty', type=str, choices=['cardiology', 'gp', 'all'], default='all', help='Medical specialty to generate')
    parser.add_argument('--model', type=str, default='gpt-4', help='GPT model to use for generation')
    parser.add_argument('--lang', type=str, choices=['en', 'fr', 'all'], default='all', help='Language to generate')
    args = parser.parse_args()
    
    # Create directory structure
    setup_directories()
    
    # Determine which specialties to process
    specialties_to_process = SPECIALTIES if args.specialty == "all" else [args.specialty]
    
    # Determine which languages to process
    languages_to_process = ["en", "fr"] if args.lang == "all" else [args.lang]
    
    # Generate conversations for each combination
    for specialty in specialties_to_process:
        for lang in languages_to_process:
            print(f"\n=== Generating {args.num} {specialty} cases in {lang} ===")
            
            # Create directory structure for this specialty and language
            lang_suffix = "en-CA" if lang == "en" else "fr-CA"
            specialty_dir = os.path.join(BASE_DIR, specialty, lang_suffix)
            
            # Set up directory structure
            for subdir in ["json", "soap", "audio", "transcripts"]:
                os.makedirs(os.path.join(specialty_dir, subdir), exist_ok=True)
            
            # Generate the cases
            for i in tqdm(range(1, args.num + 1), desc=f"{lang.upper()} {specialty} cases"):
                generate_medical_case(specialty, lang, i, model=args.model)
                time.sleep(1)  # Avoid rate limiting
    
    print("\n=== Medical Conversation Generation Complete ===")
    print(f"Generated {args.num} cases per specialty and language")
    print("Files are organized in the following structure:")
    print("- data-med/[specialty]/[language]/json: JSON conversations")
    print("- data-med/[specialty]/[language]/soap: SOAP notes in markdown")
    print("- data-med/[specialty]/[language]/audio: Audio files (to be generated)")
    print("- data-med/[specialty]/[language]/transcripts: Transcriptions (to be generated)")

if __name__ == "__main__":
    main() 