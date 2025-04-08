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
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Warning: OPENAI_API_KEY not found in environment. Set it in your .env file.")

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
- Introduction and rapport building
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
- Introduction et établissement de la relation
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
- Introduction and rapport building
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
- Introduction et établissement de la relation
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
- Allergies: Toute allergie mentionnée

Soyez spécifique à la pratique des soins primaires au Canada, en utilisant la terminologie médicale et les noms de médicaments canadiens."""
    }
}

# SOAP note prompt templates for follow-up visits
FOLLOWUP_PROMPTS = {
    "cardiology": {
        "en": """Generate a realistic cardiology follow-up visit between a doctor and a patient, along with a SOAP note.

Patient profile: {age} year old {gender} with {condition}, initially seen {weeks_ago} weeks ago. Currently on {treatment}.

The previous visit was a consultation for newly diagnosed {condition}.

Format the response as follows:
1. First, provide a conversation transcript showing exactly what the doctor and patient said to each other. Each line should start with either "Doctor:" or "Patient:" to clearly indicate the speaker.
2. Then, after the conversation, provide a detailed SOAP note for this follow-up encounter.
3. Finally, include a "Keywords / Tags" section that categorizes key aspects of the encounter.

The conversation should be detailed but realistic, covering:
- Welcome and brief review of the previous visit
- Assessment of how patient is feeling since starting treatment
- Review of any symptoms improvement or persistence
- Discussion of medication adherence and side effects
- Review of any new test results (if applicable)
- Adjustments to treatment plan if needed
- Reinforcement of lifestyle modifications
- Planning for next follow-up

The SOAP note should include:
- Subjective: Patient's reported response to treatment, symptoms, and concerns
- Objective: Updated cardiovascular examination, vital signs, and any new test results
- Assessment: Progress of the cardiac condition and response to treatment
- Plan: Adjustments to treatment, further testing if needed, and next follow-up

The Keywords / Tags section should include:
- Diagnosis: Updated cardiac diagnoses
- Treatment: Current and adjusted medications
- Symptoms: Current cardiac symptoms and changes
- Conditions: Primary and related cardiac conditions
- Allergies: Any medication allergies or intolerances noted

Be specific to cardiology follow-up practice in Canada, using Canadian medical terminology and medication names.""",
        
        "fr": """Générez une visite de suivi cardiologique réaliste entre un médecin et un patient, accompagnée d'une note SOAP.

Profil du patient : {gender} de {age} ans avec {condition}, vu initialement il y a {weeks_ago} semaines. Actuellement sous {treatment}.

La visite précédente était une consultation pour {condition} nouvellement diagnostiquée.

Formatez la réponse comme suit :
1. D'abord, fournissez une transcription de la conversation montrant exactement ce que le médecin et le patient se sont dit. Chaque ligne doit commencer par "Médecin:" ou "Patient:" pour indiquer clairement qui parle.
2. Ensuite, après la conversation, fournissez une note SOAP détaillée pour cette rencontre de suivi.
3. Enfin, incluez une section "Mots-clés / Étiquettes" qui catégorise les aspects clés de la rencontre.

La conversation doit être détaillée mais réaliste, couvrant:
- Accueil et bref examen de la visite précédente
- Évaluation de comment le patient se sent depuis le début du traitement
- Examen de l'amélioration ou de la persistance des symptômes
- Discussion sur l'adhérence aux médicaments et les effets secondaires
- Examen des nouveaux résultats de tests (le cas échéant)
- Ajustements au plan de traitement si nécessaire
- Renforcement des modifications du mode de vie
- Planification du prochain suivi

La note SOAP doit inclure:
- Subjectif: Réponse rapportée du patient au traitement, symptômes et préoccupations
- Objectif: Examen cardiovasculaire mis à jour, signes vitaux et nouveaux résultats de tests
- Évaluation: Progression de la condition cardiaque et réponse au traitement
- Plan: Ajustements du traitement, tests supplémentaires si nécessaire, et prochain suivi

La section Mots-clés / Étiquettes doit inclure:
- Diagnostic: Diagnostics cardiaques mis à jour
- Traitement: Médicaments actuels et ajustés
- Symptômes: Symptômes cardiaques actuels et changements
- Conditions: Conditions cardiaques primaires et connexes
- Allergies: Toute allergie ou intolérance aux médicaments notée

Soyez spécifique à la pratique de suivi cardiologique au Canada, en utilisant la terminologie médicale et les noms de médicaments canadiens."""
    },
    
    "gp": {
        "en": """Generate a realistic General Practitioner (GP) follow-up visit between a doctor and a patient, along with a SOAP note.

Patient profile: {age} year old {gender} with {condition}, initially seen {weeks_ago} weeks ago. Currently on {treatment}.

The previous visit was a consultation for newly diagnosed {condition}.

Format the response as follows:
1. First, provide a conversation transcript showing exactly what the doctor and patient said to each other. Each line should start with either "Doctor:" or "Patient:" to clearly indicate the speaker.
2. Then, after the conversation, provide a detailed SOAP note for this follow-up encounter.
3. Finally, include a "Keywords / Tags" section that categorizes key aspects of the encounter.

The conversation should be detailed but realistic, covering:
- Welcome and brief review of the previous visit
- Assessment of response to treatment
- Review of any symptom changes
- Discussion of medication adherence and side effects
- Any new concerns since the last visit
- Adjustments to treatment plan if needed
- Health maintenance and preventive care updates
- Planning for next follow-up

The SOAP note should include:
- Subjective: Patient's reported response to treatment, symptoms, and concerns
- Objective: Updated physical examination findings, vital signs, and any new test results
- Assessment: Progress of the condition and response to treatment
- Plan: Adjustments to treatment, preventive care, and next follow-up

The Keywords / Tags section should include:
- Diagnosis: Updated diagnoses
- Treatment: Current and adjusted medications
- Symptoms: Current symptoms and changes
- Conditions: Primary and related conditions
- Allergies: Any medication allergies or intolerances noted

Be specific to primary care follow-up practice in Canada, using Canadian medical terminology and medication names.""",
        
        "fr": """Générez une visite de suivi de médecine générale réaliste entre un médecin et un patient, accompagnée d'une note SOAP.

Profil du patient : {gender} de {age} ans avec {condition}, vu initialement il y a {weeks_ago} semaines. Actuellement sous {treatment}.

La visite précédente était une consultation pour {condition} nouvellement diagnostiquée.

Formatez la réponse comme suit :
1. D'abord, fournissez une transcription de la conversation montrant exactement ce que le médecin et le patient se sont dit. Chaque ligne doit commencer par "Médecin:" ou "Patient:" pour indiquer clairement qui parle.
2. Ensuite, après la conversation, fournissez une note SOAP détaillée pour cette rencontre de suivi.
3. Enfin, incluez une section "Mots-clés / Étiquettes" qui catégorise les aspects clés de la rencontre.

La conversation doit être détaillée mais réaliste, couvrant:
- Accueil et bref examen de la visite précédente
- Évaluation de la réponse au traitement
- Examen des changements de symptômes
- Discussion sur l'adhérence aux médicaments et les effets secondaires
- Nouvelles préoccupations depuis la dernière visite
- Ajustements au plan de traitement si nécessaire
- Mises à jour sur le maintien de la santé et les soins préventifs
- Planification du prochain suivi

La note SOAP doit inclure:
- Subjectif: Réponse rapportée du patient au traitement, symptômes et préoccupations
- Objectif: Résultats mis à jour de l'examen physique, signes vitaux et nouveaux résultats de tests
- Évaluation: Progression de la condition et réponse au traitement
- Plan: Ajustements du traitement, soins préventifs et prochain suivi

La section Mots-clés / Étiquettes doit inclure:
- Diagnostic: Diagnostics mis à jour
- Traitement: Médicaments actuels et ajustés
- Symptômes: Symptômes actuels et changements
- Conditions: Conditions primaires et connexes
- Allergies: Toute allergie ou intolérance aux médicaments notée

Soyez spécifique à la pratique de suivi des soins primaires au Canada, en utilisant la terminologie médicale et les noms de médicaments canadiens."""
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
            response = openai.ChatCompletion.create(
                model=model,
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

def process_gpt_response(response_text, lang_code):
    """Process the GPT response to extract conversation and SOAP note."""
    if not response_text:
        print("No response text received from GPT")
        return None, None
    
    # Convert lang_code to language flag
    is_english = (lang_code == "en")
    
    # Split the response into lines
    lines = response_text.split('\n')
    
    # Define language-specific patterns
    if is_english:
        soap_indicators = ["SOAP", "Subjective:", "Objective:", "Assessment:", "Plan:", "SUBJECTIVE", "OBJECTIVE", "S:", "O:", "A:", "P:", "**Subjective", "**Objective", "**Assessment", "**Plan", "# SOAP", "## Subjective"]
        keywords_indicators = ["Keywords", "Tags", "Keywords/Tags", "**Keywords", "# Keywords"]
        doctor_prefixes = ["Doctor:", "Dr:", "Dr. ", "MD:"]
        patient_prefixes = ["Patient:"]
    else:  # French
        soap_indicators = ["Note SOAP", "Subjectif:", "Objectif:", "Évaluation:", "Plan:", "SUBJECTIF", "OBJECTIF", "S:", "O:", "A:", "P:", "**Subjectif", "**Objectif", "**Évaluation", "**Plan", "# Note SOAP", "## Subjectif"]
        keywords_indicators = ["Mots-clés", "Étiquettes", "Mots-clés / Étiquettes", "**Mots-clés", "# Mots-clés"]
        doctor_prefixes = ["Médecin:", "Dr:", "Dr. ", "Docteur:", "MD:"]
        patient_prefixes = ["Patient:", "Patiente:"]

    # First, try to detect if there's a conversation part
    conversation_detected = False
    for line in lines:
        if any(line.startswith(prefix) for prefix in doctor_prefixes + patient_prefixes):
            conversation_detected = True
            break
    
    if not conversation_detected:
        print("No conversation with Doctor/Patient prefixes detected in the response")
        
        # Try to find 'conversation' section in structured response
        conversation_section = None
        in_conversation = False
        conversation_lines = []
        
        for i, line in enumerate(lines):
            if "conversation" in line.lower() or "dialogue" in line.lower():
                in_conversation = True
                continue
            
            if in_conversation:
                # Check if we've hit another section
                if any(indicator in line for indicator in soap_indicators + keywords_indicators):
                    in_conversation = False
                    break
                conversation_lines.append(line)
        
        if conversation_lines:
            # Try to parse the conversation without explicit prefixes
            structured_conversation = []
            current_speaker = "doctor"  # Assume doctor starts
            current_text = []
            
            for line in conversation_lines:
                if not line.strip():
                    if current_text:
                        structured_conversation.append({
                            "speaker": current_speaker,
                            "text": " ".join(current_text)
                        })
                        current_speaker = "patient" if current_speaker == "doctor" else "doctor"
                        current_text = []
                else:
                    current_text.append(line.strip())
            
            # Add the last utterance
            if current_text:
                structured_conversation.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            
            return structured_conversation, "\n".join([line for line in lines if not line in conversation_lines])
        
        return None, None
    
    # Extract the conversation part (before SOAP note)
    conversation_lines = []
    soap_note_lines = []
    keywords_lines = []
    
    in_conversation = True
    in_soap = False
    in_keywords = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if we've reached the Keywords/Tags section
        keyword_line = False
        for indicator in keywords_indicators:
            if indicator in line and not any(line.startswith(prefix) for prefix in doctor_prefixes + patient_prefixes):
                keyword_line = True
                in_conversation = False
                in_soap = False
                in_keywords = True
                break
                
        # Check if we've reached the SOAP note section
        soap_line = False
        if not in_keywords:
            for indicator in soap_indicators:
                if indicator in line and not any(line.startswith(prefix) for prefix in doctor_prefixes + patient_prefixes):
                    soap_line = True
                    in_conversation = False
                    in_soap = True
                    break
                
        if in_conversation:
            conversation_lines.append(line)
        elif in_soap and not in_keywords:
            soap_note_lines.append(line)
        elif in_keywords:
            keywords_lines.append(line)
    
    # Process the conversation lines to structure them properly
    structured_conversation = []
    
    current_speaker = None
    current_text = []
    
    # Extract speaker turns with explicit Doctor:/Patient: prefixes
    for line in conversation_lines:
        is_doctor_line = any(line.startswith(prefix) for prefix in doctor_prefixes)
        is_patient_line = any(line.startswith(prefix) for prefix in patient_prefixes)
        
        if is_doctor_line:
            # Save previous utterance if exists
            if current_speaker and current_text:
                structured_conversation.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            
            # Start new doctor utterance
            current_speaker = "doctor"
            # Remove the prefix
            for prefix in doctor_prefixes:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            current_text = [line]
            
        elif is_patient_line:
            # Save previous utterance if exists
            if current_speaker and current_text:
                structured_conversation.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text)
                })
            
            # Start new patient utterance
            current_speaker = "patient"
            # Remove the prefix
            for prefix in patient_prefixes:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            current_text = [line]
            
        elif current_speaker:
            # Continue previous utterance
            current_text.append(line)
    
    # Add the last utterance
    if current_speaker and current_text:
        structured_conversation.append({
            "speaker": current_speaker,
            "text": " ".join(current_text)
        })
    
    # Create the SOAP note
    soap_note = "\n".join(soap_note_lines)
    
    # Add keywords if available
    if keywords_lines:
        soap_note += "\n\n" + "\n".join(keywords_lines)
    
    # Verify the conversation has enough turns (at least 4 turns)
    if len(structured_conversation) < 4:
        print(f"Warning: Conversation has only {len(structured_conversation)} turns, which is less than expected.")
    
    return structured_conversation, soap_note

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

def generate_follow_up_with_gpt(initial_conversation, soap_note, prompt_template, params, model="gpt-4", temperature=0.7, max_retries=3):
    """Generate a follow-up medical conversation using GPT."""
    # Format the prompt template with the params
    prompt = prompt_template.format(**params)
    
    # Prepare the full context with initial conversation and SOAP note
    full_prompt = f"""Initial visit conversation and SOAP note:
    
{initial_conversation}

{soap_note}

Now, generate a follow-up visit based on the above:
{prompt}"""
    
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
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

def generate_medical_case(specialty, lang_code, case_id, model="gpt-4"):
    """Generate a medical case with initial consultation and follow-up."""
    # Format case_id with specialty
    case_id_str = f"{specialty}_{case_id}"
    
    # Get language name
    lang_name = "English" if lang_code == "en" else "French"
    lang_suffix = "en-CA" if lang_code == "en" else "fr-CA"
    
    # Set directory paths
    specialty_dir = os.path.join(BASE_DIR, specialty, lang_suffix)
    json_dir = os.path.join(specialty_dir, "json")
    soap_dir = os.path.join(specialty_dir, "soap")
    
    # Ensure directories exist
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(soap_dir, exist_ok=True)
    
    # Set file paths
    consultation_json = os.path.join(json_dir, f"{case_id_str}_consultation.json")
    followup_json = os.path.join(json_dir, f"{case_id_str}_followup.json")
    consultation_soap = os.path.join(soap_dir, f"{case_id_str}_consultation.md")
    followup_soap = os.path.join(soap_dir, f"{case_id_str}_followup.md")
    
    # Skip if files already exist
    if os.path.exists(consultation_json) and os.path.exists(followup_json):
        print(f"Files for {lang_name} {specialty} case {case_id_str} already exist. Skipping.")
        return True
    
    # Select a random condition for this specialty and language
    condition = random.choice(MEDICAL_CONDITIONS[specialty][lang_code])
    
    # Generate patient parameters
    patient_age = random.randint(30, 80)
    patient_gender = random.choice(["male", "female"])
    if lang_code == "fr":
        patient_gender = "homme" if patient_gender == "male" else "femme"
    
    # Select a random treatment
    treatment = random.choice(TREATMENTS[specialty][lang_code])
    
    # Generate time parameter for follow-up (weeks since last visit)
    weeks_ago = random.randint(4, 12)
    
    # Generate special circumstances
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
        
        # Parse consultation content
        consultation_data, soap_note = process_gpt_response(consultation_content, lang_code)
        
        if not consultation_data:
            print(f"Failed to parse consultation content for case {case_id_str}")
            return False
        
        # Save consultation as JSON
        consultation_json_data = {
            "language": lang_suffix,
            "specialty": specialty,
            "type": "consultation",
            "condition": condition,
            "patient": {
                "age": patient_age,
                "gender": patient_gender
            },
            "conversation": consultation_data
        }
        
        with open(consultation_json, 'w', encoding='utf-8') as f:
            json.dump(consultation_json_data, f, indent=2, ensure_ascii=False)
        
        # Save consultation SOAP note - check if it's not too long
        if soap_note and isinstance(soap_note, str):
            try:
                with open(consultation_soap, 'w', encoding='utf-8') as f:
                    f.write(soap_note)
            except OSError as e:
                print(f"Warning: Could not save SOAP note to {consultation_soap}. Error: {str(e)}")
                # Save a simplified version
                try:
                    with open(consultation_soap, 'w', encoding='utf-8') as f:
                        f.write(f"SOAP Note for {case_id_str} consultation\n\nThis note was too long to save directly.")
                except Exception as e2:
                    print(f"Error saving simplified SOAP note: {str(e2)}")
        
        # Get follow-up prompt
        followup_prompt = FOLLOWUP_PROMPTS[specialty][lang_code]
        
        # Generate follow-up
        followup_content = generate_follow_up_with_gpt(
            consultation_content,
            soap_note,
            followup_prompt,
            params,
            model=model
        )
        
        if not followup_content:
            print(f"Failed to generate follow-up for case {case_id_str}")
            return False
        
        # Parse follow-up content
        followup_data, followup_soap = process_gpt_response(followup_content, lang_code)
        
        if not followup_data:
            print(f"Failed to parse follow-up content for case {case_id_str}")
            return False
        
        # Save follow-up as JSON
        followup_json_data = {
            "language": lang_suffix,
            "specialty": specialty,
            "type": "followup",
            "condition": condition,
            "patient": {
                "age": patient_age,
                "gender": patient_gender
            },
            "conversation": followup_data
        }
        
        with open(followup_json, 'w', encoding='utf-8') as f:
            json.dump(followup_json_data, f, indent=2, ensure_ascii=False)
        
        # Save follow-up SOAP note - check if it's not too long
        if followup_soap and isinstance(followup_soap, str):
            try:
                with open(followup_soap, 'w', encoding='utf-8') as f:
                    f.write(followup_soap)
            except OSError as e:
                print(f"Warning: Could not save SOAP note to {followup_soap}. Error: {str(e)}")
                # Save a simplified version
                try:
                    with open(followup_soap, 'w', encoding='utf-8') as f:
                        f.write(f"SOAP Note for {case_id_str} follow-up\n\nThis note was too long to save directly.")
                except Exception as e2:
                    print(f"Error saving simplified SOAP note: {str(e2)}")
        
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