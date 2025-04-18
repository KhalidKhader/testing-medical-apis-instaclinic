#!/usr/bin/env python3

import os
import json
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics import edit_distance
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jiwer
import spacy
import warnings
from tqdm import tqdm
import argparse
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from string import punctuation

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy models
print("Loading NLP models...")
try:
    nlp_en = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")

try:
    nlp_fr = spacy.load("fr_core_news_sm")
except:
    os.system("python -m spacy download fr_core_news_sm")
    nlp_fr = spacy.load("fr_core_news_sm")

# Medical terminology datasets
medical_terms_en = set([
    # Cardiovascular terms
    "angina", "arrhythmia", "arterial", "atrial", "aortic", "aneurysm", "ablation", 
    "angioplasty", "arterial fibrillation", "atrial fibrillation", "aortic stenosis",
    "bradycardia", "bypass", "blood pressure", "cardiac", "cardiovascular", "coronary", 
    "cardiomyopathy", "catheterization", "congestive heart failure", "cholesterol",
    "diastolic", "dyslipidemia", "echocardiogram", "endocardium", "embolism", 
    "electrocardiogram", "ECG", "EKG", "egg", "fibrillation", "heartbeat", "heart rate", 
    "hyperlipidemia", "hypertension", "hypotension", "ischemia", "infarction", 
    "mitral valve", "myocardial", "myocardium", "pacemaker", "palpitation", 
    "percutaneous", "perfusion", "pericardium", "plaque", "pulmonary", "regurgitation", 
    "revascularization", "stent", "stenosis", "syncope", "systolic", "tachycardia", 
    "thrombosis", "vascular", "vasoconstriction", "vasodilation", "ventricular", 
    "artery", "vein", "ventricle", "valve", "warfarin", "clopidogrel", "aspirin",
    "beta blocker", "ace inhibitor", "statin", "diuretic", "digoxin", "nitrate",
    "anticoagulant", "antiarrhythmic", "atherosclerosis", "ventricular tachycardia",
    "catheter ablation", "stress test", "cardioversion", "holter monitor", "atrial flutter",
    "heart failure", "coronary artery disease", "palpitations", "rapid heartbeats",
    "lightheaded", "short of breath", "dizziness", "fluid retention", "heart sounds",
    "pulses", "catheter",
    
    # General practitioner terms
    "diagnosis", "prognosis", "symptom", "chronic", "acute", "condition", "disorder",
    "disease", "infection", "inflammation", "lesion", "malignant", "benign", "pain",
    "prescription", "medication", "dose", "antibiotic", "vaccine", "immunity", 
    "allergy", "asthma", "diabetes", "hypertension", "hypercholesterolemia", 
    "arthritis", "osteoporosis", "depression", "anxiety", "insomnia", "fatigue",
    "referral", "specialist", "examination", "test", "scan", "blood test", "urine test",
    "x-ray", "ultrasound", "ct scan", "mri", "biopsy", "screening", "prevention",
    "treatment", "therapy", "surgery", "procedure", "follow-up", "complication",
    "side effect", "contraindication", "obesity", "overweight", "hypertension", 
    "hypotension", "fever", "headache", "dizziness", "nausea", "vomiting",
    "cough", "shortness of breath", "rash", "mental health", "physical exam",
    "physical examination", "beta-blockers", "lightheadedness", "chest pain", "fainting",
    "swelling", "ankles", "feet", "caffeine", "alcohol", "lifestyle", "balanced diet",
    "regular exercise"
])

medical_terms_fr = set([
    # Cardiovascular terms in French
    "angine", "arythmie", "artériel", "auriculaire", "aortique", "anévrisme", "ablation",
    "angioplastie", "fibrillation auriculaire", "sténose aortique", "bradycardie",
    "pontage", "tension artérielle", "cardiaque", "cardiovasculaire", "coronaire", 
    "cardiomyopathie", "cathétérisation", "insuffisance cardiaque", "cholestérol",
    "diastolique", "dyslipidémie", "échocardiogramme", "endocarde", "embolie", 
    "électrocardiogramme", "ECG", "fibrillation", "battement de cœur", "fréquence cardiaque", 
    "hyperlipidémie", "hypertension", "hypotension", "ischémie", "infarctus", 
    "valve mitrale", "myocardique", "myocarde", "stimulateur cardiaque", "palpitation", 
    "percutané", "perfusion", "péricarde", "plaque", "pulmonaire", "régurgitation", 
    "revascularisation", "stent", "sténose", "syncope", "systolique", "tachycardie", 
    "thrombose", "vasculaire", "vasoconstriction", "vasodilatation", "ventriculaire", 
    "artère", "veine", "ventricule", "valve", "warfarine", "clopidogrel", "aspirine",
    "bêta-bloquant", "inhibiteur de l'ECA", "statine", "diurétique", "digoxine", "nitrate",
    "anticoagulant", "antiarythmique", "athérosclérose", "tachycardie ventriculaire",
    "ablation par cathéter", "test d'effort", "cardioversion", "moniteur holter", "flutter auriculaire",
    "insuffisance cardiaque", "maladie coronarienne", "palpitations", "rythme cardiaque rapide",
    "étourdissements", "essoufflement", "vertiges", "rétention d'eau", "bruits cardiaques",
    "pouls", "cathéter", "battements rapides", "souffle court",
    
    # General practitioner terms in French
    "diagnostic", "pronostic", "symptôme", "chronique", "aigu", "condition", "trouble",
    "maladie", "infection", "inflammation", "lésion", "malin", "bénin", "douleur",
    "ordonnance", "médicament", "dose", "antibiotique", "vaccin", "immunité", 
    "allergie", "asthme", "diabète", "hypertension", "hypercholestérolémie", 
    "arthrite", "ostéoporose", "dépression", "anxiété", "insomnie", "fatigue",
    "référence", "spécialiste", "examen", "test", "scan", "analyse de sang", "analyse d'urine",
    "radiographie", "échographie", "tomodensitométrie", "irm", "biopsie", "dépistage", "prévention",
    "traitement", "thérapie", "chirurgie", "procédure", "suivi", "complication",
    "effet secondaire", "contre-indication", "obésité", "surpoids", "hypertension", 
    "hypotension", "fièvre", "mal de tête", "vertige", "nausée", "vomissement",
    "toux", "essoufflement", "éruption cutanée", "santé mentale", "examen physique",
    "bêta-bloquants", "étourdissement", "douleur thoracique", "évanouissement",
    "gonflement", "chevilles", "pieds", "caféine", "alcool", "style de vie", "alimentation équilibrée",
    "exercice régulier"
])

def setup_argparse():
    parser = argparse.ArgumentParser(description='Evaluate medical transcription accuracy')
    parser.add_argument('--update-reports', action='store_true', help='Update README and report files')
    parser.add_argument('--model-name', type=str, default='speechmatics', 
                       help='Model name to use in output files')
    parser.add_argument('--eval-speechmatics', action='store_true', 
                       help='Evaluate Speechmatics transcripts (*_speechmatics.json)')
    return parser.parse_args()

def normalize_text(text, language='en'):
    """Normalize text for comparison by removing punctuation, lowercasing, etc."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def is_medical_term(token, language='en'):
    """Check if a token is a medical term based on our dictionaries"""
    token_lower = token.lower()
    
    # Handle common abbreviation variations
    if language.startswith('en'):
        # Map common transcription errors to correct terms
        token_mapping = {
            "egg": "ecg",
            "eggs": "ecg",
            "e c g": "ecg",
            "ekg": "ecg"
        }
        
        # Apply mapping if token is in mapping dictionary
        if token_lower in token_mapping:
            token_lower = token_mapping[token_lower]
            
        return token_lower in medical_terms_en
    elif language.startswith('fr'):
        # French mappings
        token_mapping = {
            "e c g": "ecg",
            "électrocardiogramme": "ecg"
        }
        
        if token_lower in token_mapping:
            token_lower = token_mapping[token_lower]
            
        return token_lower in medical_terms_fr
    
    return False

def extract_speaker_turns(data, source_type='original'):
    """Extract speaker turns from different data formats"""
    if source_type == 'original':
        # Original JSON format
        if isinstance(data, dict) and 'conversation' in data:
            return data['conversation']
        return []
    elif source_type == 'transcript':
        # Handle Speechmatics format
        if isinstance(data, dict) and 'turns' in data:
            # Map speaker keys to standardized format
            turns = []
            for turn in data['turns']:
                # Normalize speaker roles (DOCTOR/PATIENT -> doctor/patient)
                speaker = turn['speaker'].lower()
                turns.append({
                    'speaker': speaker,
                    'text': turn['text']
                })
            return turns
        # Legacy format - direct list
        elif isinstance(data, list):
            return data
        return []
    return []

def calculate_speaker_accuracy(original_turns, transcript_turns):
    """Calculate speaker identification accuracy with a more sophisticated approach that handles
    different turn counts and combined/split turns between original and transcript.
    
    This method prevents inflated accuracy when the turn structures don't match.
    """
    if not original_turns or not transcript_turns:
        return 0.0
    
    # Get total turns count for both sources
    original_count = len(original_turns)
    transcript_count = len(transcript_turns)
    
    # If turn counts don't match, this already indicates an issue with diarization
    turn_structure_similarity = min(original_count, transcript_count) / max(original_count, transcript_count)
    
    # Prepare speaker sequences for both datasets
    original_speakers = [turn['speaker'].lower() for turn in original_turns]
    transcript_speakers = [turn['speaker'].lower() for turn in transcript_turns]
    
    # Align the turns for comparison
    min_length = min(len(original_turns), len(transcript_turns))
    correct_speakers = 0
    
    for i in range(min_length):
        if original_speakers[i] == transcript_speakers[i]:
            correct_speakers += 1
    
    # Basic speaker alignment accuracy based on position
    position_accuracy = correct_speakers / min_length if min_length > 0 else 0.0
    
    # When turn counts match exactly, just use the position-based accuracy
    if original_count == transcript_count:
        return position_accuracy
    
    # Otherwise, apply a penalty based on the difference in turn structure
    # The final accuracy is weighted by the similarity in turn structure
    # This prevents claiming 100% accuracy when the turn counts are different
    final_accuracy = position_accuracy * turn_structure_similarity
    
    return final_accuracy

def extract_text_from_turns(turns):
    """Extract text from conversation turns"""
    return " ".join([turn['text'] for turn in turns])

def calculate_medical_term_accuracy(original_text, transcript_text, language='en'):
    """Calculate medical terminology recognition accuracy"""
    # Normalize texts
    original_text = normalize_text(original_text, language)
    transcript_text = normalize_text(transcript_text, language)
    
    # Tokenize texts
    original_tokens = original_text.split()
    transcript_tokens = transcript_text.split()
    
    # Check for n-grams (multi-word medical terms)
    original_med_terms = []
    transcript_med_terms = []
    
    # Single token medical terms
    for t in original_tokens:
        if is_medical_term(t, language):
            original_med_terms.append(t)
    
    for t in transcript_tokens:
        if is_medical_term(t, language):
            transcript_med_terms.append(t)
    
    # Multi-word terms (up to 3 words)
    for i in range(len(original_tokens)-1):
        bigram = original_tokens[i] + " " + original_tokens[i+1]
        if is_medical_term(bigram, language):
            original_med_terms.append(bigram)
    
    for i in range(len(transcript_tokens)-1):
        bigram = transcript_tokens[i] + " " + transcript_tokens[i+1]
        if is_medical_term(bigram, language):
            transcript_med_terms.append(bigram)
    
    # Check for trigrams (3-word terms)
    for i in range(len(original_tokens)-2):
        trigram = original_tokens[i] + " " + original_tokens[i+1] + " " + original_tokens[i+2]
        if is_medical_term(trigram, language):
            original_med_terms.append(trigram)
    
    for i in range(len(transcript_tokens)-2):
        trigram = transcript_tokens[i] + " " + transcript_tokens[i+1] + " " + transcript_tokens[i+2]
        if is_medical_term(trigram, language):
            transcript_med_terms.append(trigram)
    
    # Convert to sets for comparison
    original_med_set = set(original_med_terms)
    transcript_med_set = set(transcript_med_terms)
    
    # Calculate precision, recall, and F1 score
    true_positives = len(original_med_set.intersection(transcript_med_set))
    
    if not original_med_set:
        return {'precision': 0, 'recall': 0, 'f1_score': 0, 'medical_terms_count': 0}
    
    precision = true_positives / len(transcript_med_set) if transcript_med_set else 0
    recall = true_positives / len(original_med_set) if original_med_set else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'medical_terms_count': len(original_med_set)
    }

def calculate_text_similarity_metrics(original_text, transcript_text, language='en'):
    """Calculate various text similarity metrics"""
    # Normalize texts
    original_norm = normalize_text(original_text, language)
    transcript_norm = normalize_text(transcript_text, language)
    
    # Word Error Rate (WER)
    wer = jiwer.wer(original_norm, transcript_norm)
    
    # Character Error Rate (CER)
    cer = jiwer.cer(original_norm, transcript_norm)
    
    # Simple tokenization for other metrics
    original_tokens = original_norm.split()
    transcript_tokens = transcript_norm.split()
    
    # BLEU Score
    smoothing = SmoothingFunction().method1
    try:
        bleu = sentence_bleu([original_tokens], transcript_tokens, smoothing_function=smoothing)
    except:
        bleu = 0.0
    
    # TF-IDF Cosine Similarity
    try:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([original_norm, transcript_norm])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0.0
    
    return {
        'wer': wer,
        'cer': cer,
        'bleu': bleu,
        'cosine_similarity': cosine_sim
    }

def evaluate_file_pair(original_path, transcript_path, language):
    """Evaluate a single pair of original and transcript files"""
    try:
        # Load files
        with open(original_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Extract turns and text
        original_turns = extract_speaker_turns(original_data, 'original')
        transcript_turns = extract_speaker_turns(transcript_data, 'transcript')
        
        # Print turn count information for debugging
        original_count = len(original_turns)
        transcript_count = len(transcript_turns)
        print(f"File: {os.path.basename(original_path)}")
        print(f"Original turns: {original_count}, Transcript turns: {transcript_count}")
        
        # Normalize speaker labels for comparison
        for turn in original_turns:
            turn['speaker'] = turn['speaker'].lower()
            if turn['speaker'] == 'doctor':
                turn['speaker'] = 'doctor'
            elif turn['speaker'] == 'patient':
                turn['speaker'] = 'patient'
        
        for turn in transcript_turns:
            turn['speaker'] = turn['speaker'].lower()
            if turn['speaker'] == 'doctor':
                turn['speaker'] = 'doctor'
            elif turn['speaker'] == 'patient':
                turn['speaker'] = 'patient'
        
        original_text = extract_text_from_turns(original_turns)
        transcript_text = extract_text_from_turns(transcript_turns)
        
        # Calculate all metrics
        speaker_accuracy = calculate_speaker_accuracy(original_turns, transcript_turns)
        medical_metrics = calculate_medical_term_accuracy(original_text, transcript_text, language)
        text_metrics = calculate_text_similarity_metrics(original_text, transcript_text, language)
        
        # Get file metadata
        file_base = os.path.basename(original_path).replace('.json', '')
        specialty = 'cardiology' if 'cardiology' in file_base else 'gp'
        visit_type = 'consultation' if 'consultation' in file_base else 'followup'
        
        # Determine language name from code
        language_name = 'English' if language.startswith('en') else 'French'
        
        result = {
            'file': file_base,
            'specialty': specialty,
            'language_code': language,
            'language': language_name,
            'visit_type': visit_type,
            'speaker_accuracy': speaker_accuracy,
            'original_turn_count': original_count,
            'transcript_turn_count': transcript_count,
            **medical_metrics,
            **text_metrics
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing {original_path} & {transcript_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_file_pairs(base_dir, model_name='speechmatics'):
    """Process all file pairs for evaluation"""
    results = []
    specialties = ['cardiology', 'gp']
    
    for specialty in specialties:
        # Using fixed directory patterns for the specific structure
        language_dirs = ['en-CA-speechmatics', 'fr-CA-speechmatics']
        
        for lang_dir in language_dirs:
            language = 'en-CA' if 'en-CA' in lang_dir else 'fr-CA'
            
            # Build paths for original and transcript files
            json_dir = os.path.join(base_dir, specialty, lang_dir, 'json')
            transcript_dir = os.path.join(base_dir, specialty, lang_dir, 'transcripts')
            
            if not os.path.exists(json_dir) or not os.path.exists(transcript_dir):
                print(f"Directories not found: {json_dir} or {transcript_dir}")
                continue
            
            print(f"Processing files in {json_dir} with transcripts from {transcript_dir}")
            
            # Find all json files in the original directory
            original_files = glob.glob(os.path.join(json_dir, '*.json'))
            
            for original_path in tqdm(original_files, desc=f"Processing {specialty} {language}"):
                base_name = os.path.basename(original_path)
                file_prefix = base_name.split('.')[0]  # e.g., "cardiology_1_consultation"
                
                # Look for speechmatics transcript
                transcript_path = os.path.join(transcript_dir, f"{file_prefix}_speechmatics.json")
                
                if os.path.exists(transcript_path):
                    result = evaluate_file_pair(original_path, transcript_path, language)
                    if result:
                        # Add model info to results
                        result['model'] = model_name
                        results.append(result)
                else:
                    print(f"No transcript found for {original_path}")
    
    # Create a dataframe and ensure all required columns exist
    df = pd.DataFrame(results)
    # Fill any missing columns that might be needed later
    required_columns = ['file', 'specialty', 'language_code', 'language', 'visit_type', 'speaker_accuracy', 
                       'precision', 'recall', 'f1_score', 'medical_terms_count', 'wer', 'cer', 'bleu', 
                       'cosine_similarity', 'model', 'original_turn_count', 'transcript_turn_count']
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    return df

def generate_visualizations(results_df, output_dir='results/figures'):
    """Generate visualization charts and figures from the results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for all plots
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # 1. Overall Model Comparison by Language
    plt.figure(figsize=(15, 10))
    
    metrics = ['speaker_accuracy', 'f1_score', 'wer', 'cosine_similarity']
    metric_names = ['Speaker Accuracy', 'Medical Term F1', 'WER (lower is better)', 'Semantic Similarity']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i+1)
        
        if metric == 'wer':  # Lower is better for WER
            sns.barplot(x='language', y=metric, data=results_df, palette='viridis')
            plt.title(f"{title}")
        else:
            sns.barplot(x='language', y=metric, data=results_df, palette='viridis')
            plt.title(f"{title}")
        
        plt.xlabel('Language')
        plt.ylabel('Score')
        if metric not in ['wer', 'cer']:
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Specialty Comparison
    plt.figure(figsize=(15, 10))
    
    for i, (metric, title) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i+1)
        
        specialty_avg = results_df.groupby(['specialty', 'language'])[metric].mean().reset_index()
        
        if metric == 'wer':  # Lower is better for WER
            sns.barplot(x='specialty', y=metric, hue='language', data=specialty_avg, palette='viridis')
            plt.title(f"{title} by Specialty")
        else:
            sns.barplot(x='specialty', y=metric, hue='language', data=specialty_avg, palette='viridis')
            plt.title(f"{title} by Specialty")
        
        plt.xlabel('Specialty')
        plt.ylabel('Score')
        if metric not in ['wer', 'cer']:
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'specialty_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Visit Type Comparison
    plt.figure(figsize=(15, 10))
    
    for i, (metric, title) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i+1)
        
        visit_avg = results_df.groupby(['visit_type', 'language'])[metric].mean().reset_index()
        
        if metric == 'wer':  # Lower is better for WER
            sns.barplot(x='visit_type', y=metric, hue='language', data=visit_avg, palette='viridis')
            plt.title(f"{title} by Visit Type")
        else:
            sns.barplot(x='visit_type', y=metric, hue='language', data=visit_avg, palette='viridis')
            plt.title(f"{title} by Visit Type")
        
        plt.xlabel('Visit Type')
        plt.ylabel('Score')
        if metric not in ['wer', 'cer']:
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consultation_type_comparison_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Medical Term Accuracy
    plt.figure(figsize=(12, 8))
    
    med_metrics = ['precision', 'recall', 'f1_score']
    med_metric_names = ['Precision', 'Recall', 'F1 Score']
    
    for i, (metric, title) in enumerate(zip(med_metrics, med_metric_names)):
        plt.subplot(1, 3, i+1)
        
        lang_specialty = results_df.groupby(['language', 'specialty'])[metric].mean().reset_index()
        
        sns.barplot(x='language', y=metric, hue='specialty', data=lang_specialty, palette='viridis')
        plt.title(f"Medical Term {title}")
        plt.xlabel('Language')
        plt.ylabel('Score')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'medical_term_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Error Metrics Comparison
    plt.figure(figsize=(12, 6))
    
    error_metrics = ['wer', 'cer']
    error_metric_names = ['Word Error Rate', 'Character Error Rate']
    
    for i, (metric, title) in enumerate(zip(error_metrics, error_metric_names)):
        plt.subplot(1, 2, i+1)
        
        error_by_lang = results_df.groupby(['language', 'specialty'])[metric].mean().reset_index()
        
        sns.barplot(x='language', y=metric, hue='specialty', data=error_by_lang, palette='viridis')
        plt.title(f"{title}")
        plt.xlabel('Language')
        plt.ylabel('Error Rate (lower is better)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    
    # Select only numeric columns for correlation
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    correlation = results_df[numeric_cols].corr()
    
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Between Evaluation Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Combined Model Scores
    plt.figure(figsize=(15, 8))
    
    # Create a combined score (excluding WER and CER which are better when lower)
    results_df['combined_accuracy'] = results_df[['speaker_accuracy', 'f1_score', 'cosine_similarity']].mean(axis=1)
    
    lang_combined = results_df.groupby('language')['combined_accuracy'].mean().reset_index()
    
    sns.barplot(x='language', y='combined_accuracy', data=lang_combined, palette='viridis')
    plt.title('Combined Accuracy Score by Language')
    plt.xlabel('Language')
    plt.ylabel('Combined Score')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_model_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. NEW: Turn Count Difference Impact on Speaker Accuracy
    plt.figure(figsize=(12, 8))
    
    # Calculate turn count difference and convert to absolute value
    results_df['turn_count_diff'] = abs(results_df['original_turn_count'] - results_df['transcript_turn_count'])
    
    # Group by difference categories
    results_df['diff_category'] = pd.cut(
        results_df['turn_count_diff'], 
        bins=[-0.1, 0.1, 1.1, 2.1, np.inf], 
        labels=['No Difference', '1 Turn', '2 Turns', '3+ Turns']
    )
    
    # Create scatter plot with color-coded categories
    plt.subplot(1, 2, 1)
    sns.boxplot(x='diff_category', y='speaker_accuracy', data=results_df, palette='viridis')
    plt.title('Impact of Turn Count Difference on Speaker Accuracy')
    plt.xlabel('Turn Count Difference')
    plt.ylabel('Speaker Accuracy')
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    accuracy_by_diff = results_df.groupby(['language', 'diff_category'])['speaker_accuracy'].mean().reset_index()
    sns.barplot(x='diff_category', y='speaker_accuracy', hue='language', data=accuracy_by_diff, palette='viridis')
    plt.title('Speaker Accuracy by Turn Count Difference and Language')
    plt.xlabel('Turn Count Difference')
    plt.ylabel('Speaker Accuracy')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'turn_count_difference_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def update_readme_and_reports(results_df, args):
    """Update README.md and evaluation reports with new findings"""
    if not args.update_reports:
        return
    
    # Calculate summary statistics
    lang_summary = results_df.groupby('language')[['speaker_accuracy', 'f1_score', 'wer', 'cosine_similarity']].mean()
    specialty_summary = results_df.groupby(['specialty', 'language'])[['speaker_accuracy', 'f1_score', 'wer']].mean()
    
    # Calculate turn count match percentage
    turn_match = (results_df['original_turn_count'] == results_df['transcript_turn_count']).mean() * 100
    
    # Format as percentage for readability
    lang_summary_formatted = lang_summary.copy()
    lang_summary_formatted['speaker_accuracy'] = (lang_summary['speaker_accuracy'] * 100).round(1).astype(str) + '%'
    lang_summary_formatted['f1_score'] = (lang_summary['f1_score'] * 100).round(1).astype(str) + '%'
    lang_summary_formatted['cosine_similarity'] = (lang_summary['cosine_similarity'] * 100).round(1).astype(str) + '%'
    lang_summary_formatted['wer'] = lang_summary['wer'].round(2).astype(str)
    
    # Create additional report with updated speaker accuracy methodology
    report_content = f"""# Updated Speaker Accuracy Evaluation Report

## Methodology Change

The speaker accuracy metric has been updated to account for turn count discrepancies between original and transcribed files.
Previously, the metric only compared speaker roles in aligned turns without penalizing missing or combined turns.

## Key Findings

- **Turn Count Match Rate**: {turn_match:.1f}% of files have matching turn counts
- **Speaker Accuracy Calculation**: Now includes a penalty based on turn structure similarity
- **Overall Impact**: Accuracy scores now more accurately reflect diarization quality

## Performance Summary

| Language | Speaker Accuracy | F1 Score | WER | Similarity |
|----------|-----------------|----------|-----|------------|
"""

    for lang, row in lang_summary_formatted.iterrows():
        report_content += f"| {lang} | {row['speaker_accuracy']} | {row['f1_score']} | {row['wer']} | {row['cosine_similarity']} |\n"
    
    # Add turn count analysis section
    turn_analysis = results_df.groupby('language')[['original_turn_count', 'transcript_turn_count']].mean().round(1)
    
    report_content += f"""
## Turn Count Analysis

| Language | Original Turns (Avg) | Transcript Turns (Avg) |
|----------|---------------------|-------------------------|
"""
    
    for lang, row in turn_analysis.iterrows():
        report_content += f"| {lang} | {row['original_turn_count']} | {row['transcript_turn_count']} |\n"
    
    # Write the updated report
    with open('results/speaker_accuracy_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Update files with new statistics
    files_to_update = ['README.md', 'report.md', 'docs/Evaluation_Report.md']
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace performance metrics in tables
            for lang, row in lang_summary_formatted.iterrows():
                # For English results
                if lang == 'English':
                    pattern = r'(\| Nova-3-medical \(English\) \| )[\d.]+%( \| )[\d.]+%( \| )[\d.]+( \| )[\d.]+%( \|)'
                    replacement = f"\\1{row['f1_score']}\\2{row['speaker_accuracy']}\\3{row['wer']}\\4{row['cosine_similarity']}\\5"
                    content = re.sub(pattern, replacement, content)
                
                # For French results
                elif lang == 'French':
                    pattern = r'(\| Nova-2 \(French\) \| )[\d.]+%( \| )[\d.]+%( \| )[\d.]+( \| )[\d.]+%( \|)'
                    replacement = f"\\1{row['f1_score']}\\2{row['speaker_accuracy']}\\3{row['wer']}\\4{row['cosine_similarity']}\\5"
                    content = re.sub(pattern, replacement, content)
            
            # Add note about speaker accuracy calculation update
            if "## Key Findings" in content:
                if "Speaker Accuracy Calculation Update" not in content:
                    note = "\n- **Speaker Accuracy Calculation Update**: The speaker accuracy metric now accounts for turn count discrepancies, providing a more accurate measure of diarization quality."
                    content = content.replace("## Key Findings", "## Key Findings" + note)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated {file_path} with new evaluation results")
    
    print(f"Created new report: results/speaker_accuracy_report.md")

def main():
    args = setup_argparse()
    model_name = args.model_name
    
    base_dir = 'med-data'
    
    print(f"Starting evaluation of medical transcription files using model: {model_name}...")
    
    # Process all files
    results_df = process_file_pairs(base_dir, model_name)
    
    # Save results to CSV with model name in filename
    os.makedirs('results/tables', exist_ok=True)
    results_file = f'results/tables/transcription_evaluation_results_{model_name}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")
    
    # Generate visualizations with model name in filename
    print("Generating visualizations...")
    output_dir = f'results/figures/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    generate_visualizations(results_df, output_dir)
    print(f"Visualizations saved to {output_dir}/")
    
    # Update README and reports if requested
    if args.update_reports:
        update_readme_and_reports(results_df, args)
    
    # Print summary statistics
    print("\nEvaluation Summary:")
    print("-" * 80)
    
    # By language
    print("\nPerformance by Language:")
    language_summary = results_df.groupby('language')[['speaker_accuracy', 'f1_score', 'wer', 'cosine_similarity']].mean()
    print(language_summary.round(3))
    
    # By specialty
    print("\nPerformance by Specialty and Language:")
    specialty_summary = results_df.groupby(['specialty', 'language'])[['speaker_accuracy', 'f1_score', 'wer']].mean()
    print(specialty_summary.round(3))
    
    # Print turn count discrepancies
    print("\nTurn Count Analysis:")
    turn_analysis = results_df.groupby('language')[['original_turn_count', 'transcript_turn_count']].agg(['mean', 'min', 'max'])
    print(turn_analysis.round(1))
    
    print("\nFiles with Turn Count Mismatch:")
    mismatch_files = results_df[results_df['original_turn_count'] != results_df['transcript_turn_count']]
    if not mismatch_files.empty:
        for _, row in mismatch_files.iterrows():
            print(f"{row['file']}: Original: {int(row['original_turn_count'])} vs Transcript: {int(row['transcript_turn_count'])}")
    else:
        print("No turn count mismatches found.")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 