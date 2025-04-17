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
    "electrocardiogram", "ECG", "EKG", "fibrillation", "heartbeat", "heart rate", 
    "hyperlipidemia", "hypertension", "hypotension", "ischemia", "infarction", 
    "mitral valve", "myocardial", "myocardium", "pacemaker", "palpitation", 
    "percutaneous", "perfusion", "pericardium", "plaque", "pulmonary", "regurgitation", 
    "revascularization", "stent", "stenosis", "syncope", "systolic", "tachycardia", 
    "thrombosis", "vascular", "vasoconstriction", "vasodilation", "ventricular", 
    "artery", "vein", "ventricle", "valve", "warfarin", "clopidogrel", "aspirin",
    "beta blocker", "ace inhibitor", "statin", "diuretic", "digoxin", "nitrate",
    "anticoagulant", "antiarrhythmic", "atherosclerosis",
    
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
    "cough", "shortness of breath", "rash", "mental health", "physical exam"
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
    "anticoagulant", "antiarythmique", "athérosclérose",
    
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
    "toux", "essoufflement", "éruption cutanée", "santé mentale", "examen physique"
])

def setup_argparse():
    parser = argparse.ArgumentParser(description='Evaluate medical transcription accuracy')
    parser.add_argument('--update-reports', action='store_true', help='Update README and report files')
    parser.add_argument('--model-name', type=str, default='whisper-v3-turbo-Nvidio_Nemo', 
                       help='Model name to use in output files')
    return parser.parse_args()

def normalize_text(text, language='en'):
    """Normalize text for comparison by removing punctuation, lowercasing, etc."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def is_medical_term(token, language='en'):
    """Check if a token is a medical term based on our dictionaries"""
    if language.startswith('en'):
        return token.lower() in medical_terms_en
    elif language.startswith('fr'):
        return token.lower() in medical_terms_fr
    return False

def extract_speaker_turns(data, source_type='original'):
    """Extract speaker turns from different data formats"""
    if source_type == 'original':
        # Original JSON format
        if isinstance(data, dict) and 'conversation' in data:
            return data['conversation']
        return []
    elif source_type == 'transcript':
        # Handle new transcript format
        if isinstance(data, dict) and 'turns' in data:
            return data['turns']
        # Legacy format - direct list
        elif isinstance(data, list):
            return data
        return []
    return []

def calculate_speaker_accuracy(original_turns, transcript_turns):
    """Calculate speaker identification accuracy"""
    if not original_turns or not transcript_turns:
        return 0.0
    
    # Align the turns for comparison
    min_length = min(len(original_turns), len(transcript_turns))
    correct_speakers = 0
    
    for i in range(min_length):
        if original_turns[i]['speaker'].lower() == transcript_turns[i]['speaker'].lower():
            correct_speakers += 1
    
    return correct_speakers / min_length if min_length > 0 else 0.0

def extract_text_from_turns(turns):
    """Extract text from conversation turns"""
    return " ".join([turn['text'] for turn in turns])

def calculate_medical_term_accuracy(original_text, transcript_text, language='en'):
    """Calculate medical terminology recognition accuracy"""
    # Normalize texts
    original_text = normalize_text(original_text, language)
    transcript_text = normalize_text(transcript_text, language)
    
    # Tokenize texts - simplified to handle NLTK issues
    original_tokens = original_text.split()
    transcript_tokens = transcript_text.split()
    
    # Identify medical terms
    original_med_terms = [t for t in original_tokens if is_medical_term(t, language)]
    transcript_med_terms = [t for t in transcript_tokens if is_medical_term(t, language)]
    
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
        
        result = {
            'file': file_base,
            'specialty': specialty,
            'language_code': language,
            'language': 'English' if language.startswith('en') else 'French',
            'visit_type': visit_type,
            'speaker_accuracy': speaker_accuracy,
            **medical_metrics,
            **text_metrics
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing {original_path} & {transcript_path}: {e}")
        return None

def process_file_pairs(base_dir, language_dirs, model_name='whisper-v3-large-Nvidio_Nemo'):
    """Process all file pairs for evaluation"""
    results = []
    
    for lang_dir in language_dirs:
        language = 'en-CA' if 'en-CA' in lang_dir else 'fr-CA'
        specialty = 'cardiology' if 'cardiology' in lang_dir else 'gp'
        
        # Adjust for directories without the "- gpt4o" suffix
        json_dir = os.path.join(base_dir, specialty, language, 'json')
        if not os.path.exists(json_dir):
            # Try with specific model directory pattern
            possible_patterns = [
                os.path.join(base_dir, specialty, f"{language}-whisper-v3", 'json'),
                os.path.join(base_dir, specialty, f"{language}-whisper-v3-turbo", 'json'),
                os.path.join(base_dir, specialty, f"{language}-whisper-v3-large", 'json'),
                os.path.join(base_dir, specialty, f"{language} - gpt4o", 'json'),
            ]
            for pattern in possible_patterns:
                if os.path.exists(pattern):
                    json_dir = pattern
                    break
        
        transcript_dir = json_dir.replace('json', 'transcripts')
        if not os.path.exists(transcript_dir):
            os.makedirs(transcript_dir, exist_ok=True)
        
        print(f"Processing files in {json_dir} with transcripts from {transcript_dir}")
        
        # Find all json files in the original directory
        original_files = glob.glob(os.path.join(json_dir, '*.json'))
        
        for original_path in tqdm(original_files, desc=f"Processing {specialty} {language}"):
            base_name = os.path.basename(original_path)
            file_prefix = base_name.split('.')[0]  # e.g., "cardiology_1_consultation"
            transcript_path = os.path.join(transcript_dir, f"{file_prefix}_transcription.json")
            
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
                       'precision', 'recall', 'f1_score', 'medical_terms_count', 'wer', 'cer', 'bleu', 'cosine_similarity', 'model']
    
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
    
    return True

def update_readme_and_reports(results_df, args):
    """Update README.md and evaluation reports with new findings"""
    if not args.update_reports:
        return
    
    # Calculate summary statistics
    lang_summary = results_df.groupby('language')[['speaker_accuracy', 'f1_score', 'wer', 'cosine_similarity']].mean()
    specialty_summary = results_df.groupby(['specialty', 'language'])[['speaker_accuracy', 'f1_score', 'wer']].mean()
    
    # Format as percentage for readability
    lang_summary_formatted = lang_summary.copy()
    lang_summary_formatted['speaker_accuracy'] = (lang_summary['speaker_accuracy'] * 100).round(1).astype(str) + '%'
    lang_summary_formatted['f1_score'] = (lang_summary['f1_score'] * 100).round(1).astype(str) + '%'
    lang_summary_formatted['cosine_similarity'] = (lang_summary['cosine_similarity'] * 100).round(1).astype(str) + '%'
    lang_summary_formatted['wer'] = lang_summary['wer'].round(2).astype(str)
    
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
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated {file_path} with new evaluation results")

def main():
    args = setup_argparse()
    model_name = args.model_name
    
    base_dir = 'med-data'
    
    # Directories to process - updated to match current format
    language_dirs = [
        'cardiology/en-CA-whisper-v3',
        'cardiology/fr-CA-whisper-v3',
        'gp/en-CA-whisper-v3',
        'gp/fr-CA-whisper-v3'
    ]
    
    print(f"Starting evaluation of medical transcription files using model: {model_name}...")
    
    # Process all files
    results_df = process_file_pairs(base_dir, language_dirs, model_name)
    
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
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 