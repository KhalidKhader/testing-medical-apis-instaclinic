#!/usr/bin/env python3
"""
Evaluate transcription accuracy by comparing the transcribed text with the original conversations.
Special focus on medical terminology accuracy and speaker diarization.

Usage:
    python evaluate_transcriptions.py --specialty cardiology --lang en-CA
    python evaluate_transcriptions.py --specialty all --lang all
"""

import os
import json
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from difflib import SequenceMatcher
from collections import defaultdict

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    HAS_NLTK = True
    # Download necessary NLTK resources
    nltk.download('punkt', quiet=True)
except ImportError:
    HAS_NLTK = False
    print("NLTK not available. Some evaluation metrics will be limited.")

# Base directory for medical data
BASE_DIR = "data-med"

# Medical terminology dictionary by specialty
MEDICAL_TERMS = {
    "cardiology": [
        "hypertension", "atrial fibrillation", "heart failure", "coronary artery disease", "myocardial infarction",
        "arrhythmia", "tachycardia", "bradycardia", "palpitations", "angina", "dyspnea", "edema", "syncope",
        "hyperlipidemia", "anticoagulant", "beta blocker", "statin", "ACE inhibitor", "diuretic", "angioplasty",
        "echocardiogram", "electrocardiogram", "ECG", "EKG", "stress test", "cardiac catheterization", "stent",
        "aortic", "mitral", "tricuspid", "pulmonary", "systolic", "diastolic", "warfarin", "apixaban", "rivaroxaban",
        "metoprolol", "atorvastatin", "lisinopril", "furosemide", "spironolactone", "nitroglycerin", "aspirin",
        "clopidogrel", "ventricular", "atrium", "atria", "ventricle", "cardiomyopathy", "ischemia", "infarct",
        "stenosis", "regurgitation", "fibrillation", "flutter", "thrombosis", "embolism"
    ],
    "gp": [
        "diabetes", "hypertension", "arthritis", "asthma", "COPD", "bronchitis", "pneumonia", "influenza",
        "gastritis", "GERD", "ulcer", "irritable bowel syndrome", "urinary tract infection", "UTI", "thyroid",
        "hypothyroidism", "hyperthyroidism", "depression", "anxiety", "insomnia", "migraine", "headache", 
        "metformin", "insulin", "glucometer", "A1C", "salbutamol", "albuterol", "inhaler", "levothyroxine",
        "omeprazole", "ibuprofen", "acetaminophen", "antibiotic", "amoxicillin", "azithromycin", "ciprofloxacin",
        "blood pressure", "cholesterol", "LDL", "HDL", "triglycerides", "glucose", "potassium", "sodium",
        "vaccination", "immunization", "annual physical", "screening", "preventive", "referral", "specialist"
    ]
}

# French equivalents for medical terms
MEDICAL_TERMS_FR = {
    "cardiology": [
        "hypertension", "fibrillation auriculaire", "insuffisance cardiaque", "maladie coronarienne", "infarctus du myocarde",
        "arythmie", "tachycardie", "bradycardie", "palpitations", "angine", "dyspnée", "œdème", "syncope",
        "hyperlipidémie", "anticoagulant", "bêta-bloquant", "statine", "inhibiteur de l'ECA", "diurétique", "angioplastie",
        "échocardiographie", "électrocardiogramme", "ECG", "test d'effort", "cathétérisme cardiaque", "stent",
        "aortique", "mitral", "tricuspide", "pulmonaire", "systolique", "diastolique", "warfarine", "apixaban", "rivaroxaban",
        "métoprolol", "atorvastatine", "lisinopril", "furosémide", "spironolactone", "nitroglycérine", "aspirine",
        "clopidogrel", "ventriculaire", "oreillette", "ventricule", "cardiomyopathie", "ischémie", "infarctus",
        "sténose", "régurgitation", "fibrillation", "flutter", "thrombose", "embolie"
    ],
    "gp": [
        "diabète", "hypertension", "arthrite", "asthme", "BPCO", "bronchite", "pneumonie", "grippe",
        "gastrite", "RGO", "ulcère", "syndrome du côlon irritable", "infection urinaire", "thyroïde",
        "hypothyroïdie", "hyperthyroïdie", "dépression", "anxiété", "insomnie", "migraine", "céphalée", 
        "metformine", "insuline", "glucomètre", "A1C", "salbutamol", "inhalateur", "lévothyroxine",
        "oméprazole", "ibuprofène", "acétaminophène", "antibiotique", "amoxicilline", "azithromycine", "ciprofloxacine",
        "tension artérielle", "cholestérol", "LDL", "HDL", "triglycérides", "glucose", "potassium", "sodium",
        "vaccination", "immunisation", "examen annuel", "dépistage", "préventif", "référence", "spécialiste"
    ]
}

class TranscriptionEvaluator:
    """Class to evaluate transcription accuracy and diarization quality."""
    
    def __init__(self, specialty, language):
        """Initialize the evaluator with specialty and language selection."""
        self.specialty = specialty
        self.language = language
        self.metrics = defaultdict(list)
        self.conversation_results = {}
        
        # Load appropriate medical terms list based on specialty and language
        if language == "en-CA":
            self.medical_terms = set(MEDICAL_TERMS.get(specialty, []))
            self.medical_terms.update(MEDICAL_TERMS.get("gp", []))  # Add common terms
        else:  # fr-CA
            self.medical_terms = set(MEDICAL_TERMS_FR.get(specialty, []))
            self.medical_terms.update(MEDICAL_TERMS_FR.get("gp", []))  # Add common terms
    
    def clean_text(self, text):
        """Clean text by removing punctuation, extra spaces, and converting to lowercase."""
        if not text:
            return ""
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def calculate_similarity(self, str1, str2):
        """Calculate text similarity using sequence matcher."""
        if not str1 or not str2:
            return 0.0
        
        # Clean strings
        str1_clean = self.clean_text(str1)
        str2_clean = self.clean_text(str2)
        
        # Calculate similarity
        return SequenceMatcher(None, str1_clean, str2_clean).ratio()
    
    def calculate_wer(self, reference, hypothesis):
        """Calculate Word Error Rate between reference and hypothesis texts."""
        if not reference or not hypothesis:
            return 1.0  # Maximum error
        
        # Clean and tokenize
        ref_words = self.clean_text(reference).split()
        hyp_words = self.clean_text(hypothesis).split()
        
        if not ref_words:
            return 1.0
        
        # Calculate edit distance
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        for i in range(len(ref_words) + 1):
            d[i, 0] = i
        for j in range(len(hyp_words) + 1):
            d[0, j] = j
            
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i, j] = d[i-1, j-1]
                else:
                    d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
                    
        # Calculate WER
        return d[len(ref_words), len(hyp_words)] / len(ref_words)
    
    def calculate_bleu(self, reference, hypothesis):
        """Calculate BLEU score between reference and hypothesis."""
        if not reference or not hypothesis or not HAS_NLTK:
            return 0.0
        
        try:
            # Tokenize
            ref_tokens = word_tokenize(self.clean_text(reference))
            hyp_tokens = word_tokenize(self.clean_text(hypothesis))
            
            if not ref_tokens:
                return 0.0
            
            # Calculate BLEU
            return sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0))
        except:
            # If BLEU calculation fails, return similarity ratio as approximation
            return self.calculate_similarity(reference, hypothesis)
    
    def calculate_medical_term_accuracy(self, original_text, transcribed_text):
        """Calculate accuracy of medical term recognition in the transcription."""
        if not original_text or not transcribed_text:
            return 0.0
            
        original_lower = self.clean_text(original_text)
        transcribed_lower = self.clean_text(transcribed_text)
        
        # Find medical terms in original text
        original_terms = set()
        for term in self.medical_terms:
            term_lower = term.lower()
            if term_lower in original_lower:
                original_terms.add(term_lower)
        
        if not original_terms:
            return 1.0  # No medical terms to recognize
            
        # Count correctly recognized terms
        recognized_terms = set()
        for term in original_terms:
            if term in transcribed_lower:
                recognized_terms.add(term)
        
        # Calculate accuracy
        return len(recognized_terms) / len(original_terms) if original_terms else 1.0
    
    def evaluate_conversation(self, original_file, transcript_file, diarized_file=None):
        """Evaluate one conversation and calculate metrics."""
        try:
            # Load original conversation
            with open(original_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Load transcript
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            # Load diarized transcript if available
            diarized_transcript = None
            if diarized_file and os.path.exists(diarized_file):
                with open(diarized_file, 'r', encoding='utf-8') as f:
                    diarized_data = json.load(f)
                    diarized_transcript = diarized_data.get("conversation", [])
            
            # Get conversation ID from filename
            basename = os.path.basename(original_file)
            conversation_id = os.path.splitext(basename)[0]
            
            # Extract original conversation turns
            original_turns = original_data.get("conversation", [])
            
            # Skip evaluation if original conversation is empty
            if not original_turns:
                print(f"Warning: Empty original conversation in {conversation_id}")
                return None
            
            # Concatenate all original text
            original_full_text = " ".join([turn.get("text", "") for turn in original_turns])
            
            # Calculate basic text similarity
            similarity = self.calculate_similarity(original_full_text, transcript)
            
            # Calculate WER
            wer = self.calculate_wer(original_full_text, transcript)
            
            # Calculate BLEU
            bleu = self.calculate_bleu(original_full_text, transcript)
            
            # Calculate medical term accuracy
            medical_term_accuracy = self.calculate_medical_term_accuracy(original_full_text, transcript)
            
            # Evaluate diarization if available
            speaker_accuracy = 0.0
            if diarized_transcript:
                speaker_matches = 0
                # Limit comparison to the minimum number of turns in both conversations
                min_turns = min(len(original_turns), len(diarized_transcript))
                if min_turns > 0:
                    for i in range(min_turns):
                        original_speaker = original_turns[i].get("speaker", "").lower()
                        diarized_speaker = diarized_transcript[i].get("speaker", "").lower()
                        
                        # Check if speakers match
                        if original_speaker == diarized_speaker:
                            speaker_matches += 1
                    
                    speaker_accuracy = speaker_matches / min_turns
            
            # Store metrics
            metrics = {
                "wer": wer,
                "similarity": similarity,
                "bleu": bleu,
                "medical_term_accuracy": medical_term_accuracy,
                "speaker_accuracy": speaker_accuracy
            }
            
            # Store results for this conversation
            self.conversation_results[conversation_id] = metrics
            
            # Update aggregated metrics
            for key, value in metrics.items():
                self.metrics[key].append(value)
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {os.path.basename(original_file)}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_all_conversations(self, json_dir, transcripts_dir):
        """Evaluate all conversations in the directory."""
        # Find all original JSON files
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {json_dir}")
            return
        
        print(f"Found {len(json_files)} original conversation files in {json_dir}")
        
        # Evaluate each conversation
        results = []
        
        for json_file in tqdm(json_files, desc=f"Evaluating {self.language} {self.specialty} transcriptions"):
            # Extract basename
            basename = os.path.basename(json_file)
            filename = os.path.splitext(basename)[0]
            
            # Construct paths to transcript files
            transcript_file = os.path.join(transcripts_dir, f"{filename}_transcript.txt")
            diarized_file = os.path.join(transcripts_dir, f"{filename}_diarized.json")
            
            # Check if transcript file exists
            if not os.path.exists(transcript_file):
                print(f"Warning: Transcript file not found for {basename}")
                continue
            
            # Evaluate this conversation
            metrics = self.evaluate_conversation(json_file, transcript_file, diarized_file)
            
            if metrics:
                results.append({
                    'conversation_id': filename,
                    **metrics
                })
        
        return results
    
    def print_summary(self):
        """Print a summary of the evaluation results."""
        if not self.metrics["wer"]:
            print(f"No evaluation results for {self.language} {self.specialty}")
            return
            
        print("\n===== TRANSCRIPTION EVALUATION SUMMARY =====")
        print(f"Specialty: {self.specialty}")
        print(f"Language: {self.language}")
        print(f"Conversations evaluated: {len(self.metrics['wer'])}")
        
        print("\nMean Metrics:")
        for key, values in self.metrics.items():
            if values:
                print(f"  {key.upper()}: {np.mean(values):.4f} (std: {np.std(values):.4f})")
            
        print("\nMedical Term Accuracy:")
        medical_acc = self.metrics["medical_term_accuracy"]
        if medical_acc:
            print(f"  Mean: {np.mean(medical_acc):.4f}")
            print(f"  Median: {np.median(medical_acc):.4f}")
            print(f"  Min: {np.min(medical_acc):.4f}")
            print(f"  Max: {np.max(medical_acc):.4f}")
    
    def plot_results(self, output_dir):
        """Plot evaluation results and save to the output directory."""
        if not self.metrics["wer"]:
            print(f"No evaluation results to plot for {self.language} {self.specialty}")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(list(self.conversation_results.values()))
        
        # Plot WER distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df["wer"], kde=True)
        plt.axvline(x=results_df["wer"].mean(), color='r', linestyle='--', 
                   label=f'Mean: {results_df["wer"].mean():.4f}')
        plt.title(f'{self.language} {self.specialty} - Word Error Rate Distribution')
        plt.xlabel('Word Error Rate (WER)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{self.language}_{self.specialty}_wer.png'), dpi=300)
        plt.close()
        
        # Plot medical term accuracy distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df["medical_term_accuracy"], kde=True)
        plt.axvline(x=results_df["medical_term_accuracy"].mean(), color='r', linestyle='--', 
                   label=f'Mean: {results_df["medical_term_accuracy"].mean():.4f}')
        plt.title(f'{self.language} {self.specialty} - Medical Term Accuracy')
        plt.xlabel('Medical Term Accuracy')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{self.language}_{self.specialty}_med_acc.png'), dpi=300)
        plt.close()
        
        # Plot speaker accuracy vs medical term accuracy
        if "speaker_accuracy" in results_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(results_df["speaker_accuracy"], results_df["medical_term_accuracy"], alpha=0.7)
            plt.title(f'{self.language} {self.specialty} - Speaker vs Medical Term Accuracy')
            plt.xlabel('Speaker Accuracy')
            plt.ylabel('Medical Term Accuracy')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{self.language}_{self.specialty}_speaker_vs_med.png'), dpi=300)
            plt.close()
        
        # Save results to CSV
        results_df.to_csv(os.path.join(output_dir, f'{self.language}_{self.specialty}_results.csv'), index=False)
        
        print(f"Plots and results saved to {output_dir}")

def evaluate_specialty(specialty, language="all"):
    """Evaluate all conversations for a given specialty and language."""
    languages = ["en-CA", "fr-CA"] if language == "all" else [language]
    
    for lang in languages:
        # Set up directories
        json_dir = os.path.join(BASE_DIR, specialty, lang, "json")
        transcripts_dir = os.path.join(BASE_DIR, specialty, lang, "transcripts")
        output_dir = os.path.join(BASE_DIR, "evaluation", specialty)
        
        # Check if directories exist
        if not os.path.exists(json_dir):
            print(f"JSON directory not found: {json_dir}")
            continue
            
        if not os.path.exists(transcripts_dir):
            print(f"Transcripts directory not found: {transcripts_dir}")
            continue
        
        # Create evaluator
        evaluator = TranscriptionEvaluator(specialty, lang)
        
        # Evaluate conversations
        evaluator.evaluate_all_conversations(json_dir, transcripts_dir)
        
        # Print summary
        evaluator.print_summary()
        
        # Plot results
        evaluator.plot_results(output_dir)

def main():
    """Main function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate transcription accuracy')
    parser.add_argument('--specialty', type=str, choices=['cardiology', 'gp', 'all'], default='all', 
                        help='Medical specialty to evaluate')
    parser.add_argument('--lang', type=str, choices=['en-CA', 'fr-CA', 'all'], default='all',
                        help='Language to evaluate')
    args = parser.parse_args()
    
    # Ensure evaluation output directory exists
    os.makedirs(os.path.join(BASE_DIR, "evaluation"), exist_ok=True)
    
    # Evaluate based on arguments
    if args.specialty == "all":
        specialties = ["cardiology", "gp"]
        for specialty in specialties:
            evaluate_specialty(specialty, args.lang)
    else:
        evaluate_specialty(args.specialty, args.lang)
    
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main() 