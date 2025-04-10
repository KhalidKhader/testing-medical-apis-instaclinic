#!/usr/bin/env python3
"""
Evaluate transcription accuracy by comparing the transcribed text with the original conversations.
Special focus on medical terminology accuracy and speaker diarization.

Usage:
    python evaluate_transcriptions.py --specialty cardiology --lang en-CA
    python evaluate_transcriptions.py --specialty all --lang all
    python evaluate_transcriptions.py --data-dir "all-data"
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

# Default base directory for medical data
DEFAULT_BASE_DIR = "data-med"

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
    
    def __init__(self, specialty, language, base_dir=DEFAULT_BASE_DIR, dataset_name=""):
        """Initialize the evaluator with specialty and language selection."""
        self.specialty = specialty
        self.language = language
        self.base_dir = base_dir
        self.dataset_name = dataset_name
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
            
            # Check if transcript is suspiciously similar to original text (potential data leakage)
            suspicious_match = False
            if self.calculate_similarity(original_full_text, transcript) > 0.98:
                # Verify this isn't just a coincidence by checking for exact phrase matches
                # Get a few phrases from the original
                phrases = []
                for turn in original_turns[:3]:  # Check first 3 turns
                    text = turn.get("text", "")
                    if len(text) > 20:  # Only check substantial phrases
                        phrases.append(text[:20])  # Take first 20 chars
                
                # Check if these exact phrases appear in transcript
                matches = sum(1 for phrase in phrases if phrase in transcript)
                if matches >= 2:  # If 2 or more exact phrase matches found
                    suspicious_match = True
                    print(f"WARNING: Transcript for {conversation_id} appears to be from original text, not actual audio transcription")
            
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
                "dataset": self.dataset_name,
                "specialty": self.specialty,
                "language": self.language,
                "conversation_id": conversation_id,
                "wer": wer,
                "similarity": similarity,
                "bleu": bleu,
                "medical_term_accuracy": medical_term_accuracy,
                "speaker_accuracy": speaker_accuracy,
                "suspicious_transcript": suspicious_match
            }
            
            # Store results for this conversation
            self.conversation_results[conversation_id] = metrics
            
            # Update aggregated metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):  # Only add numeric metrics
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
            return []
        
        print(f"Found {len(json_files)} original conversation files in {json_dir}")
        
        # Evaluate each conversation
        results = []
        
        for json_file in tqdm(json_files, desc=f"Evaluating {self.language} {self.specialty} transcriptions in {self.dataset_name}"):
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
                results.append(metrics)
        
        return results
    
    def print_summary(self):
        """Print a summary of the evaluation results."""
        if not self.metrics.get("wer", []):
            print(f"No evaluation results for {self.language} {self.specialty} in {self.dataset_name}")
            return
            
        print(f"\n===== TRANSCRIPTION EVALUATION SUMMARY for {self.dataset_name} =====")
        print(f"Specialty: {self.specialty}")
        print(f"Language: {self.language}")
        print(f"Conversations evaluated: {len(self.metrics['wer'])}")
        
        # Check if there are suspicious transcripts (potentially direct copies)
        suspicious_count = sum(1 for v in self.conversation_results.values() if v.get("suspicious_transcript", False))
        if suspicious_count > 0:
            print(f"\n⚠️ WARNING: {suspicious_count} out of {len(self.conversation_results)} transcripts appear to be direct copies")
            print(f"⚠️ of the original text, rather than actual audio transcriptions.")
            print(f"⚠️ This may explain unrealistically high accuracy scores.")
        
        print("\nMean Metrics:")
        for key, values in self.metrics.items():
            if values and key not in ["dataset", "specialty", "language", "conversation_id", "suspicious_transcript"]:
                print(f"  {key.upper()}: {np.mean(values):.4f} (std: {np.std(values):.4f})")
            
        print("\nMedical Term Accuracy:")
        medical_acc = self.metrics.get("medical_term_accuracy", [])
        if medical_acc:
            print(f"  Mean: {np.mean(medical_acc):.4f}")
            print(f"  Median: {np.median(medical_acc):.4f}")
            print(f"  Min: {np.min(medical_acc):.4f}")
            print(f"  Max: {np.max(medical_acc):.4f}")
    
    def plot_results(self, output_dir):
        """Plot evaluation results and save to the output directory."""
        if not self.metrics.get("wer", []):
            print(f"No evaluation results to plot for {self.language} {self.specialty} in {self.dataset_name}")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(list(self.conversation_results.values()))
        
        # Add dataset name prefix for output files
        prefix = f"{self.dataset_name}_" if self.dataset_name else ""
        
        # Plot WER distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df["wer"], kde=True)
        plt.axvline(x=results_df["wer"].mean(), color='r', linestyle='--', 
                   label=f'Mean: {results_df["wer"].mean():.4f}')
        plt.title(f'{self.dataset_name} {self.language} {self.specialty} - Word Error Rate Distribution')
        plt.xlabel('Word Error Rate (WER)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{prefix}{self.language}_{self.specialty}_wer.png'), dpi=300)
        plt.close()
        
        # Plot medical term accuracy distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df["medical_term_accuracy"], kde=True)
        plt.axvline(x=results_df["medical_term_accuracy"].mean(), color='r', linestyle='--', 
                   label=f'Mean: {results_df["medical_term_accuracy"].mean():.4f}')
        plt.title(f'{self.dataset_name} {self.language} {self.specialty} - Medical Term Accuracy')
        plt.xlabel('Medical Term Accuracy')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{prefix}{self.language}_{self.specialty}_med_acc.png'), dpi=300)
        plt.close()
        
        # Plot speaker accuracy vs medical term accuracy
        if "speaker_accuracy" in results_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(results_df["speaker_accuracy"], results_df["medical_term_accuracy"], alpha=0.7)
            plt.title(f'{self.dataset_name} {self.language} {self.specialty} - Speaker vs Medical Term Accuracy')
            plt.xlabel('Speaker Accuracy')
            plt.ylabel('Medical Term Accuracy')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{prefix}{self.language}_{self.specialty}_speaker_vs_med.png'), dpi=300)
            plt.close()
        
        # Save results to CSV
        results_df.to_csv(os.path.join(output_dir, f'{prefix}{self.language}_{self.specialty}_results.csv'), index=False)
        
        return results_df

    def export_results(self, output_file):
        """Export the evaluation results to a CSV file."""
        results_df = pd.DataFrame(self.metrics)
        results_df.to_csv(output_file, index=False)
        print(f"Evaluation results saved to {output_file}")
        
        # Also export suspicious transcript information if any exist
        suspicious_entries = {conv_id: data for conv_id, data in self.conversation_results.items() 
                             if data.get("suspicious_transcript", False)}
        
        if suspicious_entries:
            suspicious_file = output_file.replace('.csv', '_suspicious.csv')
            suspicious_df = pd.DataFrame([
                {
                    "conversation_id": conv_id,
                    "wer": data.get("wer", "N/A"),
                    "similarity": data.get("similarity", "N/A"),
                    "medical_term_accuracy": data.get("medical_term_accuracy", "N/A"),
                    "exact_matches": data.get("exact_phrase_matches", 0),
                    "language": self.language,
                    "specialty": self.specialty,
                    "dataset": self.dataset_name
                }
                for conv_id, data in suspicious_entries.items()
            ])
            suspicious_df.to_csv(suspicious_file, index=False)
            print(f"Suspicious transcript details saved to {suspicious_file}")
            
    def compare_transcription(self, original_text, transcribed_text, id):
        """Compare original text with transcribed text and calculate metrics."""
        # Check if this might be a suspicious transcript (direct copy of original)
        is_suspicious = self.check_suspicious_transcript(original_text, transcribed_text)
        if is_suspicious:
            print(f"WARNING: Conversation {id} appears to be a direct copy rather than a genuine transcription!")
        
        # Calculate WER
        wer = self.calculate_wer(original_text, transcribed_text)
        
        # Calculate similarity
        similarity = self.calculate_similarity(original_text, transcribed_text)
        
        # Calculate medical term accuracy
        medical_terms = self.calculate_medical_term_accuracy(original_text, transcribed_text)
        
        # Calculate exact phrase matches
        phrases = self.extract_key_phrases(original_text)
        if phrases:
            exact_matches = self.count_exact_matches(phrases, transcribed_text)
        else:
            exact_matches = None
        
        # Store results for this conversation
        self.conversation_results[id] = {
            "wer": wer,
            "similarity": similarity,
            "medical_term_accuracy": medical_terms,
            "exact_phrase_matches": exact_matches,
            "suspicious_transcript": is_suspicious
        }
        
        # Update metrics
        self.metrics["conversation_id"].append(id)
        self.metrics["wer"].append(wer)
        self.metrics["similarity"].append(similarity)
        self.metrics["medical_term_accuracy"].append(medical_terms if medical_terms is not None else "N/A")
        self.metrics["exact_phrase_matches"].append(exact_matches if exact_matches is not None else 0)
        
    def check_suspicious_transcript(self, reference_text, hypothesis_text, wer=None, similarity=None):
        """Check if a transcript appears suspicious (too similar to original)."""
        if not reference_text or not hypothesis_text:
            return False
            
        # Calculate metrics if not provided
        if wer is None:
            wer = self.calculate_wer(reference_text, hypothesis_text)
        if similarity is None:
            similarity = self.calculate_similarity(reference_text, hypothesis_text)
            
        # Handle perfect or near-perfect matches
        if wer == 0.0 or similarity >= 0.99:
            return True
            
        # Check for high n-gram matches
        reference_phrases = self.extract_key_phrases(reference_text)
        matches = self.count_exact_matches(reference_phrases, hypothesis_text)
        
        if reference_phrases:
            ngram_ratio = matches / len(reference_phrases)
            suspicious_threshold = 0.8  # 80% exact matches is suspicious
            
            return (similarity > 0.98 or wer < 0.03 or ngram_ratio > suspicious_threshold)
        else:
            return (similarity > 0.98 or wer < 0.03)

    def extract_key_phrases(self, text):
        """Extract key phrases from the text for exact match evaluation."""
        if not text:
            return []
            
        # Clean the text
        clean_text = self.clean_text(text)
        words = clean_text.split()
        
        # Extract phrases of different lengths
        phrases = []
        
        # Extract 4-word phrases (good for detecting significant matches)
        if len(words) >= 4:
            for i in range(len(words) - 3):
                phrase = ' '.join(words[i:i+4])
                if len(phrase) >= 12:  # Only include substantial phrases
                    phrases.append(phrase)
        
        # Limit to a reasonable number of phrases to check
        max_phrases = 20
        if len(phrases) > max_phrases:
            # Take evenly distributed samples
            indices = np.linspace(0, len(phrases) - 1, max_phrases, dtype=int)
            phrases = [phrases[i] for i in indices]
            
        return phrases
    
    def count_exact_matches(self, phrases, text):
        """Count how many phrases have exact matches in the text."""
        if not phrases or not text:
            return 0
            
        # Clean the text
        clean_text = self.clean_text(text)
        
        # Count matches
        matches = 0
        for phrase in phrases:
            if phrase in clean_text:
                matches += 1
                
        return matches

    def evaluate_conversation_by_turn(self, original_file, transcript_file):
        """
        Evaluate a conversation by comparing each turn in the original and transcript.
        This provides a detailed turn-by-turn analysis rather than just overall metrics.
        
        Args:
            original_file: Path to the original JSON conversation file
            transcript_file: Path to the transcribed JSON file
            
        Returns:
            Dictionary with detailed turn-by-turn comparisons and metrics
        """
        try:
            # Load original conversation
            with open(original_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
                
            # Load transcribed conversation
            with open(transcript_file, 'r', encoding='utf-8') as f:
                if transcript_file.endswith('.json'):
                    # If it's a JSON file with conversation structure
                    transcript_data = json.load(f)
                    if isinstance(transcript_data, list):
                        # Direct list of turns
                        transcript_turns = transcript_data
                    elif "conversation" in transcript_data:
                        # Nested under conversation key
                        transcript_turns = transcript_data.get("conversation", [])
                    else:
                        # Unknown format
                        print(f"Unknown JSON format in {transcript_file}")
                        return None
                else:
                    # Not a structured JSON file, can't do turn-by-turn comparison
                    print(f"File {transcript_file} is not a structured JSON file")
                    return None
            
            # Get conversation ID from filename
            basename = os.path.basename(original_file)
            conversation_id = os.path.splitext(basename)[0]
            
            # Extract consultation type (consultation or followup)
            consultation_type = "consultation" if "consultation" in conversation_id.lower() else "followup"
            
            # Extract original conversation turns
            original_turns = original_data.get("conversation", [])
            
            # Skip evaluation if original conversation is empty
            if not original_turns:
                print(f"Warning: Empty original conversation in {conversation_id}")
                return None
                
            # Extract overall conversation details
            condition = original_data.get("condition", "")
            
            # Prepare results dictionary
            results = {
                "conversation_id": conversation_id,
                "specialty": self.specialty,
                "language": self.language,
                "consultation_type": consultation_type,
                "condition": condition,
                "turn_comparisons": [],
                "metrics": {}
            }
            
            # Compare turns
            # We'll compare as many turns as we can match between original and transcript
            max_turns = min(len(original_turns), len(transcript_turns))
            
            overall_wer = []
            overall_similarity = []
            overall_medical_accuracy = []
            speaker_matches = 0
            
            for i in range(max_turns):
                original_turn = original_turns[i]
                transcript_turn = transcript_turns[i]
                
                # Extract texts and speakers
                original_text = original_turn.get("text", "")
                transcript_text = transcript_turn.get("text", "")
                original_speaker = original_turn.get("speaker", "").lower()
                transcript_speaker = transcript_turn.get("speaker", "").lower()
                
                # Calculate metrics for this turn
                wer = self.calculate_wer(original_text, transcript_text)
                similarity = self.calculate_similarity(original_text, transcript_text)
                medical_accuracy = self.calculate_medical_term_accuracy(original_text, transcript_text)
                is_speaker_match = original_speaker == transcript_speaker
                
                if is_speaker_match:
                    speaker_matches += 1
                    
                # Store metrics
                overall_wer.append(wer)
                overall_similarity.append(similarity)
                if medical_accuracy is not None:  # Some turns may not have medical terms
                    overall_medical_accuracy.append(medical_accuracy)
                
                # Add comparison for this turn
                results["turn_comparisons"].append({
                    "turn_index": i,
                    "original_speaker": original_speaker,
                    "transcript_speaker": transcript_speaker,
                    "original_text": original_text,
                    "transcript_text": transcript_text,
                    "wer": wer,
                    "similarity": similarity,
                    "medical_accuracy": medical_accuracy,
                    "speaker_match": is_speaker_match
                })
            
            # Calculate overall metrics
            speaker_accuracy = speaker_matches / max_turns if max_turns > 0 else 0
            
            results["metrics"] = {
                "avg_wer": np.mean(overall_wer) if overall_wer else None,
                "avg_similarity": np.mean(overall_similarity) if overall_similarity else None,
                "avg_medical_accuracy": np.mean(overall_medical_accuracy) if overall_medical_accuracy else None,
                "speaker_accuracy": speaker_accuracy,
                "total_turns": max_turns,
                "matched_speakers": speaker_matches
            }
            
            # Check if this is a suspicious match (likely from original text)
            is_suspicious = self.check_suspicious_transcript(
                " ".join([t.get("text", "") for t in original_turns]),
                " ".join([t.get("text", "") for t in transcript_turns])
            )
            results["metrics"]["suspicious_match"] = is_suspicious
            
            return results
            
        except Exception as e:
            print(f"Error in turn-by-turn evaluation for {os.path.basename(original_file)}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_all_conversations_by_turn(self, json_dir, transcripts_dir):
        """
        Evaluate all conversations in the directory with detailed turn-by-turn analysis.
        
        Args:
            json_dir: Directory containing original JSON files
            transcripts_dir: Directory containing transcripts
            
        Returns:
            List of detailed evaluation results for each conversation
        """
        # Find all original JSON files
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {json_dir}")
            return []
        
        print(f"Found {len(json_files)} original conversation files in {json_dir}")
        
        # Evaluate each conversation
        results = []
        
        for json_file in tqdm(json_files, desc=f"Evaluating {self.language} {self.specialty} with turn-by-turn analysis"):
            # Extract basename
            basename = os.path.basename(json_file)
            filename = os.path.splitext(basename)[0]
            
            # Look for transcript JSON file
            transcript_file = os.path.join(transcripts_dir, f"{filename}.json")
            
            # Skip if transcript file doesn't exist
            if not os.path.exists(transcript_file):
                print(f"Warning: Transcript file not found for {basename}")
                continue
            
            # Evaluate this conversation
            evaluation = self.evaluate_conversation_by_turn(json_file, transcript_file)
            
            if evaluation:
                results.append(evaluation)
        
        return results
    
    def export_detailed_csv(self, evaluations, output_dir):
        """
        Export detailed CSV files from turn-by-turn evaluations.
        
        Args:
            evaluations: List of evaluation results from evaluate_all_conversations_by_turn
            output_dir: Directory to save CSV files
        """
        if not evaluations:
            print("No evaluations to export")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Conversation-level summary CSV
        conversation_rows = []
        for eval_result in evaluations:
            metrics = eval_result.get("metrics", {})
            conversation_rows.append({
                "conversation_id": eval_result.get("conversation_id", ""),
                "specialty": eval_result.get("specialty", ""),
                "language": eval_result.get("language", ""),
                "consultation_type": eval_result.get("consultation_type", ""),
                "condition": eval_result.get("condition", ""),
                "avg_wer": metrics.get("avg_wer", ""),
                "avg_similarity": metrics.get("avg_similarity", ""),
                "avg_medical_accuracy": metrics.get("avg_medical_accuracy", ""),
                "speaker_accuracy": metrics.get("speaker_accuracy", ""),
                "total_turns": metrics.get("total_turns", ""),
                "matched_speakers": metrics.get("matched_speakers", ""),
                "suspicious_match": metrics.get("suspicious_match", False)
            })
        
        # Save conversation summary
        conversation_df = pd.DataFrame(conversation_rows)
        conversation_df.to_csv(os.path.join(output_dir, "conversation_summary.csv"), index=False)
        
        # 2. Turn-by-turn details CSV
        turn_rows = []
        for eval_result in evaluations:
            conv_id = eval_result.get("conversation_id", "")
            specialty = eval_result.get("specialty", "")
            language = eval_result.get("language", "")
            consultation_type = eval_result.get("consultation_type", "")
            
            for turn in eval_result.get("turn_comparisons", []):
                turn_rows.append({
                    "conversation_id": conv_id,
                    "specialty": specialty,
                    "language": language, 
                    "consultation_type": consultation_type,
                    "turn_index": turn.get("turn_index", ""),
                    "original_speaker": turn.get("original_speaker", ""),
                    "transcript_speaker": turn.get("transcript_speaker", ""),
                    "original_text": turn.get("original_text", ""),
                    "transcript_text": turn.get("transcript_text", ""),
                    "wer": turn.get("wer", ""),
                    "similarity": turn.get("similarity", ""),
                    "medical_accuracy": turn.get("medical_accuracy", ""),
                    "speaker_match": turn.get("speaker_match", "")
                })
        
        # Save turn-by-turn details
        turn_df = pd.DataFrame(turn_rows)
        turn_df.to_csv(os.path.join(output_dir, "turn_by_turn_details.csv"), index=False)
        
        # 3. Generate aggregated analysis by consultation type
        consult_df = conversation_df[conversation_df["consultation_type"] == "consultation"]
        followup_df = conversation_df[conversation_df["consultation_type"] == "followup"]
        
        # Calculate averages by type
        consult_avg = {
            "consultation_type": "consultation",
            "count": len(consult_df),
            "avg_wer": consult_df["avg_wer"].mean(),
            "avg_similarity": consult_df["avg_similarity"].mean(),
            "avg_medical_accuracy": consult_df["avg_medical_accuracy"].mean(),
            "avg_speaker_accuracy": consult_df["speaker_accuracy"].mean()
        }
        
        followup_avg = {
            "consultation_type": "followup",
            "count": len(followup_df),
            "avg_wer": followup_df["avg_wer"].mean(),
            "avg_similarity": followup_df["avg_similarity"].mean(),
            "avg_medical_accuracy": followup_df["avg_medical_accuracy"].mean(),
            "avg_speaker_accuracy": followup_df["speaker_accuracy"].mean()
        }
        
        # Save consultation type comparison
        type_df = pd.DataFrame([consult_avg, followup_avg])
        type_df.to_csv(os.path.join(output_dir, "consultation_type_comparison.csv"), index=False)
        
        print(f"Exported detailed CSV files to {output_dir}")
        
        return {
            "conversation_df": conversation_df,
            "turn_df": turn_df,
            "type_df": type_df
        }

    def create_visualizations(self, detailed_results, output_dir, model_name="Unknown Model"):
        """
        Create visualizations from detailed evaluation results.
        
        Args:
            detailed_results: Results from export_detailed_csv
            output_dir: Directory to save visualization images
            model_name: Name of the model used for transcription
        """
        os.makedirs(output_dir, exist_ok=True)
        
        conversation_df = detailed_results.get("conversation_df", pd.DataFrame())
        turn_df = detailed_results.get("turn_df", pd.DataFrame())
        type_df = detailed_results.get("type_df", pd.DataFrame())
        
        if conversation_df.empty or turn_df.empty:
            print("Not enough data for visualizations")
            return
        
        # Set visualization style
        plt.style.use('ggplot')
        sns.set_style('whitegrid')
        
        # 1. Consultation Type Comparison - All metrics
        if not type_df.empty and len(type_df) >= 2:
            plt.figure(figsize=(12, 8))
            
            # Data preparation
            metrics = ["avg_wer", "avg_similarity", "avg_medical_accuracy", "avg_speaker_accuracy"]
            metric_names = ["Word Error Rate", "Text Similarity", "Medical Term Accuracy", "Speaker Accuracy"]
            consultation_data = type_df[type_df["consultation_type"] == "consultation"]
            followup_data = type_df[type_df["consultation_type"] == "followup"]
            
            bar_width = 0.35
            x = np.arange(len(metrics))
            
            # Plot bars for each consultation type
            consultation_values = [consultation_data[m].values[0] if not consultation_data.empty else 0 for m in metrics]
            followup_values = [followup_data[m].values[0] if not followup_data.empty else 0 for m in metrics]
            
            plt.bar(x - bar_width/2, consultation_values, bar_width, label='Consultation', color='#3498db')
            plt.bar(x + bar_width/2, followup_values, bar_width, label='Follow-up', color='#e74c3c')
            
            # Add labels and formatting
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title(f'{self.language} {self.specialty} ({model_name}) - Consultation vs Follow-up')
            plt.xticks(x, metric_names)
            plt.legend()
            
            # Add values on top of bars
            for i, v in enumerate(consultation_values):
                plt.text(i - bar_width/2, v + 0.02, f"{v:.2f}", ha='center')
            for i, v in enumerate(followup_values):
                plt.text(i + bar_width/2, v + 0.02, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{self.language}_{self.specialty}_consultation_type_comparison.png"), dpi=300)
            plt.close()
        
        # 2. Medical Term Accuracy by Specialty
        plt.figure(figsize=(10, 6))
        if self.specialty == "cardiology" or self.specialty == "gp":
            specialty_value = conversation_df["avg_medical_accuracy"].mean()
            plt.bar([self.specialty], [specialty_value], color="#3498db")
            plt.ylabel("Medical Term Accuracy")
            plt.title(f"Medical Term Accuracy for {self.specialty.upper()} using {model_name}")
            plt.ylim(0, 1.1)
            
            # Add value on bar
            plt.text(0, specialty_value + 0.02, f"{specialty_value:.2f}", ha="center")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{self.language}_{self.specialty}_medical_accuracy.png"), dpi=300)
            plt.close()
        
        # 3. Turn-by-turn WER Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(turn_df["wer"], kde=True, bins=20)
        plt.axvline(x=turn_df["wer"].mean(), color='r', linestyle='--', 
                   label=f'Mean WER: {turn_df["wer"].mean():.4f}')
        plt.title(f'{self.language} {self.specialty} - Turn-by-Turn WER Distribution ({model_name})')
        plt.xlabel('Word Error Rate (WER)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.language}_{self.specialty}_turn_wer_distribution.png"), dpi=300)
        plt.close()
        
        # 4. Speaker Matching Distribution
        plt.figure(figsize=(10, 6))
        speaker_match_counts = turn_df["speaker_match"].value_counts()
        labels = ["Correct", "Incorrect"]
        sizes = [
            speaker_match_counts.get(True, 0), 
            speaker_match_counts.get(False, 0)
        ]
        colors = ['#2ecc71', '#e74c3c']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'{self.language} {self.specialty} - Speaker Identification Accuracy ({model_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{self.language}_{self.specialty}_speaker_accuracy.png"), dpi=300)
        plt.close()
        
        # 5. Medical Term Accuracy Distribution by Turn
        plt.figure(figsize=(12, 6))
        # Filter out NaN values
        med_accuracy = turn_df["medical_accuracy"].dropna()
        if len(med_accuracy) > 0:
            sns.histplot(med_accuracy, kde=True, bins=20)
            plt.axvline(x=med_accuracy.mean(), color='r', linestyle='--', 
                       label=f'Mean: {med_accuracy.mean():.4f}')
            plt.title(f'{self.language} {self.specialty} - Medical Term Accuracy ({model_name})')
            plt.xlabel('Medical Term Accuracy')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{self.language}_{self.specialty}_medical_accuracy_distribution.png"), dpi=300)
            plt.close()
        
        return True

def evaluate_specialty(specialty, language="all", base_dir=DEFAULT_BASE_DIR, dataset_name=""):
    """Evaluate all conversations for a given specialty and language."""
    languages = ["en-CA", "fr-CA"] if language == "all" else [language]
    all_results = []
    
    for lang in languages:
        # Set up directories
        json_dir = os.path.join(base_dir, specialty, lang, "json")
        transcripts_dir = os.path.join(base_dir, specialty, lang, "transcripts")
        evaluation_dir = os.path.join(base_dir, "evaluation")
        output_dir = os.path.join(evaluation_dir, specialty)
        
        # Check if directories exist
        if not os.path.exists(json_dir):
            print(f"JSON directory not found: {json_dir}")
            continue
            
        if not os.path.exists(transcripts_dir):
            print(f"Transcripts directory not found: {transcripts_dir}")
            continue
        
        # Create evaluator
        evaluator = TranscriptionEvaluator(specialty, lang, base_dir, dataset_name)
        
        # Evaluate conversations
        results = evaluator.evaluate_all_conversations(json_dir, transcripts_dir)
        all_results.extend(results)
        
        # Print summary
        evaluator.print_summary()
        
        # Plot results
        evaluator.plot_results(output_dir)
    
    return all_results

def evaluate_specialty_detailed(specialty, language="all", base_dir=DEFAULT_BASE_DIR, dataset_name=""):
    """
    Evaluate all conversations for a given specialty and language with detailed turn-by-turn analysis.
    
    Args:
        specialty: Medical specialty (cardiology, gp)
        language: Language code (en-CA, fr-CA, all)
        base_dir: Base directory for data
        dataset_name: Name of the dataset for reporting
    
    Returns:
        Dictionary with detailed results
    """
    languages = ["en-CA", "fr-CA"] if language == "all" else [language]
    all_results = {}
    
    for lang in languages:
        # Set up directories
        json_dir = os.path.join(base_dir, specialty, lang, "json")
        transcripts_dir = os.path.join(base_dir, specialty, lang, "transcripts")
        evaluation_dir = os.path.join(base_dir, "evaluation")
        output_dir = os.path.join(evaluation_dir, f"{dataset_name}_{specialty}_{lang}")
        
        # Check if directories exist
        if not os.path.exists(json_dir):
            print(f"JSON directory not found: {json_dir}")
            continue
            
        if not os.path.exists(transcripts_dir):
            print(f"Transcripts directory not found: {transcripts_dir}")
            continue
        
        # Extract model name from transcripts directory structure
        model_name = "Unknown Model"
        if os.path.exists(transcripts_dir):
            # Try to extract model name from the directory path
            # Example: if the path includes "nova3-medical" or similar, extract it
            dir_parts = transcripts_dir.split(os.sep)
            for part in dir_parts:
                if "nova" in part.lower() or "azure" in part.lower() or "deepgram" in part.lower():
                    model_name = part
                    break
            
            # If not found in directory path, try to look at subdirectories
            if model_name == "Unknown Model":
                subfolders = [f.name for f in os.scandir(os.path.dirname(transcripts_dir)) if f.is_dir()]
                for folder in subfolders:
                    if "nova" in folder.lower() or "azure" in folder.lower() or "deepgram" in folder.lower():
                        model_name = folder
                        break
        
        # Create evaluator
        evaluator = TranscriptionEvaluator(specialty, lang, base_dir, dataset_name)
        
        # Evaluate conversations with detailed turn-by-turn analysis
        print(f"\nPerforming detailed turn-by-turn evaluation for {lang} {specialty} using {model_name}...")
        results = evaluator.evaluate_all_conversations_by_turn(json_dir, transcripts_dir)
        
        if not results:
            print(f"No valid results for {lang} {specialty}")
            continue
            
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Export detailed CSV files
        print(f"Exporting detailed CSV files for {lang} {specialty} using {model_name}...")
        detailed_results = evaluator.export_detailed_csv(results, output_dir)
        
        # Add model name to the results
        if "conversation_df" in detailed_results:
            detailed_results["conversation_df"]["model"] = model_name
        
        # Create visualizations
        print(f"Creating visualizations for {lang} {specialty} using {model_name}...")
        evaluator.create_visualizations(detailed_results, output_dir, model_name)
        
        # Store results for comparative visualizations
        all_results[(specialty, lang, model_name)] = detailed_results
        
        # Print summary
        evaluator.print_summary()
    
    return all_results

def evaluate_all_datasets_detailed(data_dir="all-data", specialties=["cardiology", "gp"]):
    """
    Evaluate all datasets in the data directory with detailed turn-by-turn analysis.
    
    Args:
        data_dir: Directory containing multiple datasets to evaluate
        specialties: List of specialties to evaluate
    
    Returns:
        Dictionary with all evaluation results
    """
    # Get all dataset directories
    dataset_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    
    if not dataset_dirs:
        print(f"No dataset directories found in {data_dir}")
        return {}
    
    print(f"Found {len(dataset_dirs)} dataset directories in {data_dir}")
    
    # Store all results
    all_results = {}
    
    # Evaluate each dataset
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        print(f"\n\n===== Evaluating dataset: {dataset_name} =====")
        
        for specialty in specialties:
            # Evaluate this specialty
            results = evaluate_specialty_detailed(specialty, "all", dataset_dir, dataset_name)
            all_results.update(results)
    
    # Create comparative visualizations
    print("\nCreating comparative visualizations across specialties and languages...")
    output_dir = os.path.join(data_dir, "evaluation", "comparative")
    create_comparative_visualizations(all_results, output_dir)
    
    return all_results

def create_comparative_visualizations(all_results, output_dir):
    """
    Create comparative visualizations across specialties, languages, and models.
    
    Args:
        all_results: Dictionary with keys as (specialty, language, model) tuples and dataframes as values
        output_dir: Directory to save visualization images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export comprehensive CSV with all model comparisons
    comprehensive_rows = []
    
    for (specialty, language, model), results in all_results.items():
        if "conversation_df" in results:
            df = results["conversation_df"]
            
            # Calculate averages
            avg_metrics = {
                "specialty": specialty,
                "language": language,
                "model": model,
                "avg_wer": df["avg_wer"].mean(),
                "avg_similarity": df["avg_similarity"].mean(),
                "avg_medical_accuracy": df["avg_medical_accuracy"].mean(),
                "speaker_accuracy": df["speaker_accuracy"].mean(),
                "count": len(df)
            }
            
            comprehensive_rows.append(avg_metrics)
    
    # Create and save comprehensive CSV
    if comprehensive_rows:
        comprehensive_df = pd.DataFrame(comprehensive_rows)
        comprehensive_df.to_csv(os.path.join(output_dir, "model_comparison_comprehensive.csv"), index=False)
        print(f"Saved comprehensive model comparison to {os.path.join(output_dir, 'model_comparison_comprehensive.csv')}")
    
    # Prepare data for specialty comparison
    models = sorted(set(result[2] for result in all_results.keys()))
    specialties = sorted(set(result[0] for result in all_results.keys()))
    languages = sorted(set(result[1] for result in all_results.keys()))
    
    # Set visualization style
    plt.style.use('ggplot')
    sns.set_style('whitegrid')
    
    # 1. Model Comparison (across all specialties and languages)
    model_metrics = []
    
    for model in models:
        model_results = {key: val for key, val in all_results.items() if key[2] == model}
        
        # Calculate average metrics for this model
        model_avg = {
            "model": model,
            "avg_wer": [],
            "avg_similarity": [],
            "avg_medical_accuracy": [],
            "speaker_accuracy": []
        }
        
        for _, results in model_results.items():
            if "conversation_df" in results:
                df = results["conversation_df"]
                model_avg["avg_wer"].append(df["avg_wer"].mean())
                model_avg["avg_similarity"].append(df["avg_similarity"].mean())
                model_avg["avg_medical_accuracy"].append(df["avg_medical_accuracy"].mean())
                model_avg["speaker_accuracy"].append(df["speaker_accuracy"].mean())
        
        # Calculate final averages
        for metric in ["avg_wer", "avg_similarity", "avg_medical_accuracy", "speaker_accuracy"]:
            if model_avg[metric]:
                model_avg[metric] = np.mean(model_avg[metric])
            else:
                model_avg[metric] = 0
        
        model_metrics.append(model_avg)
    
    # Create model comparison plot
    if model_metrics:
        plt.figure(figsize=(14, 8))
        
        metrics = ["avg_wer", "avg_similarity", "avg_medical_accuracy", "speaker_accuracy"]
        metric_names = ["Word Error Rate", "Text Similarity", "Medical Term Accuracy", "Speaker Accuracy"]
        
        x = np.arange(len(metrics))
        bar_width = 0.8 / len(model_metrics)
        
        for i, model_data in enumerate(model_metrics):
            offset = i * bar_width - (len(model_metrics) * bar_width / 2) + bar_width / 2
            values = [model_data[m] for m in metrics]
            
            plt.bar(x + offset, values, bar_width, label=model_data["model"])
            
            # Add values on top of bars
            for j, v in enumerate(values):
                plt.text(j + offset, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
        
        # Add labels and formatting
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Model Comparison - Transcription Performance')
        plt.xticks(x, metric_names)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
        plt.close()
    
    # 2. Model Comparison by Language
    for language in languages:
        language_model_metrics = []
        
        for model in models:
            model_lang_results = {key: val for key, val in all_results.items() if key[1] == language and key[2] == model}
            
            # Calculate average metrics for this model and language
            model_avg = {
                "model": model,
                "language": language,
                "avg_wer": [],
                "avg_similarity": [],
                "avg_medical_accuracy": [],
                "speaker_accuracy": []
            }
            
            for _, results in model_lang_results.items():
                if "conversation_df" in results:
                    df = results["conversation_df"]
                    model_avg["avg_wer"].append(df["avg_wer"].mean())
                    model_avg["avg_similarity"].append(df["avg_similarity"].mean())
                    model_avg["avg_medical_accuracy"].append(df["avg_medical_accuracy"].mean())
                    model_avg["speaker_accuracy"].append(df["speaker_accuracy"].mean())
            
            # Calculate final averages
            for metric in ["avg_wer", "avg_similarity", "avg_medical_accuracy", "speaker_accuracy"]:
                if model_avg[metric]:
                    model_avg[metric] = np.mean(model_avg[metric])
                else:
                    model_avg[metric] = 0
            
            language_model_metrics.append(model_avg)
        
        # Create language-specific model comparison plot
        if language_model_metrics:
            plt.figure(figsize=(14, 8))
            
            metrics = ["avg_wer", "avg_similarity", "avg_medical_accuracy", "speaker_accuracy"]
            metric_names = ["Word Error Rate", "Text Similarity", "Medical Term Accuracy", "Speaker Accuracy"]
            
            x = np.arange(len(metrics))
            bar_width = 0.8 / len(language_model_metrics)
            
            for i, model_data in enumerate(language_model_metrics):
                offset = i * bar_width - (len(language_model_metrics) * bar_width / 2) + bar_width / 2
                values = [model_data[m] for m in metrics]
                
                plt.bar(x + offset, values, bar_width, label=model_data["model"])
                
                # Add values on top of bars
                for j, v in enumerate(values):
                    plt.text(j + offset, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
            
            # Add labels and formatting
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title(f'Model Comparison for {language} - Transcription Performance')
            plt.xticks(x, metric_names)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"model_comparison_{language}.png"), dpi=300)
            plt.close()
    
    # Continue with existing visualizations...
    
    # Prepare data for specialty comparison
    cardio_results = {(lang, model): results for (specialty, lang, model), results in all_results.items() if specialty == "cardiology"}
    gp_results = {(lang, model): results for (specialty, lang, model), results in all_results.items() if specialty == "gp"}
    
    # Prepare data for language comparison
    en_results = {(specialty, model): results for (specialty, lang, model), results in all_results.items() if lang == "en-CA"}
    fr_results = {(specialty, model): results for (specialty, lang, model), results in all_results.items() if lang == "fr-CA"}
    
    # 3. Specialty Comparison (Cardiology vs GP)
    if cardio_results and gp_results:
        plt.figure(figsize=(12, 8))
        
        # Collect average metrics
        metrics = ["avg_wer", "avg_similarity", "avg_medical_accuracy", "speaker_accuracy"]
        metric_names = ["Word Error Rate", "Text Similarity", "Medical Term Accuracy", "Speaker Accuracy"]
        
        cardio_values = []
        gp_values = []
        
        # Calculate average across languages for each specialty
        for metric in metrics:
            # Cardiology average
            cardio_metric_values = []
            for (lang, model), results in cardio_results.items():
                if "conversation_df" in results:
                    cardio_metric_values.append(results["conversation_df"][metric].mean())
            cardio_values.append(np.mean(cardio_metric_values) if cardio_metric_values else 0)
            
            # GP average
            gp_metric_values = []
            for (lang, model), results in gp_results.items():
                if "conversation_df" in results:
                    gp_metric_values.append(results["conversation_df"][metric].mean())
            gp_values.append(np.mean(gp_metric_values) if gp_metric_values else 0)
        
        # Plot bars
        bar_width = 0.35
        x = np.arange(len(metrics))
        
        plt.bar(x - bar_width/2, cardio_values, bar_width, label='Cardiology', color='#3498db')
        plt.bar(x + bar_width/2, gp_values, bar_width, label='General Practice', color='#e74c3c')
        
        # Add labels and formatting
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Cardiology vs General Practice - Transcription Performance')
        plt.xticks(x, metric_names)
        plt.legend()
        
        # Add values on top of bars
        for i, v in enumerate(cardio_values):
            plt.text(i - bar_width/2, v + 0.02, f"{v:.2f}", ha='center')
        for i, v in enumerate(gp_values):
            plt.text(i + bar_width/2, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "specialty_comparison.png"), dpi=300)
        plt.close()
    
    # 4. Language Comparison (English vs French)
    if en_results and fr_results:
        plt.figure(figsize=(12, 8))
        
        # Collect average metrics
        metrics = ["avg_wer", "avg_similarity", "avg_medical_accuracy", "speaker_accuracy"]
        metric_names = ["Word Error Rate", "Text Similarity", "Medical Term Accuracy", "Speaker Accuracy"]
        
        en_values = []
        fr_values = []
        
        # Calculate average across specialties for each language
        for metric in metrics:
            # English average
            en_metric_values = []
            for (specialty, model), results in en_results.items():
                if "conversation_df" in results:
                    en_metric_values.append(results["conversation_df"][metric].mean())
            en_values.append(np.mean(en_metric_values) if en_metric_values else 0)
            
            # French average
            fr_metric_values = []
            for (specialty, model), results in fr_results.items():
                if "conversation_df" in results:
                    fr_metric_values.append(results["conversation_df"][metric].mean())
            fr_values.append(np.mean(fr_metric_values) if fr_metric_values else 0)
        
        # Plot bars
        bar_width = 0.35
        x = np.arange(len(metrics))
        
        plt.bar(x - bar_width/2, en_values, bar_width, label='English (en-CA)', color='#3498db')
        plt.bar(x + bar_width/2, fr_values, bar_width, label='French (fr-CA)', color='#e74c3c')
        
        # Add labels and formatting
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('English vs French - Transcription Performance')
        plt.xticks(x, metric_names)
        plt.legend()
        
        # Add values on top of bars
        for i, v in enumerate(en_values):
            plt.text(i - bar_width/2, v + 0.02, f"{v:.2f}", ha='center')
        for i, v in enumerate(fr_values):
            plt.text(i + bar_width/2, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "language_comparison.png"), dpi=300)
        plt.close()
    
    # 5. Consultation Type Comparison (across all specialties)
    # Collect data from all results
    consultation_rows = []
    followup_rows = []
    
    for (specialty, language, model), results in all_results.items():
        if "conversation_df" in results:
            df = results["conversation_df"]
            consult_df = df[df["consultation_type"] == "consultation"]
            followup_df = df[df["consultation_type"] == "followup"]
            
            if not consult_df.empty:
                consultation_rows.append({
                    "specialty": specialty,
                    "language": language,
                    "model": model,
                    "count": len(consult_df),
                    "avg_wer": consult_df["avg_wer"].mean(),
                    "avg_similarity": consult_df["avg_similarity"].mean(),
                    "avg_medical_accuracy": consult_df["avg_medical_accuracy"].mean(),
                    "speaker_accuracy": consult_df["speaker_accuracy"].mean()
                })
            
            if not followup_df.empty:
                followup_rows.append({
                    "specialty": specialty,
                    "language": language,
                    "model": model,
                    "count": len(followup_df),
                    "avg_wer": followup_df["avg_wer"].mean(),
                    "avg_similarity": followup_df["avg_similarity"].mean(),
                    "avg_medical_accuracy": followup_df["avg_medical_accuracy"].mean(),
                    "speaker_accuracy": followup_df["speaker_accuracy"].mean()
                })
    
    # Save detailed CSVs
    if consultation_rows:
        consult_df = pd.DataFrame(consultation_rows)
        consult_df.to_csv(os.path.join(output_dir, "consultation_details.csv"), index=False)
    
    if followup_rows:
        followup_df = pd.DataFrame(followup_rows)
        followup_df.to_csv(os.path.join(output_dir, "followup_details.csv"), index=False)
    
    if consultation_rows and followup_rows:
        consult_df = pd.DataFrame(consultation_rows)
        followup_df = pd.DataFrame(followup_rows)
        
        # Calculate overall averages
        consult_avg = {
            "avg_wer": consult_df["avg_wer"].mean(),
            "avg_similarity": consult_df["avg_similarity"].mean(),
            "avg_medical_accuracy": consult_df["avg_medical_accuracy"].mean(),
            "speaker_accuracy": consult_df["speaker_accuracy"].mean()
        }
        
        followup_avg = {
            "avg_wer": followup_df["avg_wer"].mean(),
            "avg_similarity": followup_df["avg_similarity"].mean(),
            "avg_medical_accuracy": followup_df["avg_medical_accuracy"].mean(),
            "speaker_accuracy": followup_df["speaker_accuracy"].mean()
        }
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        metrics = ["avg_wer", "avg_similarity", "avg_medical_accuracy", "speaker_accuracy"]
        metric_names = ["Word Error Rate", "Text Similarity", "Medical Term Accuracy", "Speaker Accuracy"]
        
        consult_values = [consult_avg[m] for m in metrics]
        followup_values = [followup_avg[m] for m in metrics]
        
        bar_width = 0.35
        x = np.arange(len(metrics))
        
        plt.bar(x - bar_width/2, consult_values, bar_width, label='Consultation', color='#3498db')
        plt.bar(x + bar_width/2, followup_values, bar_width, label='Follow-up', color='#e74c3c')
        
        # Add labels and formatting
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Consultation vs Follow-up - Transcription Performance (All Specialties)')
        plt.xticks(x, metric_names)
        plt.legend()
        
        # Add values on top of bars
        for i, v in enumerate(consult_values):
            plt.text(i - bar_width/2, v + 0.02, f"{v:.2f}", ha='center')
        for i, v in enumerate(followup_values):
            plt.text(i + bar_width/2, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "consultation_type_comparison_all.png"), dpi=300)
        plt.close()
    
    print(f"Comparative visualizations and CSV files saved to {output_dir}")
    return True

def generate_report(all_results, output_file="README_evaluation.md"):
    """
    Generate a detailed evaluation report in markdown format.
    
    Args:
        all_results: Dictionary with all evaluation results
        output_file: Path to save the markdown report
    """
    # Collect metrics across all datasets
    all_metrics = []
    
    for (specialty, language, model), results in all_results.items():
        if "conversation_df" in results:
            df = results["conversation_df"]
            
            # Calculate average metrics
            avg_metrics = {
                "specialty": specialty,
                "language": language,
                "model": model,
                "wer": df["avg_wer"].mean(),
                "similarity": df["avg_similarity"].mean(),
                "medical_accuracy": df["avg_medical_accuracy"].mean(),
                "speaker_accuracy": df["speaker_accuracy"].mean()
            }
            
            all_metrics.append(avg_metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Start building the report
    report = []
    report.append("# Transcription Evaluation Report\n")
    report.append("## Overview\n")
    report.append("This report provides a comprehensive analysis of the transcription accuracy across different specialties, languages, and models.\n")
    
    # Add overall metrics summary
    report.append("## Overall Metrics Summary\n")
    report.append("| Specialty | Language | Model | WER | Similarity | Medical Accuracy | Speaker Accuracy |\n")
    report.append("|-----------|----------|-------|-----|------------|------------------|------------------|\n")
    
    for _, row in metrics_df.iterrows():
        report.append(f"| {row['specialty']} | {row['language']} | {row['model']} | {row['wer']:.4f} | {row['similarity']:.4f} | {row['medical_accuracy']:.4f} | {row['speaker_accuracy']:.4f} |\n")
    
    # Add model comparison visualization
    report.append("\n## Model Comparison\n")
    report.append("Comparison of transcription accuracy between different models.\n")
    report.append("![Model Comparison](all-data/evaluation/comparative/model_comparison.png)\n")
    
    # Add language-specific model comparison
    report.append("\n## Language-Specific Model Comparison\n")
    
    # Get unique languages
    languages = metrics_df["language"].unique()
    for lang in languages:
        report.append(f"\n### {lang} Model Comparison\n")
        report.append(f"Comparison of models for {lang} language.\n")
        report.append(f"![{lang} Model Comparison](all-data/evaluation/comparative/model_comparison_{lang}.png)\n")
    
    # Add specialty comparison
    report.append("\n## Specialty Comparison\n")
    report.append("Comparison of transcription accuracy between Cardiology and General Practice specialties.\n")
    report.append("![Specialty Comparison](all-data/evaluation/comparative/specialty_comparison.png)\n")
    
    # Add consultation type comparison
    report.append("\n## Consultation vs Follow-up Comparison\n")
    report.append("Comparison of transcription accuracy between initial consultations and follow-up appointments.\n")
    report.append("![Consultation Type Comparison](all-data/evaluation/comparative/consultation_type_comparison_all.png)\n")
    
    # Add language comparison
    report.append("\n## Language Comparison\n")
    report.append("Comparison of transcription accuracy between English and French languages.\n")
    report.append("![Language Comparison](all-data/evaluation/comparative/language_comparison.png)\n")
    
    # Group models by performance
    report.append("\n## Model Performance Ranking\n")
    
    # Group by model and calculate average performance metrics
    model_perf = metrics_df.groupby("model").agg({
        "wer": "mean",
        "similarity": "mean", 
        "medical_accuracy": "mean",
        "speaker_accuracy": "mean"
    }).reset_index()
    
    # Sort by medical accuracy (higher is better)
    model_perf = model_perf.sort_values(by="medical_accuracy", ascending=False)
    
    report.append("Models ranked by average medical term accuracy:\n\n")
    report.append("| Rank | Model | Medical Accuracy | Speaker Accuracy | Similarity | WER |\n")
    report.append("|------|-------|------------------|------------------|------------|-----|\n")
    
    for idx, row in model_perf.iterrows():
        report.append(f"| {idx+1} | {row['model']} | {row['medical_accuracy']:.4f} | {row['speaker_accuracy']:.4f} | {row['similarity']:.4f} | {row['wer']:.4f} |\n")
    
    # Add key findings
    report.append("\n## Key Findings\n")
    
    # Calculate some key statistics for findings
    cardio_med = metrics_df[metrics_df["specialty"] == "cardiology"]["medical_accuracy"].mean()
    gp_med = metrics_df[metrics_df["specialty"] == "gp"]["medical_accuracy"].mean()
    
    en_med = metrics_df[metrics_df["language"] == "en-CA"]["medical_accuracy"].mean()
    fr_med = metrics_df[metrics_df["language"] == "fr-CA"]["medical_accuracy"].mean()
    
    # Get best model overall
    best_model = model_perf.iloc[0]["model"]
    best_med_acc = model_perf.iloc[0]["medical_accuracy"]
    
    # Get best model for each language
    best_en_model = metrics_df[metrics_df["language"] == "en-CA"].groupby("model").agg({"medical_accuracy": "mean"}).sort_values(by="medical_accuracy", ascending=False)
    best_fr_model = metrics_df[metrics_df["language"] == "fr-CA"].groupby("model").agg({"medical_accuracy": "mean"}).sort_values(by="medical_accuracy", ascending=False)
    
    if not best_en_model.empty:
        best_en_model_name = best_en_model.index[0]
        best_en_acc = best_en_model["medical_accuracy"].iloc[0]
        report.append(f"1. **Best Model for English**: {best_en_model_name} achieved the highest medical accuracy ({best_en_acc:.4f}) for English transcriptions.\n")
    
    if not best_fr_model.empty:
        best_fr_model_name = best_fr_model.index[0]
        best_fr_acc = best_fr_model["medical_accuracy"].iloc[0]
        report.append(f"2. **Best Model for French**: {best_fr_model_name} achieved the highest medical accuracy ({best_fr_acc:.4f}) for French transcriptions.\n")
    
    report.append(f"3. **Overall Best Model**: {best_model} achieved the highest average medical accuracy ({best_med_acc:.4f}) across all specialties and languages.\n")
    
    # Add findings based on actual results
    report.append("4. **Medical Terminology by Specialty**:\n")
    if cardio_med > gp_med:
        report.append(f"   - Cardiology transcriptions showed higher medical term accuracy ({cardio_med:.4f}) compared to General Practice ({gp_med:.4f}).\n")
    else:
        report.append(f"   - General Practice transcriptions showed higher medical term accuracy ({gp_med:.4f}) compared to Cardiology ({cardio_med:.4f}).\n")
    
    report.append("5. **Language Performance**:\n")
    if en_med > fr_med:
        report.append(f"   - English transcriptions had higher accuracy ({en_med:.4f}) compared to French ({fr_med:.4f}).\n")
    else:
        report.append(f"   - French transcriptions had higher accuracy ({fr_med:.4f}) compared to English ({en_med:.4f}).\n")
    
    report.append("6. **Speaker Identification**:\n")
    speaker_acc = metrics_df["speaker_accuracy"].mean()
    report.append(f"   - The overall speaker identification accuracy was {speaker_acc:.4f}, indicating {'excellent' if speaker_acc > 0.9 else 'good' if speaker_acc > 0.7 else 'moderate' if speaker_acc > 0.5 else 'poor'} performance in distinguishing between doctor and patient.\n")
    
    # Add model-specific findings
    report.append("7. **Model Performance Analysis**:\n")
    for idx, row in model_perf.iterrows():
        model_name = row["model"]
        med_acc = row["medical_accuracy"]
        speaker_acc = row["speaker_accuracy"]
        performance = 'excellent' if med_acc > 0.9 else 'good' if med_acc > 0.7 else 'moderate' if med_acc > 0.5 else 'poor'
        
        report.append(f"   - **{model_name}**: Showed {performance} performance with medical accuracy of {med_acc:.4f} and speaker accuracy of {speaker_acc:.4f}.\n")
    
    # Add conclusions
    report.append("\n## Conclusions\n")
    report.append("The evaluation reveals several important insights about transcription performance across different models:\n\n")
    
    # Overall performance assessment
    overall_med = metrics_df["medical_accuracy"].mean()
    report.append(f"1. **Overall Performance**: The transcription systems achieved an average medical term accuracy of {overall_med:.4f}, which is {'excellent' if overall_med > 0.9 else 'good' if overall_med > 0.7 else 'moderate' if overall_med > 0.5 else 'poor'}.\n")
    
    # Model-specific recommendations
    report.append(f"2. **Model Recommendations**: Based on the evaluation, {best_model} offers the best overall performance across specialties and languages. For language-specific tasks, use {best_en_model_name if not best_en_model.empty else 'N/A'} for English and {best_fr_model_name if not best_fr_model.empty else 'N/A'} for French.\n")
    
    # Specialty-specific challenges
    report.append("3. **Specialty-Specific Challenges**: The systems performed differently across medical specialties, likely due to the varying complexity of terminology.\n")
    
    # Language considerations
    report.append("4. **Language Considerations**: There are notable differences in transcription accuracy between English and French, highlighting the need for language-specific optimization.\n")
    
    # Write report to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("".join(report))
    
    print(f"Evaluation report saved to {output_file}")
    
    return "".join(report)

def main():
    """Main function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate transcription accuracy with detailed analysis')
    parser.add_argument('--specialty', type=str, choices=['cardiology', 'gp', 'all'], default='all', 
                        help='Medical specialty to evaluate')
    parser.add_argument('--lang', type=str, choices=['en-CA', 'fr-CA', 'all'], default='all',
                        help='Language to evaluate')
    parser.add_argument('--base-dir', type=str, default=DEFAULT_BASE_DIR,
                        help='Base directory for data')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing multiple datasets to evaluate')
    parser.add_argument('--report', action='store_true',
                        help='Generate a detailed markdown report')
    args = parser.parse_args()
    
    # Check if we should process multiple datasets
    if args.data_dir:
        all_results = evaluate_all_datasets_detailed(args.data_dir)
        
        if args.report:
            generate_report(all_results)
    else:
        # Ensure evaluation output directory exists
        os.makedirs(os.path.join(args.base_dir, "evaluation"), exist_ok=True)
        
        # Evaluate based on arguments
        if args.specialty == "all":
            specialties = ["cardiology", "gp"]
            all_results = {}
            
            for specialty in specialties:
                results = evaluate_specialty_detailed(specialty, args.lang, args.base_dir)
                all_results.update(results)
                
            # Create comparative visualizations
            output_dir = os.path.join(args.base_dir, "evaluation", "comparative")
            create_comparative_visualizations(all_results, output_dir)
            
            if args.report:
                generate_report(all_results, os.path.join(args.base_dir, "README_evaluation.md"))
        else:
            evaluate_specialty_detailed(args.specialty, args.lang, args.base_dir)
    
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main() 