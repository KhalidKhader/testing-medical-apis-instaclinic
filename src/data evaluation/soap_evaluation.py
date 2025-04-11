import os
import re
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import spacy
import pandas as pd
from collections import Counter
import seaborn as sns

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set of medical stopwords to exclude from comparison
MEDICAL_STOPWORDS = set([
    'patient', 'doctor', 'clinic', 'hospital', 'medical', 'health', 'wellness',
    'physician', 'visit', 'follow', 'follow-up', 'examination', 'exam', 'mg',
    'tablet', 'capsule', 'daily', 'twice', 'symptoms', 'diagnosis', 'treatment',
    'plan', 'assessment', 'objective', 'subjective', 'soap'
])

# Try to load spacy model for medical term extraction
try:
    nlp = spacy.load("en_core_web_md")
except:
    try:
        # If the model isn't installed, attempt to install it
        import subprocess
        subprocess.call(["python", "-m", "spacy", "download", "en_core_web_md"])
        nlp = spacy.load("en_core_web_md")
    except:
        print("Warning: SpaCy model 'en_core_web_md' could not be loaded. Some metrics may be unavailable.")
        nlp = None

# Function to calculate Word Error Rate (WER)
def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate between two texts."""
    if not reference or not hypothesis:
        return 1.0  # Complete error if either text is empty
    
    # Tokenize and lowercase the texts
    reference_words = reference.lower().split()
    hypothesis_words = hypothesis.lower().split()
    
    # Calculate Levenshtein distance at the word level
    distance = Levenshtein.distance(reference_words, hypothesis_words)
    
    # WER is the ratio of the Levenshtein distance to the length of the reference
    return distance / max(len(reference_words), 1)

# Function to calculate BLEU score
def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score between two texts."""
    if not reference or not hypothesis:
        return 0.0  # Zero score if either text is empty
    
    # Tokenize the texts
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    
    # BLEU score requires a list of references, so we create a single reference in a list
    references = [reference_tokens]
    
    # Use smoothing to avoid zero scores when there are no n-gram matches
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU score (using weights for 1-gram and 2-gram)
    try:
        return sentence_bleu(references, hypothesis_tokens, weights=(0.7, 0.3), smoothing_function=smoothing)
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

# Function to extract key medical phrases from text
def extract_key_phrases(text, top_n=20):
    """Extract key medical phrases from text using spaCy."""
    if not text or not nlp:
        return []
    
    # Process the text with spaCy
    doc = nlp(text.lower())
    
    # Extract noun phrases and filter by length
    phrases = [chunk.text for chunk in doc.noun_chunks if 2 <= len(chunk.text.split()) <= 5]
    
    # Count frequency of each phrase
    phrase_counts = Counter(phrases)
    
    # Return the most common phrases
    return [phrase for phrase, _ in phrase_counts.most_common(top_n)]

# Function to count exact phrase matches
def count_exact_matches(phrases, text):
    """Count how many phrases from a list appear exactly in a text."""
    if not phrases or not text:
        return 0
    
    text_lower = text.lower()
    matches = 0
    
    for phrase in phrases:
        if phrase.lower() in text_lower:
            matches += 1
    
    return matches

# Function to extract SOAP sections from notes
def extract_soap_sections(soap_note):
    """Extract the four SOAP sections from a note."""
    sections = {
        'subjective': '',
        'objective': '',
        'assessment': '',
        'plan': '',
        'keywords': ''
    }
    
    # Extract Subjective section
    subj_match = re.search(r'\*\*S – Subjective:\*\*(.*?)(?=\*\*O – Objective:\*\*|\Z)', soap_note, re.DOTALL)
    if subj_match:
        sections['subjective'] = subj_match.group(1).strip()
    
    # Extract Objective section
    obj_match = re.search(r'\*\*O – Objective:\*\*(.*?)(?=\*\*A – Assessment:\*\*|\Z)', soap_note, re.DOTALL)
    if obj_match:
        sections['objective'] = obj_match.group(1).strip()
    
    # Extract Assessment section
    assess_match = re.search(r'\*\*A – Assessment:\*\*(.*?)(?=\*\*P – Plan:\*\*|\Z)', soap_note, re.DOTALL)
    if assess_match:
        sections['assessment'] = assess_match.group(1).strip()
    
    # Extract Plan section
    plan_match = re.search(r'\*\*P – Plan:\*\*(.*?)(?=\*\*Keywords|\-\-\-|\Z)', soap_note, re.DOTALL)
    if plan_match:
        sections['plan'] = plan_match.group(1).strip()
    
    # Extract Keywords section
    keywords_match = re.search(r'\*\*Keywords \/ Tags\*\*(.*?)(?=\Z)', soap_note, re.DOTALL)
    if keywords_match:
        sections['keywords'] = keywords_match.group(1).strip()
    
    return sections

# Function to extract medical terms from a SOAP note
def extract_medical_terms(soap_note):
    """Extract medical terms from the SOAP note keywords section."""
    medical_terms = {
        'diagnoses': [],
        'treatments': [],
        'symptoms': [],
        'conditions': [],
        'allergies': []
    }
    
    # Extract from Keywords section if it exists
    keywords_match = re.search(r'\*\*Keywords \/ Tags\*\*(.*?)(?=\Z)', soap_note, re.DOTALL)
    if not keywords_match:
        return medical_terms
    
    keywords_section = keywords_match.group(1).strip()
    
    # Extract diagnoses
    diagnoses_match = re.search(r'\*\*Diagnosis\*\*:(.*?)(?=\*\*Treatment\*\*|\*\*Symptoms\*\*|\*\*Conditions\*\*|\*\*Allergies\*\*|\Z)', keywords_section, re.DOTALL)
    if diagnoses_match:
        diagnoses_text = diagnoses_match.group(1).strip()
        medical_terms['diagnoses'] = [term.strip() for term in diagnoses_text.split(',')]
    
    # Extract treatments
    treatments_match = re.search(r'\*\*Treatment\*\*:(.*?)(?=\*\*Diagnosis\*\*|\*\*Symptoms\*\*|\*\*Conditions\*\*|\*\*Allergies\*\*|\Z)', keywords_section, re.DOTALL)
    if treatments_match:
        treatments_text = treatments_match.group(1).strip()
        medical_terms['treatments'] = [term.strip() for term in treatments_text.split(',')]
    
    # Extract symptoms
    symptoms_match = re.search(r'\*\*Symptoms\*\*:(.*?)(?=\*\*Diagnosis\*\*|\*\*Treatment\*\*|\*\*Conditions\*\*|\*\*Allergies\*\*|\Z)', keywords_section, re.DOTALL)
    if symptoms_match:
        symptoms_text = symptoms_match.group(1).strip()
        medical_terms['symptoms'] = [term.strip() for term in symptoms_text.split(',')]
    
    # Extract conditions
    conditions_match = re.search(r'\*\*Conditions\*\*:(.*?)(?=\*\*Diagnosis\*\*|\*\*Treatment\*\*|\*\*Symptoms\*\*|\*\*Allergies\*\*|\Z)', keywords_section, re.DOTALL)
    if conditions_match:
        conditions_text = conditions_match.group(1).strip()
        medical_terms['conditions'] = [term.strip() for term in conditions_text.split(',')]
    
    # Extract allergies
    allergies_match = re.search(r'\*\*Allergies\*\*:(.*?)(?=\*\*Diagnosis\*\*|\*\*Treatment\*\*|\*\*Symptoms\*\*|\*\*Conditions\*\*|\Z)', keywords_section, re.DOTALL)
    if allergies_match:
        allergies_text = allergies_match.group(1).strip()
        medical_terms['allergies'] = [term.strip() for term in allergies_text.split(',')]
    
    # Clean up empty entries
    for category in medical_terms:
        medical_terms[category] = [term for term in medical_terms[category] if term and term != 'None' and term != 'None mentioned']
    
    return medical_terms

# Function to calculate cosine similarity between two texts
def calculate_similarity(text1, text2):
    """Calculate the cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    # Create a TF-IDF vectorizer with custom stopwords
    stop_words = list(stopwords.words('english')) + list(MEDICAL_STOPWORDS)
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    
    # Create the TF-IDF matrix
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

# Function to calculate set-based similarity metrics
def calculate_set_similarity(set1, set2):
    """Calculate precision, recall, and F1 score for two sets."""
    if not set1 and not set2:
        return 1.0, 1.0, 1.0  # Both empty means perfect match
    
    if not set1 or not set2:
        return 0.0, 0.0, 0.0  # One empty means no match
    
    intersection = len(set1.intersection(set2))
    precision = intersection / len(set1) if set1 else 0
    recall = intersection / len(set2) if set2 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Function to normalize and tokenize medical terms for comparison
def normalize_terms(terms_list):
    """Normalize and tokenize medical terms for better comparison."""
    if not terms_list:
        return set()
    
    normalized_terms = set()
    for term in terms_list:
        try:
            # Lowercase and remove punctuation
            term = re.sub(r'[^\w\s]', ' ', term.lower())
            
            # Use a simpler approach to tokenize instead of relying on nltk's punkt_tab
            # Just split by whitespace
            words = term.split()
            
            # Remove stopwords - use the existing stopwords if available, otherwise use an empty set
            try:
                stop_words = set(stopwords.words('english')).union(MEDICAL_STOPWORDS)
            except:
                stop_words = MEDICAL_STOPWORDS
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Add individual words and the whole term
            normalized_terms.update(filtered_words)
            if len(filtered_words) > 0:
                normalized_terms.add(' '.join(filtered_words))
        except Exception as e:
            print(f"Error normalizing term '{term}': {e}")
            continue
    
    return normalized_terms

# Function to evaluate a pair of SOAP notes
def evaluate_soap_pair(original_path, transcript_path, verbose=False):
    """Evaluate the similarity between a pair of SOAP notes."""
    try:
        # Read the SOAP notes
        with open(original_path, 'r') as f:
            original_soap = f.read()
        
        with open(transcript_path, 'r') as f:
            transcript_soap = f.read()
        
        # Extract SOAP sections
        original_sections = extract_soap_sections(original_soap)
        transcript_sections = extract_soap_sections(transcript_soap)
        
        # Extract medical terms
        original_terms = extract_medical_terms(original_soap)
        transcript_terms = extract_medical_terms(transcript_soap)
        
        # Calculate section similarities
        section_scores = {}
        for section in ['subjective', 'objective', 'assessment', 'plan']:
            similarity = calculate_similarity(original_sections[section], transcript_sections[section])
            section_scores[section] = similarity
        
        # Calculate medical term similarities
        term_metrics = {}
        term_f1_scores = {}
        
        for category in ['diagnoses', 'treatments', 'symptoms', 'conditions', 'allergies']:
            orig_terms = normalize_terms(original_terms[category])
            trans_terms = normalize_terms(transcript_terms[category])
            
            precision, recall, f1 = calculate_set_similarity(orig_terms, trans_terms)
            
            term_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            term_f1_scores[category] = f1
        
        # Calculate WER for the full notes
        wer_full = calculate_wer(original_soap, transcript_soap)
        
        # Calculate WER for each section
        wer_sections = {}
        for section in ['subjective', 'objective', 'assessment', 'plan']:
            wer_sections[section] = calculate_wer(original_sections[section], transcript_sections[section])
        
        # Calculate BLEU scores for the full notes
        bleu_full = calculate_bleu(original_soap, transcript_soap)
        
        # Calculate BLEU scores for each section
        bleu_sections = {}
        for section in ['subjective', 'objective', 'assessment', 'plan']:
            bleu_sections[section] = calculate_bleu(original_sections[section], transcript_sections[section])
        
        # Extract key phrases and count matches
        key_phrases = {}
        phrase_matches = {}
        
        for section in ['subjective', 'objective', 'assessment', 'plan']:
            if nlp:
                key_phrases[section] = extract_key_phrases(original_sections[section], top_n=10)
                matches = count_exact_matches(key_phrases[section], transcript_sections[section])
                phrase_matches[section] = matches / max(len(key_phrases[section]), 1)
            else:
                phrase_matches[section] = 0.0
        
        # Overall phrase match score
        phrase_match_avg = sum(phrase_matches.values()) / len(phrase_matches) if phrase_matches else 0
        
        # Calculate overall scores
        section_avg = sum(section_scores.values()) / len(section_scores) if section_scores else 0
        term_f1_avg = sum(term_f1_scores.values()) / len(term_f1_scores) if term_f1_scores else 0
        wer_avg = sum(wer_sections.values()) / len(wer_sections) if wer_sections else 1.0
        bleu_avg = sum(bleu_sections.values()) / len(bleu_sections) if bleu_sections else 0.0
        
        # Combine metrics - lower WER is better, so we use (1 - wer_avg)
        # Weight different metrics based on importance
        overall_score = (
            section_avg * 0.5 +            # Semantic similarity 50%
            term_f1_avg * 0.2 +            # Medical term extraction 20%
            (1 - wer_avg) * 0.15 +         # Word Error Rate (inverted) 15%
            bleu_avg * 0.1 +               # BLEU score 10%
            phrase_match_avg * 0.05        # Key phrase matches 5%
        )
        
        # Prepare results
        results = {
            'file_name': os.path.basename(original_path),
            'section_scores': section_scores,
            'section_average': section_avg,
            'term_metrics': term_metrics,
            'term_f1_average': term_f1_avg,
            'wer_full': wer_full,
            'wer_sections': wer_sections,
            'wer_average': wer_avg,
            'bleu_full': bleu_full,
            'bleu_sections': bleu_sections,
            'bleu_average': bleu_avg,
            'phrase_matches': phrase_matches,
            'phrase_match_average': phrase_match_avg,
            'overall_score': overall_score
        }
        
        if verbose:
            print(f"Evaluation for {os.path.basename(original_path)}:")
            print(f"  Section average: {section_avg:.2f}")
            print(f"  Term F1 average: {term_f1_avg:.2f}")
            print(f"  WER average: {wer_avg:.2f} (lower is better)")
            print(f"  BLEU average: {bleu_avg:.2f}")
            print(f"  Phrase match average: {phrase_match_avg:.2f}")
            print(f"  Overall score: {overall_score:.2f}")
        
        return results
    
    except Exception as e:
        print(f"Error evaluating {os.path.basename(original_path)}: {e}")
        return None

# Function to evaluate all SOAP notes in a dataset
def evaluate_dataset(base_path, output_dir='evaluation_results', verbose=False):
    """Evaluate all SOAP note pairs in a dataset and generate reports."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    
    # Structure of directories
    specialties = ["cardiology", "gp"]
    
    for specialty in specialties:
        specialty_path = os.path.join(base_path, specialty)
        
        # Skip if specialty directory doesn't exist
        if not os.path.exists(specialty_path):
            if verbose:
                print(f"Specialty directory not found: {specialty_path}")
            continue
        
        # Find all language-model directories
        try:
            lang_model_dirs = [d for d in os.listdir(specialty_path) 
                              if os.path.isdir(os.path.join(specialty_path, d)) 
                              and (d.startswith("en-CA") or d.startswith("fr-CA"))]
        except Exception as e:
            print(f"Error listing language directories in {specialty_path}: {e}")
            continue
        
        for lang_dir in lang_model_dirs:
            lang_path = os.path.join(specialty_path, lang_dir)
            
            # Get paths to SOAP note directories
            original_path = os.path.join(lang_path, "soap", "original")
            transcript_path = os.path.join(lang_path, "soap", "from_transcripts")
            
            # Skip if either directory doesn't exist
            if not os.path.exists(original_path) or not os.path.exists(transcript_path):
                if verbose:
                    print(f"SOAP directories not found for {lang_path}")
                continue
            
            # Extract language code
            language = 'en' if lang_dir.startswith('en-CA') else 'fr'
            
            # Extract model name properly
            model_parts = lang_dir.split(' - ')
            if len(model_parts) >= 2:
                # Join all parts after the first one (which is the language code)
                model = ' '.join(model_parts[1:]).strip()
            else:
                # If we can't parse the model name from the directory, try to infer it
                dataset_name = os.path.basename(base_path)
                if "Azure for English" in dataset_name and language == "en":
                    model = "Azure"
                elif "Azure for French" in dataset_name and language == "fr":
                    model = "Azure"
                elif "Nova-2" in dataset_name and language == "fr":
                    model = "Nova-2"
                elif "Nova-3-medical" in dataset_name and language == "en":
                    model = "Nova-3-medical"
                else:
                    model = "unknown"
                
                if verbose and model == "unknown":
                    print(f"Could not extract model name from {lang_dir}, dataset: {dataset_name}")
            
            # Find all original SOAP notes
            original_files = glob.glob(os.path.join(original_path, "*.md"))
            
            for original_file in original_files:
                # Find corresponding transcript SOAP note
                base_name = os.path.basename(original_file)
                transcript_file = os.path.join(transcript_path, base_name)
                
                # Skip if transcript SOAP note doesn't exist
                if not os.path.exists(transcript_file):
                    if verbose:
                        print(f"No matching transcript SOAP note for {base_name}")
                    continue
                
                # Evaluate the pair
                results = evaluate_soap_pair(original_file, transcript_file, verbose)
                
                if results:
                    # Add metadata
                    results['specialty'] = specialty
                    results['language'] = language
                    results['model'] = model
                    results['dataset'] = os.path.basename(base_path)
                    
                    # Add to all results
                    all_results.append(results)
    
    # Generate summary reports
    if all_results:
        generate_reports(all_results, output_dir, verbose)
    else:
        print("No SOAP notes were evaluated.")
    
    return all_results

# Function to generate evaluation reports and visualizations
def generate_reports(results, output_dir, verbose=False):
    """Generate summary reports and visualizations from evaluation results."""
    # Save all results to JSON
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate overall statistics
    overall_scores = [r['overall_score'] for r in results]
    section_scores = {
        'subjective': [r['section_scores']['subjective'] for r in results],
        'objective': [r['section_scores']['objective'] for r in results],
        'assessment': [r['section_scores']['assessment'] for r in results],
        'plan': [r['section_scores']['plan'] for r in results]
    }
    
    # Calculate averages
    avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    avg_sections = {section: sum(scores) / len(scores) if scores else 0 
                    for section, scores in section_scores.items()}
    
    # Create summary report
    summary = {
        'average_overall_score': avg_overall,
        'average_section_scores': avg_sections,
        'number_of_evaluations': len(results),
        'by_specialty': {},
        'by_language': {}
    }
    
    # Calculate scores by specialty
    specialties = set(r['specialty'] for r in results)
    for specialty in specialties:
        specialty_results = [r for r in results if r['specialty'] == specialty]
        specialty_scores = [r['overall_score'] for r in specialty_results]
        summary['by_specialty'][specialty] = sum(specialty_scores) / len(specialty_scores) if specialty_scores else 0
    
    # Calculate scores by language
    languages = set(r['language'] for r in results)
    for language in languages:
        language_results = [r for r in results if r['language'] == language]
        language_scores = [r['overall_score'] for r in language_results]
        summary['by_language'][language] = sum(language_scores) / len(language_scores) if language_scores else 0
    
    # Save summary report
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    create_visualizations(results, summary, output_dir)
    
    if verbose:
        print("\nEvaluation Summary:")
        print(f"Number of evaluations: {len(results)}")
        print(f"Average overall score: {avg_overall:.2f}")
        print("\nAverage section scores:")
        for section, avg in avg_sections.items():
            print(f"  {section.capitalize()}: {avg:.2f}")
        print("\nBy specialty:")
        for specialty, avg in summary['by_specialty'].items():
            print(f"  {specialty}: {avg:.2f}")
        print("\nBy language:")
        for language, avg in summary['by_language'].items():
            print(f"  {language}: {avg:.2f}")
        print(f"\nResults saved to {output_dir}")

# Function to create data visualizations
def create_visualizations(results, summary, output_dir):
    """Create visualizations from evaluation results."""
    # Set up plotting style
    plt.style.use('ggplot')
    
    # 1. Overall score distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist([r['overall_score'] for r in results], bins=10, alpha=0.7, color='skyblue')
    plt.axvline(summary['average_overall_score'], color='red', linestyle='dashed', linewidth=2)
    plt.title('Distribution of Overall Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Section scores comparison
    sections = ['Subjective', 'Objective', 'Assessment', 'Plan']
    avg_scores = [summary['average_section_scores'][s.lower()] for s in sections]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sections, avg_scores, color='lightseagreen')
    plt.axhline(summary['average_overall_score'], color='red', linestyle='dashed', linewidth=2, 
                label=f'Overall Avg: {summary["average_overall_score"]:.2f}')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.title('Average Similarity Score by SOAP Section')
    plt.ylabel('Average Similarity Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'section_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scores by specialty
    if summary['by_specialty']:
        specialties = list(summary['by_specialty'].keys())
        specialty_scores = [summary['by_specialty'][s] for s in specialties]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(specialties, specialty_scores, color='coral')
        plt.axhline(summary['average_overall_score'], color='blue', linestyle='dashed', linewidth=2, 
                    label=f'Overall Avg: {summary["average_overall_score"]:.2f}')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title('Average Similarity Score by Specialty')
        plt.ylabel('Average Similarity Score')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'specialty_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Scores by language
    if summary['by_language']:
        languages = list(summary['by_language'].keys())
        language_scores = [summary['by_language'][l] for l in languages]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(languages, language_scores, color='mediumpurple')
        plt.axhline(summary['average_overall_score'], color='green', linestyle='dashed', linewidth=2, 
                    label=f'Overall Avg: {summary["average_overall_score"]:.2f}')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title('Average Similarity Score by Language')
        plt.ylabel('Average Similarity Score')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'language_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Function to evaluate all datasets in a parent directory
def evaluate_all_datasets(parent_dir, output_dir='evaluation_results', verbose=False):
    """Evaluate SOAP notes across multiple dataset directories."""
    # List of dataset folders to evaluate
    datasets = [
        "Noisy-Azure for English-Nova-2- for French",
        "Noisy-Azure for French-Nova-3-medical- for English",
        "Semi-noise - Azure for English-Nova-2 for French",
        "Semi-noise - Azure for French-Nova-3-medical- for English",
        "Without-noise-Azure for English-Nova-2- for French",
        "Without-noise-Azure for French-Nova-3-medical- for English"
    ]
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Combined results across all datasets
    all_combined_results = []
    
    # Process each dataset
    for dataset in datasets:
        dataset_path = os.path.join(parent_dir, dataset)
        
        if not os.path.exists(dataset_path):
            if verbose:
                print(f"Dataset directory not found: {dataset_path}")
            continue
        
        if verbose:
            print(f"\nProcessing dataset: {dataset}")
        
        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(output_dir, dataset.replace(" ", "_"))
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Evaluate the dataset
        dataset_results = evaluate_dataset(dataset_path, dataset_output_dir, verbose)
        
        # Add to combined results
        if dataset_results:
            all_combined_results.extend(dataset_results)
    
    # Generate combined reports if there are results
    if all_combined_results:
        # Create a combined output directory
        combined_output_dir = os.path.join(output_dir, "combined_results")
        os.makedirs(combined_output_dir, exist_ok=True)
        
        # Generate combined reports
        generate_combined_reports(all_combined_results, combined_output_dir, verbose)
    else:
        print("No SOAP notes were evaluated across all datasets.")

# Function to generate combined evaluation reports and visualizations
def generate_combined_reports(results, output_dir, verbose=False):
    """Generate summary reports and visualizations from combined evaluation results."""
    # Save all results to JSON
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate overall statistics
    overall_scores = [r['overall_score'] for r in results]
    section_scores = {
        'subjective': [r['section_scores']['subjective'] for r in results],
        'objective': [r['section_scores']['objective'] for r in results],
        'assessment': [r['section_scores']['assessment'] for r in results],
        'plan': [r['section_scores']['plan'] for r in results]
    }
    
    # Calculate averages
    avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    avg_sections = {section: sum(scores) / len(scores) if scores else 0 
                    for section, scores in section_scores.items()}
    
    # Create summary report
    summary = {
        'average_overall_score': avg_overall,
        'average_section_scores': avg_sections,
        'number_of_evaluations': len(results),
        'by_specialty': {},
        'by_language': {},
        'by_dataset': {},
        'by_noise_type': {},
        'by_model': {}
    }
    
    # Calculate scores by specialty
    specialties = set(r['specialty'] for r in results)
    for specialty in specialties:
        specialty_results = [r for r in results if r['specialty'] == specialty]
        specialty_scores = [r['overall_score'] for r in specialty_results]
        summary['by_specialty'][specialty] = sum(specialty_scores) / len(specialty_scores) if specialty_scores else 0
    
    # Calculate scores by language
    languages = set(r['language'] for r in results)
    for language in languages:
        language_results = [r for r in results if r['language'] == language]
        language_scores = [r['overall_score'] for r in language_results]
        summary['by_language'][language] = sum(language_scores) / len(language_scores) if language_scores else 0
    
    # Calculate scores by dataset
    datasets = set(r['dataset'] for r in results)
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        dataset_scores = [r['overall_score'] for r in dataset_results]
        summary['by_dataset'][dataset] = sum(dataset_scores) / len(dataset_scores) if dataset_scores else 0
    
    # Calculate scores by noise type (Noisy, Semi-noise, Without-noise)
    for result in results:
        dataset = result['dataset']
        if "Noisy" in dataset and not "Without" in dataset:
            noise_type = "Noisy"
        elif "Semi-noise" in dataset:
            noise_type = "Semi-noise"
        elif "Without-noise" in dataset:
            noise_type = "Without-noise"
        else:
            noise_type = "Unknown"
        
        result['noise_type'] = noise_type
    
    noise_types = set(r['noise_type'] for r in results)
    for noise_type in noise_types:
        noise_results = [r for r in results if r['noise_type'] == noise_type]
        noise_scores = [r['overall_score'] for r in noise_results]
        summary['by_noise_type'][noise_type] = sum(noise_scores) / len(noise_scores) if noise_scores else 0
    
    # Calculate scores by model
    models = set(r['model'] for r in results)
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        model_scores = [r['overall_score'] for r in model_results]
        summary['by_model'][model] = sum(model_scores) / len(model_scores) if model_scores else 0
    
    # Calculate scores by model and language combination
    summary['by_model_and_language'] = {}
    for model in models:
        for language in languages:
            combo_key = f"{model}_{language}"
            combo_results = [r for r in results if r['model'] == model and r['language'] == language]
            combo_scores = [r['overall_score'] for r in combo_results]
            if combo_scores:
                summary['by_model_and_language'][combo_key] = sum(combo_scores) / len(combo_scores)
    
    # Calculate average WER by model and language
    summary['wer_by_model'] = {}
    summary['wer_by_language'] = {}
    summary['wer_by_model_and_language'] = {}
    
    for model in models:
        model_wers = [r.get('wer_average', 1.0) for r in results if r.get('model') == model]
        if model_wers:
            summary['wer_by_model'][model] = sum(model_wers) / len(model_wers)
    
    for language in languages:
        lang_wers = [r.get('wer_average', 1.0) for r in results if r.get('language') == language]
        if lang_wers:
            summary['wer_by_language'][language] = sum(lang_wers) / len(lang_wers)
    
    for model in models:
        for language in languages:
            combo_key = f"{model}_{language}"
            combo_wers = [r.get('wer_average', 1.0) for r in results 
                         if r.get('model') == model and r.get('language') == language]
            if combo_wers:
                summary['wer_by_model_and_language'][combo_key] = sum(combo_wers) / len(combo_wers)
    
    # Save summary report
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    create_combined_visualizations(results, summary, output_dir)
    
    # Create a detailed model and language performance report
    create_model_language_report(results, output_dir)
    
    if verbose:
        print("\nCombined Evaluation Summary:")
        print(f"Number of evaluations: {len(results)}")
        print(f"Average overall score: {avg_overall:.2f}")
        print("\nAverage section scores:")
        for section, avg in avg_sections.items():
            print(f"  {section.capitalize()}: {avg:.2f}")
        print("\nBy specialty:")
        for specialty, avg in summary['by_specialty'].items():
            print(f"  {specialty}: {avg:.2f}")
        print("\nBy language:")
        for language, avg in summary['by_language'].items():
            print(f"  {language}: {avg:.2f}")
        print("\nBy dataset:")
        for dataset, avg in summary['by_dataset'].items():
            print(f"  {dataset}: {avg:.2f}")
        print("\nBy noise type:")
        for noise_type, avg in summary['by_noise_type'].items():
            print(f"  {noise_type}: {avg:.2f}")
        print("\nBy model:")
        for model, avg in summary['by_model'].items():
            print(f"  {model}: {avg:.2f}")
        print(f"\nCombined results saved to {output_dir}")

# Function to create combined data visualizations
def create_combined_visualizations(results, summary, output_dir):
    """Create visualizations from combined evaluation results."""
    # Set up plotting style
    plt.style.use('ggplot')
    
    # 1. Overall score distribution histogram
    plt.figure(figsize=(12, 7))
    plt.hist([r['overall_score'] for r in results], bins=15, alpha=0.7, color='skyblue')
    plt.axvline(summary['average_overall_score'], color='red', linestyle='dashed', linewidth=2)
    plt.title('Distribution of Overall Similarity Scores Across All Datasets')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'combined_score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Section scores comparison
    sections = ['Subjective', 'Objective', 'Assessment', 'Plan']
    avg_scores = [summary['average_section_scores'][s.lower()] for s in sections]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(sections, avg_scores, color='lightseagreen')
    plt.axhline(summary['average_overall_score'], color='red', linestyle='dashed', linewidth=2, 
                label=f'Overall Avg: {summary["average_overall_score"]:.2f}')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.title('Average Similarity Score by SOAP Section Across All Datasets')
    plt.ylabel('Average Similarity Score')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'combined_section_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scores by specialty
    if summary['by_specialty']:
        specialties = list(summary['by_specialty'].keys())
        specialty_scores = [summary['by_specialty'][s] for s in specialties]
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(specialties, specialty_scores, color='coral')
        plt.axhline(summary['average_overall_score'], color='blue', linestyle='dashed', linewidth=2, 
                    label=f'Overall Avg: {summary["average_overall_score"]:.2f}')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title('Average Similarity Score by Specialty Across All Datasets')
        plt.ylabel('Average Similarity Score')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'combined_specialty_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Scores by language
    if summary['by_language']:
        languages = list(summary['by_language'].keys())
        language_scores = [summary['by_language'][l] for l in languages]
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(languages, language_scores, color='mediumpurple')
        plt.axhline(summary['average_overall_score'], color='green', linestyle='dashed', linewidth=2, 
                    label=f'Overall Avg: {summary["average_overall_score"]:.2f}')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title('Average Similarity Score by Language Across All Datasets')
        plt.ylabel('Average Similarity Score')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'combined_language_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Scores by dataset
    if summary['by_dataset']:
        datasets = list(summary['by_dataset'].keys())
        dataset_scores = [summary['by_dataset'][d] for d in datasets]
        
        plt.figure(figsize=(16, 8))
        bars = plt.bar(datasets, dataset_scores, color='teal')
        plt.axhline(summary['average_overall_score'], color='red', linestyle='dashed', linewidth=2, 
                    label=f'Overall Avg: {summary["average_overall_score"]:.2f}')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title('Average Similarity Score by Dataset')
        plt.ylabel('Average Similarity Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'combined_dataset_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Scores by noise type
    if summary['by_noise_type']:
        noise_types = list(summary['by_noise_type'].keys())
        noise_scores = [summary['by_noise_type'][n] for n in noise_types]
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(noise_types, noise_scores, color='darkgoldenrod')
        plt.axhline(summary['average_overall_score'], color='red', linestyle='dashed', linewidth=2, 
                    label=f'Overall Avg: {summary["average_overall_score"]:.2f}')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title('Average Similarity Score by Noise Type')
        plt.ylabel('Average Similarity Score')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'combined_noise_type_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Scores by model
    if summary['by_model']:
        models = list(summary['by_model'].keys())
        model_scores = [summary['by_model'][m] for m in models]
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(models, model_scores, color='darkviolet')
        plt.axhline(summary['average_overall_score'], color='red', linestyle='dashed', linewidth=2, 
                    label=f'Overall Avg: {summary["average_overall_score"]:.2f}')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title('Average Similarity Score by Model')
        plt.ylabel('Average Similarity Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'combined_model_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 8. Create a heatmap of language vs noise type
    if summary['by_language'] and summary['by_noise_type']:
        languages = list(summary['by_language'].keys())
        noise_types = list(summary['by_noise_type'].keys())
        
        # Create data matrix for heatmap
        heatmap_data = []
        for language in languages:
            row = []
            for noise_type in noise_types:
                # Get results that match both language and noise type
                matching_results = [r for r in results if r['language'] == language and r['noise_type'] == noise_type]
                if matching_results:
                    avg_score = sum(r['overall_score'] for r in matching_results) / len(matching_results)
                    row.append(avg_score)
                else:
                    row.append(0)  # No data for this combination
            heatmap_data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, cmap='viridis')
        
        # Add labels and colorbar
        plt.xticks(np.arange(len(noise_types)), noise_types, rotation=45)
        plt.yticks(np.arange(len(languages)), languages)
        plt.colorbar(label='Average Similarity Score')
        
        # Add text annotations
        for i in range(len(languages)):
            for j in range(len(noise_types)):
                text = plt.text(j, i, f"{heatmap_data[i][j]:.2f}", ha="center", va="center", color="w")
        
        plt.title('Average Similarity Score by Language and Noise Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'language_noise_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 9. Medical term extraction accuracy
    # Calculate average F1 scores for each term category
    term_categories = ['diagnoses', 'treatments', 'symptoms', 'conditions', 'allergies']
    term_f1_scores = {category: [] for category in term_categories}
    
    for result in results:
        term_metrics = result.get('term_metrics', {})
        for category in term_categories:
            if category in term_metrics and 'f1' in term_metrics[category]:
                term_f1_scores[category].append(term_metrics[category]['f1'])
    
    # Calculate averages
    avg_term_f1 = {category: sum(scores) / len(scores) if scores else 0 
                  for category, scores in term_f1_scores.items()}
    
    plt.figure(figsize=(12, 7))
    categories = list(avg_term_f1.keys())
    scores = list(avg_term_f1.values())
    
    bars = plt.bar(categories, scores, color='darkturquoise')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.title('Medical Term Extraction Accuracy (F1 Score) by Term Category')
    plt.ylabel('Average F1 Score')
    plt.savefig(os.path.join(output_dir, 'medical_term_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 10. WER, BLEU, and Phrase Match metrics
    plt.figure(figsize=(14, 8))
    
    # Create a dataframe for additional metrics
    metrics_data = {
        'WER (lower is better)': [r.get('wer_average', 1.0) for r in results],
        'BLEU Score': [r.get('bleu_average', 0.0) for r in results],
        'Phrase Match': [r.get('phrase_match_average', 0.0) for r in results]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create boxplots
    sns.boxplot(data=metrics_df)
    plt.title('Distribution of Additional Metrics Across All Datasets')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'additional_metrics_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 11. Correlation heatmap between metrics
    plt.figure(figsize=(12, 10))
    
    # Create a dataframe for correlation analysis
    corr_data = {
        'Semantic Similarity': [r.get('section_average', 0.0) for r in results],
        'Term F1': [r.get('term_f1_average', 0.0) for r in results],
        'WER': [r.get('wer_average', 1.0) for r in results],
        'BLEU': [r.get('bleu_average', 0.0) for r in results], 
        'Phrase Match': [r.get('phrase_match_average', 0.0) for r in results],
        'Overall Score': [r.get('overall_score', 0.0) for r in results]
    }
    corr_df = pd.DataFrame(corr_data)
    
    # Create a correlation matrix
    corr_matrix = corr_df.corr()
    
    # Create a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Evaluation Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 12. WER comparison by model and language
    if summary['by_model'] and summary['by_language']:
        # Calculate average WER by model
        wer_by_model = {}
        for model in summary['by_model'].keys():
            model_wers = [r.get('wer_average', 1.0) for r in results if r.get('model') == model]
            if model_wers:
                wer_by_model[model] = sum(model_wers) / len(model_wers)
        
        # Calculate average WER by language
        wer_by_language = {}
        for language in summary['by_language'].keys():
            lang_wers = [r.get('wer_average', 1.0) for r in results if r.get('language') == language]
            if lang_wers:
                wer_by_language[language] = sum(lang_wers) / len(lang_wers)
        
        # Plot WER by model
        plt.figure(figsize=(14, 7))
        models = list(wer_by_model.keys())
        wer_scores = list(wer_by_model.values())
        
        bars = plt.bar(models, wer_scores, color='orchid')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Word Error Rate by Model (Lower is Better)')
        plt.ylabel('Average WER')
        plt.ylim(0, min(1.1, max(wer_scores) + 0.2))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'wer_by_model.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot WER by language
        plt.figure(figsize=(12, 7))
        languages = list(wer_by_language.keys())
        wer_scores = list(wer_by_language.values())
        
        bars = plt.bar(languages, wer_scores, color='lightcoral')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Word Error Rate by Language (Lower is Better)')
        plt.ylabel('Average WER')
        plt.ylim(0, min(1.1, max(wer_scores) + 0.2))
        plt.savefig(os.path.join(output_dir, 'wer_by_language.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 13. Metric breakdown by section (add to existing Combined Evaluation Summary)
    if 'average_section_scores' in summary:
        # Create a markdown table with all metrics by section
        metrics_by_section = pd.DataFrame({
            'Section': ['Subjective', 'Objective', 'Assessment', 'Plan'],
            'Semantic Similarity': [summary['average_section_scores']['subjective'], 
                                    summary['average_section_scores']['objective'],
                                    summary['average_section_scores']['assessment'], 
                                    summary['average_section_scores']['plan']]
        })
        
        # Calculate average WER by section
        wer_by_section = {'subjective': [], 'objective': [], 'assessment': [], 'plan': []}
        for r in results:
            if 'wer_sections' in r:
                for section, wer in r['wer_sections'].items():
                    if section in wer_by_section:
                        wer_by_section[section].append(wer)
        
        # Calculate average BLEU by section
        bleu_by_section = {'subjective': [], 'objective': [], 'assessment': [], 'plan': []}
        for r in results:
            if 'bleu_sections' in r:
                for section, bleu in r['bleu_sections'].items():
                    if section in bleu_by_section:
                        bleu_by_section[section].append(bleu)
        
        # Calculate average phrase match by section
        phrase_by_section = {'subjective': [], 'objective': [], 'assessment': [], 'plan': []}
        for r in results:
            if 'phrase_matches' in r:
                for section, match in r['phrase_matches'].items():
                    if section in phrase_by_section:
                        phrase_by_section[section].append(match)
        
        # Add columns to the dataframe
        metrics_by_section['WER (lower is better)'] = [
            sum(wer_by_section['subjective']) / len(wer_by_section['subjective']) if wer_by_section['subjective'] else 1.0,
            sum(wer_by_section['objective']) / len(wer_by_section['objective']) if wer_by_section['objective'] else 1.0,
            sum(wer_by_section['assessment']) / len(wer_by_section['assessment']) if wer_by_section['assessment'] else 1.0,
            sum(wer_by_section['plan']) / len(wer_by_section['plan']) if wer_by_section['plan'] else 1.0
        ]
        
        metrics_by_section['BLEU Score'] = [
            sum(bleu_by_section['subjective']) / len(bleu_by_section['subjective']) if bleu_by_section['subjective'] else 0.0,
            sum(bleu_by_section['objective']) / len(bleu_by_section['objective']) if bleu_by_section['objective'] else 0.0,
            sum(bleu_by_section['assessment']) / len(bleu_by_section['assessment']) if bleu_by_section['assessment'] else 0.0,
            sum(bleu_by_section['plan']) / len(bleu_by_section['plan']) if bleu_by_section['plan'] else 0.0
        ]
        
        metrics_by_section['Phrase Match'] = [
            sum(phrase_by_section['subjective']) / len(phrase_by_section['subjective']) if phrase_by_section['subjective'] else 0.0,
            sum(phrase_by_section['objective']) / len(phrase_by_section['objective']) if phrase_by_section['objective'] else 0.0,
            sum(phrase_by_section['assessment']) / len(phrase_by_section['assessment']) if phrase_by_section['assessment'] else 0.0,
            sum(phrase_by_section['plan']) / len(phrase_by_section['plan']) if phrase_by_section['plan'] else 0.0
        ]
        
        # Export as CSV
        metrics_by_section.to_csv(os.path.join(output_dir, 'metrics_by_section.csv'), index=False)
        
        # Create a visualization
        plt.figure(figsize=(16, 10))
        
        # Prepare data for grouped bar chart
        section_names = metrics_by_section['Section']
        x = np.arange(len(section_names))
        width = 0.2
        
        # Create grouped bars
        plt.bar(x - width*1.5, metrics_by_section['Semantic Similarity'], width, label='Semantic Similarity', color='lightseagreen')
        plt.bar(x - width*0.5, metrics_by_section['WER (lower is better)'], width, label='WER (lower is better)', color='coral')
        plt.bar(x + width*0.5, metrics_by_section['BLEU Score'], width, label='BLEU Score', color='mediumpurple')
        plt.bar(x + width*1.5, metrics_by_section['Phrase Match'], width, label='Phrase Match', color='gold')
        
        # Add labels and legend
        plt.xlabel('Section')
        plt.ylabel('Score')
        plt.title('Multiple Metrics by SOAP Section')
        plt.xticks(x, section_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multiple_metrics_by_section.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Function to create model and language performance report
def create_model_language_report(results, output_dir):
    """Create a detailed report on model and language performance."""
    # Group results by model and language
    model_lang_results = {}
    
    for r in results:
        model = r.get('model', 'unknown')
        language = r.get('language', 'unknown')
        key = f"{model}_{language}"
        
        if key not in model_lang_results:
            model_lang_results[key] = []
        
        model_lang_results[key].append(r)
    
    # Create a report for each model and language combination
    report_data = []
    
    for key, entries in model_lang_results.items():
        model, language = key.split('_')
        
        # Calculate metrics
        overall_scores = [e.get('overall_score', 0) for e in entries]
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        
        wer_scores = [e.get('wer_average', 1.0) for e in entries]
        avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 1.0
        
        bleu_scores = [e.get('bleu_average', 0) for e in entries]
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        
        # Calculate section scores
        section_avgs = {}
        for section in ['subjective', 'objective', 'assessment', 'plan']:
            scores = [e.get('section_scores', {}).get(section, 0) for e in entries]
            section_avgs[section] = sum(scores) / len(scores) if scores else 0
        
        # Calculate term F1 average
        term_f1_avgs = [e.get('term_f1_average', 0) for e in entries]
        avg_term_f1 = sum(term_f1_avgs) / len(term_f1_avgs) if term_f1_avgs else 0
        
        # Count the number of files
        num_files = len(entries)
        
        # Get noise levels represented
        noise_types = set(e.get('noise_type', 'Unknown') for e in entries)
        
        # Get specialties represented
        specialties = set(e.get('specialty', 'Unknown') for e in entries)
        
        # Add to report data
        report_data.append({
            'model': model,
            'language': language,
            'num_files': num_files,
            'overall_score': avg_overall,
            'wer_score': avg_wer,
            'bleu_score': avg_bleu,
            'term_f1_score': avg_term_f1,
            'subjective_score': section_avgs.get('subjective', 0),
            'objective_score': section_avgs.get('objective', 0),
            'assessment_score': section_avgs.get('assessment', 0),
            'plan_score': section_avgs.get('plan', 0),
            'noise_types': list(noise_types),
            'specialties': list(specialties)
        })
    
    # Save report as JSON
    with open(os.path.join(output_dir, 'model_language_report.json'), 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Create a markdown report
    md_report = "# Model and Language Performance Report\n\n"
    
    # Sort by overall score
    report_data.sort(key=lambda x: x['overall_score'], reverse=True)
    
    md_report += "## Performance Rankings (Overall Score)\n\n"
    md_report += "| Rank | Model | Language | Overall Score | WER (lower is better) | BLEU | Term F1 | Files |\n"
    md_report += "|------|-------|----------|--------------|------------------------|------|---------|-------|\n"
    
    for i, entry in enumerate(report_data):
        md_report += f"| {i+1} | {entry['model']} | {entry['language']} | {entry['overall_score']:.2f} | {entry['wer_score']:.2f} | {entry['bleu_score']:.2f} | {entry['term_f1_score']:.2f} | {entry['num_files']} |\n"
    
    md_report += "\n## Detailed Section Performance\n\n"
    md_report += "| Model | Language | Subjective | Objective | Assessment | Plan |\n"
    md_report += "|-------|----------|------------|-----------|------------|------|\n"
    
    for entry in report_data:
        md_report += f"| {entry['model']} | {entry['language']} | {entry['subjective_score']:.2f} | {entry['objective_score']:.2f} | {entry['assessment_score']:.2f} | {entry['plan_score']:.2f} |\n"
    
    md_report += "\n## Performance by Metric\n\n"
    
    # WER Rankings
    md_report += "### WER Rankings (Lower is Better)\n\n"
    sorted_by_wer = sorted(report_data, key=lambda x: x['wer_score'])
    md_report += "| Rank | Model | Language | WER Score |\n"
    md_report += "|------|-------|----------|-----------|\n"
    for i, entry in enumerate(sorted_by_wer):
        md_report += f"| {i+1} | {entry['model']} | {entry['language']} | {entry['wer_score']:.2f} |\n"
    
    # BLEU Rankings
    md_report += "\n### BLEU Rankings\n\n"
    sorted_by_bleu = sorted(report_data, key=lambda x: x['bleu_score'], reverse=True)
    md_report += "| Rank | Model | Language | BLEU Score |\n"
    md_report += "|------|-------|----------|------------|\n"
    for i, entry in enumerate(sorted_by_bleu):
        md_report += f"| {i+1} | {entry['model']} | {entry['language']} | {entry['bleu_score']:.2f} |\n"
    
    # Term F1 Rankings
    md_report += "\n### Medical Term Extraction (F1) Rankings\n\n"
    sorted_by_f1 = sorted(report_data, key=lambda x: x['term_f1_score'], reverse=True)
    md_report += "| Rank | Model | Language | Term F1 Score |\n"
    md_report += "|------|-------|----------|---------------|\n"
    for i, entry in enumerate(sorted_by_f1):
        md_report += f"| {i+1} | {entry['model']} | {entry['language']} | {entry['term_f1_score']:.2f} |\n"
    
    # Save markdown report
    with open(os.path.join(output_dir, 'model_language_report.md'), 'w') as f:
        f.write(md_report)

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SOAP notes generated from transcripts against original SOAP notes.')
    parser.add_argument('--path', help='Path to the dataset root directory')
    parser.add_argument('--output', default='evaluation_results', help='Output directory for evaluation results')
    parser.add_argument('--original', help='Path to a specific original SOAP note file to evaluate')
    parser.add_argument('--transcript', help='Path to a specific transcript SOAP note file to evaluate')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--all', help='Path to the parent directory containing all dataset folders')
    args = parser.parse_args()
    
    # Evaluate a specific pair of SOAP notes
    if args.original and args.transcript:
        results = evaluate_soap_pair(args.original, args.transcript, args.verbose)
        if results:
            # Save results to JSON
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'single_evaluation.json'), 'w') as f:
                json.dump(results, f, indent=2)
    
    # Evaluate all datasets
    elif args.all:
        evaluate_all_datasets(args.all, args.output, args.verbose)
    
    # Evaluate a single dataset
    elif args.path:
        evaluate_dataset(args.path, args.output, args.verbose)
    
    # No arguments provided
    else:
        parser.print_help() 