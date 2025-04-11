import os
import json
import argparse
import glob
from pathlib import Path
from openai import AzureOpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://notegen-openai-eastca.openai.azure.com")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://59d8fe10.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "OeU2iAGXPk2SzYGMYu-IPK4lSxeh72smucUVBt_U96s")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Prompt templates for SOAP note generation with markdown formatting
SOAP_PROMPT = """
Based on the following medical conversation, generate a professional SOAP note in markdown format:

Conversation:
{conversation}

Patient Information:
- Age: {age}
- Gender: {gender}
- Condition: {condition}
- Specialty: {specialty}
- Visit Type: {visit_type}

{medical_context}

Generate a structured SOAP note with clear markdown formatting using this exact format:

**SOAP Note**

**S – Subjective:**  
[Include patient's complaints, symptoms, medical history, medications, allergies]

**O – Objective:**  
[Include physical examination findings, vital signs, test results]

**A – Assessment:**  
[Include diagnosis or differential diagnoses]

**P – Plan:**  
[Include treatment plan, medications, follow-up instructions]

---

**Keywords / Tags**  
**Diagnosis**: [Include relevant diagnoses]  
**Treatment**: [Include prescribed treatments and medications]  
**Symptoms**: [Include primary symptoms]  
**Conditions**: [Include relevant medical conditions]  
**Allergies**: [Include allergies if mentioned]  

Use bullet points for clarity and ensure the information is organized professionally.
"""

# Initialize Neo4j connection with error handling
def get_neo4j_driver():
    try:
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        print(f"Failed to create Neo4j driver: {e}")
        return None

# Function to query Neo4j for medical knowledge enhancement
def query_neo4j_for_medical_knowledge(condition, specialty):
    try:
        driver = get_neo4j_driver()
        if not driver:
            return {}
            
        with driver.session() as session:
            # Query for condition-related information
            try:
                result = session.run("""
                    MATCH (d:Diagnosis {name: $condition})
                    OPTIONAL MATCH (d)-[:HAS_TREATMENT]->(t:Treatment)
                    OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                    RETURN d, collect(distinct t) as treatments, collect(distinct s) as symptoms
                """, condition=condition)
                
                record = result.single()
                if record:
                    return {
                        "diagnosis": record["d"],
                        "treatments": record["treatments"],
                        "symptoms": record["symptoms"]
                    }
                return {}
            except Exception as e:
                print(f"Neo4j query error: {e}")
                return {}
    except Exception as e:
        print(f"Neo4j connection error: {e}")
        return {}

# Function to extract medical terms from Neo4j knowledge graph
def extract_medical_terms(condition, specialty):
    try:
        driver = get_neo4j_driver()
        if not driver:
            return {}
            
        with driver.session() as session:
            # Query for related medical terms
            try:
                result = session.run("""
                    MATCH (d:Diagnosis) 
                    WHERE d.name =~ $pattern
                    OPTIONAL MATCH (d)-[:HAS_TREATMENT]->(t:Treatment)
                    OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
                    OPTIONAL MATCH (d)-[:RELATED_TO]->(r:Diagnosis)
                    RETURN d.name as diagnosis, 
                           collect(distinct t.name) as treatments, 
                           collect(distinct s.name) as symptoms,
                           collect(distinct r.name) as related_conditions
                """, pattern=f"(?i).*{condition}.*")
                
                # Process results
                medical_terms = {
                    "diagnoses": [],
                    "treatments": [],
                    "symptoms": [],
                    "related_conditions": []
                }
                
                for record in result:
                    if record["diagnosis"] and record["diagnosis"] not in medical_terms["diagnoses"]:
                        medical_terms["diagnoses"].append(record["diagnosis"])
                    
                    for treatment in record["treatments"]:
                        if treatment and treatment not in medical_terms["treatments"]:
                            medical_terms["treatments"].append(treatment)
                    
                    for symptom in record["symptoms"]:
                        if symptom and symptom not in medical_terms["symptoms"]:
                            medical_terms["symptoms"].append(symptom)
                    
                    for related in record["related_conditions"]:
                        if related and related not in medical_terms["related_conditions"]:
                            medical_terms["related_conditions"].append(related)
                
                return medical_terms
            except Exception as e:
                print(f"Neo4j query error for medical terms: {e}")
                return {}
    except Exception as e:
        print(f"Neo4j connection error for medical terms: {e}")
        return {}

# Function to generate SOAP note with RAG enhancement
def generate_soap_note(conversation, age, gender, condition, specialty, visit_type, language="en", verbose=False):
    # Format conversation for prompt
    formatted_conversation = ""
    for turn in conversation:
        # Check if the conversation item has the expected structure
        if isinstance(turn, dict) and 'speaker' in turn and 'text' in turn:
            formatted_conversation += f"{turn['speaker'].capitalize()}: {turn['text']}\n"
        elif isinstance(turn, str):
            # Handle case where the conversation item is a string
            formatted_conversation += turn + "\n"
        else:
            # Handle unexpected formats
            try:
                formatted_conversation += f"Content: {str(turn)}\n"
            except:
                formatted_conversation += "Error: Could not format conversation turn\n"
    
    # Try to get medical knowledge from Neo4j for RAG enhancement
    if verbose:
        print(f"Querying Neo4j for medical knowledge about '{condition}'...")
    
    medical_knowledge = query_neo4j_for_medical_knowledge(condition, specialty)
    
    # Try to extract related medical terms from Neo4j
    medical_terms = extract_medical_terms(condition, specialty)
    
    # Prepare RAG context
    medical_context = ""
    
    # Add medical knowledge if available
    if medical_knowledge and (medical_knowledge.get('treatments', []) or medical_knowledge.get('symptoms', [])):
        medical_context += "Medical Context (from Knowledge Graph):\n"
        if medical_knowledge.get('treatments', []):
            treatments = [t.get('name', '') for t in medical_knowledge.get('treatments', [])]
            if treatments:
                medical_context += f"- Common treatments for {condition}: {', '.join(treatments)}\n"
        if medical_knowledge.get('symptoms', []):
            symptoms = [s.get('name', '') for s in medical_knowledge.get('symptoms', [])]
            if symptoms:
                medical_context += f"- Common symptoms of {condition}: {', '.join(symptoms)}\n"
    
    # Add medical terms if available
    if medical_terms:
        if not medical_context:
            medical_context += "Medical Context (from Knowledge Graph):\n"
        
        if medical_terms.get('diagnoses', []):
            medical_context += f"- Related diagnoses: {', '.join(medical_terms['diagnoses'])}\n"
        
        if medical_terms.get('treatments', []) and not medical_knowledge.get('treatments', []):
            medical_context += f"- Potential treatments: {', '.join(medical_terms['treatments'])}\n"
        
        if medical_terms.get('symptoms', []) and not medical_knowledge.get('symptoms', []):
            medical_context += f"- Associated symptoms: {', '.join(medical_terms['symptoms'])}\n"
        
        if medical_terms.get('related_conditions', []):
            medical_context += f"- Related conditions: {', '.join(medical_terms['related_conditions'])}\n"
    
    if verbose and medical_context:
        print("Found medical context from Neo4j:")
        print(medical_context)
    elif verbose:
        print("No medical context found in Neo4j.")
    
    # Format prompt with patient data, conversation, and medical context if available
    prompt = SOAP_PROMPT.format(
        conversation=formatted_conversation,
        age=age,
        gender=gender,
        condition=condition,
        specialty=specialty,
        visit_type=visit_type,
        medical_context=medical_context
    )
    
    # Call Azure OpenAI
    if verbose:
        print("Generating SOAP note using Azure OpenAI...")
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a medical professional specializing in generating detailed, accurate, and well-structured SOAP notes in markdown format for healthcare providers in Canada. Your notes follow standard medical documentation practices and are formatted for clarity."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Azure OpenAI API error: {e}")
        return f"Error generating SOAP note: {e}"

# Function to process a JSON file
def process_json_file(json_file_path, output_file, verbose=False):
    try:
        if verbose:
            print(f"Processing JSON file: {json_file_path}")
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract patient and conversation info
        language_code = data.get('language', 'en-CA').split('-')[0]
        specialty = data.get('specialty', 'gp')
        visit_type = data.get('type', 'consultation')
        condition = data.get('condition', '')
        patient = data.get('patient', {})
        age = patient.get('age', 0)
        gender = patient.get('gender', '')
        conversation = data.get('conversation', [])
        
        if verbose:
            print(f"Patient: {age} year old {gender}")
            print(f"Condition: {condition}")
            print(f"Visit type: {visit_type}")
            print(f"Specialty: {specialty}")
        
        # Generate SOAP note
        soap_note = generate_soap_note(
            conversation, 
            age, 
            gender, 
            condition, 
            specialty, 
            visit_type, 
            language=language_code,
            verbose=verbose
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save SOAP note to file
        with open(output_file, 'w') as f:
            f.write(soap_note)
        
        print(f"Generated SOAP note saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return False

# Function to process a transcript file
def process_transcript_file(transcript_file_path, output_file, verbose=False):
    try:
        if verbose:
            print(f"Processing transcript file: {transcript_file_path}")
        
        with open(transcript_file_path, 'r') as f:
            conversation_data = json.load(f)
        
        # Check if conversation data is a string (direct transcription) or a list (structured conversation)
        if isinstance(conversation_data, str):
            # Handle string transcripts by creating a structured conversation format
            conversation = [{"speaker": "transcript", "text": conversation_data}]
        elif isinstance(conversation_data, list):
            # Already in expected format
            conversation = conversation_data
        else:
            # Handle the case where conversation is neither a string nor a list
            # Try to extract conversation field if it exists in a dictionary
            if isinstance(conversation_data, dict) and 'conversation' in conversation_data:
                conversation = conversation_data['conversation']
            else:
                print(f"Unsupported transcript format in {transcript_file_path}. Expected string or list.")
                return False
        
        # Extract info from file name
        base_name = os.path.basename(transcript_file_path)
        parts = base_name.replace('.json', '').split('_')
        
        if len(parts) >= 3:
            specialty = parts[0]
            visit_type = parts[2]  # consultation or followup
            
            # Determine language based on directory path
            language_code = "en"
            if "fr-CA" in transcript_file_path:
                language_code = "fr"
            
            # Try to determine condition from file name
            condition = "unknown condition"
            
            # Try to infer condition from json file with same name pattern in json folder
            json_path = transcript_file_path.replace('transcripts', 'json')
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                        if 'condition' in json_data:
                            condition = json_data['condition']
                except:
                    pass
            
            if verbose:
                print(f"Inferred condition: {condition}")
                print(f"Specialty: {specialty}")
                print(f"Visit type: {visit_type}")
            
            # Generate SOAP note with defaults for missing info
            soap_note = generate_soap_note(
                conversation, 
                50,  # default age 
                "unknown",  # default gender
                condition,
                specialty, 
                visit_type, 
                language=language_code,
                verbose=verbose
            )
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save SOAP note to file
            with open(output_file, 'w') as f:
                f.write(soap_note)
            
            print(f"Generated SOAP note saved to: {output_file}")
            return True
        else:
            print(f"Could not extract required info from filename: {base_name}")
            return False
    except Exception as e:
        print(f"Error processing {transcript_file_path}: {e}")
        return False

# Function to process all files in a dataset
def process_dataset(base_path, verbose=False):
    # Structure of directories
    specialties = ["cardiology", "gp"]
    
    for specialty in specialties:
        specialty_path = os.path.join(base_path, specialty)
        
        # Find all language-model directories
        lang_model_dirs = [d for d in os.listdir(specialty_path) 
                          if os.path.isdir(os.path.join(specialty_path, d)) 
                          and (d.startswith("en-CA") or d.startswith("fr-CA"))]
        
        for lang_dir in lang_model_dirs:
            lang_path = os.path.join(specialty_path, lang_dir)
            
            # Process JSON files
            json_path = os.path.join(lang_path, "json")
            if os.path.exists(json_path):
                # Create output directory
                soap_original_path = os.path.join(lang_path, "soap", "original")
                os.makedirs(soap_original_path, exist_ok=True)
                
                # Find all JSON files
                json_files = glob.glob(os.path.join(json_path, "*.json"))
                
                for json_file in json_files:
                    base_name = os.path.basename(json_file)
                    output_file = os.path.join(soap_original_path, base_name.replace('.json', '.md'))
                    
                    # Skip if output file already exists
                    if os.path.exists(output_file) and not args.force:
                        if verbose:
                            print(f"Skipping {base_name} - output file already exists")
                        continue
                    
                    # Process JSON file
                    process_json_file(json_file, output_file, verbose)
            
            # Process transcript files
            transcript_path = os.path.join(lang_path, "transcripts")
            if os.path.exists(transcript_path):
                # Create output directory
                soap_transcript_path = os.path.join(lang_path, "soap", "from_transcripts")
                os.makedirs(soap_transcript_path, exist_ok=True)
                
                # Find all transcript files
                transcript_files = glob.glob(os.path.join(transcript_path, "*.json"))
                
                for transcript_file in transcript_files:
                    base_name = os.path.basename(transcript_file)
                    output_file = os.path.join(soap_transcript_path, base_name.replace('.json', '.md'))
                    
                    # Skip if output file already exists
                    if os.path.exists(output_file) and not args.force:
                        if verbose:
                            print(f"Skipping {base_name} - output file already exists")
                        continue
                    
                    # Process transcript file
                    process_transcript_file(transcript_file, output_file, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SOAP notes from medical conversations using Azure OpenAI with Neo4j knowledge graph.')
    parser.add_argument('--path', help='Path to the dataset')
    parser.add_argument('--file', help='Path to a specific JSON file to process')
    parser.add_argument('--output', help='Output file path for the generated SOAP note (when using --file)')
    parser.add_argument('--transcript', help='Path to a specific transcript file to process')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if output file exists')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Process a single file
    if args.file:
        output_file = args.output
        if not output_file:
            # Generate output file path if not provided
            base_name = os.path.basename(args.file)
            dir_name = os.path.dirname(args.file)
            soap_dir = os.path.join(os.path.dirname(dir_name), "soap", "original")
            os.makedirs(soap_dir, exist_ok=True)
            output_file = os.path.join(soap_dir, base_name.replace('.json', '.md'))
        
        process_json_file(args.file, output_file, args.verbose)
    
    # Process a single transcript file
    elif args.transcript:
        output_file = args.output
        if not output_file:
            # Generate output file path if not provided
            base_name = os.path.basename(args.transcript)
            dir_name = os.path.dirname(args.transcript)
            soap_dir = os.path.join(os.path.dirname(dir_name), "soap", "from_transcripts")
            os.makedirs(soap_dir, exist_ok=True)
            output_file = os.path.join(soap_dir, base_name.replace('.json', '.md'))
        
        process_transcript_file(args.transcript, output_file, args.verbose)
    
    # Process an entire dataset
    elif args.path:
        process_dataset(args.path, args.verbose)
    
    # No arguments provided
    else:
        parser.print_help() 