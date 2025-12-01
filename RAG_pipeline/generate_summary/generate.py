import json
import os
from openai import OpenAI

# -----------------------
# CONFIGURE API CLIENT
# -----------------------

# Get API key and strip any whitespace/newlines
api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=api_key)

MODEL_NAME = "gpt-4o-mini"  # or "gpt-4.1" or "o3-mini"

# -----------------------s
# METADATA EXTRACTION PROMPT
# -----------------------

METADATA_PROMPT = """
You are an expert legal-privacy analyst.

Extract high-quality metadata from the following privacy policy or privacy statement.
Return ONLY valid JSON. No explanations.

Document text:
---
{DOCUMENT_TEXT}
---

Return JSON with this schema:

{
  "document_title": "",
  "document_type": "privacy_policy | privacy_statement | terms_of_service",
  "organization": "",
  "jurisdiction": [],
  "publication_date": "",
  "last_updated": "",
  "categories_of_information_collected": [],
  "purposes_of_processing": [],
  "legal_bases": [],
  "data_subjects": [],
  "data_recipients": [],
  "security_measures": [],
  "user_rights": [],
  "retention_policies": "",
  "third_party_services_mentioned": [],
  "key_entities": [],
  "high_level_summary": "",
  "section_summaries": {
      "definitions": "",
      "data_collection": "",
      "tracking_and_cookies": "",
      "genetic_or_sensitive_data": "",
      "data_sharing": "",
      "research_use": "",
      "security": "",
      "user_rights_and_controls": "",
      "account_deletion": ""
  }
}
"""
# -----------------------
# FUNCTION TO EXTRACT METADATA
# -----------------------

def extract_metadata_from_document(doc_text: str) -> dict:
    """Send doc text to LLM and get metadata JSON."""
    prompt = METADATA_PROMPT.replace("{DOCUMENT_TEXT}", doc_text)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You must return valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    raw_output = response.choices[0].message.content.strip()

    # Parse JSON safely
    try:
        metadata = json.loads(raw_output)
    except json.JSONDecodeError:
        # Retry by cleaning stray characters
        cleaned = raw_output.strip("```json").strip("```")
        metadata = json.loads(cleaned)

    return metadata

# -----------------------
# MAIN SCRIPT
# -----------------------

def process_file(input_path: str, output_path: str):
    """Reads a PrivacyQA document, extracts metadata, saves .json file."""
    # Read document
    with open(input_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    # Extract metadata
    print(f"Extracting metadata from {input_path}...")
    metadata = extract_metadata_from_document(doc_text)

    # Save output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"Metadata saved to {output_path}")


def process_all_documents(input_folder: str, output_folder: str):
    """Process all .txt files in the input folder"""
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all .txt files
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    print(f"Found {len(txt_files)} documents to process\n")
    
    for i, filename in enumerate(txt_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace('.txt', '_metadata.json')
        output_path = os.path.join(output_folder, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"[{i}/{len(txt_files)}] Skipping {filename} (already processed)")
            continue
        
        print(f"[{i}/{len(txt_files)}] Processing {filename}...")
        
        try:
            process_file(input_path, output_path)
            print(f"  ✓ Saved to {output_filename}\n")
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            continue


if __name__ == "__main__":
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: $env:OPENAI_API_KEY = 'your-key-here'")
        exit(1)
    
    # Process all documents
    input_folder = "rag/data/privacy_qa"
    output_folder = "RAG_pipeline/generate_summary/metadata"
    
    try:
        process_all_documents(input_folder, output_folder)
        print("\n✓ All documents processed!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
