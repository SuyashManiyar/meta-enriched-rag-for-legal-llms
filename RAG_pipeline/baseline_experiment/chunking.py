import os
import json
from transformers import AutoTokenizer

def chunk_folder_llama_json(
    input_folder: str,
    output_json: str,
    chunk_size: int = 500,
    window: int = 50,
    model_name: str = "thenlper/gte-large"
):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Final dictionary to save
    result = {}

    # Identify folder name for prefix
    folder_name = os.path.basename(os.path.normpath(input_folder))

    # Iterate over all .txt files
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(input_folder, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Remove extension for chunk_id
        base_name = os.path.splitext(filename)[0]

        # Tokenize with offsets
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        n = len(token_ids)

        start = 0
        chunk_idx = 1

        # Sliding window chunking
        while start < n:
            end = min(start + chunk_size, n)

            chunk_offsets = offsets[start:end]

            # Exact character span in original raw text
            char_start = chunk_offsets[0][0]
            char_end = chunk_offsets[-1][1]

            chunk_text = text[char_start:char_end]

            # Construct chunk ID (no .txt)
            chunk_id = f"{folder_name}_{base_name}_chunk{chunk_idx}"

            # Save into dictionary
            result[chunk_id] = {
                "chunk_text": chunk_text,
                "span": [char_start, char_end]
            }

            chunk_idx += 1
            start += chunk_size - window

    # Write entire dictionary as one JSON
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(result, jf, ensure_ascii=False, indent=2)

chunk_folder_llama_json(
    input_folder="/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/muad",
    output_json="/home/smaniyar_umass_edu/BioNLP_Ontology/other/nlp/RAG_data/muad_chunks.json",
    chunk_size=500,
    window=50,
    model_name="thenlper/gte-large"
)

