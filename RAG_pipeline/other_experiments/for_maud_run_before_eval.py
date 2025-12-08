# import json
# import re
# from pathlib import Path

# # ============================
# # INPUT AND OUTPUT PATHS
# # ============================

RETRIEVAL_JSON_IN  = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_inference/retrieval_results_recur_dense_window_metadata_n_doc_name.json"
RETRIEVAL_JSON_OUT = "/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/maud_subset_inference/retrieval_results_recur_dense_window_metadata_n_doc_name_corrected.json"


# # ============================
# # FIX FUNCTION
# # ============================

# def fix_retrieved_doc_id_from_chunkid(chunk_id: str) -> str:
#     if not chunk_id.startswith("muad_"):
#         return None

#     core = chunk_id[len("muad_"):]          # drop muad_
#     core = re.sub(r"_chunk\d+$", "", core)  # drop _chunk###
#     return f"maud/{core}_.txt"


# # ============================
# # MAIN FIXER
# # ============================

# def main():
#     data = json.loads(Path(RETRIEVAL_JSON_IN).read_text())

#     for qid, entry in data.items():
#         for rank, r in entry["retrieved_chunks"].items():

#             fixed = fix_retrieved_doc_id_from_chunkid(r["chunk_id"])
#             if fixed is not None:
#                 r["retrieved_doc_id"] = fixed

#     Path(RETRIEVAL_JSON_OUT).write_text(json.dumps(data, indent=2))
#     print(f"Saved fixed retrieval JSON to: {RETRIEVAL_JSON_OUT}")


# if __name__ == "__main__":
#     main()

import json
import re

def convert_chunk_id(chunk_id):
    # Remove the _chunk{number} suffix
    base_name = re.sub(r'_chunk\d+$', '', chunk_id)
    # Replace the first underscore with a forward slash
    if base_name.startswith("maud_"):
        base_name = base_name.replace("maud_", "maud/", 1)
    return f"{base_name}.txt"

# Load the json (replace 'data.json' with your actual filename)
# For the purpose of this script, assume 'data' contains the dictionary you provided.
with open(RETRIEVAL_JSON_IN, 'r') as f:
    data = json.load(f)

# Iterate through the JSON structure and update IDs
for query_key, query_val in data.items():
    if "retrieved_chunks" in query_val:
        chunks = query_val["retrieved_chunks"]
        for rank, chunk_data in chunks.items():
            original_chunk_id = chunk_data.get("chunk_id")
            if original_chunk_id:
                # Generate the correct doc_id
                new_doc_id = convert_chunk_id(original_chunk_id)
                
                # Update the retrieved_doc_id field
                chunk_data["retrieved_doc_id"] = new_doc_id

# Save the corrected JSON
with open(RETRIEVAL_JSON_OUT, 'w') as f:
    json.dump(data, f, indent=2)

print("JSON correction complete. Saved.")
