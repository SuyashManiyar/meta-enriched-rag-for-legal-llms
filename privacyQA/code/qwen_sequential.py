import pandas as pd
from vllm import LLM, SamplingParams
import json
import re
import time


# =========================================================
# === 0. USER-DEFINED PATHS ===============================
# =========================================================
INPUT_FILE = "/home/smaniyar_umass_edu/BioNLP_Ontology/nlp/privacyQA/data/PrivacyQA_EMNLP/sampled_data/stratified_test_sample_1000.csv"
OUTPUT_FILE = "/home/smaniyar_umass_edu/BioNLP_Ontology/nlp/privacyQA/test_results/qwen_2_5_privacy_qa.csv"
LOG_FILE = "/home/smaniyar_umass_edu/BioNLP_Ontology/nlp/privacyQA/test_results/qwen_2_5_privacy_qa.log"
# =========================================================


# === 1. Load dataset ===
df = pd.read_csv(INPUT_FILE)

# === 2. Create unique ID ===
df["id"] = (
    df["DocID"].astype(str) + "_" +
    df["QueryID"].astype(str) + "_" +
    df["SentID"].astype(str)
)

# === 3. Build classification prompt ===
def make_prompt(query: str, segment: str) -> str:
    return f"""
You are a legal expert, classifier that determines if a segment of text from a document is relevant to a user query.

Query: "{query}"
Segment: "{segment}"

Decide whether the segment is RELEVANT or IRRELEVANT to the query.

Respond in JSON only, in this format:
{{
  "reasoning": "Your reasoning here in one or two sentences",
  "final_label": "Relevant" | "Irrelevant"
}}
""".strip()


# === 4. Extract reasoning + label ===
def extract_reasoning_and_label(output_text: str):
    output_text = output_text.strip()

    # Try to isolate the JSON part
    match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if not match:
        return None, None

    json_str = match.group(0)

    # Try parsing JSON safely
    try:
        data = json.loads(json_str)
        reasoning = data.get("reasoning")
        final_label = data.get("final_label")
        return reasoning, final_label
    except json.JSONDecodeError:
        # fallback: regex extraction if JSON is invalid or duplicated
        reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"]+)"', json_str)
        label_match = re.search(r'"?final_label"?\s*:\s*"([^"]+)"', json_str)

        reasoning = reasoning_match.group(1) if reasoning_match else None
        final_label = label_match.group(1) if label_match else None
        return reasoning, final_label



# === 5. Initialize model ===
llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    dtype="bfloat16",            # for A100 or Ampere GPUs
    gpu_memory_utilization=0.7,
    enforce_eager=True,
)

# === 6. Sampling params ===
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# === 7. Prepare dataframe columns ===
df["model_output"] = None
df["reasoning"] = None
df["final_label"] = None

# === 8. Initialize log file ===
with open(LOG_FILE, "w", encoding="utf-8") as log:
    log.write("========================================\n")
    log.write("Qwen 2.5 Privacy QA Classification Log\n")
    log.write("========================================\n\n")

# === 9. Sequential processing ===
for i, row in df.iterrows():
    query = str(row["Query"])
    segment = str(row["Segment"])
    doc_id = row["id"]
    prompt = make_prompt(query, segment)

    print(f"\n--- Processing {i+1}/{len(df)} ---")
    print(f"ID: {doc_id}")
    print(f"Query: {query}")

    # Run model inference
    output = llm.generate(prompt, sampling_params=params)
    raw_output = output[0].outputs[0].text.strip()

    # Extract reasoning + label
    reasoning, final_label = extract_reasoning_and_label(raw_output)

    # Store results
    df.at[i, "model_output"] = raw_output
    df.at[i, "reasoning"] = reasoning
    df.at[i, "final_label"] = final_label

    # Log full record
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"ID: {doc_id}\n")
        log.write(f"PROMPT:\n{prompt}\n")
        log.write(f"RAW OUTPUT:\n{raw_output}\n")
        log.write(f"REASONING: {reasoning}\n")
        log.write(f"FINAL LABEL: {final_label}\n")
        log.write("------------------------------------------------------------\n\n")

    # === Clear GPU cache after each prompt ===
    if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "llm_cache"):
        llm.llm_engine.llm_cache.clear()

    time.sleep(0.5)

# === 10. Save results ===
df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Done!")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"Log saved to: {LOG_FILE}")
print(df[["id", "reasoning", "final_label"]].head())
