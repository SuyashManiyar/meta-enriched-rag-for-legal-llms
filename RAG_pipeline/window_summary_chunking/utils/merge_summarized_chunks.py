"""
Merge all summarized chunk JSON files into a single file
"""
import os
import json


def merge_json_files(input_folder, output_file):
    """
    Merge all JSON files from input_folder into a single JSON file
    
    Args:
        input_folder: Folder containing individual JSON files
        output_file: Output path for merged JSON file
    """
    merged_data = {}
    
    # Get all JSON files
    json_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])
    
    print(f"Found {len(json_files)} JSON files to merge")
    print("=" * 70)
    
    total_chunks = 0
    
    for json_file in json_files:
        filepath = os.path.join(input_folder, json_file)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Merge into main dictionary
            merged_data.update(data)
            
            print(f"[OK] {json_file}: {len(data)} chunks")
            total_chunks += len(data)
            
        except Exception as e:
            print(f"[ERROR] Error reading {json_file}: {e}")
    
    # Save merged file
    print("\n" + "=" * 70)
    print(f"Saving merged file with {total_chunks} total chunks...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Saved to: {output_file}")
    print(f"[OK] Total chunks: {total_chunks}")
    print(f"[OK] File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    import sys
    
    # Default paths
    input_folder = "RAG_pipeline/window_summary_chunking/Summaries"
    output_file = "RAG_pipeline/window_summary_chunking/all_maud_chunks_summaries.json"
    
    # Allow custom paths via command line
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("Merging Summarized Chunks")
    print("=" * 70)
    print(f"Input folder: {input_folder}")
    print(f"Output file: {output_file}")
    print()
    
    merge_json_files(input_folder, output_file)
    
    print("\n[OK] Merge complete!")
