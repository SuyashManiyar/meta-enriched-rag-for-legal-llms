"""
Add document-level summary to chunks from generate_summary/metadata/ folder
Only adds the high_level_summary field
"""
import os
import json


def add_doc_summary_to_chunks(chunks_folder, metadata_folder, output_folder):
    """
    Add document-level summary to chunks
    
    Args:
        chunks_folder: Folder with chunk JSON files
        metadata_folder: Folder with metadata JSON files
        output_folder: Output folder for chunks with doc summary
    """

    os.makedirs(output_folder, exist_ok=True)

    # Get all chunk files
    chunk_files = [f for f in os.listdir(chunks_folder) if f.endswith(".json")]

    print(f"Processing {len(chunk_files)} chunk files")
    print("=" * 70)

    for chunk_file in chunk_files:
        # Extract document name from chunk filename
        # e.g., "23andMe_with_summaries.json" -> "23andMe"
        doc_name = chunk_file.replace("_with_summaries.json", "").replace(".json", "")

        # Find corresponding metadata file
        metadata_file = os.path.join(metadata_folder, f"{doc_name}_metadata.json")

        if not os.path.exists(metadata_file):
            print(f"[WARNING] No metadata found for {doc_name}, skipping...")
            continue

        # Load metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Get document summary
        doc_summary = metadata.get("high_level_summary", "")

        if not doc_summary:
            print(f"[WARNING] No high_level_summary found for {doc_name}, skipping...")
            continue

        # Load chunks
        chunk_path = os.path.join(chunks_folder, chunk_file)
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Add document summary to each chunk
        for chunk_id, chunk_data in chunks.items():
            chunk_text = chunk_data.get("chunk_text", "")

            # Create metadata section with document summary
            metadata_section = f"<metadata>\nDocument Summary: {doc_summary}\n</metadata>"

            # Update chunk
            chunk_data["chunk_text_with_metadata"] = f"{chunk_text}\n\n{metadata_section}"
            chunk_data["document_summary"] = doc_summary

            # Remove window_chunks property if it exists
            if "window_chunks" in chunk_data:
                del chunk_data["window_chunks"]

        # Save updated chunks
        output_path = os.path.join(output_folder, chunk_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"[OK] {chunk_file}: {len(chunks)} chunks updated")

    print("\n" + "=" * 70)
    print("[OK] All chunks updated with document summary!")


if __name__ == "__main__":
    import sys

    # Default paths
    chunks_folder = "RAG_pipeline/window_summary_chunking/chunks_with_summaries"
    metadata_folder = "RAG_pipeline/generate_summary/metadata"
    output_folder = "RAG_pipeline/window_summary_chunking/chunks_with_doc_summary"

    # Allow custom paths
    if len(sys.argv) > 1:
        chunks_folder = sys.argv[1]
    if len(sys.argv) > 2:
        metadata_folder = sys.argv[2]
    if len(sys.argv) > 3:
        output_folder = sys.argv[3]

    print("Adding Document-Level Summary to Chunks")
    print("=" * 70)
    print(f"Chunks folder: {chunks_folder}")
    print(f"Metadata folder: {metadata_folder}")
    print(f"Output folder: {output_folder}")
    print()

    add_doc_summary_to_chunks(
        chunks_folder=chunks_folder,
        metadata_folder=metadata_folder,
        output_folder=output_folder,
    )
