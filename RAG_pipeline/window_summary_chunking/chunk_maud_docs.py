"""
Chunk MAUD legal documents using the same recursive chunking approach
"""
import os
import sys

# Add parent directory to path to import chunking_with_span
sys.path.insert(0, os.path.dirname(__file__))

from chunking_with_span import chunk_folder_llama_json_recursive_docwise


if __name__ == "__main__":
    print("Chunking MAUD legal documents...")
    print("=" * 60)
    
    chunk_folder_llama_json_recursive_docwise(
        input_folder="maud",
        output_folder="RAG_pipeline/window_summary_chunking/maud_chunks_only",
        chunk_size=380,
        window=50,
        model_name="thenlper/gte-large"
    )
    
    print("\n" + "=" * 60)
    print("✓ All MAUD documents chunked successfully!")
    print(f"✓ Output: RAG_pipeline/window_summary_chunking/maud_chunks_only/")
