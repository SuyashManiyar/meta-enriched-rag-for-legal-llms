"""
Verify that MAUD chunk spans are correct
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from verify_spans import verify_all_chunks


if __name__ == "__main__":
    print("Verifying MAUD chunk spans...")
    print("=" * 70)
    
    verify_all_chunks(
        chunks_folder="RAG_pipeline/window_summary_chunking/maud_chunks_only",
        docs_folder="maud"
    )
