"""
Verify that chunk spans are correct
Checks if text extracted using spans matches the chunk_text
"""
import os
import json
from typing import Dict, List, Tuple


def verify_chunk_spans(chunk_file: str, original_doc: str) -> Dict:
    """Verify spans for a single document"""
    
    # Load chunks
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Load original document
    with open(original_doc, "r", encoding="utf-8") as f:
        original_text = f.read()
    
    results = {
        "total_chunks": len(chunks),
        "correct": 0,
        "incorrect": 0,
        "errors": []
    }
    
    # Sort chunks by ID
    chunk_list = [(k, v) for k, v in chunks.items()]
    chunk_list.sort(key=lambda x: int(x[0].split("_chunk")[-1]))
    
    for chunk_id, chunk_data in chunk_list:
        span = chunk_data["span"]
        chunk_text = chunk_data["chunk_text"]
        
        # Extract text using span
        extracted_text = original_text[span[0]:span[1]]
        
        # Compare
        if extracted_text == chunk_text:
            results["correct"] += 1
        else:
            results["incorrect"] += 1
            
            # Find differences
            if len(extracted_text) != len(chunk_text):
                error = f"Length mismatch: extracted={len(extracted_text)}, chunk={len(chunk_text)}"
            else:
                # Find first difference
                for i, (c1, c2) in enumerate(zip(extracted_text, chunk_text)):
                    if c1 != c2:
                        error = f"First diff at pos {i}: extracted='{c1}' vs chunk='{c2}'"
                        break
                else:
                    error = "Unknown difference"
            
            results["errors"].append({
                "chunk_id": chunk_id,
                "span": span,
                "error": error,
                "extracted_preview": extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text,
                "chunk_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
            })
    
    return results


def verify_all_chunks(chunks_folder: str, docs_folder: str):
    """Verify all chunk files"""
    
    json_files = [f for f in os.listdir(chunks_folder) if f.endswith('.json')]
    
    print(f"Verifying {len(json_files)} chunk files\n")
    print("=" * 70)
    
    all_correct = 0
    all_incorrect = 0
    
    for json_file in json_files:
        chunk_file = os.path.join(chunks_folder, json_file)
        
        # Find corresponding original document
        doc_name = json_file.replace('.json', '.txt')
        original_doc = os.path.join(docs_folder, doc_name)
        
        if not os.path.exists(original_doc):
            print(f"[WARNING] Original document not found for {json_file}")
            continue
        
        # Verify
        results = verify_chunk_spans(chunk_file, original_doc)
        
        all_correct += results["correct"]
        all_incorrect += results["incorrect"]
        
        # Print results
        status = "[OK]" if results["incorrect"] == 0 else "[ERROR]"
        print(f"{status} {json_file}")
        print(f"  Correct: {results['correct']}/{results['total_chunks']}")
        
        if results["incorrect"] > 0:
            print(f"  [ERROR] Incorrect: {results['incorrect']}")
            print(f"  Errors:")
            for error in results["errors"][:3]:  # Show first 3 errors
                print(f"    - {error['chunk_id']}: {error['error']}")
                print(f"      Extracted: {error['extracted_preview']}")
                print(f"      Chunk:     {error['chunk_preview']}")
            if len(results["errors"]) > 3:
                print(f"    ... and {len(results['errors']) - 3} more errors")
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total chunks: {all_correct + all_incorrect}")
    print(f"[OK] Correct: {all_correct} ({all_correct/(all_correct+all_incorrect)*100:.1f}%)")
    print(f"[ERROR] Incorrect: {all_incorrect} ({all_incorrect/(all_correct+all_incorrect)*100:.1f}%)")
    
    if all_incorrect == 0:
        print("\n[SUCCESS] All spans are correct!")
    else:
        print(f"\n[WARNING] {all_incorrect} chunks have incorrect spans")


def quick_test_single_chunk(chunk_file: str, original_doc: str, chunk_id: str):
    """Quick test for a single chunk"""
    
    # Load chunks
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Load original document
    with open(original_doc, "r", encoding="utf-8") as f:
        original_text = f.read()
    
    if chunk_id not in chunks:
        print(f"Chunk {chunk_id} not found!")
        return
    
    chunk_data = chunks[chunk_id]
    span = chunk_data["span"]
    chunk_text = chunk_data["chunk_text"]
    
    # Extract using span
    extracted_text = original_text[span[0]:span[1]]
    
    print(f"Chunk ID: {chunk_id}")
    print(f"Span: {span}")
    print(f"Match: {extracted_text == chunk_text}")
    print(f"\nExtracted text (first 200 chars):")
    print(extracted_text[:200])
    print(f"\nChunk text (first 200 chars):")
    print(chunk_text[:200])
    
    if extracted_text == chunk_text:
        print("\n[OK] Span is correct!")
    else:
        print("\n[ERROR] Span is incorrect!")
        print(f"Length: extracted={len(extracted_text)}, chunk={len(chunk_text)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test mode
        quick_test_single_chunk(
            chunk_file="RAG_pipeline/window_summary_chunking/chunks_only/23andMe.json",
            original_doc="rag/data/privacy_qa/23andMe.txt",
            chunk_id="privacy_qa_23andMe_chunk1"
        )
    else:
        # Full verification
        verify_all_chunks(
            chunks_folder="RAG_pipeline/window_summary_chunking/maud_chunks_only",
            docs_folder="maud"
        )
