#!/usr/bin/env python3
"""
Add proper span information to ground truth for accurate RAG evaluation.
This ensures ground truth spans are compatible with chunk spans.
"""

import json
from typing import Dict, List, Tuple, Optional

class GroundTruthSpanEnhancer:
    def __init__(self):
        self.final_test_path = "Final_test_with_spans.json"
        self.chunks_path = "australian_legal_text_recursive_chunking.json"
        self.query_ids_path = "australian_legal_w_query_ids.json"
        self.output_path = "ground_truth_with_proper_spans.json"
        
    def load_data(self):
        """Load all required data files"""
        print("📂 Loading data files...")
        
        # Load Final_test with spans
        with open(self.final_test_path, 'r', encoding='utf-8') as f:
            self.final_test_data = json.load(f)
        print(f"✅ Loaded {len(self.final_test_data)} QA pairs from Final_test")
        
        # Load chunks
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            self.chunks_data = json.load(f)
        print(f"✅ Loaded {len(self.chunks_data)} chunks")
        
        # Load query IDs mapping
        with open(self.query_ids_path, 'r', encoding='utf-8') as f:
            self.query_ids_data = json.load(f)
        print(f"✅ Loaded {len(self.query_ids_data)} query mappings")
    
    def find_overlapping_chunks(self, ground_truth_span: List[int], doc_id: str) -> List[Dict]:
        """
        Find chunks that overlap with the ground truth span for a specific document
        """
        overlapping_chunks = []
        
        # Extract document number from doc_id (e.g., "001" from various formats)
        doc_number = self.extract_doc_number(doc_id)
        if not doc_number:
            return overlapping_chunks
        
        gt_start, gt_end = ground_truth_span
        
        # Find chunks for this document
        for chunk_id, chunk_data in self.chunks_data.items():
            if f"_{doc_number}_" in chunk_id:  # Match document number
                chunk_span = chunk_data.get('span', [])
                if len(chunk_span) == 2:
                    chunk_start, chunk_end = chunk_span
                    
                    # Check for overlap
                    overlap_start = max(gt_start, chunk_start)
                    overlap_end = min(gt_end, chunk_end)
                    
                    if overlap_start < overlap_end:  # There is overlap
                        overlap_length = overlap_end - overlap_start
                        chunk_length = chunk_end - chunk_start
                        gt_length = gt_end - gt_start
                        
                        # Calculate overlap percentages
                        overlap_pct_chunk = overlap_length / chunk_length if chunk_length > 0 else 0
                        overlap_pct_gt = overlap_length / gt_length if gt_length > 0 else 0
                        
                        overlapping_chunks.append({
                            'chunk_id': chunk_id,
                            'chunk_span': chunk_span,
                            'overlap_span': [overlap_start, overlap_end],
                            'overlap_length': overlap_length,
                            'overlap_pct_chunk': overlap_pct_chunk,
                            'overlap_pct_gt': overlap_pct_gt,
                            'chunk_text': chunk_data.get('chunk_text', '')[:200] + '...'  # Preview
                        })
        
        # Sort by overlap percentage with ground truth (descending)
        overlapping_chunks.sort(key=lambda x: x['overlap_pct_gt'], reverse=True)
        return overlapping_chunks
    
    def extract_doc_number(self, doc_id: str) -> Optional[str]:
        """Extract document number from various doc_id formats"""
        if not doc_id:
            return None
            
        # Try different patterns
        import re
        
        # Pattern 1: Direct number like "001", "003"
        if re.match(r'^\d{3}$', doc_id):
            return doc_id
            
        # Pattern 2: URL with year and case number
        url_match = re.search(r'/(\d{4})/(\d{4}[a-z]+\d+)', doc_id.lower())
        if url_match:
            year, case = url_match.groups()
            # Try to find matching document by year and case
            for chunk_id in self.chunks_data.keys():
                if year in chunk_id and case in chunk_id:
                    # Extract doc number from chunk_id
                    parts = chunk_id.split('_')
                    if len(parts) >= 4:
                        return parts[3]  # Should be 001, 003, etc.
        
        # Pattern 3: Citation matching
        # This would require more sophisticated matching
        
        return None
    
    def create_ground_truth_with_spans(self):
        """Create enhanced ground truth with proper span information"""
        print("\n🔧 Creating ground truth with proper spans...")
        
        enhanced_ground_truth = {}
        
        # Create reverse mapping from original QA ID to query ID
        qa_to_query_mapping = {}
        for query_id, query_data in self.query_ids_data.items():
            original_qa_id = query_data.get('original_qa_id')
            if original_qa_id:
                qa_to_query_mapping[original_qa_id] = query_id
        
        successful_mappings = 0
        failed_mappings = 0
        
        for qa_id, qa_data in self.final_test_data.items():
            print(f"\n📊 Processing QA {qa_id}")
            
            # Get span information
            span_info = qa_data.get('span_info', {})
            if span_info.get('status') != 'success':
                print(f"⚠️  No valid span for QA {qa_id}")
                failed_mappings += 1
                continue
            
            ground_truth_span = span_info.get('span', [])
            if len(ground_truth_span) != 2:
                print(f"⚠️  Invalid span format for QA {qa_id}")
                failed_mappings += 1
                continue
            
            # Get document ID
            doc_id = qa_data.get('source', {}).get('url', '') or qa_data.get('source', {}).get('citation', '')
            
            # Find overlapping chunks
            overlapping_chunks = self.find_overlapping_chunks(ground_truth_span, doc_id)
            
            if not overlapping_chunks:
                print(f"❌ No overlapping chunks found for QA {qa_id}")
                failed_mappings += 1
                continue
            
            print(f"✅ Found {len(overlapping_chunks)} overlapping chunks")
            
            # Get corresponding query ID
            query_id = qa_to_query_mapping.get(qa_id)
            
            # Create enhanced entry
            enhanced_entry = {
                'original_qa_id': qa_id,
                'query_id': query_id,
                'question': qa_data.get('question', ''),
                'answer': qa_data.get('answer', ''),
                'citation': qa_data.get('source', {}).get('citation', ''),
                'ground_truth_span': ground_truth_span,
                'ground_truth_similarity': span_info.get('similarity_score', 0.0),
                'document_path': span_info.get('document_path', ''),
                'overlapping_chunks': overlapping_chunks[:5],  # Top 5 overlapping chunks
                'best_chunk_match': overlapping_chunks[0] if overlapping_chunks else None,
                'total_overlapping_chunks': len(overlapping_chunks)
            }
            
            enhanced_ground_truth[qa_id] = enhanced_entry
            successful_mappings += 1
        
        # Save enhanced ground truth
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_ground_truth, f, indent=2, ensure_ascii=False)
        
        # Summary
        total = len(self.final_test_data)
        success_rate = (successful_mappings / total) * 100
        
        print(f"\n📊 GROUND TRUTH ENHANCEMENT SUMMARY:")
        print(f"   ✅ Successful mappings: {successful_mappings}/{total} ({success_rate:.1f}%)")
        print(f"   ❌ Failed mappings: {failed_mappings}")
        print(f"   📄 Output saved to: {self.output_path}")
        
        return enhanced_ground_truth
    
    def analyze_span_coverage(self, enhanced_ground_truth: Dict):
        """Analyze how well chunks cover ground truth spans"""
        print(f"\n📈 SPAN COVERAGE ANALYSIS:")
        
        high_coverage = 0  # >80% overlap
        medium_coverage = 0  # 50-80% overlap
        low_coverage = 0  # <50% overlap
        
        overlap_scores = []
        
        for qa_id, data in enhanced_ground_truth.items():
            best_match = data.get('best_chunk_match')
            if best_match:
                overlap_pct = best_match['overlap_pct_gt']
                overlap_scores.append(overlap_pct)
                
                if overlap_pct >= 0.8:
                    high_coverage += 1
                elif overlap_pct >= 0.5:
                    medium_coverage += 1
                else:
                    low_coverage += 1
        
        total_with_matches = len(overlap_scores)
        if total_with_matches > 0:
            avg_coverage = sum(overlap_scores) / total_with_matches
            print(f"   📊 Average overlap coverage: {avg_coverage:.1%}")
            print(f"   🟢 High coverage (≥80%): {high_coverage} ({high_coverage/total_with_matches:.1%})")
            print(f"   🟡 Medium coverage (50-79%): {medium_coverage} ({medium_coverage/total_with_matches:.1%})")
            print(f"   🔴 Low coverage (<50%): {low_coverage} ({low_coverage/total_with_matches:.1%})")
    
    def run(self):
        """Run the complete ground truth enhancement process"""
        print("🚀 GROUND TRUTH SPAN ENHANCEMENT")
        print("=" * 50)
        
        self.load_data()
        enhanced_ground_truth = self.create_ground_truth_with_spans()
        self.analyze_span_coverage(enhanced_ground_truth)
        
        print(f"\n🎉 Ground truth enhancement complete!")
        print(f"💡 Use '{self.output_path}' for evaluation with proper span alignment")

def main():
    enhancer = GroundTruthSpanEnhancer()
    enhancer.run()

if __name__ == "__main__":
    main()