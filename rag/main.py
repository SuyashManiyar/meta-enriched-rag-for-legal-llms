"""Main CLI entry point"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from pipeline.build_corpus import build_corpus
from pipeline.build_vectorstore import build_vectorstore
from pipeline.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument('command', choices=['build', 'index', 'query'], 
                       help='Command to execute')
    parser.add_argument('--query', type=str, help='Query text for query command')
    parser.add_argument('--top-k', type=int, default=5, help='Number of documents to retrieve')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        print("Building corpus...")
        build_corpus()
        
    elif args.command == 'index':
        print("Building vector store...")
        build_vectorstore()
        
    elif args.command == 'query':
        if not args.query:
            print("Error: --query required for query command")
            sys.exit(1)
        
        pipeline = RAGPipeline()
        result = pipeline.query(args.query, top_k=args.top_k)
        
        print("\n" + "="*80)
        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print("="*80)
        
        pipeline.close()


if __name__ == "__main__":
    main()
