"""End-to-end RAG pipeline"""

import sys
sys.path.append('..')

from vectorstore.vector_store_factory import VectorStoreFactory
from generator.llama_infer import LlamaGenerator
from generator.formatter import PromptFormatter
from utils.timing import timing_decorator
import yaml


class RAGPipeline:
    """Complete RAG pipeline"""
    
    def __init__(self):
        # Load configs
        with open("../config/rag_config.yaml", 'r') as f:
            self.rag_config = yaml.safe_load(f)
        
        with open("../config/model_config.yaml", 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        with open("../config/paths.yaml", 'r') as f:
            self.paths = yaml.safe_load(f)
        
        # Initialize retriever based on index type
        full_config = {**self.rag_config, 'paths': self.paths}
        self.retriever = VectorStoreFactory.create_querier(
            index_type=self.rag_config['retrieval']['index_type'],
            config=full_config,
            embedding_model=self.rag_config['embedding']['model_name']
        )
        
        self.generator = LlamaGenerator(
            model_name=self.model_config['model']['name'],
            max_new_tokens=self.model_config['generation']['max_new_tokens'],
            temperature=self.model_config['generation']['temperature']
        )
        
        self.formatter = PromptFormatter()
    
    @timing_decorator
    def query(self, question: str, top_k: int = None) -> dict:
        """Execute RAG query"""
        if top_k is None:
            top_k = self.rag_config['retrieval']['top_k']
        
        # Retrieve
        print(f"Retrieving top-{top_k} documents...")
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
        
        # Format prompt
        prompt = self.formatter.format_rag_prompt(question, retrieved_docs)
        
        # Generate
        print("Generating answer...")
        answer = self.generator.generate(prompt)
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs
        }
    
    def close(self):
        """Cleanup"""
        self.retriever.close()


if __name__ == "__main__":
    pipeline = RAGPipeline()
    
    # Test query
    result = pipeline.query("What are the key terms in merger agreements?")
    
    print("\n" + "="*80)
    print(f"Question: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print("="*80)
    
    pipeline.close()
