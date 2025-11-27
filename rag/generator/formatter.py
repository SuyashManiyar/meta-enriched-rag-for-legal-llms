"""Format retrieved documents into prompts"""

from typing import List, Dict


class PromptFormatter:
    """Format context and query into prompt"""
    
    @staticmethod
    def format_rag_prompt(query: str, retrieved_docs: List[Dict], template: str = None) -> str:
        """Format RAG prompt"""
        if template is None:
            template = """Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        prompt = template.format(context=context, query=query)
        return prompt
