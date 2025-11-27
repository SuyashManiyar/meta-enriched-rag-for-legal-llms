"""Factory for creating vector stores"""

from .faiss_index import FAISSIndexBuilder, FAISSQuerier


class VectorStoreFactory:
    """Factory to create vector store based on config"""
    
    @staticmethod
    def create_builder(index_type: str, config: dict, embedding_model: str):
        """Create index builder"""
        if index_type == "faiss":
            index_path = config['paths']['data']['index']
            return FAISSIndexBuilder(config.get('faiss', {}), embedding_model, index_path)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    @staticmethod
    def create_querier(index_type: str, config: dict, embedding_model: str):
        """Create index querier"""
        if index_type == "faiss":
            index_path = config['paths']['data']['index']
            return FAISSQuerier(config.get('faiss', {}), embedding_model, index_path)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
