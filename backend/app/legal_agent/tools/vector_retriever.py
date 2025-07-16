import logging
import chromadb
import torch
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class VectorRetriever:
    def __init__(self, settings):
        self.settings = settings
        self.device = self._get_device()
        self.embedding_function = self._get_embedding_function()
        self.client = None
        self.collection = None

        try:
            logger.info(f"Connecting to ChromaDB at: {self.settings.CHROMA_PERSIST_PATH}")
            self.client = chromadb.PersistentClient(path=str(self.settings.CHROMA_PERSIST_PATH))
            
            self.collection = self.client.get_collection(
                name=self.settings.CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_function,
            )
            
            logger.info(
                f"✅ Connected to collection '{self.settings.CHROMA_COLLECTION_NAME}'. "
                f"Total items: {self.collection.count()}. "
                f"Using device: '{self.device.upper()}'."
            )
        except Exception as e:
            logger.error(f"❌ Critical error connecting to ChromaDB: {e}", exc_info=True)
            logger.error("Please ensure you have run the build_vector_database script and the config path is correct.")
            raise

    def _get_device(self):
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected. Using 'cuda'.")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Silicon (MPS) detected. Using 'mps'.")
            return "mps"
        logger.info("No GPU detected. Using 'cpu'.")
        return "cpu"

    def _get_embedding_function(self):
        logger.info(f"Using embedding model: {self.settings.EMBEDDING_MODEL_NAME}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.settings.EMBEDDING_MODEL_NAME,
            device=self.device,
            normalize_embeddings=False,
        )

    def search(self, query_text: str, n_results: int = 5) -> list[dict]:
        if not self.collection:
            logger.warning("Collection does not exist, search cannot be performed.")
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["metadatas", "distances"],
            )
            
            formatted_results = []
            if results and results["ids"][0]:
                for i, res_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    
                    similarity = 1 - distance
                    
                    
                    if similarity > 0.4:
                        clean_result = {
                            "id": res_id,
                            "document_name": metadata.get("document_name"),
                            "content": metadata.get("content"),
                            "context": metadata.get("context"),
                            "similarity": round(similarity, 4),
                        }
                        formatted_results.append(clean_result)
        
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}", exc_info=True)
            return []
