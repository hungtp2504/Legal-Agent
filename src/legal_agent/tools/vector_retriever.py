
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
            logger.info(
                f"Đang kết nối tới ChromaDB tại: {self.settings.CHROMA_PERSIST_PATH}"
            )
            
            self.client = chromadb.PersistentClient(
                path=str(self.settings.CHROMA_PERSIST_PATH)
            )
            

            self.collection = self.client.get_collection(
                name=self.settings.CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_function,
            )

            logger.info(
                f"✅ Kết nối thành công tới collection '{self.settings.CHROMA_COLLECTION_NAME}'. "
                f"Tổng số mục: {self.collection.count()}. "
                f"Đang sử dụng device: '{self.device.upper()}'."
            )

        except Exception as e:
            logger.error(
                f"❌ Lỗi nghiêm trọng khi kết nối tới ChromaDB: {e}", exc_info=True
            )
            logger.error(
                "Vui lòng đảm bảo bạn đã chạy script `03_build_vector_database.py` và đường dẫn cấu hình là chính xác."
            )
            raise

    def _get_device(self):
        if torch.cuda.is_available():
            logger.info("Phát hiện GPU CUDA. Sẽ sử dụng 'cuda'.")
            return "cuda"
        
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Phát hiện Apple Silicon (MPS). Sẽ sử dụng 'mps'.")
            return "mps"
        logger.info("Không phát hiện GPU. Sẽ sử dụng 'cpu'.")
        return "cpu"

    def _get_embedding_function(self):
        logger.info(f"Sử dụng embedding model: {self.settings.EMBEDDING_MODEL_NAME}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.settings.EMBEDDING_MODEL_NAME,
            device=self.device,
            normalize_embeddings=False,
        )

    def search(self, query_text: str, n_results: int = 5) -> list[dict]:
        if not self.collection:
            logger.warning("Collection không tồn tại, không thể tìm kiếm.")
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
            logger.error(f"Lỗi khi thực hiện query trên ChromaDB: {e}", exc_info=True)
            return []
