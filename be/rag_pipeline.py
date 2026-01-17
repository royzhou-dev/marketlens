import google.generativeai as genai
import faiss
import numpy as np
import json
import os
import traceback
from pathlib import Path
from config import (
    GEMINI_API_KEY,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    RAG_TOP_K
)


class EmbeddingGenerator:
    """Generates embeddings using Google's Gemini embedding API (FREE)"""

    def __init__(self, api_key=None, model=EMBEDDING_MODEL):
        genai.configure(api_key=api_key or GEMINI_API_KEY)
        self.model = model

    def generate_embedding(self, text):
        """
        Generate embedding vector for text

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Truncate text if too long (Google has ~10k token limit)
            if len(text) > 25000:
                text = text[:25000]

            result = genai.embed_content(
                model=f"models/{self.model}",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def generate_query_embedding(self, text):
        """
        Generate embedding vector for a query (uses retrieval_query task type)

        Args:
            text: Query text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            result = genai.embed_content(
                model=f"models/{self.model}",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return None


class VectorStore:
    """Manages FAISS vector database for RAG"""

    def __init__(self, index_path=None):
        self.index_path = Path(index_path or FAISS_INDEX_PATH)
        self.index_file = self.index_path / "index.faiss"
        self.metadata_file = self.index_path / "metadata.json"
        self.doc_ids_file = self.index_path / "doc_ids.json"

        self.dimension = 768  # Google text-embedding-004
        self.index = None
        self.metadata = {}  # {internal_id: metadata_dict}
        self.doc_id_to_index = {}  # {doc_id: internal_id}
        self.next_id = 0  # Counter for internal IDs

        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load FAISS index from disk"""
        try:
            # Create directory if doesn't exist
            self.index_path.mkdir(parents=True, exist_ok=True)

            # Check if index files exist
            if self.index_file.exists():
                try:
                    # Load existing index
                    self.index = faiss.read_index(str(self.index_file))

                    # Load metadata
                    if self.metadata_file.exists():
                        with open(self.metadata_file, 'r') as f:
                            # Convert string keys back to int
                            loaded = json.load(f)
                            self.metadata = {int(k): v for k, v in loaded.items()}

                    # Load doc_id mapping
                    if self.doc_ids_file.exists():
                        with open(self.doc_ids_file, 'r') as f:
                            self.doc_id_to_index = json.load(f)

                    # Set next_id to max existing ID + 1
                    if self.metadata:
                        self.next_id = max(self.metadata.keys()) + 1

                    print(f"Loaded FAISS index from {self.index_path} with {self.index.ntotal} vectors")
                except Exception as e:
                    print(f"Warning: Corrupted index file, creating fresh index: {e}")
                    # Delete corrupted files
                    if self.index_file.exists():
                        self.index_file.unlink()
                    if self.metadata_file.exists():
                        self.metadata_file.unlink()
                    if self.doc_ids_file.exists():
                        self.doc_ids_file.unlink()
                    # Create fresh index
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.metadata = {}
                    self.doc_id_to_index = {}
                    self.next_id = 0
            else:
                # Create new index
                # Use IndexFlatIP for cosine similarity (requires L2 normalized vectors)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = {}
                self.doc_id_to_index = {}
                self.next_id = 0
                print(f"Created new FAISS index at {self.index_path}")

        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
            raise

    def upsert_document(self, doc_id, embedding, metadata, namespace="news"):
        """
        Store document embedding with metadata in FAISS

        Args:
            doc_id: Unique document identifier
            embedding: Embedding vector (1536 dimensions)
            metadata: Dictionary of metadata
            namespace: Namespace for organization (stored in doc_id prefix)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add namespace to doc_id for organization
            full_doc_id = f"{namespace}:{doc_id}"

            # Check if document already exists (update case)
            if full_doc_id in self.doc_id_to_index:
                # For FAISS, we'll skip true updates and just log
                # (Alternative: remove old and add new, but requires index rebuild)
                print(f"Document {full_doc_id} already exists, skipping update")
                return True

            # Convert to numpy array and normalize for cosine similarity
            vector = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(vector)  # L2 normalize for IndexFlatIP

            # Add to FAISS index
            self.index.add(vector)

            # Store metadata
            internal_id = self.next_id
            self.metadata[internal_id] = metadata.copy()
            self.metadata[internal_id]['doc_id'] = full_doc_id  # Store doc_id in metadata

            # Update doc_id mapping
            self.doc_id_to_index[full_doc_id] = internal_id

            # Increment counter
            self.next_id += 1

            return True

        except Exception as e:
            print(f"Error upserting document {doc_id}: {e}")
            traceback.print_exc()
            return False

    def search(self, query_embedding, ticker=None, doc_type=None, top_k=None, namespace="news"):
        """
        Search for similar documents

        Args:
            query_embedding: Query embedding vector
            ticker: Filter by ticker symbol
            doc_type: Filter by document type
            top_k: Number of results to return
            namespace: Namespace to search

        Returns:
            List of matching documents with metadata (Pinecone-compatible format)
        """
        try:
            if self.index.ntotal == 0:
                return []

            # Normalize query vector for cosine similarity
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            # Calculate search_k (fetch more to allow for filtering)
            k = top_k or RAG_TOP_K
            # Fetch 5x more results to account for filtering
            search_k = min(k * 5, self.index.ntotal)

            # Search FAISS index
            distances, indices = self.index.search(query_vector, search_k)

            # Build results list with filtering
            matches = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue

                if idx not in self.metadata:
                    continue

                meta = self.metadata[idx].copy()
                doc_id = meta.get('doc_id', '')

                # Apply namespace filter
                if not doc_id.startswith(f"{namespace}:"):
                    continue

                # Apply ticker filter
                if ticker and meta.get('ticker') != ticker:
                    continue

                # Apply doc_type filter
                if doc_type and meta.get('type') != doc_type:
                    continue

                # Remove doc_id from metadata (stored separately)
                meta.pop('doc_id', None)

                # Create Pinecone-compatible match object
                match = type('Match', (), {
                    'id': doc_id.replace(f"{namespace}:", ""),  # Remove namespace prefix
                    'score': float(dist),  # Cosine similarity score
                    'metadata': meta
                })()

                matches.append(match)

                # Stop if we have enough matches
                if len(matches) >= k:
                    break

            return matches

        except Exception as e:
            print(f"Error searching FAISS: {e}")
            return []

    def document_exists(self, doc_id, namespace="news"):
        """
        Check if document already exists in index

        Args:
            doc_id: Document identifier
            namespace: Namespace

        Returns:
            Boolean indicating if document exists
        """
        try:
            full_doc_id = f"{namespace}:{doc_id}"
            return full_doc_id in self.doc_id_to_index
        except Exception as e:
            print(f"Error checking document existence: {e}")
            return False

    def delete_by_ticker(self, ticker, namespace="news"):
        """
        Delete all documents for a ticker
        Note: FAISS doesn't support deletion, so we rebuild the index

        Args:
            ticker: Ticker symbol
            namespace: Namespace
        """
        try:
            # Find indices to keep
            indices_to_keep = []
            metadata_to_keep = {}
            doc_ids_to_keep = {}
            new_id = 0

            for internal_id, meta in self.metadata.items():
                doc_id = meta.get('doc_id', '')

                # Skip if wrong namespace
                if not doc_id.startswith(f"{namespace}:"):
                    indices_to_keep.append(internal_id)
                    metadata_to_keep[new_id] = meta.copy()
                    doc_ids_to_keep[doc_id] = new_id
                    new_id += 1
                    continue

                # Skip if matching ticker
                if meta.get('ticker') == ticker:
                    continue

                # Keep this document
                indices_to_keep.append(internal_id)
                metadata_to_keep[new_id] = meta.copy()
                doc_ids_to_keep[doc_id] = new_id
                new_id += 1

            if len(indices_to_keep) == len(self.metadata):
                print(f"No documents found for ticker {ticker}")
                return

            # Rebuild index with remaining vectors
            new_index = faiss.IndexFlatIP(self.dimension)

            # Extract and re-add vectors
            for old_id in indices_to_keep:
                # Get vector from old index
                vector = self.index.reconstruct(old_id)
                # Reshape and add to new index
                vector = vector.reshape(1, -1)
                new_index.add(vector)

            # Replace old index and metadata
            self.index = new_index
            self.metadata = metadata_to_keep
            self.doc_id_to_index = doc_ids_to_keep
            self.next_id = new_id

            deleted_count = len(self.metadata) - len(metadata_to_keep)
            print(f"Deleted {deleted_count} documents for {ticker}")

        except Exception as e:
            print(f"Error deleting documents for {ticker}: {e}")

    def save(self):
        """
        Manually save index and metadata to disk
        Call this after batch operations or on graceful shutdown
        """
        try:
            # Ensure directory exists
            self.index_path.mkdir(parents=True, exist_ok=True)

            # Write to temporary files first for atomic saves
            temp_index = str(self.index_file) + ".tmp"
            temp_metadata = str(self.metadata_file) + ".tmp"
            temp_docids = str(self.doc_ids_file) + ".tmp"

            # Save FAISS index
            faiss.write_index(self.index, temp_index)

            # Save metadata (convert int keys to strings for JSON)
            with open(temp_metadata, 'w') as f:
                json.dump({str(k): v for k, v in self.metadata.items()}, f, indent=2)

            # Save doc_id mapping
            with open(temp_docids, 'w') as f:
                json.dump(self.doc_id_to_index, f, indent=2)

            # Atomic rename (overwrite old files)
            os.replace(temp_index, str(self.index_file))
            os.replace(temp_metadata, str(self.metadata_file))
            os.replace(temp_docids, str(self.doc_ids_file))

            print(f"Saved FAISS index with {self.index.ntotal} vectors to {self.index_path}")
            return True

        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            # Clean up temp files
            for temp_file in [temp_index, temp_metadata, temp_docids]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            return False

    def get_stats(self):
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_metadata": len(self.metadata),
            "index_type": "IndexFlatIP",
            "metric": "cosine"
        }


class ContextRetriever:
    """High-level interface for retrieving relevant context"""

    def __init__(self, vector_store=None):
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()

    def retrieve_context(self, query, ticker, doc_type=None, top_k=None):
        """
        Retrieve relevant documents for a query

        Args:
            query: User query text
            ticker: Ticker symbol to filter by
            doc_type: Optional document type filter
            top_k: Number of results to return

        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding (use query-specific embedding for better retrieval)
        query_embedding = self.embedding_gen.generate_query_embedding(query)

        if not query_embedding:
            return []

        # Search vector store
        matches = self.vector_store.search(
            query_embedding=query_embedding,
            ticker=ticker,
            doc_type=doc_type,
            top_k=top_k
        )

        # Format results
        contexts = []
        for match in matches:
            contexts.append({
                'score': match.score,
                'metadata': match.metadata,
                'id': match.id
            })

        return contexts
