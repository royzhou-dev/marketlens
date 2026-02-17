"""
ReAct-style agent service that replaces the static RAG chat pipeline.
Uses Gemini function calling to dynamically fetch data through tools.
"""

import json
import logging

from google.genai import types as genai_types
from llm_client import AgentLLMClient, ConversationManager
from agent_tools import TOOL_DECLARATIONS, ToolExecutor
from rag_pipeline import VectorStore, ContextRetriever
from polygon_api import PolygonAPI
from config import AGENT_MAX_ITERATIONS

logger = logging.getLogger(__name__)


class AgentService:
    """Orchestrates the ReAct agent loop with tool calling."""

    def __init__(self):
        self.polygon = PolygonAPI()
        self.vector_store = VectorStore()
        self.context_retriever = ContextRetriever(vector_store=self.vector_store)
        self.llm_client = AgentLLMClient()
        self.conversation_manager = ConversationManager()
        self.tool_executor = ToolExecutor(
            polygon_api=self.polygon,
            context_retriever=self.context_retriever,
            vector_store=self.vector_store,
        )

    def process_message(self, ticker, message, frontend_context, conversation_id):
        """
        Process a user message through the ReAct agent loop.

        Yields (event_type, data) tuples for SSE streaming:
          - ("tool_call", {"tool": str, "args": dict, "status": "calling"|"complete"|"error"})
          - ("text", str)
          - ("done", {})

        Args:
            ticker: Stock ticker symbol
            message: User message
            frontend_context: Cached data from frontend (Layer 1 cache)
            conversation_id: Conversation session ID
        """
        try:
            ticker = ticker.upper() if ticker else ""

            # Prime the tool executor with frontend context (Layer 1)
            self.tool_executor.set_context(frontend_context, ticker)

            # Build conversation contents from history
            history = self.conversation_manager.get_history(conversation_id)
            contents = self.llm_client.history_to_contents(history)

            # Append user message
            contents.append(self.llm_client.make_user_content(message))

            # Build config with tools
            config = self.llm_client.build_config(TOOL_DECLARATIONS, ticker)

            # ReAct loop
            for iteration in range(AGENT_MAX_ITERATIONS):
                logger.info(f"Agent iteration {iteration + 1}/{AGENT_MAX_ITERATIONS}")

                response = self.llm_client.generate(contents, config)
                function_calls, text_parts, response_content = self.llm_client.extract_parts(response)

                if not function_calls:
                    # Final text response — stream it in chunks
                    final_text = "".join(text_parts)
                    if not final_text:
                        final_text = "I wasn't able to generate a response. Please try again."

                    chunk_size = 20
                    for i in range(0, len(final_text), chunk_size):
                        yield ("text", final_text[i:i + chunk_size])

                    yield ("done", {})

                    # Save to conversation history (only user message + final text)
                    self.conversation_manager.add_message(conversation_id, "user", message)
                    self.conversation_manager.add_message(conversation_id, "assistant", final_text)
                    return

                # Process function calls
                contents.append(response_content)

                tool_response_parts = []
                for part in function_calls:
                    fc = part.function_call
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}

                    # Notify frontend: tool call starting
                    yield ("tool_call", {
                        "tool": tool_name,
                        "args": tool_args,
                        "status": "calling",
                    })

                    # Execute the tool
                    result = self.tool_executor.execute(tool_name, tool_args)

                    # Notify frontend: tool call complete
                    if "error" in result:
                        yield ("tool_call", {
                            "tool": tool_name,
                            "status": "error",
                            "error": result["error"],
                        })
                    else:
                        yield ("tool_call", {
                            "tool": tool_name,
                            "status": "complete",
                        })

                    # Build function response Part for Gemini
                    tool_response_parts.append(
                        genai_types.Part.from_function_response(
                            name=tool_name, response={"result": result}
                        )
                    )

                # Add all tool responses as a single Content with multiple Parts
                # Gemini requires all function responses for a turn in one Content object
                contents.append(genai_types.Content(role="tool", parts=tool_response_parts))

            # Hit max iterations — provide a fallback
            yield ("text", "I gathered some information but reached the maximum number of analysis steps. Please try a more specific question.")
            yield ("done", {})

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            yield ("error", {"message": f"An error occurred: {str(e)}"})

    def scrape_and_embed_articles(self, ticker, articles):
        """
        Background job to scrape and embed news articles into FAISS.
        Delegates to the same logic as the old ChatService.
        """
        from scraper import ArticleScraper
        from rag_pipeline import EmbeddingGenerator
        import concurrent.futures
        import hashlib

        scraper = ArticleScraper()
        embedding_gen = EmbeddingGenerator()

        results = {"scraped": 0, "embedded": 0, "failed": 0, "skipped": 0}

        def process_article(article):
            try:
                article_url = article.get("article_url", "")
                doc_id = f"{ticker}_news_{hashlib.md5(article_url.encode()).hexdigest()[:12]}"

                if self.vector_store.document_exists(doc_id):
                    return "skipped"

                content = scraper.scrape_article(article_url)
                if not content:
                    content = article.get("description", "")
                    if not content or len(content) < 50:
                        return "failed"

                embedding = embedding_gen.generate_embedding(content)
                if not embedding:
                    return "failed"

                metadata = {
                    "ticker": ticker,
                    "type": "news_article",
                    "title": article.get("title", ""),
                    "url": article_url,
                    "published_date": article.get("published_utc", ""),
                    "source": article.get("publisher", {}).get("name", "Unknown"),
                    "content_preview": content[:200],
                    "full_content": content,
                }

                success = self.vector_store.upsert_document(doc_id, embedding, metadata)
                return "embedded" if success else "failed"

            except Exception as e:
                logger.error(f"Error processing article: {e}")
                return "failed"

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_article, a) for a in articles[:20]]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status == "embedded":
                    results["embedded"] += 1
                    results["scraped"] += 1
                elif status == "skipped":
                    results["skipped"] += 1
                elif status == "failed":
                    results["failed"] += 1

        if results["embedded"] > 0:
            self.vector_store.save()

        return results
