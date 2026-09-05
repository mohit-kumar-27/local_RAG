"""
End-to-end Local RAG pipeline:
- Grounded, hallucination-resistant system prompt.
- Strict inline citation enforcement.
- Context window budgeting.
- Streaming token generator from local Ollama.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from config import CHAT_HISTORY_WINDOW_SIZE, get_active_llm_model
from ingestion.base import Document
from rag.hybrid_search import HybridSearcher
from rag.ollama_client import OllamaClient

SYSTEM_PROMPT = """You are an expert software engineering assistant working with proprietary, confidential company data.
Your job is to answer user queries with extreme technical precision based ONLY on the provided context passages.

CRITICAL OPERATIONAL RULES:
1. STRICT GROUNDEDNESS: Answer strictly and exclusively from the provided context. If the context does not contain the answer, or if information is ambiguous, state clearly: "Based on the provided context, I cannot find information regarding [specific topic]." Do NOT hallucinate, infer unmentioned APIs, or invent ticket numbers.
2. MANDATORY INLINE CITATIONS: Every claim, explanation, code reference, or status update MUST be followed immediately by an inline citation using the exact citation tag specified for each context passage, e.g.:
   - "[Source: src/auth.py:12-45]" for source code
   - "[Source: ADO Bug #1042 - Null pointer in auth]" for Azure DevOps work items
   - "[Source: Confluence - Architecture Guide > Storage Layer]" for Confluence wiki pages
3. CODE SNIPPETS: When providing code, use markdown syntax with appropriate language identifiers.
4. CONFIDENTIALITY: Do not mention any external cloud LLM providers. All data remains strictly local.
"""


def format_context_passages(documents_with_scores: List[Tuple[Document, float]]) -> str:
    """Formats retrieved context documents with explicit citation headers for the prompt."""
    if not documents_with_scores:
        return "No relevant context found in local database.\n"

    formatted_parts: List[str] = []
    for idx, (doc, score) in enumerate(documents_with_scores, start=1):
        citation = doc.get_citation_tag()
        header = (
            f"=== CONTEXT PASSAGE {idx} ===\n"
            f"Citation: {citation}\n"
            f"Source Type: {doc.doc_type}\n"
            f"Source URL: {doc.source_url}\n"
            f"Relevance Score: {score:.4f}\n"
        )
        if doc.file_path:
            header += f"File: {doc.file_path}\n"
        if doc.sprint_id:
            header += f"Sprint/Iteration: {doc.sprint_id}\n"
        if doc.work_item_id:
            header += f"Work Item ID: {doc.work_item_id}\n"

        passage_str = f"{header}\n{doc.content.strip()}\n"
        formatted_parts.append(passage_str)

    return "\n" + "\n----------------------------------------\n".join(formatted_parts) + "\n"


class RAGPipeline:
    """
    Coordinates Hybrid Search, Prompt Assembly, and Ollama Streaming Chat.
    """

    def __init__(self, hybrid_searcher: HybridSearcher, ollama_client: OllamaClient):
        self.searcher = hybrid_searcher
        self.ollama = ollama_client

    async def retrieve_context(
        self,
        query: str,
        doc_type_filter: Optional[str] = None,
        sprint_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieves top reranked context documents for the query."""
        return await self.searcher.search(
            query=query,
            doc_type_filter=doc_type_filter,
            sprint_filter=sprint_filter,
            source_filter=source_filter,
        )

    async def answer_stream(
        self,
        query: str,
        chat_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        doc_type_filter: Optional[str] = None,
        sprint_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> Tuple[List[Tuple[Document, float]], AsyncGenerator[str, None]]:
        """
        Retrieves context, formats prompt, and returns (retrieved_documents, token_stream).
        Includes sliding-window prior chat history for multi-turn continuity.
        """
        # 1. Retrieve top passages
        retrieved_docs = await self.retrieve_context(
            query=query,
            doc_type_filter=doc_type_filter,
            sprint_filter=sprint_filter,
            source_filter=source_filter,
        )

        # 2. Build augmented user message
        context_text = format_context_passages(retrieved_docs)
        user_prompt = (
            f"Here is the context retrieved from local repositories and documents:\n"
            f"{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Provide a thorough, grounded answer with inline citations for every statement."
        )

        # 3. Assemble chat messages with sliding-window history
        messages: List[Dict[str, str]] = []
        if chat_history is not None:
            for msg in chat_history[-CHAT_HISTORY_WINDOW_SIZE:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        elif chat_id and hasattr(self.searcher, "store"):
            prior_records = self.searcher.store.get_chat_messages(chat_id, limit=CHAT_HISTORY_WINDOW_SIZE + 2)
            # If the current query was already inserted into the DB before streaming, exclude it from prior context
            if prior_records and prior_records[-1].role == "user" and prior_records[-1].content.strip() == query.strip():
                prior_records = prior_records[:-1]
            for record in prior_records[-CHAT_HISTORY_WINDOW_SIZE:]:
                messages.append({"role": record.role, "content": record.content})

        messages.append({"role": "user", "content": user_prompt})

        # 4. Return retrieved documents and token generator
        stream_generator = self.ollama.stream_chat(
            messages=messages,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        )

        return retrieved_docs, stream_generator
