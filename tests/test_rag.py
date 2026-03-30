"""
Tests for the RAG pipeline (embeddings, vector store, retriever).
"""

import pytest
from unittest.mock import patch, MagicMock


class TestDisruptionRetriever:
    """Tests for the DisruptionRetriever class."""

    @patch("src.rag.retriever.DisruptionVectorStore")
    def test_query_returns_documents(self, mock_vstore_cls):
        """Retriever.query should call similarity_search and return results."""
        from src.rag.retriever import DisruptionRetriever
        from langchain_core.documents import Document

        # Setup mock
        mock_vstore = MagicMock()
        mock_vstore.similarity_search.return_value = [
            Document(page_content="Port strike in Rotterdam", metadata={"source": "test"}),
        ]
        mock_vstore_cls.return_value = mock_vstore

        retriever = DisruptionRetriever()
        results = retriever.query("rotterdam strike", k=3)

        assert len(results) == 1
        assert "Rotterdam" in results[0].page_content
        mock_vstore.similarity_search.assert_called_once()

    @patch("src.rag.retriever.DisruptionVectorStore")
    def test_format_docs_produces_string(self, mock_vstore_cls):
        """format_docs should produce a formatted string from documents."""
        from src.rag.retriever import DisruptionRetriever
        from langchain_core.documents import Document

        mock_vstore_cls.return_value = MagicMock()
        retriever = DisruptionRetriever()

        docs = [
            Document(
                page_content="Typhoon hits Shanghai",
                metadata={"source": "WeatherAPI", "date": "2025-01-01", "type": "weather"},
            ),
            Document(
                page_content="Strike at Rotterdam",
                metadata={"source": "NewsAPI", "date": "2025-01-02", "type": "news"},
            ),
        ]

        result = retriever.format_docs(docs)

        assert "Typhoon hits Shanghai" in result
        assert "Strike at Rotterdam" in result
        assert "---" in result  # separator between docs

    @patch("src.rag.retriever.DisruptionVectorStore")
    def test_query_with_filters(self, mock_vstore_cls):
        """Retriever.query should pass filters through to the vector store."""
        from src.rag.retriever import DisruptionRetriever

        mock_vstore = MagicMock()
        mock_vstore.similarity_search.return_value = []
        mock_vstore_cls.return_value = mock_vstore

        retriever = DisruptionRetriever()
        retriever.query("test query", filters={"region": "Europe"}, k=3)

        mock_vstore.similarity_search.assert_called_once_with(
            "test query", k=3, filter_dict={"region": "Europe"}
        )

    @patch("src.rag.retriever.DisruptionVectorStore")
    def test_empty_docs_formatting(self, mock_vstore_cls):
        """format_docs should handle empty list gracefully."""
        from src.rag.retriever import DisruptionRetriever

        mock_vstore_cls.return_value = MagicMock()
        retriever = DisruptionRetriever()

        result = retriever.format_docs([])
        assert result == ""
