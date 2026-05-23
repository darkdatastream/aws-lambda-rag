"""Streamlit UI for the AWS Lambda RAG demo.

Run:
    streamlit run app/streamlit_app.py

The app loads the existing ChromaDB index and the configured LLM client.
"""

from __future__ import annotations

import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

st.set_page_config(
    page_title="AWS Lambda RAG",
    page_icon="λ",
    layout="wide",
)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource
def get_store_stats() -> int:
    return VectorStore().count()


def main() -> None:
    st.title("λ AWS Lambda Documentation Assistant")
    st.caption(
        "Ask technical questions about AWS Lambda. Answers are grounded in selected "
        "official AWS Lambda Developer Guide Markdown pages and cite the retrieved source files."
    )

    with st.sidebar:
        st.subheader("Index")
        try:
            count = get_store_stats()
            st.metric("Chunks indexed", f"{count:,}")
            st.caption("Source: selected AWS Lambda Developer Guide Markdown pages")
        except Exception as exc:
            st.error(f"Index not built. Run the fetch and build scripts first. {exc}")
            return

        st.subheader("Try these")
        examples = [
            "What is the maximum Lambda function timeout?",
            "What is the maximum memory for a Lambda function?",
            "What Python runtimes does Lambda support?",
            "How do I configure environment variables?",
            "What are AWS Lambda best practices for database connections?",
            "How does Lambda scaling work?",
        ]
        for example in examples:
            if st.button(example, key=example, use_container_width=True):
                st.session_state["question"] = example

        st.divider()
        st.caption(
            "Stack: ChromaDB, sentence-transformers, BM25 + RRF, DeepSeek API, Streamlit"
        )

    question = st.text_area(
        "Your question",
        value=st.session_state.get("question", ""),
        height=80,
        placeholder="e.g. What is the maximum Lambda timeout?",
    )

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving relevant documentation and generating answer..."):
            pipeline = get_pipeline()
            response = pipeline.ask(question)

        st.subheader("Answer")
        st.markdown(response.answer)

        st.subheader(f"Retrieved sources ({len(response.sources)})")
        for i, src in enumerate(response.sources, start=1):
            score_pct = src.score * 100
            header = f"[{i}] {src.source} — {src.section} — relevance {score_pct:.2f}%"
            with st.expander(header):
                st.text(src.text)


if __name__ == "__main__":
    main()
