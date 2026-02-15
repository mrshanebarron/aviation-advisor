import anthropic
from app.config import ANTHROPIC_API_KEY
from app.services.vector_store import query_documents

_client = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """You are an expert advisor on private aviation. You help high-net-worth individuals make smarter, data-driven decisions about flying private.

Answer questions based ONLY on the provided context documents. If the context doesn't contain enough information to answer fully, say so honestly and explain what you can infer from the available data.

When referencing specific information, mention the source document name so users can verify.

Be thorough but concise. Use clear formatting with bullet points or numbered lists when comparing options. Speak with authority — you are the expert."""


def build_context(query: str) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks and build context string."""
    matches = query_documents(query, n_results=5)

    if not matches:
        return "", []

    context_parts = []
    for i, match in enumerate(matches, 1):
        source = match["metadata"]["source"]
        context_parts.append(f"[Source: {source}]\n{match['text']}")

    context = "\n\n---\n\n".join(context_parts)
    return context, matches


def stream_response(query: str, conversation_history: list[dict] = None):
    """Stream a response from Claude using RAG context."""
    client = get_client()
    context, matches = build_context(query)

    messages = []

    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)

    # Build the user message with context
    if context:
        user_content = f"""Based on the following knowledge base documents, answer the user's question.

<context>
{context}
</context>

<question>
{query}
</question>"""
    else:
        user_content = f"""The knowledge base has no documents yet. Let the user know that documents need to be uploaded before you can answer aviation questions.

<question>
{query}
</question>"""

    messages.append({"role": "user", "content": user_content})

    with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def get_response(query: str, conversation_history: list[dict] = None) -> str:
    """Get a complete (non-streaming) response."""
    parts = []
    for text in stream_response(query, conversation_history):
        parts.append(text)
    return "".join(parts)
