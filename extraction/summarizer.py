from core.llm import call_model
import config


CHUNK_SUMMARY_PROMPT = """
Summarize the following document excerpt in 2-3 sentences.
Focus only on factual content, main ideas, and key points.
Do not include opinions or meta commentary.

Excerpt:
{chunk_text}

Summary:
"""

GLOBAL_SUMMARY_PROMPT = """
Given the following chunk summaries from a document, produce a concise overall summary of 4-6 sentences.
Also list 3-5 key points as bullet points.

Chunk summaries:
{summaries}

Output format:
Summary: <overall summary>
Key Points:
- <point 1>
- <point 2>
...
"""


def summarize_chunk(chunk_text: str, model: str | None = None) -> str:
    prompt = CHUNK_SUMMARY_PROMPT.format(chunk_text=chunk_text)
    return call_model(prompt, model=model, max_tokens=256)


def summarize_document(chunks: list[str], model: str | None = None) -> tuple[str, list[str]]:
    """Generate chunk summaries, then global summary and key points."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    chunk_summaries = []
    if chunks:
        with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as executor:
            futures = {executor.submit(summarize_chunk, chunk, model): idx for idx, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                summary = future.result()
                if summary:
                    chunk_summaries.append(summary)

    if not chunk_summaries:
        return "", []

    combined = "\n".join(f"- {s}" for s in chunk_summaries)
    prompt = GLOBAL_SUMMARY_PROMPT.format(summaries=combined)
    raw = call_model(prompt, model=model, max_tokens=512)

    # Parse
    summary_part = ""
    key_points = []
    lines = raw.splitlines()
    in_key_points = False
    for line in lines:
        if line.startswith("Summary:"):
            summary_part = line[len("Summary:"):].strip()
        elif line.startswith("Key Points:"):
            in_key_points = True
        elif in_key_points and line.strip().startswith("-"):
            key_points.append(line.strip()[1:].strip())
        elif summary_part == "" and line.strip() and not in_key_points:
            summary_part = line.strip()

    return summary_part, key_points