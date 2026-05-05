"""
rag_triad.py — RAG Triad scorer using a dedicated Ollama call.

Measures the three industry-standard RAG quality dimensions:
    1. Faithfulness        — Does the answer come ONLY from the retrieved context?
    2. Answer Relevance    — Does the response actually address the user's question?
    3. Context Precision   — Did the system retrieve the RIGHT chunks to answer?

Each score uses a 0–5 Likert scale, normalized to 0.0–1.0 for storage.
Overall score = mean of the three normalized scores.

References:
    - RAG Triad (TruLens / RAGAS standard)
    - Likert normalization: score / 5.0
    - TAR (Total Agreement Rate) logged per question for instability tracking
"""

import json
import re
import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2"
OLLAMA_TIMEOUT  = 120.0

TRIAD_SYSTEM_PROMPT = """You are a RAG quality evaluator. You score chatbot answers on three dimensions.
You respond ONLY with a raw JSON object. No markdown, no backticks, no explanation.
Start your response with { and end with }.
Use only integers 0 through 5 for scores."""

TRIAD_PROMPT_TEMPLATE = """Score this RAG chatbot interaction on three dimensions using a 0-5 Likert scale.

QUESTION:
{question}

RETRIEVED CONTEXT CHUNKS:
{context}

ANSWER GIVEN:
{answer}

SCORING RUBRIC:

FAITHFULNESS (Does the answer come ONLY from the retrieved context? No hallucination?):
  5 = Every claim in the answer is directly supported by the context chunks.
  4 = Almost all claims supported, one minor inference not directly stated.
  3 = Mostly supported but one claim is not traceable to the context.
  2 = Several claims not in the context — partial hallucination.
  1 = Most of the answer is not supported by the context.
  0 = Answer is completely fabricated or contradicts the context.

ANSWER RELEVANCE (Does the response actually address the user's question?):
  5 = Directly and completely answers exactly what was asked.
  4 = Answers the question well with minor irrelevant content.
  3 = Partially answers the question but misses key aspects.
  2 = Tangentially related but does not answer the question.
  1 = Barely related to the question.
  0 = Completely off-topic.

CONTEXT PRECISION (Did the system retrieve the RIGHT chunks to answer this question?):
  5 = All retrieved chunks are highly relevant and necessary to answer the question.
  4 = Most chunks are relevant, one is marginally useful.
  3 = Mixed — some relevant, some irrelevant chunks retrieved.
  2 = Mostly irrelevant chunks retrieved, better chunks likely exist.
  1 = Retrieved chunks are almost entirely wrong for this question.
  0 = Retrieved chunks have no relevance to the question.

Return ONLY this JSON:
{{
  "faithfulness": 0-5,
  "answer_relevance": 0-5,
  "context_precision": 0-5,
  "faithfulness_reason": "one sentence",
  "answer_relevance_reason": "one sentence",
  "context_precision_reason": "one sentence"
}}"""


class RAGTriadScorer:
    """
    Scores every RAG interaction on the three standard quality dimensions.
    Uses a separate Ollama call with temperature=0 for reproducibility.
    """

    def score(
        self,
        question: str,
        answer: str,
        sources: list[dict],
    ) -> dict:
        """
        Compute RAG Triad scores for a single interaction.

        Args:
            question: The user's original question.
            answer:   The answer produced by the RAG pipeline.
            sources:  The source chunks used (each with source, page, excerpt).

        Returns:
            dict with raw Likert scores (0-5), normalized scores (0.0-1.0),
            overall score, reasons, and raw scores for TAR tracking.
        """
        context_text = ""
        for i, s in enumerate(sources, 1):
            context_text += (
                f"[Chunk {i}: {s.get('source','unknown')}, "
                f"page {s.get('page', 0)}]\n"
                f"{s.get('excerpt', '')}\n\n"
            )
        if not context_text:
            context_text = "No context chunks were retrieved."

        prompt = TRIAD_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer,
            context=context_text,
        )

        raw = self._call_ollama(prompt)
        return self._parse_response(raw)

    def _call_ollama(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": TRIAD_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            raw = response.json()["response"]
            print(f"TRIAD RAW: {raw[:300]}")  # ← add this line only
            return raw
        except Exception:
            return '{"faithfulness":3,"answer_relevance":3,"context_precision":3,"faithfulness_reason":"Scorer unavailable.","answer_relevance_reason":"Scorer unavailable.","context_precision_reason":"Scorer unavailable."}'

    def _parse_response(self, raw: str) -> dict:
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                f  = max(0, min(5, int(data.get("faithfulness",        3))))
                ar = max(0, min(5, int(data.get("answer_relevance",     3))))
                cp = max(0, min(5, int(data.get("context_precision",    3))))

                # Normalize to 0.0–1.0 (Likert / 5)
                f_norm  = round(f  / 5.0, 4)
                ar_norm = round(ar / 5.0, 4)
                cp_norm = round(cp / 5.0, 4)
                overall = round((f_norm + ar_norm + cp_norm) / 3.0, 4)

                return {
                    # Raw Likert (0-5) — used for TAR tracking
                    "faithfulness_raw":        f,
                    "answer_relevance_raw":    ar,
                    "context_precision_raw":   cp,
                    # Normalized (0.0-1.0) — stored in DB
                    "faithfulness":            f_norm,
                    "answer_relevance":        ar_norm,
                    "context_precision":       cp_norm,
                    "overall_score":           overall,
                    # Reasons
                    "faithfulness_reason":       data.get("faithfulness_reason",       ""),
                    "answer_relevance_reason":   data.get("answer_relevance_reason",   ""),
                    "context_precision_reason":  data.get("context_precision_reason",  ""),
                    "scorer_available":          True,
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Safe fallback
        return {
            "faithfulness_raw":       3,
            "answer_relevance_raw":   3,
            "context_precision_raw":  3,
            "faithfulness":           0.6,
            "answer_relevance":       0.6,
            "context_precision":      0.6,
            "overall_score":          0.6,
            "faithfulness_reason":    "Score unavailable.",
            "answer_relevance_reason":"Score unavailable.",
            "context_precision_reason":"Score unavailable.",
            "scorer_available":       False,
        }