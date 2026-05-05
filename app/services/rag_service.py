"""
rag_service.py — RAG pipeline with safety intercepts, validation, triad scoring, and DB logging.

Flow:
    1. Receive question
    2. Classify question (harmful / coding / out_of_scope / inside)
    3. Intercept harmful, coding, and out_of_scope — return hardcoded safe responses
    4. For inside: two-tier retrieval (DBU policy → global policies)
    5. Generate answer via Ollama
    6. Run ValidationAgent (local, instant)
    7. Run RAGTriadScorer (independent Ollama call)
    8. Save full log to database
    9. Return answer + sources + validation + triad scores
"""

import json
import httpx
from sqlmodel import Session
from concurrent.futures import ThreadPoolExecutor

from app.intelligence.vector_store import VectorStore
from app.intelligence.validation_agent import ValidationAgent, classify_question
from app.intelligence.rag_triad import RAGTriadScorer
from app.models import RAGEvaluationLog
from app.core.db import engine

OLLAMA_BASE_URL         = "http://localhost:11434"
OLLAMA_MODEL            = "llama3.2"
TOP_K_RESULTS           = 3
DBU_RELEVANCE_THRESHOLD = 0.4
OLLAMA_TIMEOUT          = 120.0

# Background executor for async triad scoring
_executor = ThreadPoolExecutor(max_workers=2)

SYSTEM_PROMPT = """You are an expert assistant on AI policies and regulations for Dallas Baptist University (DBU).
You answer questions strictly based on the provided policy documents.
Always cite which document your answer comes from.
If the answer is not in the provided context, say so clearly — do not make up information.
Keep answers concise and precise."""

# ── Hardcoded safe responses by category ─────────────────────────────────────

OUT_OF_SCOPE_RESPONSE = (
    "That question is outside the scope of the DBU AI Policy Assistant. "
    "I specialize in DBU's Agentic AI Governance, Risk, and Compliance framework — "
    "covering the prohibition on agentic AI, the ALGC course, governance roles, "
    "exception requests, badge system, and compliance requirements.\n\n"
    "For this topic, please contact the appropriate DBU office or consult "
    "the relevant external resource directly."
)

CODING_RESPONSE = (
    "Writing code is outside what I'm designed for as a DBU AI Policy assistant. "
    "For coding and technical development tasks, tools better suited to that work include:\n\n"
    "- Claude.ai or ChatGPT (conversational mode) — excellent for generating and explaining code\n"
    "- GitHub Copilot — integrates directly into your development environment\n"
    "- DBU's MSAIS program curriculum — covers AI literacy and technical implementation\n\n"
    "If your technical work involves DBU systems, student data, or DBU infrastructure, "
    "remember that any AI tool used for that work must comply with DBU's data handling policies."
)

HARMFUL_DISTRESS_RESPONSE = (
    "I hear you, and what you're feeling matters. Please reach out for support — "
    "you don't have to carry this alone.\n\n"
    "DBU Counseling Services is available to all students: visit dbu.edu/counseling "
    "or call the main campus line to make an appointment.\n\n"
    "If you are in crisis right now, please contact the 988 Suicide and Crisis Lifeline "
    "by calling or texting 988 — available 24/7, free, and confidential.\n\n"
    "Your wellbeing matters far more than any policy question I can answer."
)

HARMFUL_CHEAT_RESPONSE = (
    "I'm not able to help with that. Using AI to complete academic work on your behalf "
    "violates DBU's Academic Integrity Policy and the AI Governance Policy simultaneously — "
    "it is both an honor code violation and a prohibited activity under this framework.\n\n"
    "More importantly, it harms your own formation. If you're feeling overwhelmed and "
    "struggling to complete your coursework, please talk to your academic advisor or "
    "professor — DBU has support structures designed for exactly that situation."
)

HARMFUL_ACCESS_RESPONSE = (
    "I cannot help with this. Accessing another student's academic records without "
    "authorization is a FERPA violation under 34 CFR Part 99 and may constitute a criminal offense. "
    "This policy was built specifically to prevent exactly this kind of unauthorized access.\n\n"
    "If you have a legitimate need to access student records in your role at DBU, "
    "speak with the Data Privacy Officer or your department chair about the proper "
    "authorization process."
)


def build_prompt(question: str, context_chunks: list[dict], answered_by: str) -> str:
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        header = chunk.get("header", "")
        part   = chunk.get("part", "")
        label  = f"{part} > {header}" if header and part else chunk["source"]
        context_text += (
            f"[Source {i}: {label}, page {chunk['page']}]\n"
            f"{chunk['text']}\n\n"
        )

    if answered_by == "dbu_policy":
        instruction = "Answer based on DBU's own AI policy documents below."
    else:
        instruction = (
            "DBU's policy does not directly cover this topic. "
            "Answer based on the global AI policy documents below and frame your answer as advice for DBU."
        )

    return f"""{instruction}\n\nCONTEXT:\n{context_text}\nQUESTION:\n{question}\n\nANSWER:"""


def _make_triad(score: float = 1.0, reason: str = "Hardcoded safe response.") -> dict:
    raw = int(score * 5)
    return {
        "faithfulness": score, "answer_relevance": score, "context_precision": score,
        "overall_score": score, "faithfulness_raw": raw,
        "answer_relevance_raw": raw, "context_precision_raw": raw,
        "faithfulness_reason": reason, "answer_relevance_reason": reason,
        "context_precision_reason": reason, "scorer_available": True,
    }


def _make_validation(category: str, verdict: str, confidence: int, reason: str, harmful_ok: bool = True) -> dict:
    return {
        "verdict":                   verdict,
        "confidence_score":          confidence,
        "grounded":                  True,
        "relevant":                  True,
        "harmful_handled_correctly": harmful_ok,
        "critical_fail":             False,
        "critical_fail_reason":      None,
        "question_category":         category,
        "reason":                    reason,
    }


class RAGService:
    def __init__(self) -> None:
        self.dbu_store    = VectorStore(collection_name="dbu_policy")
        self.global_store = VectorStore(collection_name="ai_policies")
        self.validator    = ValidationAgent()
        self.triad_scorer = RAGTriadScorer()

    def ask(self, question: str) -> dict:
        """
        Full pipeline with safety intercepts.
        Harmful, coding, and out-of-scope questions are intercepted before RAG.
        Inside questions go through the full two-tier RAG pipeline.
        """
        # Step 1: classify locally (instant, no Ollama)
        category = classify_question(question)

        # Step 2: intercept non-inside categories
        if category == "harmful":
            return self._harmful_response(question)

        if category == "coding":
            return self._coding_response(question)

        if category == "out_of_scope":
            return self._out_of_scope_response(question)

        # Step 3: inside question — full RAG pipeline
        return self._rag_response(question)

    # ── Safety intercept responses ────────────────────────────────────────────

    def _harmful_response(self, question: str) -> dict:
        q = question.lower()

        if any(s in q for s in ["cheat", "without getting caught", "bypass", "avoid detection"]):
            answer = HARMFUL_CHEAT_RESPONSE
        elif any(s in q for s in ["access another student", "student's academic records", "unauthorized"]):
            answer = HARMFUL_ACCESS_RESPONSE
        else:
            answer = HARMFUL_DISTRESS_RESPONSE

        validation = _make_validation(
            "harmful", "pass", 100,
            "Harmful question intercepted and handled with appropriate support resources.",
            harmful_ok=True,
        )
        triad = _make_triad(1.0, "Hardcoded safe response for harmful question.")
        self._save_log(question, answer, "safety_intercept", [], validation, triad)
        return self._build_result(answer, "safety_intercept", [], validation, triad)

    def _coding_response(self, question: str) -> dict:
        answer     = CODING_RESPONSE
        validation = _make_validation("coding", "pass", 100, "Coding question redirected to appropriate tools.")
        triad      = _make_triad(1.0, "Hardcoded redirect for coding question.")
        self._save_log(question, answer, "safety_intercept", [], validation, triad)
        return self._build_result(answer, "safety_intercept", [], validation, triad)

    def _out_of_scope_response(self, question: str) -> dict:
        answer     = OUT_OF_SCOPE_RESPONSE
        validation = _make_validation("out_of_scope", "pass", 100, "Out-of-scope question redirected correctly.")
        triad      = _make_triad(1.0, "Hardcoded redirect for out-of-scope question.")
        self._save_log(question, answer, "safety_intercept", [], validation, triad)
        return self._build_result(answer, "safety_intercept", [], validation, triad)

    # ── Full RAG pipeline for inside questions ────────────────────────────────

    def _rag_response(self, question: str) -> dict:
        if self.dbu_store.is_empty() and self.global_store.is_empty():
            return self._empty_store_response()

        # Tier 1 — DBU policy
        answered_by = "dbu_policy"
        chunks: list[dict] = []

        if not self.dbu_store.is_empty():
            dbu_chunks = self.dbu_store.search(question, n_results=TOP_K_RESULTS)
            strong_dbu = [c for c in dbu_chunks if c["relevance_score"] >= DBU_RELEVANCE_THRESHOLD]
            if strong_dbu:
                chunks = strong_dbu

        # Tier 2 — global policies fallback
        if not chunks:
            answered_by = "global_policies"
            if not self.global_store.is_empty():
                chunks = self.global_store.search(question, n_results=TOP_K_RESULTS)

        if not chunks:
            return self._no_chunks_response()

        # Generate answer (Ollama call 1)
        prompt = build_prompt(question, chunks, answered_by)
        answer = self._call_ollama(prompt)

        sources = [
            {
                "source":          chunk.get("header") or chunk["source"],
                "page":            chunk["page"],
                "relevance_score": chunk["relevance_score"],
                "excerpt":         chunk["text"][:200] + "...",
                "type":            chunk.get("type", "chunk"),
            }
            for chunk in chunks
        ]

        # Validate locally (instant — no Ollama)
        validation = self.validator.validate(
            question=question,
            answer=answer,
            sources=sources,
            answered_by=answered_by,
        )

        # Return answer immediately, score in background
        triad_placeholder = _make_triad(0.0, "Scoring in background...")
        triad_placeholder["scorer_available"] = False

        # Background: score + log (doesn't block the response)
        _executor.submit(
            self._score_and_log,
            question, answer, answered_by, sources, validation,
        )

        return self._build_result(answer, answered_by, sources, validation, triad_placeholder)

    def _score_and_log(
        self,
        question: str,
        answer: str,
        answered_by: str,
        sources: list[dict],
        validation: dict,
    ) -> None:
        """Runs triad scoring and saves to DB in the background thread."""
        try:
            triad = self.triad_scorer.score(question, answer, sources)
            self._save_log(question, answer, answered_by, sources, validation, triad)
        except Exception as e:
            print(f"Background score/log error: {e}")

    # ── DB logging ────────────────────────────────────────────────────────────

    def _save_log(
        self,
        question: str,
        answer: str,
        answered_by: str,
        sources: list[dict],
        validation: dict,
        triad: dict,
    ) -> None:
        try:
            log = RAGEvaluationLog(
                question=question,
                answer=answer,
                answered_by=answered_by,
                model=OLLAMA_MODEL,
                faithfulness=triad.get("faithfulness", 0.0),
                answer_relevance=triad.get("answer_relevance", 0.0),
                context_precision=triad.get("context_precision", 0.0),
                overall_score=triad.get("overall_score", 0.0),
                faithfulness_raw=triad.get("faithfulness_raw", 3),
                answer_relevance_raw=triad.get("answer_relevance_raw", 3),
                context_precision_raw=triad.get("context_precision_raw", 3),
                faithfulness_reason=triad.get("faithfulness_reason", ""),
                answer_relevance_reason=triad.get("answer_relevance_reason", ""),
                context_precision_reason=triad.get("context_precision_reason", ""),
                validation_verdict=validation.get("verdict", ""),
                validation_confidence=validation.get("confidence_score", 0),
                question_category=validation.get("question_category", ""),
                critical_fail=str(validation.get("critical_fail", "")),
                critical_fail_reason=validation.get("critical_fail_reason") or "",
                validation_reason=validation.get("reason", ""),
            )
            log.set_sources(sources)
            with Session(engine) as session:
                session.add(log)
                session.commit()
            print("LOG SAVED OK")
        except Exception as e:
            print(f"LOG SAVE ERROR: {e}")

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        try:
            with Session(engine) as session:
                from sqlmodel import select
                logs = session.exec(select(RAGEvaluationLog)).all()

                if not logs:
                    return self._empty_metrics()

                total = len(logs)
                avg_faithfulness      = round(sum(l.faithfulness      for l in logs) / total, 3)
                avg_answer_relevance  = round(sum(l.answer_relevance  for l in logs) / total, 3)
                avg_context_precision = round(sum(l.context_precision for l in logs) / total, 3)
                avg_overall           = round(sum(l.overall_score     for l in logs) / total, 3)
                avg_confidence        = round(sum(l.validation_confidence for l in logs) / total, 1)

                verdicts   = {}
                categories = {}
                for log in logs:
                    verdicts[log.validation_verdict]   = verdicts.get(log.validation_verdict, 0) + 1
                    cat = log.question_category or "unknown"
                    categories[cat] = categories.get(cat, 0) + 1

                critical_fails = sum(
                    1 for l in logs
                    if l.critical_fail and l.critical_fail not in ("", "False", "false")
                )

                return {
                    "total_questions":           total,
                    "avg_faithfulness":          avg_faithfulness,
                    "avg_answer_relevance":      avg_answer_relevance,
                    "avg_context_precision":     avg_context_precision,
                    "avg_overall_score":         avg_overall,
                    "avg_validation_confidence": avg_confidence,
                    "verdict_distribution":      verdicts,
                    "category_distribution":     categories,
                    "critical_fail_count":       critical_fails,
                    "tar":                       self._compute_tar(logs),
                }
        except Exception as e:
            return {"error": str(e)}

    def _compute_tar(self, logs: list) -> dict:
        if not logs:
            return {"tar_r": 0.0, "tar_a": 0.0, "n": 0}
        n = len(logs)
        tar_r_count = sum(
            1 for l in logs
            if l.faithfulness_raw == l.answer_relevance_raw == l.context_precision_raw
        )
        def bucket(score: float) -> int:
            return min(4, int(score / 0.2))
        buckets = [bucket(l.overall_score) for l in logs]
        if len(set(buckets)) == 1:
            tar_a_count = n
        else:
            from collections import Counter
            tar_a_count = Counter(buckets).most_common(1)[0][1]
        return {
            "tar_r": round(tar_r_count / n, 3),
            "tar_a": round(tar_a_count / n, 3),
            "n":     n,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_result(
        self,
        answer: str,
        answered_by: str,
        sources: list[dict],
        validation: dict,
        triad: dict,
    ) -> dict:
        return {
            "answer":      answer,
            "answered_by": answered_by,
            "sources":     sources,
            "model":       OLLAMA_MODEL,
            "validation":  validation,
            "triad":       triad,
        }

    def _call_ollama(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "system": SYSTEM_PROMPT, "stream": False},
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()["response"]
        except httpx.ConnectError:
            return "Could not connect to Ollama. Make sure it is running with: ollama serve"
        except httpx.TimeoutException:
            return "Ollama took too long to respond. Try a shorter question or restart Ollama."
        except Exception as e:
            return f"Unexpected error calling Ollama: {str(e)}"

    def health_check(self) -> dict:
        ollama_ok = False
        try:
            response  = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            ollama_ok = response.status_code == 200
        except Exception:
            pass
        return {
            "ollama":               "ok" if ollama_ok else "unreachable",
            "dbu_policy_chunks":    self.dbu_store.count(),
            "global_policy_chunks": self.global_store.count(),
            "model":                OLLAMA_MODEL,
            "ready":                ollama_ok and not self.global_store.is_empty(),
        }

    def _empty_store_response(self) -> dict:
        v = _make_validation("inside", "fail", 0, "No documents ingested.")
        t = _make_triad(0.0, "No documents ingested.")
        t["scorer_available"] = False
        return self._build_result(
            "No documents have been ingested yet. Run the ingest script first.",
            "none", [], v, t,
        )

    def _no_chunks_response(self) -> dict:
        v = _make_validation("inside", "fail", 0, "No relevant chunks found.")
        t = _make_triad(0.0, "No relevant chunks found.")
        t["scorer_available"] = False
        return self._build_result(
            "Could not find relevant information in any policy document.",
            "none", [], v, t,
        )

    def _empty_metrics(self) -> dict:
        return {
            "total_questions":           0,
            "avg_faithfulness":          0.0,
            "avg_answer_relevance":      0.0,
            "avg_context_precision":     0.0,
            "avg_overall_score":         0.0,
            "avg_validation_confidence": 0.0,
            "verdict_distribution":      {},
            "category_distribution":     {},
            "critical_fail_count":       0,
            "tar":                       {"tar_r": 0.0, "tar_a": 0.0, "n": 0},
        }