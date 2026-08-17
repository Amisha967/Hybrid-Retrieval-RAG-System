import os
import re
import threading
from typing import List, Dict, Any, Tuple, Optional
from backend.app.config import settings
from backend.app.api.schemas import Citation, RetrievedChunk, LatencyBreakdown
from backend.app.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from backend.app.core.metrics import LatencyTimer

class CitationGenerator:
    """
    Generates citation-backed answers with strict factual grounding and latency tracking.
    Supports Local Extractive Grounded Synthesizer, Gemini, OpenAI, and HuggingFace pipelines.
    """

    def __init__(self):
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_lock = threading.Lock()

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        provider: Optional[str] = None,
        custom_api_key: Optional[str] = None
    ) -> Tuple[str, List[Citation], float, float]:
        """
        Returns: (answer_text, citations_list, groundedness_score, generation_latency_ms)
        """
        chosen_provider = provider or settings.LLM_PROVIDER
        
        if not retrieved_chunks:
            return (
                "No relevant context chunks were found in the knowledge base to answer your question.",
                [],
                0.0,
                0.0
            )

        with LatencyTimer() as timer:
            # Build context representation
            context_blocks = []
            citations: List[Citation] = []
            
            for idx, c in enumerate(retrieved_chunks):
                citation_num = idx + 1
                doc_name = c.get("filename", "unknown")
                chunk_idx = c.get("chunk_index", 0)
                chunk_id = c.get("chunk_id", f"chunk_{idx}")
                text = c.get("text", "").strip()
                score = c.get("rerank_score", c.get("fused_score", 0.0))
                
                label = f"[{citation_num}]"
                context_blocks.append(f"[{citation_num}] (File: {doc_name}, Chunk: {chunk_idx}):\n{text}")
                
                # Create short snippet
                snippet = text[:150] + "..." if len(text) > 150 else text
                citations.append(
                    Citation(
                        citation_id=citation_num,
                        label=label,
                        chunk_id=chunk_id,
                        doc_id=c.get("doc_id", ""),
                        filename=doc_name,
                        snippet=snippet,
                        relevance_score=round(float(score), 4)
                    )
                )

            formatted_context = "\n\n".join(context_blocks)
            prompt = build_rag_prompt(query=query, chunks_context=formatted_context)

            # Generate via chosen provider
            if chosen_provider == "gemini":
                answer = self._generate_gemini(prompt, custom_api_key)
            elif chosen_provider == "openai":
                answer = self._generate_openai(prompt, custom_api_key)
            elif chosen_provider == "huggingface":
                answer = self._generate_hf(prompt, citations)
            else:
                # Default high-performance extractive grounded synthesizer
                answer = self._generate_extractive(query, retrieved_chunks, citations)

            # Calculate Groundedness Score
            groundedness = self._compute_groundedness(answer, retrieved_chunks)

        return answer, citations, groundedness, round(timer.elapsed_ms, 2)

    def _generate_extractive(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        citations: List[Citation]
    ) -> str:
        """
        Deterministic, low-latency grounded synthesizer extracting most salient
        answers directly from top-k reranked passages with citation markers.
        """
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        answer_sentences = []
        
        for idx, (chunk, cit) in enumerate(zip(retrieved_chunks, citations)):
            text = chunk.get("text", "")
            sentences = re.split(r'(?<=[.?!])\s+', text)
            
            scored_sentences = []
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                s_words = set(re.findall(r'\b\w+\b', s_clean.lower()))
                overlap = len(query_words.intersection(s_words))
                scored_sentences.append((overlap, s_clean))
            
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [s for score, s in scored_sentences[:2] if score > 0]
            
            if top_sentences:
                combined_sent = " ".join(top_sentences)
                answer_sentences.append(f"{combined_sent} {cit.label}")
            elif idx == 0 and sentences:
                answer_sentences.append(f"{sentences[0]} {cit.label}")

        if not answer_sentences:
            return f"Based on the retrieved context, {retrieved_chunks[0].get('text', '')[:200]}... {citations[0].label}"

        header = f"According to the indexed documents, regarding '{query}':\n\n"
        body = "\n\n".join([f"• {stmt}" for stmt in answer_sentences[:4]])
        return header + body

    def _generate_gemini(self, prompt: str, api_key: Optional[str]) -> str:
        key = api_key or settings.GEMINI_API_KEY
        if not key:
            return "[Gemini Error] GEMINI_API_KEY is not configured. Please provide your Gemini API key in the sidebar."
        try:
            from google import genai
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"system_instruction": RAG_SYSTEM_PROMPT}
            )
            return response.text or ""
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                return f"[Gemini Quota Exceeded]: {err_str}. You can switch to 'local_extractive' for offline, zero-quota execution."
            return f"[Gemini Exception]: {err_str}"

    def _generate_openai(self, prompt: str, api_key: Optional[str]) -> str:
        key = api_key or settings.OPENAI_API_KEY
        if not key:
            return "[OpenAI Error] OPENAI_API_KEY is not configured. Please provide your OpenAI API key in the sidebar."
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e)
            if "insufficient_quota" in err_str or "429" in err_str:
                return "[OpenAI Quota Exceeded]: Your OpenAI account has exceeded its billing quota. Switch to 'local_extractive' or 'huggingface' for free, zero-quota execution."
            return f"[OpenAI Exception]: {err_str}"

    def _generate_hf(self, prompt: str, citations: List[Citation]) -> str:
        try:
            with self._hf_lock:
                if self._hf_tokenizer is None or self._hf_model is None:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    self._hf_tokenizer = AutoTokenizer.from_pretrained(settings.HF_MODEL_NAME)
                    self._hf_model = AutoModelForSeq2SeqLM.from_pretrained(settings.HF_MODEL_NAME)

            inputs = self._hf_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            outputs = self._hf_model.generate(**inputs, max_new_tokens=200, num_beams=2, early_stopping=True)
            gen_text = self._hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Attach top citation badge if missing
            if citations and not any(c.label in gen_text for c in citations):
                gen_text = f"{gen_text} {citations[0].label}"
            
            return gen_text
        except Exception as e:
            return f"[HuggingFace Exception]: {e}"

    def _compute_groundedness(self, answer: str, chunks: List[Dict[str, Any]]) -> float:
        """
        Computes context grounding alignment score in [0.0, 1.0].
        """
        if not answer or not chunks:
            return 0.0
        
        answer_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
        if not answer_tokens:
            return 0.0
        
        all_chunk_text = " ".join([c.get("text", "") for c in chunks]).lower()
        chunk_tokens = set(re.findall(r'\b\w+\b', all_chunk_text))
        
        common = answer_tokens.intersection(chunk_tokens)
        score = len(common) / len(answer_tokens)
        return round(min(1.0, score * 1.2), 2)

generator = CitationGenerator()
