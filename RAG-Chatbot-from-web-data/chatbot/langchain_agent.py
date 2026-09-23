"""
LangChain-based Agentic RAG Module
-----------------------------------
Implements an Agentic RAG pipeline using LangChain primitives.

Architecture Flow:
  1. Supervisor Router (Fused Call): Determines intent, standalone query, sub-domain, sentiment, and 3 search queries.
  2. Early-Exit Routes: 'greeting' and 'identity' bypass retrieval entirely.
  3. Multi-Query Retrieval: Retrives documents using expanded queries via HybridRetriever.
  4. Document Grading: Batch evaluates document relevance in 1 LLM call.
  5. Corrective Rewriting: Rewrites queries if 0 documents pass grading.
  6. Sub-Agent Execution: Routes to specialized expert sub-agents.
  7. Faithfulness Check: Verifies generation is grounded in facts.
  8. Caching & Tracing: Returns answer, trace steps, and caches the result.
"""

import os
import re
import time
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

import json
from api_key_manager import api_key_manager
from prompt import (
    get_prompt,
    get_router_prompt,
    get_doc_grader_prompt,
    get_batch_doc_grader_prompt,
    get_query_rewrite_prompt,
    get_hallucination_prompt,
    get_multi_query_prompt,
    get_supervisor_router_prompt,
    get_contextual_query_prompt,
    get_combined_supervisor_prompt,
)
from query_cache import query_cache
from specialized_agents import get_specialized_agent
from core_utils import (
    is_greeting,
    is_identity_query,
    is_out_of_scope_query,
    clean_json_markdown,
    clean_meta_talk,
    format_live_currency_context,
)


class LangChainAgenticRAG:
    """
    Agentic RAG Engine powered by LangChain tools, LCEL chains, and key-rotation failovers.
    """

    def __init__(self, retriever_getter):
        """
        :param retriever_getter: Function returning the initialized HybridRetriever instance.
        """
        self.retriever_getter = retriever_getter

    def _get_llm(self, temperature: float = 0.0):
        """
        Instantiate LLM using Gemini primary key-rotation with ChatOllama fallback.
        """
        tried_keys = set()
        max_attempts = max(1, len(api_key_manager.keys)) if hasattr(api_key_manager, "keys") else 1
        fallback_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash",
            "gemini-3.5-flash-lite",
        ]

        for _ in range(max_attempts):
            key = api_key_manager.get_next_available_key()
            if not key or key in tried_keys:
                break
            tried_keys.add(key)

            for model_name in fallback_models:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=key,
                        temperature=temperature,
                        max_retries=0,
                    )
                    api_key_manager.mark_key_success(key)
                    return llm
                except Exception as e:
                    msg = str(e).lower()
                    if api_key_manager.is_client_error(e):
                        print(f"🛑 Client/Input Error during LLM init: {e} - Skipping key rotation.")
                        raise e
                    if any(x in msg for x in ["404", "not_found", "not found"]):
                        print(f"⚠️ Model '{model_name}' not available. Trying next model...")
                        continue
                    if api_key_manager.is_quota_error(e):
                        api_key_manager.mark_key_failed(key, error=e)
                        break
                    continue

        # Ollama Fallback
        try:
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=temperature,
                timeout=30,
            )
        except Exception as e:
            print(f"⚠️ Ollama init fallback error in Agent: {e}")

        raise RuntimeError("No LLM models (Gemini or Ollama) available for Agentic RAG.")

    def _safe_str(self, val: Any) -> str:
        """Safely extracts clean response text from string, list, nested list, or dict."""
        if val is None:
            return ""

        def extract_text(obj: Any) -> List[str]:
            results = []
            if isinstance(obj, str):
                results.append(obj)
            elif isinstance(obj, dict):
                if "text" in obj and isinstance(obj["text"], str):
                    results.append(obj["text"])
                else:
                    for v in obj.values():
                        results.extend(extract_text(v))
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    results.extend(extract_text(item))
            elif hasattr(obj, "content"):
                results.extend(extract_text(getattr(obj, "content")))
            return results

        extracted = extract_text(val)
        if extracted:
            return "\n".join(extracted).strip()
        return str(val).strip()

    def route_intent(self, question: str) -> str:
        """
        Classifies intent into 'greeting', 'identity', or 'retrieve'.
        Uses fast heuristic matching followed by LCEL router chain.
        """
        if is_greeting(question):
            return "greeting"

        if is_identity_query(question):
            return "identity"

        try:
            llm = self._get_llm(temperature=0.0)
            chain = get_router_prompt() | llm
            res = chain.invoke({"question": question})
            content = self._safe_str(res).lower()
            if "greeting" in content:
                return "greeting"
            elif "identity" in content:
                return "identity"
            else:
                return "retrieve"
        except Exception as e:
            print(f"⚠️ Intent Routing fallback to 'retrieve': {e}")
            return "retrieve"

    def grade_document(self, question: str, doc: Document) -> bool:
        """
        Grades whether a single document chunk is relevant to the question.
        """
        try:
            llm = self._get_llm(temperature=0.0)
            chain = get_doc_grader_prompt() | llm
            res = chain.invoke({
                "question": question,
                "document": doc.page_content[:1500]
            })
            content = self._safe_str(res).lower()
            return "yes" in content
        except Exception as e:
            print(f"⚠️ Document Grader error (defaulting to keep chunk): {e}")
            return True

    def grade_documents_batch(self, question: str, docs: List[Document]) -> List[Document]:
        """
        Grades document relevance in 1 single LLM API call .
        """
        if not docs:
            return []

        eval_docs = docs[:6]
        formatted_snippets = []
        for i, d in enumerate(eval_docs, 1):
            formatted_snippets.append(f"[Snippet {i}]\n{d.page_content[:1000]}")

        documents_text = "\n\n".join(formatted_snippets)

        try:
            llm = self._get_llm(temperature=0.0)
            chain = get_batch_doc_grader_prompt() | llm
            res = chain.invoke({
                "question": question,
                "documents_text": documents_text
            })
            content = self._safe_str(res).lower()

            if "none" in content or not content:
                return eval_docs[:3]

            selected_indices = set(map(int, re.findall(r'\b[1-6]\b', content)))
            filtered = [doc for idx, doc in enumerate(eval_docs, 1) if idx in selected_indices]
            return filtered if filtered else eval_docs[:3]
        except Exception as e:
            print(f"⚠️ Batch Document Grader error (defaulting to keep top chunks): {e}")
            return eval_docs[:3]

    def rewrite_query(self, question: str) -> str:
        """
        Rewrites question into an optimized keyword search query.
        """
        try:
            llm = self._get_llm(temperature=0.2)
            chain = get_query_rewrite_prompt() | llm
            res = chain.invoke({"question": question})
            rewritten = self._safe_str(res)
            return rewritten if rewritten else question
        except Exception as e:
            print(f"⚠️ Query Rewrite error: {e}")
            return question

    def generate_multi_queries(self, question: str) -> List[str]:
        """
        Generates 3 proactive search query variations to improve retrieval recall.
        """
        queries = [question]
        try:
            llm = self._get_llm(temperature=0.2)
            chain = get_multi_query_prompt() | llm
            res = chain.invoke({"question": question})
            raw_text = self._safe_str(res)
            parsed = [q.strip() for q in raw_text.split("\n") if q.strip()]
            for p in parsed:
                if p not in queries:
                    queries.append(p)
            return queries[:4]  # Original + up to 3 variations
        except Exception as e:
            print(f"⚠️ Multi-Query Generation fallback: {e}")
            return queries

    def route_subdomain(self, question: str) -> str:
        """
        Supervisor routing: classifies query into specialized expert domain
        ('itinerary', 'culture', 'logistics', or 'general').
        """
        try:
            llm = self._get_llm(temperature=0.0)
            chain = get_supervisor_router_prompt() | llm
            res = chain.invoke({"question": question})
            domain = self._safe_str(res).lower()
            if any(d in domain for d in ["itinerary", "culture", "logistics"]):
                for d in ["itinerary", "culture", "logistics"]:
                    if d in domain:
                        return d
            return "general"
        except Exception as e:
            print(f"⚠️ Subdomain Routing fallback to 'general': {e}")
            return "general"

    def contextualize_question(self, question: str, chat_history_text: str = "") -> str:
        """
        Resolves coreferences, ambiguous pronouns ('there', 'it'), and retains user preferences,
        interests, and constraints stated in previous conversation turns.
        """
        if not chat_history_text or not chat_history_text.strip():
            return question

        try:
            llm = self._get_llm(temperature=0.0)
            chain = get_contextual_query_prompt() | llm
            res = chain.invoke({
                "question": question,
                "chat_history": chat_history_text[-3000:]
            })
            standalone = self._safe_str(res)
            if standalone and len(standalone) > 3:
                print(f"🔄 CONTEXTUAL REPHRASE: '{question}' → '{standalone}'", flush=True)
                return standalone
            return question
        except Exception as e:
            print(f"⚠️ Contextualize Question fallback: {e}")
            return question

    def supervisor_plan(self, question: str, chat_history_text: str = "") -> Dict[str, Any]:
        """
        Fused 1-call Supervisor Planner:
        Combines Intent Classification, Contextual Rephrase, Sub-Domain Routing,
        and Multi-Query Expansion in 1 single fast LLM call.
        """
        # Fast Heuristic Greetings & Identity check (0 LLM calls!)
        if is_greeting(question):
            return {"intent": "greeting", "standalone_query": question, "domain": "general", "sentiment": "positive", "search_queries": [question]}

        if is_identity_query(question):
            return {"intent": "identity", "standalone_query": question, "domain": "general", "sentiment": "neutral", "search_queries": [question]}

        if is_out_of_scope_query(question):
            return {"intent": "out_of_scope", "standalone_query": question, "domain": "general", "sentiment": "neutral", "search_queries": [question]}

        try:
            llm = self._get_llm(temperature=0.0)
            chain = get_combined_supervisor_prompt() | llm
            res = chain.invoke({
                "question": question,
                "chat_history": chat_history_text[-3000:] if chat_history_text else "None"
            })
            raw_content = getattr(res, "content", str(res)).strip()
            # Clean JSON markdown fences using core_utils helper
            raw_content = clean_json_markdown(raw_content)

            parsed = json.loads(raw_content)
            intent = parsed.get("intent", "retrieve")
            standalone = parsed.get("standalone_query", question)
            domain = parsed.get("domain", "general")
            sentiment = parsed.get("sentiment", "neutral")
            search_queries = parsed.get("search_queries", [standalone])
            if not isinstance(search_queries, list) or not search_queries:
                search_queries = [standalone]

            return {
                "intent": intent,
                "standalone_query": standalone if standalone else question,
                "domain": domain,
                "sentiment": sentiment,
                "search_queries": search_queries[:4]
            }
        except Exception as e:
            print(f"[WARNING] Combined Supervisor Plan fallback: {e}", flush=True)
            contextualized = self.contextualize_question(question, chat_history_text)
            domain = self.route_subdomain(contextualized)
            return {
                "intent": "retrieve",
                "standalone_query": contextualized,
                "domain": domain,
                "sentiment": "neutral",
                "search_queries": [contextualized]
            }



    def check_faithfulness(self, context: str, generation: str) -> bool:
        """
        Evaluates whether answer generation is grounded in retrieved facts.
        """
        if not context.strip():
            return True
        try:
            llm = self._get_llm(temperature=0.0)
            chain = get_hallucination_prompt() | llm
            res = chain.invoke({
                "context": context[:3000],
                "generation": generation
            })
            content = getattr(res, "content", str(res)).strip().lower()
            return "yes" in content
        except Exception as e:
            print(f"⚠️ Faithfulness check error: {e}")
            return True

    def run(
        self,
        question: str,
        organization_name: str,
        organization_info: str,
        contact_info: str,
        chat_history_text: str = "",
        memory_context: str = "",
    ) -> Dict[str, Any]:
        """
        Executes the Multi-Agent RAG execution cycle with fused 1-call Supervisor planning,
        caching, sub-agent delegation, and multi-query search.
        """
        step_logs: List[str] = []
        start_time = time.time()

        # Step 0: Check Query Cache (0 LLM Calls!)
        cached = query_cache.get(question)
        if cached:
            execution_time = round(time.time() - start_time, 3)
            step_logs.append(f"⚡ Served from Semantic Query Cache in {execution_time}s.")
            cached["trace_logs"] = step_logs
            cached["execution_time_sec"] = execution_time
            return cached

        print("\n" + "#" * 70, flush=True)
        print("🔍 AGENTIC RAG DETAILED DEBUG TRACE START", flush=True)
        print("#" * 70, flush=True)
        print(f"👤 RAW QUESTION: {question}", flush=True)

        # Step 1: Fused Supervisor Planning (1 Single Fast LLM Call!)
        plan = self.supervisor_plan(question, chat_history_text)
        intent = plan.get("intent", "retrieve")
        standalone_q = plan.get("standalone_query", question)
        domain_key = plan.get("domain", "general")
        sentiment = plan.get("sentiment", "neutral")
        search_queries = plan.get("search_queries", [standalone_q])

        step_logs.append(f"🧭 Supervisor classified intent as: **'{intent}'**")
        step_logs.append(f"🎭 Supervisor detected user sentiment as: **'{sentiment}'**")

        if intent == "greeting":
            answer = "🇪🇹 **Selam! Welcome to the Visit Ethiopia Travel Assistant!** I'm here to help you explore the Land of Origins. What would you like to explore today? ✨"
            res = {
                "answer": answer,
                "route": intent,
                "trace_logs": step_logs,
                "docs": [],
                "is_grounded": True,
                "execution_time_sec": round(time.time() - start_time, 2)
            }
            query_cache.set(question, res)
            return res

        if intent in ["identity", "out_of_scope"]:
            answer = "I am the Visit Ethiopia Travel Assistant, an official digital guide designed specifically to showcase the tourism, culture, and government infrastructure of Ethiopia. 🇪🇹 I cannot assist with software development, coding, or topics outside of Ethiopia."
            res = {
                "answer": answer,
                "route": intent,
                "trace_logs": step_logs,
                "docs": [],
                "is_grounded": True,
                "execution_time_sec": round(time.time() - start_time, 2)
            }
            query_cache.set(question, res)
            return res

        if standalone_q != question:
            step_logs.append(f"🔄 Contextual Query Reformulator resolved query to: *\"{standalone_q}\"*")

        specialized_agent = get_specialized_agent(domain_key)
        step_logs.append(f"👥 Supervisor assigned request to **[{specialized_agent.name}]** (Domain: *'{domain_key}'*)")
        print(f"👥 SUPERVISOR DELEGATION: [{specialized_agent.name}] ({domain_key})", flush=True)

        # Step 2: Hybrid Search across Multi-Queries
        retriever = self.retriever_getter()
        step_logs.append(f"🔄 Proactive Multi-Query Search generated {len(search_queries)} search query variations.")


        if hasattr(retriever, "multi_search"):
            retrieved_docs = retriever.multi_search(search_queries, k=4)
        else:
            retrieved_docs = retriever.search(query=standalone_q, k=4)

        step_logs.append(f"📚 Retrieved {len(retrieved_docs)} raw candidate document chunks.")
        print(f"📚 RETRIEVED CHUNKS: {len(retrieved_docs)} candidates across multi-queries", flush=True)

        # Step 3: Fast Relevance Check (Skip extra LLM grader call if <= 4 high-confidence chunks)
        if len(retrieved_docs) > 4:
            step_logs.append("⚖️ Batch-grading document chunks...")
            filtered_docs = self.grade_documents_batch(standalone_q, retrieved_docs)
            step_logs.append(f"✅ Filtered to {len(filtered_docs)} relevant document chunks.")
        else:
            filtered_docs = retrieved_docs

        effective_docs = filtered_docs if filtered_docs else retrieved_docs
        raw_context_text = "\n\n".join([d.page_content for d in effective_docs])
        live_currency_info = format_live_currency_context()
        combined_context = f"{live_currency_info}\n\n{raw_context_text}"
        context_text = specialized_agent.format_domain_context(combined_context)[:6000]

        # Step 4: Specialized Sub-Agent Answer Generation
        step_logs.append(f"🤖 Generating answer with **{specialized_agent.name}**...")
        llm = self._get_llm(temperature=0.0)
        gen_prompt = get_prompt()
        gen_chain = gen_prompt | llm

        gen_res = gen_chain.invoke({
            "question": question,
            "chat_history": chat_history_text,
            "memory_context": memory_context,
            "organization_name": organization_name,
            "organization_info": organization_info,
            "contact_info": contact_info,
            "context": context_text,
        })
        answer = self._safe_str(gen_res)
        answer = clean_meta_talk(answer)

        # Step 5: Answer Faithfulness Verification
        is_grounded = self.check_faithfulness(context_text, answer)
        if not is_grounded:
            step_logs.append("⚠️ Faithfulness check flagged potential hallucination in generation.")

        execution_time = round(time.time() - start_time, 2)
        step_logs.append(f"⚡ Multi-Agent RAG pipeline finished in {execution_time}s.")

        serializable_docs = [
            {"page_content": d.page_content, "metadata": getattr(d, "metadata", {})} if hasattr(d, "page_content") else d
            for d in effective_docs
        ]

        res = {
            "answer": answer,
            "route": f"{intent}_{domain_key}",
            "retrieval_query": standalone_q,
            "sub_agent": specialized_agent.name,
            "raw_doc_count": len(retrieved_docs),
            "filtered_doc_count": len(effective_docs),
            "docs": serializable_docs,
            "is_grounded": is_grounded,
            "trace_logs": step_logs,
            "execution_time_sec": execution_time,
        }

        # Cache response for future queries
        query_cache.set(question, res)
        return res






