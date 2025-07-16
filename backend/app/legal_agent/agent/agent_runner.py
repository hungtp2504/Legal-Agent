import glob
import json
import logging
import os
import re
from functools import lru_cache
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from ...core.settings import settings
from ..tools.vector_retriever import VectorRetriever
from .agent_graph import build_agent_graph
from .prompt import (
    FactAnalysisPrompt, FinalReasoningPrompt, FrameworkGenerationPrompt,
    KeywordExtractionPrompt, ResponseGenerationPrompt, RouterPrompt,
    SimpleKeywordExtractionPrompt, SimpleRAGPrompt
)
from .state import AgentState

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_agent_runner():
    logger.info(f"Initializing or retrieving cached LegalAgentRunner.")
    return LegalAgentRunner(settings=settings)

class LegalAgentRunner:
    def __init__(self, settings):
        self.settings = settings
        if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
            self.langfuse = Langfuse(
                secret_key=settings.LANGFUSE_SECRET_KEY,
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                host=settings.LANGFUSE_HOST,
            )

        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            openai_api_key=self.settings.OPENAI_API_KEY,
            temperature=0,
            streaming=True,
        )

        self.vector_retriever = VectorRetriever(settings)
        self.units_map = self._preload_all_units(settings)
        self.app = build_agent_graph(self)
        logger.info("✅ LegalAgentRunner initialized successfully.")

    def _preload_all_units(self, settings):
        logger.info("Preloading all JSON data into memory...")
        all_units = {}
        folder_path = settings.PARSED_JSON_DIR
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        for filepath in json_files:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                doc_name = data.get("document", {}).get("name", "Unknown Document")
                for unit in data.get("units", []):
                    unit["document_name"] = doc_name
                    all_units[unit["id"]] = unit
        logger.info(f"✅ Preloaded {len(all_units)} units.")
        return all_units

    async def stream_run(self, query: str):
        session_id = str(uuid4())
        initial_state = {"original_query": query}
        
        callbacks = []
        if hasattr(self, 'langfuse'):
            langfuse_handler = CallbackHandler()
            callbacks.append(langfuse_handler)

        config = {"configurable": {"thread_id": session_id}, "callbacks": callbacks}

        async for event in self.app.astream_events(initial_state, config=config, version="v1"):
            kind = event["event"]
            
            if kind == "on_chain_end":
                node_name = event["name"]
                if node_name not in ["LangGraph", "responder", "simple_rag"]:
                    yield {
                        "type": "node_result",
                        "node_name": node_name,
                        "data": event["data"].get("output"),
                    }
            
            if kind == "on_chat_model_stream":
                if "final_answer" in event["tags"]:
                    content = event["data"]["chunk"].content
                    if content:
                        yield {"type": "final_chunk", "data": content}

    def router_node(self, state: AgentState) -> dict:
        prompt = RouterPrompt.format(query=state["original_query"])
        response = self.llm.invoke(prompt)
        route = response.content.strip().lower()
        if "case_analysis" in route:
            route = "case_analysis"
        else: 
            route = "simple_rag"

        return {"route_decision": route}

    def simple_keyword_extractor_node(self, state: AgentState) -> dict:
        prompt = SimpleKeywordExtractionPrompt.format(query=state["original_query"])
        response_text = self.llm.invoke(prompt).content
        try:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            json_str = match.group(1) if match else response_text
            keywords = json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            keywords = [state["original_query"]]
        return {"extracted_keywords": keywords}

    def analyze_case_node(self, state: AgentState) -> dict:
        prompt = FactAnalysisPrompt.format(query=state["original_query"])
        response = self.llm.invoke(prompt)
        return {"fact_analysis": response.content}

    def generate_reasoning_framework_node(self, state: AgentState) -> dict:
        prompt = FrameworkGenerationPrompt.format(fact_analysis=state["fact_analysis"])
        response = self.llm.invoke(prompt)
        return {"reasoning_framework": response.content}

    def keyword_extraction_node(self, state: AgentState) -> dict:
        prompt = KeywordExtractionPrompt.format(
            fact_analysis=state["fact_analysis"],
            reasoning_framework=state["reasoning_framework"],
        )
        response_text = self.llm.invoke(prompt).content
        try:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            json_str = match.group(1) if match else response_text
            keywords = json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            keywords = []
        return {"extracted_keywords": keywords}

    def information_retrieval_node(self, state: AgentState) -> dict:
        keywords = state.get("extracted_keywords", [])
        if not keywords: return {"retrieved_context": "[]"}
        
        unique_hits = {}
        for k in keywords:
            search_results = self.vector_retriever.search(k, n_results=3)
            for hit in search_results:
                if hit['id'] not in unique_hits:
                    unique_hits[hit['id']] = hit

        sorted_hits = sorted(unique_hits.values(), key=lambda x: x['similarity'], reverse=True)

        context_json_str = json.dumps(sorted_hits, indent=2, ensure_ascii=False)
        
        # parent_article_ids = set()
        # for hit in sorted_hits:
        #       match = re.search(r"(_điều-[\w.-]+)", hit["id"])
        #       if match:
        #           parent_article_ids.add(hit["id"][: match.end()])
        # # print("----------------------------------------------------------------------------------------------len(parent_article_ids): ",len(parent_article_ids), parent_article_ids)
        # hydrated_context_units = []
        # retrieved_ids = set()
        # for article_id in parent_article_ids:
        #     for unit_id, unit_data in self.units_map.items():
        #         if unit_id.startswith(article_id) and unit_id not in retrieved_ids:
        #             hydrated_context_units.append(unit_data)
        #             retrieved_ids.add(unit_id)
        # # print("----------------------------------------------------------------------------------------------len(hydrated_context_units): ",len(hydrated_context_units), hydrated_context_units)
        # context_json_str = json.dumps(hydrated_context_units, indent=2, ensure_ascii=False)
        return {"retrieved_context": context_json_str}

    def final_reasoning_node(self, state: AgentState) -> dict:
        prompt = FinalReasoningPrompt.format(
            reasoning_framework=state["reasoning_framework"],
            context=state["retrieved_context"],
        )
        response = self.llm.invoke(prompt)
        return {"final_analysis": response.content}

    def response_generation_node(self, state: AgentState) -> dict:
        prompt = ResponseGenerationPrompt.format(
            query=state["original_query"],
            final_analysis=state["final_analysis"],
            retrieved_context=state["retrieved_context"],
        )
        response = self.llm.with_config(tags=["final_answer"]).invoke(prompt)
        return {"final_response": response.content}

    def simple_rag_node(self, state: AgentState) -> dict:
        prompt = SimpleRAGPrompt.format(context=state["retrieved_context"], query=state["original_query"])
        response = self.llm.with_config(tags=["final_answer"]).invoke(prompt)
        return {"final_response": response.content}