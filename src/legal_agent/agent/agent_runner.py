import asyncio
import glob
import json
import logging
import os
import re
from uuid import uuid4

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ..tools.vector_retriever import VectorRetriever
from .agent_graph import build_agent_graph
from .prompt import (
    FactAnalysisPrompt,
    FinalReasoningPrompt,
    FrameworkGenerationPrompt,
    KeywordExtractionPrompt,
    ResponseGenerationPrompt,
    RouterPrompt,
    SimpleKeywordExtractionPrompt,
)
from .state import AgentState

logger = logging.getLogger(__name__)


class LegalAgentRunner:
    def __init__(self, settings, api_key: str):
        logger.info("Initializing LegalAgentRunner...")

        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=api_key,
            temperature=0,
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
                doc_name = data.get("document", {}).get("name", "Không rõ tên văn bản")
                for unit in data.get("units", []):
                    unit["document_name"] = doc_name
                    all_units[unit["id"]] = unit
        logger.info(f"✅ Preloaded {len(all_units)} units.")
        return all_units

    async def stream_run(self, query: str):
        config_run = {"configurable": {"thread_id": str(uuid4())}}
        initial_state = {"original_query": query}

        async for event in self.app.astream_events(initial_state, version="v1"):
            kind = event["event"]

            if kind == "on_chain_start" and event["name"] != "LangGraph":
                yield {"type": "log", "data": f"--- Bắt đầu: {event['name']} ---"}

            if kind == "on_chain_end" and event["name"] != "LangGraph":
                yield {
                    "type": "node_result",
                    "node_name": event["name"],
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
        if route not in ["case_analysis", "simple_rag"]:
            route = "simple_rag"
        return {"route_decision": route}

    def simple_rag_node(self, state: AgentState) -> dict:
        context_str = state.get("retrieved_context", "[]")
        query = state["original_query"]

        rag_prompt_template = PromptTemplate.from_template(
            "Bạn là một trợ lý pháp lý AI. Dựa vào các thông tin pháp lý được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng một cách chính xác và dễ hiểu. Nếu thông tin không đủ, hãy nói rõ là không tìm thấy thông tin.\n\n"
            "Thông tin pháp lý tham khảo:\n{context}\n\n"
            "Câu hỏi của người dùng:\n{query}\n\n"
            "Câu trả lời của bạn:"
        )
        prompt = rag_prompt_template.format(context=context_str, query=query)
        response = self.llm.with_config(tags=["final_answer"]).invoke(prompt)
        return {"final_response": response.content}

    def simple_keyword_extractor_node(self, state: AgentState) -> dict:
        """Trích xuất từ khóa từ một câu hỏi đơn giản."""
        prompt = SimpleKeywordExtractionPrompt.format(query=state["original_query"])
        response_text = self.llm.invoke(prompt).content
        keywords = []
        try:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            json_str = match.group(1) if match else response_text
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, list):
                keywords = parsed_json
        except (json.JSONDecodeError, AttributeError):

            keywords = [state["original_query"]]
            logger.warning(
                f"Failed to decode keywords for simple_rag. Falling back to original query."
            )
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
        keywords = []
        try:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
            json_str = match.group(1) if match else response_text
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, list):
                keywords = parsed_json
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"Failed to decode keywords from LLM response.")
        return {"extracted_keywords": keywords}

    def information_retrieval_node(self, state: AgentState) -> dict:

        keywords = state.get("extracted_keywords", [])
        if not keywords:
            return {"retrieved_context": "[]"}
        initial_hits = []
        for k in keywords:
            initial_hits.extend(self.vector_retriever.search(k, n_results=1))
        parent_article_ids = set()
        for hit in initial_hits:
            if not hit or "id" not in hit:
                continue
            match = re.search(r"(_điều-[\w.-]+)", hit["id"])
            if match:
                parent_article_ids.add(hit["id"][: match.end()])
        hydrated_context_units = []
        retrieved_ids = set()
        for article_id in parent_article_ids:
            for unit_id, unit_data in self.units_map.items():
                if unit_id.startswith(article_id) and unit_id not in retrieved_ids:
                    hydrated_context_units.append(unit_data)
                    retrieved_ids.add(unit_id)
        context_json_str = json.dumps(
            hydrated_context_units, indent=2, ensure_ascii=False
        )
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
