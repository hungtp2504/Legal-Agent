from typing import List, TypedDict


class AgentState(TypedDict):

    original_query: str

    route_decision: str

    fact_analysis: str

    reasoning_framework: str

    extracted_keywords: List[str]

    retrieved_context: str

    final_analysis: str

    final_response: str
