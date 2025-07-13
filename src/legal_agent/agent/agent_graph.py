from langgraph.graph import END, StateGraph

from .state import AgentState


def decide_route(state: AgentState):

    route_decision = state.get("route_decision")
    if route_decision == "case_analysis":

        return "analyzer"
    else:

        return "simple_keyword_extractor"


def after_retrieval_router(state: AgentState):

    if state.get("route_decision") == "case_analysis":
        return "reasoner"
    else:
        return "simple_rag"


def build_agent_graph(runner):

    workflow = StateGraph(AgentState)

    workflow.add_node("router", runner.router_node)

    workflow.add_node("analyzer", runner.analyze_case_node)
    workflow.add_node("framework_generator", runner.generate_reasoning_framework_node)
    workflow.add_node("keyword_extractor", runner.keyword_extraction_node)
    workflow.add_node("reasoner", runner.final_reasoning_node)
    workflow.add_node("responder", runner.response_generation_node)

    workflow.add_node("simple_keyword_extractor", runner.simple_keyword_extractor_node)
    workflow.add_node("retriever", runner.information_retrieval_node)
    workflow.add_node("simple_rag", runner.simple_rag_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        decide_route,
        {
            "analyzer": "analyzer",
            "simple_keyword_extractor": "simple_keyword_extractor",
        },
    )

    workflow.add_edge("analyzer", "framework_generator")
    workflow.add_edge("framework_generator", "keyword_extractor")
    workflow.add_edge("keyword_extractor", "retriever")

    workflow.add_edge("simple_keyword_extractor", "retriever")

    workflow.add_conditional_edges(
        "retriever",
        after_retrieval_router,
        {"reasoner": "reasoner", "simple_rag": "simple_rag"},
    )

    workflow.add_edge("reasoner", "responder")
    workflow.add_edge("responder", END)

    workflow.add_edge("simple_rag", END)

    return workflow.compile()
