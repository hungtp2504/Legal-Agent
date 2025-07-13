import asyncio
import json
from collections import OrderedDict

import streamlit as st

from src.legal_agent.agent.agent_runner import LegalAgentRunner
from src.legal_agent.core.settings import settings

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

st.set_page_config(page_title="LegalAgent Chatbot", page_icon="⚖️", layout="wide")


@st.cache_resource
def get_agent_runner(_api_key):

    if not _api_key:
        return None

    print(f"Initializing Agent Runner for a new API key...")
    try:
        return LegalAgentRunner(settings=settings, api_key=_api_key)
    except Exception as e:

        st.sidebar.error(
            f"Lỗi khởi tạo Agent: API Key không hợp lệ hoặc có vấn đề kết nối."
        )
        print(f"Agent initialization error: {e}")
        return None


def on_api_key_change():

    print("API Key input changed. Clearing agent runner cache.")
    get_agent_runner.clear()


st.title("⚖️ Trợ lý Pháp lý AI")


with st.sidebar:
    st.header("Cấu hình")

    if "api_key_from_input" not in st.session_state:
        st.session_state.api_key_from_input = settings.GEMINI_API_KEY or ""

    st.text_input(
        "Nhập Google Gemini API Key của bạn:",
        type="password",
        key="api_key_from_input",
        on_change=on_api_key_change,
        help="API key của bạn sẽ không được lưu trữ.",
    )


api_key = st.session_state.api_key_from_input

agent_runner = get_agent_runner(api_key)


def format_and_render_step(node_name, raw_data):
    st.markdown("---")
    formatted_node_name = node_name.replace("_", " ").title()
    st.subheader(f"Bước: {formatted_node_name}")

    if node_name in ["keyword_extractor", "simple_keyword_extractor"] and isinstance(
        raw_data, dict
    ):
        keywords = raw_data.get("extracted_keywords", [])
        if keywords:
            with st.expander("Xem danh sách từ khóa đã trích xuất"):
                for keyword in keywords:
                    st.markdown(f"- `{keyword}`")
        else:
            st.info("Không có từ khóa nào được trích xuất.")
    elif node_name == "retriever" and isinstance(raw_data, dict):
        context_str = raw_data.get("retrieved_context", "[]")
        try:
            retrieved_docs = json.loads(context_str)
            if retrieved_docs:
                with st.expander("Xem các điều luật đã được tra cứu"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Kết quả tra cứu {i+1}:**")
                        st.json(doc)
            else:
                st.info("Không tìm thấy điều luật liên quan.")
        except json.JSONDecodeError:
            st.error("Lỗi khi đọc kết quả tra cứu.")
    else:
        content_to_display = None
        if isinstance(raw_data, dict) and raw_data:
            content_to_display = next(iter(raw_data.values()), None)
        elif isinstance(raw_data, str):
            content_to_display = raw_data
        if isinstance(content_to_display, str) and content_to_display.strip():
            st.markdown(content_to_display)
        else:
            st.info("Không có kết quả văn bản để hiển thị cho bước này.")


async def stream_and_render_response(
    user_prompt: str, response_placeholder, expander_container, agent_runner_instance
):
    full_response = ""
    reasoning_details_raw = OrderedDict()
    expander = expander_container.expander(
        "Xem chi tiết quá trình suy luận và nguồn", expanded=True
    )

    with st.status("Agent đang suy nghĩ...", expanded=True) as status:
        async for event in agent_runner_instance.stream_run(user_prompt):
            if event["type"] == "log":
                status.update(label=event["data"])
            elif event["type"] == "node_result":
                node_name = event["node_name"]
                raw_data = event["data"]
                if node_name not in reasoning_details_raw:
                    reasoning_details_raw[node_name] = raw_data
                    with expander:
                        format_and_render_step(node_name, raw_data)
            elif event["type"] == "final_chunk":
                full_response += event["data"]
                response_placeholder.markdown(full_response + "▌")

        status.update(label="Hoàn tất suy luận!", state="complete", expanded=False)
        expander.expanded = False

    response_placeholder.markdown(full_response)
    return full_response, reasoning_details_raw


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
            and "details" in message
            and message["details"]
        ):
            with st.expander(
                "Xem chi tiết quá trình suy luận và nguồn", expanded=False
            ):
                ordered_keys = [
                    "router",
                    "simple_keyword_extractor",
                    "analyzer",
                    "framework_generator",
                    "keyword_extractor",
                    "retriever",
                    "reasoner",
                    "responder",
                ]
                sorted_details = {
                    key: message["details"][key]
                    for key in ordered_keys
                    if key in message["details"]
                }
                for node_name, raw_data in sorted_details.items():
                    format_and_render_step(node_name, raw_data)

if agent_runner:
    if prompt := st.chat_input("Nhập tình huống hoặc câu hỏi pháp lý của bạn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            expander_container = st.container()
            full_response, reasoning_details = asyncio.run(
                stream_and_render_response(
                    prompt, response_placeholder, expander_container, agent_runner
                )
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "details": reasoning_details,
                }
            )
else:
    st.info(
        "Vui lòng nhập API Key hợp lệ vào thanh sidebar bên trái để bắt đầu trò chuyện."
    )
    st.chat_input("Nhập câu hỏi của bạn...", disabled=True)
