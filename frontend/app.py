import json
import os
import requests
import streamlit as st
from collections import OrderedDict

BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
STREAM_ENDPOINT = f"{BACKEND_URL}/api/v1/chat/stream"

st.set_page_config(page_title="LegalAgent Chatbot", page_icon="⚖️", layout="wide")

def stream_from_backend(query: str):
    payload = {"query": query}
    try:
        with requests.post(STREAM_ENDPOINT, json=payload, stream=True, timeout=600) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data:"):
                        try:
                            json_str = decoded_line[len("data:") :].strip()
                            yield json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối đến backend: {e}")
        yield {"type": "error", "data": f"Connection Error: {e}"}

def format_and_render_step(container, node_name, raw_data):
    """Hiển thị kết quả của một bước suy luận trong expander."""
    if not raw_data:
        return

    formatted_node_name = node_name.replace("_", " ").title()
    container.subheader(f"Bước: {formatted_node_name}")
    
    if isinstance(raw_data, dict):
        if node_name in ["keyword_extractor", "simple_keyword_extractor"]:
            keywords = raw_data.get("extracted_keywords", [])
            if keywords:
                container.markdown("- **Từ khóa được trích xuất:** " + ", ".join(f"`{k}`" for k in keywords))
        elif node_name == "retriever":
            context_str = raw_data.get("retrieved_context", "[]")
            try:
                retrieved_docs = json.loads(context_str)
                if retrieved_docs:
                    with container.expander(f"Xem chi tiết {len(retrieved_docs)} điều luật đã tra cứu"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Kết quả tra cứu {i+1}:**")
                            st.json(doc, expanded=False)
                else:
                    container.info("Không tìm thấy điều luật liên quan.")
            except json.JSONDecodeError:
                container.error("Lỗi khi đọc kết quả tra cứu.")
        else:
            content_to_display = next(iter(raw_data.values()), None)
            if isinstance(content_to_display, str) and content_to_display.strip():
                container.markdown(content_to_display)

    elif isinstance(raw_data, str) and raw_data.strip():
        container.markdown(raw_data)
    
    container.markdown("---")

st.title("⚖️ Trợ lý Pháp lý AI")



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "details" in message:
                with st.expander("Xem chi tiết quá trình suy luận", expanded=False):
                    for node_name, raw_data in message["details"].items():
                        format_and_render_step(st, node_name, raw_data)

if prompt := st.chat_input("Nhập tình huống hoặc câu hỏi pháp lý của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        reasoning_details = OrderedDict()
        
        with st.expander("🕵️‍♂️ **Quá trình suy luận...**", expanded=True) as reasoning_expander:
            reasoning_container = st.container()

            full_response = ""
            try:
            
                for event in stream_from_backend(prompt):
                    event_type = event.get("type")

                    if event_type == "node_result":
                        node_name = event.get("node_name")
                        if node_name and node_name not in reasoning_details:
                            data = event.get("data")
                            reasoning_details[node_name] = data
                            format_and_render_step(reasoning_container, node_name, data)
                    
                    elif event_type == "final_chunk":
                        chunk = event.get("data", "")
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    elif event_type == "error":
                        st.error(f"Lỗi từ Backend: {event.get('data')}")
                        break
            except Exception as e:
                st.error(f"Đã có lỗi xảy ra trong quá trình xử lý: {e}")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "details": reasoning_details}
            )