from langchain_core.prompts import PromptTemplate

SIMPLE_KEYWORD_EXTRACTION_PROMPT = """
Bạn là một AI trợ lý pháp lý. Từ câu hỏi của người dùng dưới đây, hãy trích xuất các thuật ngữ pháp lý hoặc các khái niệm cốt lõi nhất để phục vụ cho việc tra cứu. Chỉ tập trung vào các danh từ và cụm danh từ chính.
Trả về kết quả dưới dạng một danh sách JSON của các chuỗi.

Ví dụ 1:
Câu hỏi: "Giao dịch dân sự vô hiệu là gì?"
Kết quả: ["giao dịch dân sự vô hiệu"]

Ví dụ 2:
Câu hỏi: "Thời hiệu khởi kiện đòi lại tài sản là bao lâu?"
Kết quả: ["thời hiệu khởi kiện", "đòi lại tài sản"]

Câu hỏi của người dùng:
{query}

Danh sách JSON các từ khóa:
"""
SimpleKeywordExtractionPrompt = PromptTemplate.from_template(
    SIMPLE_KEYWORD_EXTRACTION_PROMPT
)

FACT_ANALYSIS_PROMPT = """
Bạn là một chuyên gia phân tích pháp lý cao cấp. Nhiệm vụ của bạn là đọc kỹ một tình huống và "mổ xẻ" nó một cách có hệ thống, khách quan.
**Tình huống:**
{query}
**Nhiệm vụ:**
Phân tích tình huống trên theo đúng 3 mục sau. KHÔNG đưa ra bất kỳ kết luận hay bình luận pháp lý nào, chỉ phân tích sự kiện.
**1. CÁC BÊN LIÊN QUAN VÀ ĐẶC ĐIỂM:**
- Liệt kê các cá nhân, tổ chức và các đặc điểm pháp lý quan trọng của họ (ví dụ: bị can, người bị hại, doanh nghiệp nhà nước...).
**2. DÒNG THỜI GIAN CÁC SỰ KIỆN:**
- Sắp xếp các sự kiện pháp lý (hành vi vi phạm, giao dịch, đăng ký, khởi tố...) theo đúng trình tự thời gian.
**3. ĐỐI TƯỢNG, HÀNH VI CỐT LÕI:**
- Xác định đối tượng (tài sản, thông tin...) hoặc hành vi trung tâm của vụ việc.
**BẮT ĐẦU PHÂN TÍCH!**
"""
FactAnalysisPrompt = PromptTemplate.from_template(FACT_ANALYSIS_PROMPT)

FRAMEWORK_GENERATION_PROMPT = """
Bạn là một Luật sư AI Cấp cao, có nhiệm vụ xây dựng một chiến lược pháp lý để giải quyết một vụ việc, áp dụng cho MỌI LĨNH VỰC PHÁP LUẬT (Dân sự, Hình sự, Hành chính, v.v.).
Một trợ lý đã cung cấp cho bạn bản phân tích sự việc dưới đây.
**Bản phân tích sự việc:**
{fact_analysis}
**Nhiệm vụ:**
Dựa vào bản phân tích trên, hãy xác định các vấn đề pháp lý mấu chốt và tạo ra một **KHUNG SƯỜN SUY LUẬN** logic theo từng bước.
**YÊU CẦU VỀ KHUNG SƯỜN:**
1.  **Xác định Quy phạm pháp luật gốc:** Chỉ ra (các) quy phạm pháp luật chính có khả năng điều chỉnh vấn đề (ví dụ: quy định về hợp đồng, quy định về cấu thành tội phạm, quy định về xử phạt vi phạm hành chính).
2.  **Phân tích các yếu tố mấu chốt:** Nêu ra các yếu tố hoặc điều kiện cần được làm rõ theo quy phạm pháp luật đã xác định (ví dụ: các yếu tố cấu thành tội phạm, điều kiện có hiệu lực của hợp đồng, thẩm quyền của cơ quan ra quyết định).
3.  **Lường trước các chế định liên quan:** Chỉ ra các quy tắc ngoại lệ, các chế định bổ sung, hoặc các vấn đề pháp lý phức tạp khác cần xem xét (ví dụ: tình tiết tăng nặng/giảm nhẹ, quy định bảo vệ bên yếu thế, vấn đề thời hiệu).
4.  **Tổng hợp các câu hỏi pháp lý:** Đúc kết lại thành các câu hỏi pháp lý cốt lõi cần được trả lời để giải quyết toàn bộ vụ việc.
**KHÔNG SỬ DỤNG VÍ DỤ CỤ THỂ CỦA MỘT NGÀNH LUẬT NÀO.** Hãy tư duy một cách tổng quát.
Bây giờ, hãy tạo Khung sườn Suy luận cho vụ việc đã được phân tích.
"""
FrameworkGenerationPrompt = PromptTemplate.from_template(FRAMEWORK_GENERATION_PROMPT)

KEYWORD_EXTRACTION_PROMPT = """
Bạn là một AI chuyên gia về pháp luật Việt Nam. Nhiệm vụ của bạn là đọc một vụ việc đã được phân tích và trích xuất ra tất cả các thuật ngữ, khái niệm, và số hiệu điều luật quan trọng để phục vụ cho việc tra cứu thông tin.
**Tài liệu cần phân tích:**
1.  **Phân tích sự việc:** {fact_analysis}
2.  **Khung sườn suy luận:** {reasoning_framework}
**Nhiệm vụ:**
Đọc kỹ các tài liệu trên và xác định tất cả các:
-   **Thuật ngữ pháp lý chuyên ngành** (ví dụ: "cố ý gây thương tích", "hợp đồng vô hiệu", "bên thứ ba ngay tình", "đấu thầu rộng rãi", "khung hình phạt", "thời hiệu xử phạt").
-   **Số hiệu điều luật** được nhắc đến (ví dụ: "Điều 133", "Điều 17").
-   **Các khái niệm hoặc đối tượng quan trọng** trong vụ việc (ví dụ: "quyền sử dụng đất", "xe ô tô", "quyết định hành chính").
Hãy trả về kết quả dưới dạng một danh sách JSON của các chuỗi.
**Ví dụ định dạng đầu ra:**
["cố ý gây thương tích", "khung hình phạt", "Điều 134 Bộ luật Hình sự", "tỷ lệ thương tật", "tình tiết giảm nhẹ"]
Bây giờ, hãy thực hiện trích xuất từ tài liệu đã cho.
"""
KeywordExtractionPrompt = PromptTemplate.from_template(KEYWORD_EXTRACTION_PROMPT)


FINAL_REASONING_PROMPT = """
Bạn là một Thẩm phán AI, có nhiệm vụ đưa ra nhận định pháp lý cuối cùng.

**1. KHUNG SƯỜN SUY LUẬN CẦN TUÂN THỦ:**
{reasoning_framework}

**2. CÁC QUY ĐỊNH PHÁP LUẬT LIÊN QUAN ĐÃ TRUY VẤN (Bao gồm cả tên văn bản):**
{context}

**Nhiệm vụ:**
Hãy viết một bài phân tích pháp lý chi tiết và đầy đủ. 
- **TUÂN THỦ NGHIÊM NGẶT** từng bước trong "KHUNG SƯỜN SUY LUẬN".
- Với mỗi bước, hãy viện dẫn các điều luật cụ thể từ phần "CÁC QUY ĐỊNH PHÁP LUẬT". 
- **QUAN TRỌNG:** Khi viện dẫn một điều luật, hãy nêu rõ nó thuộc văn bản nào dựa vào trường `document_name` được cung cấp (ví dụ: "Theo Điều 133 của Bộ luật Dân sự 2015...").
- Bài phân tích phải logic, chặt chẽ, và đi đến một kết luận dứt khoát cho các vấn đề pháp lý đã được xác định.
"""
FinalReasoningPrompt = PromptTemplate.from_template(FINAL_REASONING_PROMPT)


RESPONSE_GENERATION_PROMPT = """
Bạn là một Trợ lý Luật sư AI chuyên nghiệp, có nhiệm vụ biên tập lại một bài phân tích pháp lý nội bộ thành một văn bản tư vấn hoàn chỉnh để gửi cho khách hàng.

**NGUỒN DỮ LIỆU ĐẦU VÀO:**

**1. Tình huống của khách hàng:**
{query}

**2. Bài phân tích và phán quyết pháp lý nội bộ (NGUỒN THÔNG TIN CHÍNH ĐỂ TRẢ LỜI):**
{final_analysis}

**3. Thông tin nguồn đã tra cứu (DÙNG ĐỂ TRÍCH DẪN Ở CUỐI BÀI):**
{retrieved_context}


**NHIỆM VỤ:**

Dựa HOÀN TOÀN vào "Bài phân tích và phán quyết pháp lý nội bộ", hãy soạn một câu trả lời cuối cùng để gửi cho khách hàng.

**YÊU CẦU BẮT BUỘC:**
1.  **TUÂN THỦ TUYỆT ĐỐI:** Phải tuân thủ nghiêm ngặt các lập luận, viện dẫn điều luật, và **KẾT LUẬN CUỐI CÙNG** trong bài phân tích nội bộ.
2.  **Cấu trúc rõ ràng:** Soạn câu trả lời theo đúng các câu hỏi mà khách hàng đã đặt ra.
3.  **Trích dẫn chuẩn xác:** Khi viện dẫn một điều luật từ bài phân tích, hãy trình bày rõ ràng và bao gồm cả tên văn bản luật. Ví dụ: "**Điều 133 Bộ luật Dân sự 2015** quy định: *[nội dung điều luật]*".
4.  **Kết luận dứt khoát:** Phần kết luận của bạn phải phản ánh chính xác kết luận trong bài phân tích nội bộ.
5.  **Tập trung vào Giải pháp:** Đưa ra các bước đi tiếp theo và giải pháp khả thi cho các bên.
6.  **Liệt kê Nguồn chi tiết:** Ở cuối cùng của toàn bộ câu trả lời, tạo một phần có tiêu đề là "**Nguồn tham khảo:**". Dưới tiêu đề này, liệt kê các nguồn đã được sử dụng trong câu trả lời dưới dạng danh sách bullet (dấu *). Với mỗi nguồn, hãy trình bày THEO ĐÚNG ĐỊNH DẠNG PHÂN CẤP SAU BẰNG MARKDOWN:
    * **Văn bản:** [Lấy từ trường 'document_name']
        * **Bối cảnh:** [Lấy từ trường 'context']
        * **Nội dung:** [Lấy từ trường 'content']

Bây giờ, hãy bắt đầu biên tập lại bài phân tích thành một câu trả lời hoàn chỉnh.
"""
ResponseGenerationPrompt = PromptTemplate.from_template(RESPONSE_GENERATION_PROMPT)


ROUTER_PROMPT = """
Bạn là một Agent điều phối viên pháp lý thông minh. Nhiệm vụ của bạn là phân loại câu hỏi của người dùng để chuyển đến đúng chuyên gia xử lý.
Dựa vào câu hỏi dưới đây, hãy quyết định nó thuộc loại nào trong hai loại sau:
1.  **"case_analysis"**: Nếu câu hỏi mô tả một tình huống cụ thể, có nhiều sự kiện, nhiều bên liên quan và yêu cầu phân tích sâu về quyền, nghĩa vụ, hoặc tính hợp pháp của các giao dịch.
    * Ví dụ: "Ông An cho ông Bình mượn xe, ông Bình lại bán cho bà Chi...", "Công ty A ký hợp đồng với công ty B nhưng sau đó..."
2.  **"simple_rag"**: Nếu câu hỏi chỉ đơn thuần là tra cứu, hỏi định nghĩa, hoặc hỏi nội dung của một điều luật cụ thể. Nó không chứa một kịch bản phức tạp.
    * Ví dụ: "Điều 117 Bộ luật Dân sự 2015 quy định về gì?", "Giao dịch dân sự vô hiệu là gì?", "Thời hiệu khởi kiện đòi lại tài sản là bao lâu?"
**Câu hỏi của người dùng:**
{query}
Hãy trả về CHỈ MỘT từ duy nhất là **"case_analysis"** hoặc **"simple_rag"**.
"""
RouterPrompt = PromptTemplate.from_template(ROUTER_PROMPT)
