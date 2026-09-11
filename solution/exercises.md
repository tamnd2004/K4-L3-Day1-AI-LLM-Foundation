# K4 — Ngày 1: Bài Tập & Phản Ánh

## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature

Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)

> Thí nghiệm dùng `nex-agi/nex-n2.5-pro:free`, reasoning thì được tắt, `top_p=0.9`, `max_tokens=256`: temperature 0.0 và 1.5 trả lời về cà phê, còn 0.5 và 1.0 trả lời về hang Sơn Đoòng; bản 1.5 diễn giải dài hơn bản 0.0. Temperature cao làm phân phối lấy mẫu bớt tập trung, nhưng bốn kết quả này không cho thấy độ đa dạng tăng đều theo temperature và không chứng minh tính đúng sai của thông tin. Mỗi mức mới chạy được một lần nên cần lặp lại nhiều lần với cùng điều kiện để kết luận được xác đáng; phản hồi gốc và latency được lưu trong `experiment_results.json`.

### Câu 1.2 — Chọn temperature cho sản phẩm

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**

> Em chọn khoảng 0.2 để câu trả lời về chính sách, đơn hàng và quy trình ít biến động giữa các lượt, đồng thời vẫn giữ được giọng văn tự nhiên. Khi thử temperature, em giữ top_p cố định để dễ xác định nguyên nhân cho sự thay đổi. Temperature thấp không bảo đảm câu trả lời đúng: chatbot vẫn cần dữ liệu chính sách được kiểm chứng, chỉ dẫn không tự bịa và chuyển cho nhân viên khi thiếu thông tin.

### Câu 1.3 — Đánh đổi chi phí

Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**

> Theo bảng giá học tập trong `template.py`, tổng output mỗi ngày là `10.000 × 3 × 350 = 10.500.000 token`. GPT-4o: `10.500.000 / 1.000 × 0,010 = 105 USD/ngày`; GPT-4o-mini: `10.500.000 / 1.000 × 0,0006 = 6,30 USD/ngày`. GPT-4o đắt hơn khoảng `16,67 lần`, chênh `98,70 USD/ngày`, chỉ tính output vì đề không cho lượng input. Có thể chọn GPT-4o cho phân tích khiếu nại một cách phức tạp cần tổng hợp nhiều điều kiện; mini phù hợp cho phân loại yêu cầu hoặc trả lời FAQ đơn giản. Đây là kịch bản giá GPT của đề, không phải hóa đơn của hai endpoint Nex miễn phí dùng trong thí nghiệm.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona

Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:

- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)

> Với cùng câu hỏi và model Nex Pro, persona giáo viên dùng hình ảnh “cuốn sổ ghi chép” và ví dụ các bạn trao kẹo, nên từ vựng gần gũi và dễ hình dung cho trẻ. Persona chuyên gia dùng các thuật ngữ distributed ledger, hash, node, consensus, PoW/PoS và smart contract, tổ chức câu trả lời thành nhiều mục kỹ thuật. Bản giáo viên trả về sau khoảng 5,07 giây, bản chuyên gia sau 7,85 giây; cả hai được đặt `max_tokens=512`, nhưng bản chuyên gia dài và kết thúc giữa bảng nên không được coi là câu trả lời hoàn chỉnh. System prompt thay đổi cách trình bày, ví dụ và mức độ chi tiết rõ rệt, còn độ chính xác của các khẳng định vẫn cần kiểm tra riêng.

### Câu 2.2 — tiktoken vs đếm từ

Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**

> Đoạn thử: “Mỗi buổi sáng, em dành thời gian đọc sách và ghi lại những điều mới học. Hôm nay, em tìm hiểu cách một chương trình Python gửi câu hỏi đến mô hình ngôn ngữ thông qua API. Trước khi gọi dịch vụ, em kiểm tra khóa truy cập, chọn mô hình và đặt giới hạn độ dài câu trả lời. Sau đó, em so sánh nội dung nhận được, thời gian chờ và số token đã dùng. Việc lưu lịch sử hội thoại giúp trợ lý hiểu câu hỏi tiếp theo, nhưng cũng làm tăng lượng dữ liệu gửi đi.” Đoạn này có 100 đơn vị tách bằng khoảng trắng; `count_tokens(text, model="gpt-4o")` dùng encoding `o200k_base` trả 121 token, còn `100 / 0,75 ≈ 133,33 token`. Lấy ước lượng làm mẫu số, chênh lệch là `(121 - 133,33) / 133,33 × 100 = -9,25%`: số đếm thực thấp hơn ước lượng 9,25%. Dấu tiếng Việt và cách tokenizer học các mảnh từ có thể khiến một từ bị tách thành nhiều token; hơn nữa một từ tiếng Việt có thể gồm nhiều âm tiết cách nhau bằng khoảng trắng, nên đếm bằng split không tương đương đếm từ ngôn ngữ học. Không thể khẳng định mọi đoạn tiếng Việt đều tốn hơn tiếng Anh chỉ từ thí nghiệm này; phải so sánh văn bản tương đương với cùng tokenizer. Encoding trên là của OpenAI để làm đúng bài tiktoken, không phải tokenizer Nex; khi để model Nex mặc định, hàm rơi về ước lượng ký tự và trả 113.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming

**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)

> Streaming hữu ích nhất cho chatbot tương tác và câu trả lời dài vì người dùng thấy nội dung đầu tiên sớm, biết ứng dụng đang hoạt động và có thể đọc trong khi phần còn lại tiếp tục sinh. Nó không bảo đảm giảm tổng thời gian hoàn thành. Trong demo thật, trợ lý in dần hai câu trả lời và lượt thứ hai vẫn nhớ tên An cùng ngôn ngữ Python nhờ history. Non-streaming phù hợp cho tác vụ nền, phân loại hoặc nhận một JSON cần kiểm tra đầy đủ trước khi dùng; với streaming phải xử lý cả chunk metadata rỗng, content=None và khả năng đứt kết nối giữa chừng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**

> Exponential backoff tăng khoảng chờ sau mỗi lần lỗi, giảm tốc độ gửi lại khi server đang quá tải và cho dịch vụ thời gian hồi phục; lab dùng 0,1 → 0,2 → 0,4 giây cho ba lần retry. Nếu hàng nghìn client cùng chờ đúng một giây, chúng có thể đồng loạt gửi lại và tạo các đợt quá tải lặp đi lặp lại. Backoff thuần túy vẫn có thể đồng bộ giữa các client, nên sản phẩm thực nên thêm jitter, giới hạn số lần thử, tôn trọng Retry-After và chỉ retry lỗi tạm thời. Trong lần thử Gemma, 429 xuất phát từ pool dùng chung phía provider; retry ngắn của lab không đủ khắc phục, vì vậy đã chuyển sang hai endpoint Nex đáp ứng được.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona

**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**

> Persona sử dụng: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.” “Trợ giảng thân thiện” định hướng giọng giải thích gần gũi cho người mới; “ngắn gọn bằng tiếng Việt” giúp câu trả lời phù hợp thời lượng lab và người học. Persona được đặt trong message system ở đầu mọi request, tách khỏi history, nên không bị loại khi chỉ giữ ba lượt cuối. Demo hai lượt xác nhận trợ lý dùng tiếng Việt và trả lời đúng “An” cùng “Python” từ ngữ cảnh trước.

### Câu 4.2 — Hạn chế & cải thiện

**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**

> Hạn chế lớn nhất là history chỉ giữ ba lượt: thông tin ở các lượt cũ sẽ mất, còn thống kê token/chi phí hiện chỉ cộng câu hỏi mới và câu trả lời, chưa tính toàn bộ persona/history gửi lại hoặc token reasoning. Cải thiện bộ nhớ bằng cách tóm tắt các lượt sắp bị loại thành một bản ghi ngắn, giữ riêng với persona và gửi lại cùng ba lượt gần nhất; cần giới hạn token của bản tóm tắt và coi đó là dữ liệu hội thoại, không phải chỉ dẫn hệ thống. Về thống kê thực tế, có thể lưu trường usage từ provider và bảng giá đúng model, tách khỏi giá tham chiếu của bài. Free endpoint còn phụ thuộc hạn mức: demo hiện dừng và báo lỗi nếu stream đứt sau khi đã in một phần, tránh tự phát lại nội dung trùng.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
