# Bản hoàn thiện — Lab 01

Đã triển khai Part 1–4 và hai hàm bonus. Giữ nguyên chữ ký hàm,
các key trả về và toàn bộ test gốc; thêm 5 test cho các tình huống biên.
Câu trả lời trong `exercises.md` dựa trên thí nghiệm API thật ngày 11/09/2026.

## Chạy trên Windows

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider
.venv\Scripts\python.exe grade.py
.venv\Scripts\python.exe solution\solution.py
```

Nếu chưa có `.env`, copy `.env.example` thành `.env` rồi nhập key OpenRouter.
Mẫu dùng `nex-agi/nex-n2.5-pro:free` và `nex-agi/nex-n2.5-mini:free`.
`LAB_REASONING_EFFORT=none` tắt reasoning cho cả hai, để so sánh câu trả lời
ngắn với cùng temperature/top_p/max_tokens. Extension này chỉ gửi khi base URL
là OpenRouter. Restart Python sau khi đổi `.env` vì tên model được nạp lúc import.

## Kết quả đã đo

Gemma 4 31B và 26B A4B được thử trước nhưng đều trả 429 từ pool Google AI
Studio dùng chung; key vẫn hợp lệ. Hai model Nex trả lời thành công,
bao gồm temperature 0.0/0.5/1.0/1.5, hai persona và hội thoại streaming hai lượt.

| Thí nghiệm | Kết quả |
|---|---|
| Cùng prompt phân biệt temperature và top_p | Pro 1,360 giây; Mini 0,823 giây |
| Persona giáo viên / chuyên gia | 5,072 / 7,850 giây |
| History | Nhớ đúng tên An và ngôn ngữ Python ở lượt hai |
| Tokenizer `o200k_base` | Đoạn 100 đơn vị cách trắng → 121 token |
| Ước lượng `số từ / 0.75` | 133,33 token; số đếm thật thấp hơn 9,25% |

Đây là một lần đo mỗi trường hợp, không phải benchmark kết luận Pro/Mini
luôn nhanh hoặc chính xác hơn. Bản persona chuyên gia kết thúc giữa bảng
với ngân sách 512 token; câu trả lời này chưa hoàn chỉnh.

Phản hồi model được giữ nguyên để có thể kiểm tra: câu Pro về top_p dùng
ký hiệu “≤ p” chưa chính xác. Nucleus sampling lấy tập token nhỏ nhất có tổng
xác suất đạt hoặc vượt ngưỡng p; vì xác suất rời rạc, tổng có thể lớn hơn p.
Các nội dung model nói về Việt Nam cũng là output thí nghiệm, chưa được kiểm
chứng như tài liệu sự kiện. Temperature cao/thấp không bảo đảm tính đúng đắn.

Kết quả gốc: `experiment_results.json`. Lần thử Gemma thất bại được giữ trong
`experiment_gemma_unavailable.json`; các file này không chứa key/header.
Để chạy lại thí nghiệm (gọi API thật và ghi đè kết quả hiện tại):

```powershell
.venv\Scripts\python.exe run_experiments.py
```

## Chi phí và token trong bài

Các endpoint `:free` có giá token bằng 0 theo catalog khi chạy. Tuy nhiên,
`gpt4o_cost_estimate` và `estimate_cost()` giữ bảng giá tham chiếu của đề,
đúng Phụ lục B. Chúng không phải hóa đơn OpenRouter; bài test scenario còn
yêu cầu `total_cost > 0`. Không sửa bảng giá thành 0 để rồi làm sai hợp đồng test.

`tiktoken` không nhận model ID Nex: mặc định dùng fallback ký tự. Bài 2.2
gọi rõ `count_tokens(text, model="gpt-4o")` và kiểm tra encoding thật để
không nhầm fallback với phép đếm tiktoken. Stats phiên chat theo yêu cầu lab
chỉ cộng nội dung user/assistant mới, chưa phản ánh toàn bộ request đã gửi lại.

Free endpoint có thể hết hạn mức tạm thời. Retry trong lab chờ
0,1/0,2/0,4 giây; không bảo đảm vượt qua quota ngày hoặc pool đang bận.
Stream đã in một phần sẽ không tự retry để tránh lặp output.

## Bản nộp

`solution/solution.py` và `solution/exercises.md` là hai file được bộ chấm
ưu tiên. Kết quả thí nghiệm cũng được copy vào `solution/` để đọc cùng câu trả lời.
Nếu sửa bản ở gốc, copy lại trước khi chấm/nộp:

```powershell
Copy-Item template.py solution\solution.py -Force
Copy-Item exercises.md solution\exercises.md -Force
Copy-Item experiment_results.json solution\experiment_results.json -Force
.venv\Scripts\python.exe grade.py
```

Việc push lên GitHub và xác nhận nộp link trên VLearn là hai bước riêng biệt.
Checklist nộp trên VLearn chỉ được đánh dấu sau khi đã thực sự xác nhận.

Nguồn cấu hình: [OpenRouter Pro](https://openrouter.ai/nex-agi/nex-n2.5-pro:free),
[OpenRouter Mini](https://openrouter.ai/nex-agi/nex-n2.5-mini:free).
