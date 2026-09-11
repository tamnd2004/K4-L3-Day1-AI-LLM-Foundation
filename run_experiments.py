"""Thí nghiệm API thật cho exercises.md; chỉ chạy khi có key trong .env.

Chạy: python run_experiments.py
Ghi kết quả không chứa key vào experiment_results.json sau từng thí nghiệm.
Không được pytest/grade.py tự động gọi.
"""

import contextlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import template as lab


TOKEN_TEXT = (
    "Mỗi buổi sáng, tôi dành thời gian đọc sách và ghi lại những điều mới học. "
    "Hôm nay, tôi tìm hiểu cách một chương trình Python gửi câu hỏi đến mô hình "
    "ngôn ngữ thông qua API. Trước khi gọi dịch vụ, tôi kiểm tra khóa truy cập, "
    "chọn mô hình và đặt giới hạn độ dài câu trả lời. Sau đó, tôi so sánh nội "
    "dung nhận được, thời gian chờ và số token đã dùng. Việc lưu lịch sử hội "
    "thoại giúp trợ lý hiểu câu hỏi tiếp theo, nhưng cũng làm tăng lượng dữ "
    "liệu gửi đi."
)
PERSONA = "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."


def main():
    path = Path(__file__).with_name("experiment_results.json")
    results = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "main_model": lab.OPENAI_MODEL,
        "mini_model": lab.OPENAI_MINI_MODEL,
        "reasoning_effort": os.getenv("LAB_REASONING_EFFORT"),
        "cost_note": "Lab estimates use GPT-4o reference pricing, not actual provider charges.",
        "experiments": {},
    }

    def record(name, fn):
        try:
            results["experiments"][name] = {"status": "ok", "result": fn()}
        except Exception as exc:
            # Không lưu request/header hay raw exception có thể chứa bí mật.
            results["experiments"][name] = {
                "status": "error", "error_type": type(exc).__name__,
                "http_status": getattr(exc, "status_code", None),
            }
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, results["experiments"][name]["status"], flush=True)

    def token_experiment():
        import tiktoken
        # Gọi trực tiếp để bảo đảm đây là encoding thật, không phải fallback.
        tokens = len(tiktoken.encoding_for_model("gpt-4o").encode(TOKEN_TEXT))
        words = len(TOKEN_TEXT.split())
        return {
            "text": TOKEN_TEXT, "whitespace_words": words, "encoding": "o200k_base",
            "actual_tiktoken_count": tokens,
            "count_tokens_explicit_gpt4o": lab.count_tokens(TOKEN_TEXT, model="gpt-4o"),
            "word_estimate": words / 0.75,
            "difference_percent_vs_word_estimate": (tokens - words / 0.75) / (words / 0.75) * 100,
            "configured_model_fallback_count": lab.count_tokens(TOKEN_TEXT),
        }

    record("token_count", token_experiment)
    prompt = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."
    for temperature in (0.0, 0.5, 1.0, 1.5):
        record(f"temperature_{temperature}", lambda t=temperature: {
            "prompt": prompt, "temperature": t, "top_p": 0.9, "max_tokens": 256,
            "response_and_latency": lab.call_openai(prompt, temperature=t),
        })
    for name, persona in (
        ("teacher", "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."),
        ("expert", "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."),
    ):
        record(f"persona_{name}", lambda p=persona: {
            "persona": p, "prompt": "Giải thích blockchain là gì?", "max_tokens": 512,
            "response_and_latency": lab.chat_with_system_prompt(
                p, "Giải thích blockchain là gì?", max_tokens=512,
            ),
        })
    record("model_comparison", lambda: lab.batch_compare([
        "Giải thích khác biệt giữa temperature và top_p trong một câu.",
    ]))

    def conversation():
        prompts = iter([
            "Tên tôi là An. Tôi đang học Python. Hãy chào tôi bằng một câu.",
            "Tôi tên gì và đang học ngôn ngữ lập trình nào? Trả lời một câu.",
            "quit",
        ])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            stats = lab.run_assistant(PERSONA, get_input=lambda: next(prompts))
        return {"persona": PERSONA, "stats": stats, "streamed_text": output.getvalue()}

    record("streaming_history", conversation)


if __name__ == "__main__":
    main()
