"""Các tình huống chưa có trong bộ test gốc; không gọi API thật."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests._loader import MOD


def test_sampling_parameters_are_forwarded():
    with patch("openai.OpenAI") as factory:
        client = factory.return_value
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))]
        )
        MOD.call_openai("prompt", model="test-model", temperature=1.5, top_p=0.4, max_tokens=37)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert (kwargs["model"], kwargs["temperature"], kwargs["top_p"], kwargs["max_tokens"]) == (
            "test-model", 1.5, 0.4, 37,
        )
        client.close.assert_called_once()


def test_retry_delays_and_original_exception():
    error = RuntimeError("offline")
    fn = MagicMock(side_effect=error)
    with patch.object(MOD.time, "sleep") as sleep:
        with pytest.raises(RuntimeError) as caught:
            MOD.retry_with_backoff(fn, max_retries=3, base_delay=0.1)
    assert caught.value is error
    assert fn.call_count == 4
    assert [call.args[0] for call in sleep.call_args_list] == [0.1, 0.2, 0.4]


def test_stream_skips_metadata_and_closes(capsys):
    class Stream:
        closed = False

        def __iter__(self):
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Xin chào"))])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])

        def close(self):
            self.closed = True

    stream = Stream()
    assert MOD._print_stream(stream) == "Xin chào"
    assert stream.closed
    assert capsys.readouterr().out == "Xin chào\n"


def test_stream_failure_is_not_replayed():
    def broken_stream():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))])
        raise ConnectionError("connection interrupted")

    with patch("openai.OpenAI") as factory:
        client = factory.return_value
        client.chat.completions.create.return_value = broken_stream()
        with pytest.raises(ConnectionError):
            MOD.run_assistant("persona", get_input=lambda: "hello", max_turns=1)
        client.chat.completions.create.assert_called_once()
        client.close.assert_called_once()


def test_batch_preserves_prompt_order_and_formats_table():
    result = {"gpt4o_response": "word " * 20, "mini_response": "short\nreply",
              "gpt4o_latency": 0.5, "mini_latency": 0.2, "gpt4o_cost_estimate": 0.01}
    with patch.object(MOD, "compare_models", return_value=result):
        rows = MOD.batch_compare(["first", "second"])
    assert [row["prompt"] for row in rows] == ["first", "second"]
    assert "prompt" not in result
    table = MOD.format_comparison_table(rows)
    assert "short reply" in table and "..." in table
    assert "0.500s" in table
    assert len(table.splitlines()) == 4
    assert len(MOD.format_comparison_table([]).splitlines()) == 2
