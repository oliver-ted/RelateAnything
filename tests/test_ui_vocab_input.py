"""Vocabulary box parsing and error text (deploy/ui/vocab_input.py)."""
from deploy.ui.vocab_input import describe_error, parse_terms


def test_multi_word_entries_stay_intact():
    assert parse_terms("bottle, cap, label, conveyor belt") == [
        "bottle", "cap", "label", "conveyor belt"]
    assert parse_terms("on, attached to, next to, part of, on top of") == [
        "on", "attached to", "next to", "part of", "on top of"]


def test_newlines_whitespace_empties_and_duplicates():
    assert parse_terms(" bottle ,\n\nconveyor   belt\r\nBottle,, ,cap ") == [
        "bottle", "conveyor belt", "cap"]
    assert parse_terms("") == [] and parse_terms(None) == []


def test_describe_error_keeps_real_text_and_explains_known_causes():
    hint, detail = describe_error(AssertionError(
        "Prompt-free model does not support setting classes. Please try with "
        "Text/Visual prompt models."))
    assert "custom object list" in hint
    assert detail.startswith("AssertionError: Prompt-free model")
    hint, _ = describe_error(RuntimeError("ZeroGPU quota exceeded"))
    assert "GPU unavailable on this link" in hint
    _, detail = describe_error(ValueError("x" * 1000))
    assert len(detail) <= 300 and detail.endswith("...")


def test_ui_copy_has_no_em_dashes():
    import inspect
    import deploy.ui.vocab_input as m
    assert "—" not in inspect.getsource(m)
