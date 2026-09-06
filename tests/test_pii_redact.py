"""PII redaction uses synthetic identifiers only (no real personal data)."""
from core.pii import redact, _luhn_ok


def test_email_masked():
    out, hits = redact("Contact jane.doe99@example.org for details.")
    assert "[EMAIL]" in out and "jane.doe99" not in out
    assert "email" in hits


def test_phone_masked():
    out, hits = redact("Call +1 (555) 123-4567 tomorrow.")
    assert "[PHONE]" in out and "555" not in out
    assert "phone" in hits


def test_ssn_masked():
    out, hits = redact("SSN 123-45-6789 on file.")
    assert "[SSN]" in out and "123-45-6789" not in out


def test_card_luhn_gated():
    assert _luhn_ok("4111111111111111")
    out, hits = redact("Card 4111111111111111 charged.")
    assert "[CARD]" in out and "4111111111111111" not in out
    # Non-Luhn digit runs are order numbers, not cards: left intact.
    out2, hits2 = redact("Order 1234567890123 shipped.")
    assert "[CARD]" not in out2


def test_plain_text_untouched():
    out, hits = redact("Marie Curie discovered radium in Paris.")
    assert out == "Marie Curie discovered radium in Paris."
    assert hits == []


def test_names_off_by_default():
    out, _ = redact("Remember to call Alice tomorrow.")
    assert "Alice" in out


def test_names_on_with_backend():
    import config
    old = getattr(config, "MEMORY_PII_REDACT_NAMES", False)
    config.MEMORY_PII_REDACT_NAMES = True
    try:
        out, hits = redact("Remember to call Alice tomorrow.")
        assert "Alice" not in out and "person-name" in hits
    finally:
        config.MEMORY_PII_REDACT_NAMES = old


def test_disabled_flag_passes_through():
    import config
    old = getattr(config, "MEMORY_PII_REDACT", True)
    config.MEMORY_PII_REDACT = False
    try:
        out, hits = redact("Mail a@b.co now.")
        assert out == "Mail a@b.co now." and hits == []
    finally:
        config.MEMORY_PII_REDACT = old
