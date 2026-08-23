"""
Normalization utilities for facts and verification standards.
Provides stable keys for exact matching and contradiction detection.
"""
import re


def normalize_name(name: str) -> str:
    name = re.sub(r'\s+', ' ', str(name).lower()).strip()
    name = re.sub(r'[^\w\s]', '', name)
    return name


def normalize_statement(statement: str) -> str:
    return normalize_name(statement)


def normalize_triple(subject: str, predicate: str, obj: str, negation: int = 0):
    return (
        normalize_name(subject),
        normalize_name(predicate),
        normalize_name(obj),
        int(negation or 0)
    )


def make_standard_key(standard: dict) -> tuple:
    """Return a stable tuple key for a standard / fact."""
    return normalize_triple(
        standard.get("subject", ""),
        standard.get("predicate", ""),
        standard.get("object", ""),
        standard.get("negation", 0)
    )


def make_statement_key(statement: str) -> str:
    return normalize_statement(statement)
