"""Text processing utilities"""

import re


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_special_chars(text: str) -> str:
    """Remove special characters but keep punctuation"""
    text = re.sub(r'[^\w\s.,;:!?-]', '', text)
    return text


def split_into_sentences(text: str) -> list:
    """Simple sentence splitter"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]
