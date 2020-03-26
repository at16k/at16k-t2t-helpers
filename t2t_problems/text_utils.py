import re


def clean_text(text):
    text = re.sub(r"\[.*\]", " ", text)
    text = re.sub(r"\(.*\)", " ", text)
    text = re.sub("’", "'", text)
    text = re.sub(r"[^a-zA-Z0-9.,?!$%&@\\s'\*\/:#\-]", " ", text)
    tokens = [t for t in text.split() if t]
    text = ' '.join(tokens)
    return text


def clean_punkt(text):
    text = text.strip('?!.,')
    text = re.sub("([A-Za-z]+[\\s]?)(,)([\\s]?[A-Za-z]+[\\s]?)", "\\1\\3", text)
    return text


def format_text(string):
    text = clean_text(string)
    text = text.lower()
    text = clean_punkt(text)
    return text
