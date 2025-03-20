import emoji

# Code inspired by https://medium.com/@ebimsv/nlp-series-day-5-handling-emojis-strategies-and-code-implementation-0f8e77e3a25c
def normalize_emojis(text: str) -> str:
    """
    Normalize emojis in the given text.

    Args:
        text (str): The text containing emojis to be normalized.

    Returns:
        str: The text with emojis normalized to their textual representation.
    
    Author:
        Kelvin Mock
    """
    return emoji.demojize(text)

# Code inspired by https://ai.plainenglish.io/how-to-perform-hashtag-analysis-using-natural-language-processing-and-machine-learning-in-python-ea6da817b3a4
def normalize_symbols(text: str) -> str:
    """
    Normalize symbols in the given text by removing URLs, mentions, and hashtags.

    Args:
        text (str): The text containing symbols to be normalized.

    Returns:
        str: The text with URLs, mentions, and hashtags removed.

    Author:
        Kelvin Mock
    """
    return ' '.join(word for word in text.split() if not (word.startswith('http') or word.startswith('@') or word.startswith('#')))

# Code inspired by https://ai.plainenglish.io/how-to-perform-hashtag-analysis-using-natural-language-processing-and-machine-learning-in-python-ea6da817b3a4
def normalize_punctuation(text: str) -> str:
    """
    Normalize punctuation in the given text by removing all punctuation marks and returning the text in lowercase.

    Args:
        text (str): The text containing punctuation to be normalized.

    Returns:
        str: The text with all punctuation marks removed."
    
    Author:
        Kelvin Mock
    """
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    text = text.lower()
    return text

def normalize(text: str) -> str:
    """
    Normalize the given text with all possible methods.

    Args:
        text (str): The text to be normalized.

    Returns:
        str: The normalized text."
    
    Author:
        Kelvin Mock
    """
    text = normalize_emojis(text)
    text = normalize_symbols(text)
    text = normalize_punctuation(text)
    return text