import re
import emoji


def clean_emoji(text: str):
    emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([s for s in text.split() if not any(i in s for i in emoji_list)])

    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)

    clean_text = emoji_pattern.sub(r'', clean_text)
    return clean_text


def clean_url(text: str):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE)
    return text


def normalize(text: str, **kwargs):
    if "lower" in kwargs:
        if kwargs['lower']:
            text = text.lower()
    if "cleaned_emoji" in kwargs:
        if kwargs['cleaned_emoji']:
            text = clean_emoji(text)

    text = re.sub(r'[\n]+', '', text)
    text = re.sub(r'[\t]+', ' ', text)
    text = re.sub(r'[\t\+]+', ' ', text)
    text = re.sub(r'[ ]{2}', ' ', text)
    text = re.sub(r'"', '', text)
    return text
