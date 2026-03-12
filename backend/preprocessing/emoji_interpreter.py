import emoji


def expand_emojis(text: str) -> str:
    """
    Converts all emojis to English text using the emoji library.
    Handles all 5000+ emojis automatically — no hardcoding needed.
    Example: "you're 🔪😈" → "you're kitchen knife smiling face with horns"
    """
    return emoji.replace_emoji(
        text,
        replace=lambda chars, data: f" {data.get('en', '').replace('_', ' ').strip(':')} ",
    ).strip()


def has_emoji(text: str) -> bool:
    return emoji.emoji_count(text) > 0
