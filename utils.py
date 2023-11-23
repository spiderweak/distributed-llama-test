import json

def extract_and_parse_json(text: str):
    """Extract and parse JSON from a string.

    Args:
        text (str): String containing embedded JSON data.

    Returns:
        list: Parsed JSON data if valid, otherwise an empty list.
    """
    try:
        start = text.index('[')
        end = text.rindex(']') + 1
        json_string = text[start:end]
        return json.loads(json_string)
    except (ValueError, json.JSONDecodeError):
        print("Could not find or parse JSON")
        return []
