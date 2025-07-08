def parse_user_prompt(prompt: str) -> dict:
    import re
    match = re.search(r"in ([A-Za-z\s]+)", prompt)
    location = match.group(1).strip() if match else "Dubai"
    return {"location": location}
