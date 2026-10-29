
def triage_ticket(ticket: str) -> dict:
    text = ticket.lower()

    if any(word in text for word in ["security breach", "data leak"]):
        category = "security"
        priority = "critical"
        product_area = "security"

    elif any(word in text for word in ["crash", "error", "broken", "fail"]):
        category = "bug"
        priority = "high"
        product_area = "application"

    elif any(word in text for word in ["slow", "latency", "performance"]):
        category = "performance"
        priority = "medium"
        product_area = "performance"

    else:
        category = "other"
        priority = "low"
        product_area = "general"

    return {
        "category": category,
        "priority": priority,
        "product_area": product_area,
        "summary": " ".join(ticket.split())[:200],
    }
