import re


class DLPScanner:
    """Data Loss Prevention — detects PII in messages"""

    PATTERNS = {
        "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "phone": r"\b[6-9]\d{9}\b",
        "email": r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
        "address_hint": r"\b(flat|house|near|street|nagar|colony|road)\b",
        "location_share": r"(my location|find me at|i am at|come to)",
    }

    def check(self, text: str) -> dict:
        flags = []
        for ptype, pattern in self.PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(ptype)
        return {
            "dlp_flag": len(flags) > 0,
            "detected_types": flags,
            "severity": "high" if ("aadhaar" in flags or "pan" in flags) else "medium",
        }
