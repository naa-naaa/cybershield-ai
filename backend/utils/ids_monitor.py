from db.postgres import fetch_one, execute
from datetime import datetime, timedelta


class IDSMonitor:
    """Intrusion Detection — rate spike and bot behavior detection"""

    RATE_LIMIT = 20  # messages per 60 seconds
    WINDOW_SECONDS = 60

    async def check(self, username: str) -> dict:
        try:
            since = datetime.now() - timedelta(seconds=self.WINDOW_SECONDS)
            result = await fetch_one(
                "SELECT COUNT(*) as msg_count FROM messages WHERE sender=$1 AND timestamp > $2",
                username, since,
            )
            count = result["msg_count"] if result else 0
            rate_exceeded = count > self.RATE_LIMIT

            if rate_exceeded:
                await execute(
                    "INSERT INTO security_logs (log_type, username, detail, severity) VALUES ('ids', $1, $2, 'high')",
                    username,
                    f"Rate spike: {count} messages in {self.WINDOW_SECONDS}s",
                )
            return {"rate_exceeded": rate_exceeded, "message_count": count, "window_seconds": self.WINDOW_SECONDS}
        except Exception:
            return {"rate_exceeded": False, "message_count": 0}
