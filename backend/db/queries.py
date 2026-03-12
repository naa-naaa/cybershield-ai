from db.postgres import fetch_one, fetch_all, execute, notify_channel
from datetime import datetime


# ── INCIDENTS ─────────────────────────────────────────────────────
async def insert_incident(data: dict) -> str:
    result = await fetch_one(
        """
        INSERT INTO incidents
            (sender, receiver, content, medium, action, harm_score,
             category, severity, agent_t1, agent_t2, agent_t3,
             explanation, ip_address, ip_risk, platform, women_risk_flag)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
        RETURNING id::text
        """,
        data["sender"], data["receiver"], data["content"],
        data.get("medium", "text"), data["action"],
        data["harm_score"], data["category"], data["severity"],
        data.get("agent_t1", 0.0), data.get("agent_t2", 0.0),
        data.get("agent_t3", 0.0), data.get("explanation", ""),
        data.get("ip_address", ""), data.get("ip_risk", "unknown"),
        data.get("platform", "SafeChat"),
        data.get("women_risk_flag", False)
    )
    incident_id = result["id"]

    # Notify admin dashboard via pg NOTIFY
    await notify_channel("new_incident", {**data, "id": incident_id})

    # Women safety alert — separate channel
    if data.get("women_risk_flag") and data["harm_score"] > 0.80:
        await notify_channel("women_safety_alert", {**data, "id": incident_id})

    return incident_id


async def get_incidents(limit: int = 50, offset: int = 0):
    return await fetch_all(
        "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT $1 OFFSET $2",
        limit, offset
    )


async def get_incidents_by_user(username: str):
    return await fetch_all(
        "SELECT * FROM incidents WHERE sender=$1 OR receiver=$1 ORDER BY timestamp DESC",
        username
    )


# ── USERS ─────────────────────────────────────────────────────────
async def get_user(username: str):
    return await fetch_one("SELECT * FROM users WHERE username = $1", username)


async def get_all_users():
    return await fetch_all(
        """
        SELECT u.*,
            COUNT(CASE WHEN i.sender = u.username THEN 1 END) as sent_flags,
            COUNT(CASE WHEN i.receiver = u.username THEN 1 END) as received_flags
        FROM users u
        LEFT JOIN incidents i ON (i.sender = u.username OR i.receiver = u.username)
            AND i.action IN ('alert', 'block')
        GROUP BY u.id
        ORDER BY u.strike_count DESC
        """
    )


async def update_user_risk_label(username: str):
    sent = await fetch_one(
        "SELECT COUNT(*) as c FROM incidents WHERE sender=$1 AND action IN ('alert','block')",
        username
    )
    received = await fetch_one(
        "SELECT COUNT(*) as c FROM incidents WHERE receiver=$1 AND action IN ('alert','block')",
        username
    )
    sent_count = sent["c"]
    received_count = received["c"]

    if sent_count >= 5:
        label = "predator"
    elif received_count >= 5:
        label = "victim"
    elif sent_count >= 2:
        label = "warning"
    else:
        label = "safe"

    await execute("UPDATE users SET risk_label=$1 WHERE username=$2", label, username)
    return label


# ── STRIKES ───────────────────────────────────────────────────────
async def get_strike_count(username: str) -> int:
    result = await fetch_one(
        "SELECT strike_count FROM users WHERE username=$1", username
    )
    return result["strike_count"] if result else 0


async def increment_strike(username: str, incident_id: str) -> int:
    result = await fetch_one(
        """
        UPDATE users SET strike_count = strike_count + 1
        WHERE username = $1 RETURNING strike_count
        """,
        username
    )
    new_count = result["strike_count"]
    await execute(
        "INSERT INTO strikes (username, incident_id, reason) VALUES ($1, $2, $3)",
        username, incident_id, "automated_flag"
    )
    if new_count >= 3:
        await execute("UPDATE users SET status='blocked' WHERE username=$1", username)
    return new_count


# ── EVIDENCE ──────────────────────────────────────────────────────
async def create_evidence_package(
    victim: str, predator: str, incident_ids: list,
    pdf_path: str, case_id: str, pocso: bool = False
):
    await execute(
        """
        INSERT INTO evidence_packages
            (case_id, victim, predator, incident_ids, pdf_path,
             it_act_sections, pocso_applicable)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        """,
        case_id, victim, predator, incident_ids, pdf_path,
        ["Section 66A", "Section 67", "Section 67A"], pocso
    )


# ── WOMEN SAFETY ──────────────────────────────────────────────────
async def create_women_escalation(data: dict):
    await execute(
        """
        INSERT INTO women_safety_escalations
            (incident_id, victim_username, escalation_type,
             authority_name, auto_triggered, complaint_text)
        VALUES ($1,$2,$3,$4,$5,$6)
        """,
        data["incident_id"], data["victim_username"],
        data["escalation_type"], data["authority_name"],
        data.get("auto_triggered", False), data.get("complaint_text", "")
    )


# ── NOTIFICATIONS ─────────────────────────────────────────────────
async def create_notification(target_user: str, ntype: str, title: str, body: str):
    await execute(
        "INSERT INTO notifications (target_user, type, title, body) VALUES ($1,$2,$3,$4)",
        target_user, ntype, title, body
    )


async def get_notifications(username: str):
    return await fetch_all(
        "SELECT * FROM notifications WHERE target_user=$1 ORDER BY timestamp DESC LIMIT 20",
        username
    )


# ── SECURITY LOGS ─────────────────────────────────────────────────
async def log_security_event(log_type: str, username: str, detail: str, severity: str):
    await execute(
        "INSERT INTO security_logs (log_type, username, detail, severity) VALUES ($1,$2,$3,$4)",
        log_type, username, detail, severity
    )
