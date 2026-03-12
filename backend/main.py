import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from orchestrators.text_orchestrator import TextOrchestrator
from db.postgres import get_pool, close_pool, execute
from db.queries import (
    insert_incident, get_incidents, get_all_users,
    get_user, get_strike_count, increment_strike,
    update_user_risk_label, create_evidence_package,
    create_notification, get_notifications, log_security_event,
    get_incidents_by_user,
)
from utils.evidence_pdf import generate_evidence_pdf
from utils.women_safety_router import WomenSafetyRouter
from utils.dlp_scanner import DLPScanner
from utils.ids_monitor import IDSMonitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("CyberShield starting up...")
    await get_pool()
    print("Database pool ready")
    yield
    await close_pool()
    print("CyberShield shut down")


app = FastAPI(
    title="CyberShield AI",
    description="Real-Time Cyberbullying Detection and Remediation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_orchestrator = TextOrchestrator()
women_router = WomenSafetyRouter()
dlp_scanner = DLPScanner()
ids_monitor = IDSMonitor()


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────
class TextAnalyzeRequest(BaseModel):
    text: str
    sender: str
    receiver: str
    thread_id: Optional[str] = "default"
    timestamp: Optional[str] = None
    thread_history: Optional[List[str]] = []
    platform: Optional[str] = "SafeChat"
    sender_gender: Optional[str] = None
    receiver_gender: Optional[str] = None
    receiver_age_group: Optional[str] = "adult"


class AnalyzeResponse(BaseModel):
    action: str
    harm_score: float
    category: str
    severity: str
    explanation: str
    strike_count: int
    agent_scores: dict
    women_risk_flag: bool
    victim_distress_flag: bool
    incident_id: str


class AdminActionRequest(BaseModel):
    admin_username: str
    target_user: str
    action_type: str
    reason: Optional[str] = ""
    incident_id: Optional[str] = None


# ── HEALTH ────────────────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "CyberShield online", "version": "1.0.0", "timestamp": datetime.now().isoformat()}


# ── MAIN WEBHOOK — TEXT ───────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: TextAnalyzeRequest):
    """Main webhook endpoint — called by chat app on every message send."""
    dlp_result = dlp_scanner.check(request.text)
    ids_result = await ids_monitor.check(request.sender)
    strike_count = await get_strike_count(request.sender)

    result = await text_orchestrator.analyze(
        text=request.text,
        thread_history=request.thread_history,
        sender=request.sender,
        strike_count=strike_count,
    )

    if ids_result.get("rate_exceeded"):
        result["harm_score"] = max(result["harm_score"], 0.85)
        result["action"] = "block"
        result["explanation"] += " | Automated rate spike detected"

    action = result["action"]

    incident_data = {
        "sender": request.sender,
        "receiver": request.receiver,
        "content": request.text,
        "medium": "text",
        "action": action,
        "harm_score": result["harm_score"],
        "category": result["category"],
        "severity": result["severity"],
        "agent_t1": result["agent_scores"]["toxicity"],
        "agent_t2": result["agent_scores"]["context"],
        "agent_t3": result["agent_scores"]["emotion"],
        "explanation": result["explanation"],
        "platform": request.platform,
        "women_risk_flag": result.get("women_risk_flag", False),
    }

    incident_id = await insert_incident(incident_data)

    if action in ["alert", "block"]:
        strike_count = await increment_strike(request.sender, incident_id)

    await update_user_risk_label(request.sender)

    if result.get("women_risk_flag") and result["harm_score"] > 0.65:
        await women_router.route(
            incident_id=incident_id,
            harm_score=result["harm_score"],
            category=result["category"],
            sender=request.sender,
            receiver=request.receiver,
            receiver_gender=request.receiver_gender,
            victim_age_group=request.receiver_age_group,
        )

    if action in ["alert", "block"]:
        await create_notification(
            target_user=request.receiver,
            ntype=action,
            title="🛡️ Message Protected" if action == "block" else "⚠️ Message Flagged",
            body=f"CyberShield {action}ed a message from {request.sender}",
        )

    return AnalyzeResponse(
        action=action,
        harm_score=round(result["harm_score"], 3),
        category=result["category"],
        severity=result["severity"],
        explanation=result["explanation"],
        strike_count=strike_count,
        agent_scores=result["agent_scores"],
        women_risk_flag=result.get("women_risk_flag", False),
        victim_distress_flag=result.get("victim_distress_flag", False),
        incident_id=incident_id,
    )


# ── VOICE WEBHOOK ─────────────────────────────────────────────────
@app.post("/analyze/voice")
async def analyze_voice(
    audio: UploadFile = File(...),
    sender: str = Form(...),
    receiver: str = Form(...),
    platform: str = Form("SafeChat"),
):
    from agents.voice.agent_v1_stt import AgentV1STT
    from agents.voice.agent_v2_acoustic import AgentV2Acoustic

    audio_bytes = await audio.read()
    v1 = AgentV1STT()
    v2 = AgentV2Acoustic()

    stt_result, acoustic_result = await asyncio.gather(v1.analyze(audio_bytes), v2.analyze(audio_bytes))

    text_result = {"harm_score": 0.0, "category": "safe", "severity": "none",
                   "action": "allow", "agent_scores": {}, "explanation": ""}
    if stt_result.get("transcript"):
        text_result = await text_orchestrator.analyze(text=stt_result["transcript"], sender=sender)

    fused_score = 0.50 * text_result["harm_score"] + 0.50 * acoustic_result["score"]
    action = "allow" if fused_score < 0.40 else "alert" if fused_score < 0.80 else "block"

    incident_data = {
        "sender": sender, "receiver": receiver,
        "content": stt_result.get("transcript", "[voice message]"),
        "medium": "voice", "action": action, "harm_score": fused_score,
        "category": text_result.get("category", "unknown"),
        "severity": text_result.get("severity", "none"),
        "agent_t1": acoustic_result["score"], "agent_t2": 0.0, "agent_t3": text_result["harm_score"],
        "explanation": f"Voice: {acoustic_result.get('explanation','')} | Text: {text_result.get('explanation','')}",
        "platform": platform, "women_risk_flag": text_result.get("women_risk_flag", False),
    }
    incident_id = await insert_incident(incident_data)
    return {"action": action, "harm_score": round(fused_score, 3),
            "transcript": stt_result.get("transcript", ""), "incident_id": incident_id}


# ── IMAGE WEBHOOK ─────────────────────────────────────────────────
@app.post("/analyze/image")
async def analyze_image(
    image: UploadFile = File(...),
    sender: str = Form(...),
    receiver: str = Form(...),
    platform: str = Form("SafeChat"),
):
    from agents.image.agent_i1_ocr import AgentI1OCR
    from agents.image.agent_i3_nsfw import AgentI3NSFW

    image_bytes = await image.read()
    i1 = AgentI1OCR()
    i3 = AgentI3NSFW()

    ocr_result, nsfw_result = await asyncio.gather(i1.analyze(image_bytes), i3.analyze(image_bytes))

    text_result = {"harm_score": 0.0, "category": "safe", "severity": "none",
                   "action": "allow", "agent_scores": {}, "explanation": ""}
    if ocr_result.get("has_text"):
        text_result = await text_orchestrator.analyze(text=ocr_result["extracted_text"], sender=sender)

    fused_score = max(text_result["harm_score"], nsfw_result["score"])
    women_risk = nsfw_result.get("women_risk_flag", False) or text_result.get("women_risk_flag", False)
    action = "allow" if fused_score < 0.40 else "alert" if fused_score < 0.80 else "block"

    incident_data = {
        "sender": sender, "receiver": receiver,
        "content": ocr_result.get("extracted_text", "[image]"),
        "medium": "image", "action": action, "harm_score": fused_score,
        "category": "explicit" if nsfw_result["is_explicit"] else text_result.get("category", "safe"),
        "severity": text_result.get("severity", "none"),
        "agent_t1": nsfw_result["score"], "agent_t2": 0.0, "agent_t3": text_result["harm_score"],
        "explanation": f"NSFW:{nsfw_result['is_explicit']} | OCR:{ocr_result.get('has_text',False)}",
        "platform": platform, "women_risk_flag": women_risk,
    }
    incident_id = await insert_incident(incident_data)
    return {"action": action, "harm_score": round(fused_score, 3),
            "extracted_text": ocr_result.get("extracted_text", ""),
            "is_explicit": nsfw_result["is_explicit"], "incident_id": incident_id}


# ── ADMIN ENDPOINTS ───────────────────────────────────────────────
@app.get("/admin/incidents")
async def get_all_incidents(limit: int = 50, offset: int = 0):
    incidents = await get_incidents(limit, offset)
    return {"incidents": [dict(i) for i in incidents]}


@app.get("/admin/users")
async def get_users():
    users = await get_all_users()
    return {"users": [dict(u) for u in users]}


@app.get("/admin/user/{username}")
async def get_user_profile(username: str):
    user = await get_user(username)
    incidents = await get_incidents_by_user(username)
    return {"user": dict(user) if user else None, "incidents": [dict(i) for i in incidents]}


@app.post("/admin/action")
async def admin_action(request: AdminActionRequest):
    if request.action_type == "block":
        await execute("UPDATE users SET status='blocked' WHERE username=$1", request.target_user)
    elif request.action_type == "unblock":
        await execute("UPDATE users SET status='active', strike_count=0 WHERE username=$1", request.target_user)
    elif request.action_type == "send_resources":
        await create_notification(
            target_user=request.target_user, ntype="resources",
            title="💙 Safety Resources",
            body="iCall: 9152987821 | NCW: 7827170170 | Cyber Crime: 1930 | Women: 181",
        )
    await log_security_event(
        log_type="admin_action", username=request.target_user,
        detail=f"Admin {request.admin_username}: {request.action_type} — {request.reason}",
        severity="medium",
    )
    return {"success": True, "action": request.action_type}


@app.get("/admin/helplines")
async def get_helplines(category: Optional[str] = None):
    from db.postgres import fetch_all
    if category:
        rows = await fetch_all("SELECT * FROM helpline_contacts WHERE category=$1", category)
    else:
        rows = await fetch_all("SELECT * FROM helpline_contacts ORDER BY category")
    return {"helplines": [dict(r) for r in rows]}


# ── EVIDENCE ──────────────────────────────────────────────────────
@app.post("/evidence/generate")
async def generate_evidence(victim: str, predator: str, pocso: bool = False):
    case_id = f"CS-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    incidents = await get_incidents_by_user(victim)
    incident_list = [dict(i) for i in incidents if dict(i)["action"] in ["alert", "block"]]

    pdf_path = f"./evidence/{case_id}.pdf"
    generate_evidence_pdf(
        case_id=case_id, victim=victim, predator=predator,
        incidents=incident_list, output_path=pdf_path, pocso_applicable=pocso,
    )

    incident_ids = [i["id"] for i in incident_list]
    await create_evidence_package(
        victim=victim, predator=predator, incident_ids=incident_ids,
        pdf_path=pdf_path, case_id=case_id, pocso=pocso,
    )

    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=f"CyberShield_Evidence_{case_id}.pdf")


# ── NOTIFICATIONS ─────────────────────────────────────────────────
@app.get("/notifications/{username}")
async def get_user_notifications(username: str):
    notifications = await get_notifications(username)
    return {"notifications": [dict(n) for n in notifications]}


# ── METRICS ───────────────────────────────────────────────────────
@app.get("/metrics")
async def get_metrics():
    from db.postgres import fetch_one
    today = datetime.now().date()
    stats = await fetch_one(
        """
        SELECT COUNT(*) as total,
            COUNT(CASE WHEN action='block' THEN 1 END) as blocked,
            COUNT(CASE WHEN action='alert' THEN 1 END) as alerted,
            COUNT(CASE WHEN women_risk_flag=TRUE THEN 1 END) as women_flagged,
            AVG(harm_score) as avg_score
        FROM incidents WHERE DATE(timestamp) = $1
        """,
        today,
    )
    return dict(stats) if stats else {}


# ── WEBHOOK TEST ──────────────────────────────────────────────────
@app.post("/webhook/test")
async def test_webhook():
    """Test endpoint to verify backend is reachable from chat app"""
    return {"status": "ok", "message": "CyberShield webhook is reachable", "timestamp": datetime.now().isoformat()}
