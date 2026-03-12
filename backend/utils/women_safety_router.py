from db.queries import create_women_escalation, create_notification, log_security_event


class WomenSafetyRouter:
    """
    Handles women-specific safety routing and auto-escalation.
    Runs after main analysis when women_risk_flag is True.
    """

    async def route(
        self,
        incident_id: str,
        harm_score: float,
        category: str,
        sender: str,
        receiver: str,
        receiver_gender: str = None,
        victim_age_group: str = None,
    ):
        actions_taken = []
        is_crisis = harm_score > 0.85
        is_high = harm_score > 0.70
        is_minor = victim_age_group == "minor"

        if is_high:
            await create_notification(
                target_user=receiver,
                ntype="women_safety",
                title="🛡️ CyberShield Protected You",
                body="A harmful message was blocked. You are safe. Help is available.",
            )
            actions_taken.append("victim_notified")

        if is_crisis:
            await create_women_escalation({
                "incident_id": incident_id,
                "victim_username": receiver,
                "escalation_type": "ncw",
                "authority_name": "National Commission for Women",
                "auto_triggered": True,
                "complaint_text": self._build_ncw_complaint(incident_id, sender, receiver, harm_score, category),
            })
            actions_taken.append("ncw_escalation_created")

        if is_minor and category in ["threat", "identity_hate", "harassment"]:
            await create_women_escalation({
                "incident_id": incident_id,
                "victim_username": receiver,
                "escalation_type": "childline",
                "authority_name": "CHILDLINE India — 1098",
                "auto_triggered": True,
                "complaint_text": f"POCSO-applicable incident. Minor victim. Case: {incident_id}",
            })
            actions_taken.append("pocso_escalation_created")

        await log_security_event(
            log_type="women_safety",
            username=receiver,
            detail=f"Women safety routing triggered. Actions: {', '.join(actions_taken)}",
            severity="high" if is_crisis else "medium",
        )
        return {
            "women_safety_routing": True,
            "actions_taken": actions_taken,
            "crisis_level": is_crisis,
            "pocso_applicable": is_minor,
        }

    def _build_ncw_complaint(self, case_id, sender, receiver, score, category) -> str:
        return f"""NATIONAL COMMISSION FOR WOMEN — CYBER HARASSMENT COMPLAINT
Case ID: {case_id}
COMPLAINT DETAILS:
The victim (ID: {receiver}) has been subjected to online {category}
by user (ID: {sender}) with severity score {score:.0%}.
Auto-flagged by CyberShield AI. Evidence package available.
Submit at: ncw.nic.in | Helpline: 7827170170""".strip()
