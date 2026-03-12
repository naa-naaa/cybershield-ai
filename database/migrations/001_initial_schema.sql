-- ═══════════════════════════════════════════════════════════════
-- CYBERSHIELD AI — COMPLETE POSTGRESQL SCHEMA
-- ═══════════════════════════════════════════════════════════════
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── USERS ──────────────────────────────────────────────────────
CREATE TABLE users (
    id               SERIAL PRIMARY KEY,
    username         VARCHAR(50) UNIQUE NOT NULL,
    password_hash    VARCHAR(255),
    gender           VARCHAR(20),
    age_group        VARCHAR(20),
    strike_count     INT DEFAULT 0,
    status           VARCHAR(20) DEFAULT 'active',
    risk_label       VARCHAR(20) DEFAULT 'safe',
    protection_level VARCHAR(20) DEFAULT 'normal',
    kavach_score     INT DEFAULT 100,
    suraksha_mode    BOOLEAN DEFAULT FALSE,
    trusted_contact  VARCHAR(100),
    platform         VARCHAR(50) DEFAULT 'SafeChat',
    webhook_enabled  BOOLEAN DEFAULT FALSE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    last_active      TIMESTAMPTZ DEFAULT NOW()
);

-- ── MESSAGES ───────────────────────────────────────────────────
CREATE TABLE messages (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    sender           VARCHAR(50) NOT NULL,
    receiver         VARCHAR(50) NOT NULL,
    content          TEXT,
    medium           VARCHAR(20) DEFAULT 'text',
    status           VARCHAR(20) DEFAULT 'sent',
    harm_score       FLOAT DEFAULT 0,
    category         VARCHAR(50),
    conversation_id  VARCHAR(100),
    platform         VARCHAR(50),
    timestamp        TIMESTAMPTZ DEFAULT NOW()
);

-- ── INCIDENTS ──────────────────────────────────────────────────
CREATE TABLE incidents (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    sender           VARCHAR(50) NOT NULL,
    receiver         VARCHAR(50) NOT NULL,
    content          TEXT,
    medium           VARCHAR(20) DEFAULT 'text',
    action           VARCHAR(20) NOT NULL,
    harm_score       FLOAT,
    category         VARCHAR(50),
    severity         VARCHAR(20),
    agent_t1         FLOAT,
    agent_t2         FLOAT,
    agent_t3         FLOAT,
    explanation      TEXT,
    ip_address       VARCHAR(50),
    ip_risk          VARCHAR(20),
    platform         VARCHAR(50),
    women_risk_flag  BOOLEAN DEFAULT FALSE,
    escalated_to     VARCHAR(50),
    timestamp        TIMESTAMPTZ DEFAULT NOW()
);

-- ── STRIKES ────────────────────────────────────────────────────
CREATE TABLE strikes (
    id               SERIAL PRIMARY KEY,
    username         VARCHAR(50) NOT NULL,
    incident_id      UUID REFERENCES incidents(id),
    reason           TEXT,
    timestamp        TIMESTAMPTZ DEFAULT NOW()
);

-- ── EVIDENCE PACKAGES ──────────────────────────────────────────
CREATE TABLE evidence_packages (
    id                      UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    case_id                 VARCHAR(50) UNIQUE,
    victim                  VARCHAR(50) NOT NULL,
    predator                VARCHAR(50),
    incident_ids            UUID[],
    pdf_path                TEXT,
    it_act_sections         TEXT[],
    ipc_sections            TEXT[],
    pocso_applicable        BOOLEAN DEFAULT FALSE,
    submitted_cybercrime    BOOLEAN DEFAULT FALSE,
    submitted_ncw           BOOLEAN DEFAULT FALSE,
    case_status             VARCHAR(30) DEFAULT 'open',
    generated_at            TIMESTAMPTZ DEFAULT NOW()
);

-- ── REHABILITATION ─────────────────────────────────────────────
CREATE TABLE rehabilitation (
    id                  SERIAL PRIMARY KEY,
    username            VARCHAR(50) UNIQUE NOT NULL,
    stage               INT DEFAULT 1,
    stage_1_complete    BOOLEAN DEFAULT FALSE,
    stage_2_complete    BOOLEAN DEFAULT FALSE,
    stage_3_complete    BOOLEAN DEFAULT FALSE,
    reinstated          BOOLEAN DEFAULT FALSE,
    quiz_score          INT,
    started_at          TIMESTAMPTZ DEFAULT NOW(),
    completed_at        TIMESTAMPTZ
);

-- ── ADMIN ACTIONS ──────────────────────────────────────────────
CREATE TABLE admin_actions (
    id              SERIAL PRIMARY KEY,
    admin_username  VARCHAR(50),
    target_user     VARCHAR(50),
    action_type     VARCHAR(50),
    reason          TEXT,
    incident_id     UUID REFERENCES incidents(id),
    timestamp       TIMESTAMPTZ DEFAULT NOW()
);

-- ── WOMEN SAFETY ESCALATIONS ───────────────────────────────────
CREATE TABLE women_safety_escalations (
    id                  SERIAL PRIMARY KEY,
    incident_id         UUID REFERENCES incidents(id),
    victim_username     VARCHAR(50),
    escalation_type     VARCHAR(50),
    authority_name      VARCHAR(100),
    auto_triggered      BOOLEAN DEFAULT FALSE,
    admin_triggered     BOOLEAN DEFAULT FALSE,
    complaint_text      TEXT,
    reference_number    VARCHAR(50),
    status              VARCHAR(30) DEFAULT 'pending',
    timestamp           TIMESTAMPTZ DEFAULT NOW()
);

-- ── HELPLINE CONTACTS ──────────────────────────────────────────
CREATE TABLE helpline_contacts (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    number          VARCHAR(30),
    url             TEXT,
    category        VARCHAR(50),
    available_24x7  BOOLEAN DEFAULT TRUE,
    language        VARCHAR(50) DEFAULT 'Hindi, English',
    description     TEXT
);

-- ── NOTIFICATIONS ──────────────────────────────────────────────
CREATE TABLE notifications (
    id              SERIAL PRIMARY KEY,
    target_user     VARCHAR(50),
    type            VARCHAR(50),
    title           TEXT,
    body            TEXT,
    read            BOOLEAN DEFAULT FALSE,
    timestamp       TIMESTAMPTZ DEFAULT NOW()
);

-- ── SECURITY LOGS ──────────────────────────────────────────────
CREATE TABLE security_logs (
    id              SERIAL PRIMARY KEY,
    log_type        VARCHAR(20),
    username        VARCHAR(50),
    detail          TEXT,
    severity        VARCHAR(20),
    timestamp       TIMESTAMPTZ DEFAULT NOW()
);

-- ── INDEXES ────────────────────────────────────────────────────
CREATE INDEX idx_incidents_sender    ON incidents(sender);
CREATE INDEX idx_incidents_receiver  ON incidents(receiver);
CREATE INDEX idx_incidents_timestamp ON incidents(timestamp DESC);
CREATE INDEX idx_incidents_action    ON incidents(action);
CREATE INDEX idx_incidents_women     ON incidents(women_risk_flag);
CREATE INDEX idx_messages_conv       ON messages(conversation_id);

-- ── POSTGRES NOTIFY TRIGGERS ───────────────────────────────────
CREATE OR REPLACE FUNCTION notify_new_incident()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('new_incident', row_to_json(NEW)::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER incident_notify_trigger
AFTER INSERT ON incidents
FOR EACH ROW EXECUTE FUNCTION notify_new_incident();

CREATE OR REPLACE FUNCTION notify_women_safety()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.women_risk_flag = TRUE AND NEW.harm_score > 0.80 THEN
        PERFORM pg_notify('women_safety_alert', row_to_json(NEW)::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER women_safety_trigger
AFTER INSERT ON incidents
FOR EACH ROW EXECUTE FUNCTION notify_women_safety();
