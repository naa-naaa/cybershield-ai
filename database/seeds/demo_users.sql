-- ── DEMO USERS ────────────────────────────────────────────────
INSERT INTO users (username) VALUES ('user1'), ('user2'), ('user3');

-- ── HELPLINE SEED DATA ────────────────────────────────────────
INSERT INTO helpline_contacts (name, number, url, category, description) VALUES
('Cyber Crime Helpline', '1930', 'cybercrime.gov.in', 'cyber', 'National cyber crime reporting'),
('NCW Helpline (Women)', '7827170170', 'ncw.nic.in', 'women', 'National Commission for Women'),
('Women Helpline', '181', NULL, 'women', 'Women in distress - all states'),
('CHILDLINE', '1098', 'childlineindia.org', 'child', 'Children in distress 24/7'),
('iCall TISS', '9152987821', 'icallhelpline.org', 'mental_health', 'Free counselling'),
('Vandrevala Foundation', '1860-2662-345', NULL, 'mental_health', '24/7 mental health support'),
('NIMHANS', '080-46110007', 'nimhansdigitalacademy.in', 'mental_health', 'Mental health institute'),
('SNEHI', '044-24640050', NULL, 'mental_health', 'Emotional support'),
('Police Emergency', '100', NULL, 'emergency', 'Police emergency'),
('Women SOS', '1091', NULL, 'women', 'Women safety emergency'),
('NALSA Legal Aid', '15100', 'nalsa.gov.in', 'legal', 'Free legal aid'),
('NCPCR Child Rights', '1800-121-2830', 'ncpcr.gov.in', 'child', 'Child rights commission'),
('One Stop Centre (Sakhi)', '181', 'wcd.nic.in', 'women', 'Integrated support for women');
