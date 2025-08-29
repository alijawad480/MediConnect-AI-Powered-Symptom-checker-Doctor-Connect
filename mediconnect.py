import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from urllib.parse import quote_plus
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
import textwrap
import base64
import json
import os
import math
import random
from typing import List, Tuple, Optional, Dict, Any

REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(
    page_title="MediConnect — Symptom Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* App background gradient */
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #eef7ff 100%);
        color: #0f172a;
    }

    /* App header */
    .app-header {
        background: linear-gradient(90deg,#06b6d4,#3b82f6);
        color: white;
        padding: 22px;
        border-radius: 14px;
        box-shadow: 0 12px 30px rgba(2,6,23,0.08);
        text-align: center;
        font-weight: 800;
        margin-bottom: 18px;
    }

    /* Card style */
    .card {
        background: linear-gradient(180deg, #ffffff, #fbfdff);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(2,6,23,0.05);
        margin-bottom: 18px;
    }

    /* Sidebar gradient and text */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #0f172a);
        color: white;
        padding-top: 18px;
    }
    [data-testid="stSidebar"] .css-1d391kg { color: white !important; }
    [data-testid="stSidebar"] .stText { color: white !important; }

    /* Sidebar header */
    .sidebar-title {
        font-weight: 900; font-size: 1.2rem; color: #fee08b;
        margin-bottom: 8px;
    }

    /* Buttons */
    .btn {
        display:inline-block;
        padding: 10px 16px;
        border-radius: 12px;
        font-weight:800;
        text-decoration:none;
        margin: 4px 0px;
    }
    .btn-green { background: linear-gradient(90deg,#10b981,#06b6d4); color:white; }
    .btn-red { background: linear-gradient(90deg,#ef4444,#f97316); color:white; }
    .btn-yellow { background: linear-gradient(90deg,#f59e0b,#f97316); color:white; }
    .btn-blue { background: linear-gradient(90deg,#3b82f6,#06b6d4); color:white; }

    /* severity badges */
    .severity-badge { padding:12px; border-radius:12px; font-weight:900; display:flex; gap:12px; align-items:center; margin-bottom:12px; }
    .sev-em { background: linear-gradient(90deg,#fee2e2,#fecaca); border-left:8px solid #dc2626; color:#7f1d1d; }
    .sev-mod { background: linear-gradient(90deg,#fffbeb,#fef3c7); border-left:8px solid #f59e0b; color:#92400e; }
    .sev-mild { background: linear-gradient(90deg,#ecfeff,#cffafe); border-left:8px solid #06b6d4; color:#075985; }

    /* doctor card */
    .doctor-card { background: linear-gradient(180deg,#fff,#f8fafc); padding:14px; border-radius:12px; box-shadow: 0 6px 24px rgba(2,6,23,0.04); margin-bottom:10px; }

    /* small muted */
    .muted { color: #475569; font-size:0.95rem; }

    /* responsive */
    @media (max-width: 720px) {
        .app-header { font-size: 1.1rem; padding: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

#pretty json display for debugging

def pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

# Training data creation
@st.cache_resource(show_spinner=False)
def create_training_data() -> Tuple[List[str], List[str]]:
    """
    Create a basic labeled symptom -> condition dataset.
    This is intentionally small and is meant for demo / prototype use.
    Replace with a production dataset for real-world deployments.
    """
    # Basic symptom sentences covering many conditions
    texts = [
        # ENT / respiratory
        "runny nose sneezing sore throat mild cough",
        "stuffy nose sore throat sneezing and congestion",
        "nasal congestion runny nose itchy eyes",
        "itchy eyes sneezing nasal congestion and watery eyes",
        "sore throat, fever and cough for two days",
        "persistent cough with phlegm and chest congestion",
        # Influenza / viral
        "high fever body aches severe tiredness cough",
        "fever chills muscle pain cough fatigue",
        "fever dry cough loss of smell difficulty breathing",
        # COVID-like
        "loss of taste fever cough trouble breathing",
        "fever cough loss of smell and extreme tiredness",
        # Migraine / neurological
        "severe one-sided headache nausea sensitivity to light",
        "throbbing headache nausea visual aura",
        "headache dizziness blurred vision and pounding",
        # Gastroenteritis
        "diarrhea vomiting stomach pain abdominal cramps",
        "nausea vomiting watery stools stomach ache",
        # Hypertension / general cardiac
        "headache dizziness blurred vision high blood pressure readings",
        "dizziness headache pounding in head associated with high bp",
        # Diabetes symptoms
        "increased thirst frequent urination fatigue blurred vision",
        "extreme thirst weight loss frequent urination and fatigue",
        # UTI
        "burning urination frequent urination lower abdominal pain",
        "painful urination cloudy urine urgent need to urinate",
        # Bronchitis/respiratory
        "persistent cough phlegm chest discomfort wheeze",
        "productive cough chest congestion shortness of breath fever",
        # Heart attack / chest emergencies
        "severe chest pain pressure in chest sweating nausea shortness of breath",
        "sharp chest pain radiating to left arm jaw nausea fainting",
        # Stroke-like
        "sudden weakness on one side slurred speech facial droop sudden confusion",
        # misc short samples for robustness
        "mild sore throat and runny nose",
        "fever and body aches with cough",
        "shortness of breath and persistent cough",
        "sharp abdominal cramps with vomiting",
        "itchy rash on arms after eating nuts",
        "red itchy eyes with watery discharge",
        "severe abdominal pain with guarding and vomiting",
        "chest tightness during exercise and short breath"
    ]

    labels = [
        "Common Cold","Common Cold","Allergy","Allergy","Influenza","Bronchitis",
        "Influenza","Influenza","COVID-19","COVID-19","Migraine","Migraine","Hypertension",
        "Hypertension","Diabetes","Diabetes","UTI","UTI","Bronchitis","Bronchitis",
        "Heart Attack","Heart Attack","Stroke","Common Cold","Influenza","Bronchitis",
        "Gastroenteritis","Allergy","Allergy","Gastroenteritis","Heart Attack"
    ]

    # Ensure equal length
    if len(texts) != len(labels):
        # Truncate if mismatched
        min_len = min(len(texts), len(labels))
        texts = texts[:min_len]
        labels = labels[:min_len]

    return texts, labels

# Train baseline model (TF-IDF + RandomForest)

@st.cache_resource(show_spinner=False)
def train_baseline_model() -> Any:
    """
    Train a TF-IDF + RandomForest pipeline.
    For demo use — training is fast due to small dataset.
    """
    texts, labels = create_training_data()
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), max_features=4000),
        RandomForestClassifier(n_estimators=200, random_state=42)
    )
    pipeline.fit(texts, labels)
    return pipeline


# Instantiate model once
model = train_baseline_model()

# Mappings: specialties, severity, care advice

EMERGENCY_KEYWORDS = [
    "chest pain", "severe chest pain", "difficulty breathing", "shortness of breath",
    "unconscious", "not breathing", "no pulse", "severe bleeding", "heavy bleeding",
    "faint", "collapse", "sudden weakness", "sudden numbness", "sudden confusion",
    "slurred speech", "facial droop", "loss of consciousness", "passed out", "not responsive"
]

DISEASE_TO_SPECIALTY = {
    "Common Cold": "ENT",
    "Allergy": "ENT",
    "Influenza": "General Physician",
    "COVID-19": "Infectious Disease",
    "Migraine": "Neurologist",
    "Gastroenteritis": "Gastroenterologist",
    "Hypertension": "Cardiologist",
    "Diabetes": "Endocrinologist",
    "UTI": "Urologist",
    "Bronchitis": "Pulmonologist",
    "Heart Attack": "Cardiologist",
    "Stroke": "Neurologist",
}

SEVERITY_MAP = {
    "Heart Attack": "Emergency",
    "Stroke": "Emergency",
    "COVID-19": "Moderate",
    "Influenza": "Moderate",
    "Bronchitis": "Moderate",
    "Gastroenteritis": "Moderate",
    "Hypertension": "Moderate",
    "Diabetes": "Moderate",
    "UTI": "Mild",
    "Allergy": "Mild",
    "Common Cold": "Mild",
    "Migraine": "Mild",
}


CARE_ADVICE = {
    "Common Cold": ("Rest, hydration, and over-the-counter cold remedies.", ["Paracetamol", "Antihistamines"]),
    "Allergy": ("Avoid allergens, use antihistamines and nasal saline.", ["Loratadine", "Cetirizine"]),
    "Influenza": ("Rest, fluids, and antiviral medications if prescribed by a doctor.", ["Paracetamol", "Oseltamivir"]),
    "COVID-19": ("Isolate, monitor symptoms, test if advised, seek care if breathing worsens.", ["Paracetamol", "Vitamin C"]),
    "Migraine": ("Rest in a dark room, hydration, avoid triggers, take prescribed pain-relief.", ["Ibuprofen", "Paracetamol"]),
    "Gastroenteritis": ("Hydration (ORS), bland diet, and rest. Seek care if dehydrated.", ["Oral Rehydration Salts", "Loperamide"]),
    "Hypertension": ("Lifestyle changes, regular monitoring, and prescribed antihypertensives.", ["Amlodipine", "Lisinopril"]),
    "Diabetes": ("Diet control, exercise, blood glucose monitoring, and medications.", ["Metformin", "Insulin (if prescribed)"]),
    "UTI": ("Hydration, medical evaluation, and antibiotics if bacterial infection.", ["Nitrofurantoin", "Trimethoprim"]),
    "Bronchitis": ("Rest, fluids, and cough relief as advised.", ["Dextromethorphan", "Paracetamol"]),
    "Heart Attack": ("Call emergency services immediately — urgent medical intervention needed.", []),
    "Stroke": ("Call emergency services immediately — time-sensitive treatment is critical.", []),
}

# Seeded doctors database (extensive list to be useful)

@st.cache_data(show_spinner=False)
def load_default_doctors() -> pd.DataFrame:
    """
    Returns a seeded doctors dataframe with many entries.
    Users can upload CSV to append more entries (admin in sidebar).
    """
    data = [
        {"name":"Dr. Ayesha Khan","specialty":"General Physician","hospital":"Faisalabad General Hospital","city":"Faisalabad","phone":"+92-300-1111111","email":"ayesha.khan@example.com","rating":4.6},
        {"name":"Dr. Yasir Malik","specialty":"General Physician","hospital":"Lahore Central Hospital","city":"Lahore","phone":"+92-300-1212121","email":"yasir.malik@example.com","rating":4.5},
        {"name":"Dr. Hamza Ahmed","specialty":"General Physician","hospital":"Faisalabad Clinic","city":"Faisalabad","phone":"+92-300-4444444","email":"hamza.ahmed@example.com","rating":4.3},
        {"name":"Dr. Bilal Ahmed","specialty":"Pulmonologist","hospital":"City Pulmo Care","city":"Lahore","phone":"+92-300-2222222","email":"bilal.ahmed@example.com","rating":4.7},
        {"name":"Dr. Fatima Shah","specialty":"Pulmonologist","hospital":"Lung Care","city":"Multan","phone":"+92-300-5555555","email":"fatima.shah@example.com","rating":4.4},
        {"name":"Dr. Sana Iqbal","specialty":"Infectious Disease","hospital":"Islamabad Infectious Center","city":"Islamabad","phone":"+92-300-3333333","email":"sana.iqbal@example.com","rating":4.8},
        {"name":"Dr. Ali Hassan","specialty":"Infectious Disease","hospital":"Karachi Infectious Care","city":"Karachi","phone":"+92-300-1010101","email":"ali.hassan@example.com","rating":4.5},
        {"name":"Dr. Omar Riaz","specialty":"Neurologist","hospital":"NeuroCare Clinic","city":"Faisalabad","phone":"+92-300-4444444","email":"omar.riaz@example.com","rating":4.5},
        {"name":"Dr. Ayesha Noor","specialty":"Neurologist","hospital":"Neuro Clinic","city":"Karachi","phone":"+92-300-3333333","email":"ayesha.noor@example.com","rating":4.6},
        {"name":"Dr. Mehwish Tariq","specialty":"Gastroenterologist","hospital":"Digestive Health","city":"Lahore","phone":"+92-300-5555555","email":"mehwish.tariq@example.com","rating":4.4},
        {"name":"Dr. Kamran Malik","specialty":"Gastroenterologist","hospital":"Gastro Care","city":"Islamabad","phone":"+92-300-1313131","email":"kamran.malik@example.com","rating":4.3},
        {"name":"Dr. Ahmed Zafar","specialty":"Cardiologist","hospital":"HeartCare Hospital","city":"Islamabad","phone":"+92-300-6666666","email":"ahmed.zafar@example.com","rating":4.9},
        {"name":"Dr. Sarah Khan","specialty":"Cardiologist","hospital":"Private Clinic","city":"Lahore","phone":"+92-300-1111112","email":"sarah.khan@example.com","rating":4.7},
        {"name":"Dr. Nadia Farooq","specialty":"Endocrinologist","hospital":"Diabetes Center","city":"Faisalabad","phone":"+92-300-7777777","email":"nadia.farooq@example.com","rating":4.6},
        {"name":"Dr. Imran Ali","specialty":"Endocrinologist","hospital":"Endocrine Care","city":"Karachi","phone":"+92-300-1414141","email":"imran.ali@example.com","rating":4.4},
        {"name":"Dr. Imran Qureshi","specialty":"Urologist","hospital":"Kidney & Urology Clinic","city":"Lahore","phone":"+92-300-8888888","email":"imran.qureshi@example.com","rating":4.3},
        {"name":"Dr. Sana Malik","specialty":"Urologist","hospital":"Uro Care","city":"Islamabad","phone":"+92-300-1515151","email":"sana.malik@example.com","rating":4.5},
        {"name":"Dr. Rabia Shah","specialty":"ENT","hospital":"Ear Nose Throat Center","city":"Faisalabad","phone":"+92-300-9999999","email":"rabia.shah@example.com","rating":4.2},
        {"name":"Dr. Aliya Khan","specialty":"ENT","hospital":"ENT Care","city":"Lahore","phone":"+92-300-1616161","email":"aliya.khan@example.com","rating":4.3},
        {"name":"Dr. Hina Malik","specialty":"Dermatologist","hospital":"Skin Health","city":"Karachi","phone":"+92-300-1717171","email":"hina.malik@example.com","rating":4.4},
        {"name":"Dr. Javed Iqbal","specialty":"Cardiologist","hospital":"Heart and Vessels","city":"Lahore","phone":"+92-300-1818181","email":"javed.iqbal@example.com","rating":4.6},
        {"name":"Dr. M. Arif","specialty":"Pulmonologist","hospital":"BreathWell Clinic","city":"Islamabad","phone":"+92-300-1919191","email":"arif@example.com","rating":4.5},
        {"name":"Dr. Saba Khan","specialty":"Pediatrician","hospital":"Children Care","city":"Karachi","phone":"+92-300-2020202","email":"saba.khan@example.com","rating":4.7},
        {"name":"Dr. Farah Aziz","specialty":"OB/GYN","hospital":"Women's Health","city":"Lahore","phone":"+92-300-2121212","email":"farah.aziz@example.com","rating":4.6},
        {"name":"Dr. Tahir Shah","specialty":"Orthopedist","hospital":"Bone & Joint","city":"Faisalabad","phone":"+92-300-2323232","email":"tahir.shah@example.com","rating":4.3},
        {"name":"Dr. Noor ul Amin","specialty":"Psychiatrist","hospital":"Mind Care","city":"Islamabad","phone":"+92-300-2424242","email":"noor.amin@example.com","rating":4.4},
        {"name":"Dr. Asma Riaz","specialty":"ENT","hospital":"ENT Specialists","city":"Multan","phone":"+92-300-2525252","email":"asma.riaz@example.com","rating":4.1},
        {"name":"Dr. Khalid Noor","specialty":"Urologist","hospital":"Urology Center","city":"Multan","phone":"+92-300-2626262","email":"khalid.noor@example.com","rating":4.2},
        {"name":"Dr. Zainab Mir","specialty":"Nephrologist","hospital":"Kidney Care","city":"Lahore","phone":"+92-300-2727272","email":"zainab.mir@example.com","rating":4.5},
        {"name":"Dr. Bilqis Zahid","specialty":"Dermatologist","hospital":"Skin Solutions","city":"Faisalabad","phone":"+92-300-2828282","email":"bilqis.zahid@example.com","rating":4.3},
    ]

    df = pd.DataFrame(data)
    return df


# initialize session state variables used across the app
if 'doctors_df' not in st.session_state:
    st.session_state['doctors_df'] = load_default_doctors()
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'patient_name' not in st.session_state:
    st.session_state['patient_name'] = "Patient"
if 'last_report' not in st.session_state:
    st.session_state['last_report'] = None


# Utility functions for model & predictions

def predict_top_k(text: str, k: int = 3) -> List[Tuple[str, float]]:
    """
    Predict top-k conditions for a given symptom text.
    Returns list of (condition, probability) pairs.
    If the model fails, returns empty list.
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        probs = model.predict_proba([text])[0]
        # classes order from pipeline
        classes = model.named_steps['randomforestclassifier'].classes_
        # sort probs descending
        idx = np.argsort(probs)[::-1][:k]
        return [(classes[i], float(probs[i])) for i in idx]
    except Exception as e:
        # model failure fallback
        return []


def detect_emergency(text: str) -> Tuple[bool, Optional[str]]:
    """
    Checks whether the text contains any emergency keywords.
    Returns tuple (is_emergency, matched_keyword_or_none).
    """
    s = (text or "").lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in s:
            return True, kw
    return False, None


def estimate_severity(text: str, top_conditions: List[str]) -> Tuple[str, str]:
    """
    Estimate severity using red-flag detection and severity mapping.
    Returns severity label and a short note.
    """
    is_em, kw = detect_emergency(text)
    if is_em:
        return "Emergency", f"Red-flag phrase matched: '{kw}'"
    if any(SEVERITY_MAP.get(c) == "Emergency" for c in top_conditions):
        return "Emergency", "Condition indicates emergency"
    if any(SEVERITY_MAP.get(c) == "Moderate" for c in top_conditions):
        return "Moderate", "Recommend seeing a specialist soon"
    return "Mild", "Home care and monitoring recommended"


def get_care_and_tablets(condition: str) -> Tuple[str, List[str]]:
    """
    Returns care advice and common medications for the condition.
    """
    return CARE_ADVICE.get(condition, ("General care and consultation recommended.", []))


def rule_based_check(symptoms: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Additional simple rule-based heuristics for low-confidence situations.
    Returns (human-readable label, suggested_specialty)
    """
    s = (symptoms or "").lower()
    if "chest pain" in s or "pressure in chest" in s or "radiat" in s:
        return "Possible Cardiac issue (urgent)", "Cardiologist"
    if "shortness of breath" in s or "difficulty breathing" in s or "wheeze" in s:
        return "Possible Respiratory issue", "Pulmonologist"
    if "rash" in s or "itch" in s or "hives" in s:
        return "Possible Allergy / Dermatology issue", "Dermatologist"
    if "headache" in s or "migraine" in s or "aura" in s:
        return "Possible Migraine / Neurological issue", "Neurologist"
    if "vomit" in s or "diarrhea" in s or "stomach pain" in s:
        return "Possible Gastrointestinal issue", "Gastroenterologist"
    if "urine" in s or "burning" in s or "painful urination" in s:
        return "Possible Urinary Tract issue", "Urologist"
    return None, None


def predict_doctors_for_condition(condition: str, city: Optional[str] = None) -> pd.DataFrame:
    """
    Returns doctors matching the condition's specialty.
    If city is provided, try to filter by city first; otherwise return all matching specialty doctors.
    """
    doctors_df = st.session_state['doctors_df']
    specialty = DISEASE_TO_SPECIALTY.get(condition, "General Physician")
    if city:
        matches = doctors_df[(doctors_df['specialty'] == specialty) & (doctors_df['city'] == city)]
        if matches.empty:
            matches = doctors_df[doctors_df['specialty'] == specialty]
    else:
        matches = doctors_df[doctors_df['specialty'] == specialty]
    return matches

# Report generation (text + optional PDF)

def generate_text_report(patient_name: str, city: str, symptoms: str, predictions: List[Tuple[str, float]], severity: str, notes: str = "") -> str:
    """
    Generate a plain-text report summarizing the check.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = []
    lines.append("MediConnect — Symptom Report")
    lines.append(f"Generated: {now}")
    lines.append(f"Patient: {patient_name}")
    lines.append(f"City: {city}")
    lines.append("")
    lines.append("Symptoms:")
    lines.append(symptoms)
    lines.append("")
    lines.append("Top Predictions:")
    for cond, prob in predictions:
        # For transparency we include probability in the report file (but main UI table will not show prob lines)
        lines.append(f"- {cond} — probability: {prob:.1%} (Specialist: {DISEASE_TO_SPECIALTY.get(cond,'General Physician')})")
    lines.append("")
    lines.append(f"Estimated severity: {severity}")
    if notes:
        lines.append("")
        lines.append("Notes:")
        lines.append(notes)
    lines.append("")
    lines.append("Care advice / common medications:")
    top_cond = predictions[0][0] if predictions else "Unknown"
    care_text, meds = get_care_and_tablets(top_cond)
    lines.append(f"- {top_cond}: {care_text}")
    if meds:
        lines.append(f"- Common meds: {', '.join(meds)}")
    lines.append("")
    lines.append("Disclaimer: This is an AI-based screening tool and not a substitute for professional medical advice.")
    lines.append("If you detect any red-flag symptoms (chest pain, difficulty breathing, sudden weakness, loss of consciousness), call your local emergency number immediately.")
    return "\n".join(lines)


def generate_pdf_bytes(text_content: str) -> bytes:
    """
    Generate a PDF in-memory if reportlab is available; otherwise return bytes of text.
    """
    if REPORTLAB_AVAILABLE:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        # simple layout: write lines with wrap
        y = height - 72
        for raw_line in text_content.splitlines():
            # wrap long lines
            wrapped = textwrap.wrap(raw_line, 100)
            if not wrapped:
                wrapped = [""]
            for line in wrapped:
                c.drawString(72, y, line)
                y -= 12
                if y < 72:
                    c.showPage()
                    y = height - 72
        c.save()
        buffer.seek(0)
        return buffer.read()
    else:
        # return utf-8 encoded text
        return text_content.encode('utf-8')


# UI: Header and top-level layout

st.markdown('<div class="app-header">🚑 MediConnect — AI Symptom Checker & Doctor Connect System</div>', unsafe_allow_html=True)

# Create three columns layout for the top area
col_left, col_mid, col_right = st.columns([1.6, 1.0, 1.0])



# Left column: Describe your symptoms (big, attractive area)

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📝 Describe your symptoms", unsafe_allow_html=True)
    st.markdown('<div class="muted">Be specific: include onset (e.g., "2 hours ago"), severity, and any worrying signs.</div>', unsafe_allow_html=True)

    # Provide a text_area where user writes symptoms
    symptoms_input = st.text_area(
        label="Enter symptoms here",
        placeholder="e.g. sudden chest pain and sweating, started 30 minutes ago",
        height=200,
        key="symptoms_area"
    )

    # Quick example buttons to help users try the app
    st.markdown("<div style='margin-top:8px; margin-bottom:8px;'><strong>Quick examples:</strong></div>", unsafe_allow_html=True)
    example_cols = st.columns([1, 1, 1, 1])
    examples = [
        "sudden chest pain and sweating",
        "runny nose, sneezing, itchy eyes",
        "severe headache with nausea",
        "diarrhea and vomiting last 24h"
    ]
    for ex_col, ex in zip(example_cols, examples):
        if ex_col.button(ex, key=f"ex_{ex}"):
            st.session_state['symptoms_area'] = ex
            symptoms_input = ex

    # Checkbox to ask whether to include previous history details
    include_history_details = st.checkbox("Include brief medical history (optional)", key="include_history")
    if include_history_details:
        history_text = st.text_input("Add history (e.g., diabetes, meds, allergies):", key="history_text")
    else:
        history_text = ""

    check_btn = st.button("🔍 Check Symptoms", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Mid column: Quick actions & metrics

with col_mid:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🚨 Quick Actions", unsafe_allow_html=True)
    st.markdown('<div class="muted">Emergency numbers & fast links use your phone to call directly.</div>', unsafe_allow_html=True)

    # Provide clickable links styled as buttons for 115 and 911 and local ambulance
    # Note: In a browser, clicking a tel: link will attempt to open the phone app on mobile.
    st.markdown("<div style='margin-top:8px; margin-bottom:8px;'>", unsafe_allow_html=True)

    # Buttons as clickable HTML anchors (styled)
    st.markdown(f'<a class="btn btn-red" href="tel:115">🚑 Call 115 (Ambulance)</a><br>', unsafe_allow_html=True)
    st.markdown(f'<a class="btn btn-red" href="tel:911">🚨 Call 911 (Emergency)</a><br>', unsafe_allow_html=True)
    st.markdown(f'<a class="btn btn-yellow" href="tel:1122">🚑 Call 1122 (Rescue)</a><br>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Show some helpful metrics
    st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
    st.metric(label="Checks this session", value=len(st.session_state['history']))
    st.markdown("</div>", unsafe_allow_html=True)

    # Additional quick tools
    st.markdown("---")
    st.markdown("**Self-care quick tips**")
    st.markdown("- Stay hydrated\n- Rest\n- Avoid heavy meals\n- Use paracetamol if feverish (follow dosing instructions)")

    st.markdown('</div>', unsafe_allow_html=True)

# Right column: Nearby services (searchable)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📍 Nearby Services", unsafe_allow_html=True)
    st.markdown('<div class="muted">Pick a seeded city to show local doctor counts and quick map links.</div>', unsafe_allow_html=True)

    # List cities in seeded DB
    doctors_df_local = st.session_state['doctors_df']
    cities = sorted(doctors_df_local['city'].fillna("Unknown").unique().tolist())
    selected_city = st.selectbox("Choose city (seeded DB)", options=cities, index=0)
    city_count = doctors_df_local[doctors_df_local['city'] == selected_city].shape[0]
    st.markdown(f"<div class='muted' style='margin-top:6px;'>Doctors in {selected_city}: <strong>{city_count}</strong></div>", unsafe_allow_html=True)

    # Provide quick Google Maps link
    if st.button("🗺️ Open hospitals near city", use_container_width=True):
        map_link = f"https://www.google.com/maps/search/hospitals+near+{quote_plus(selected_city)}"
        st.markdown(f"[Open hospitals in {selected_city} — Google Maps]({map_link})", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Process input when user clicks Check

if 'symptoms_area' not in st.session_state:
    st.session_state['symptoms_area'] = ""

# Real value used for processing
current_symptoms = st.session_state.get('symptoms_area', "").strip()

# When user hits check button, analyze
if check_btn:
    if not current_symptoms:
        st.warning("⚠️ Please enter symptoms before checking.")
    else:
        # Run analysis inside spinner
        with st.spinner("Analyzing symptoms using AI + rules..."):
            # Primary model predictions
            preds = predict_top_k(current_symptoms, k=3)
            # Top condition names for severity check
            top_condition_names = [p[0] for p in preds]

            # If model returns nothing, add fallback empty list
            if not preds:
                preds = []

            # If model is low confidence, try rule-based augmentation
            if preds and preds[0][1] < 0.30:
                # Try to augment with rules
                rb_label, rb_spec = rule_based_check(current_symptoms)
                if rb_label:
                    # Insert rule-based guess at front (with pseudo probability)
                    preds.insert(0, (rb_label, 0.35))
                    top_condition_names = [p[0] for p in preds]

            # Estimate severity
            severity_label, severity_notes = estimate_severity(current_symptoms, top_condition_names)

            # Save to session for later
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            st.session_state['history'].append({
                'timestamp': timestamp,
                'symptoms': current_symptoms,
                'preds': preds,
                'severity': severity_label
            })

        # ---- UI Output area for results (large area below top columns) ----
        st.markdown("---")

        # Severity badge
        if severity_label == "Emergency":
            st.markdown('<div class="severity-badge sev-em">🚨 EMERGENCY — Immediate action advised. Call emergency services or go to the nearest ED.</div>', unsafe_allow_html=True)
        elif severity_label == "Moderate":
            st.markdown('<div class="severity-badge sev-mod">⚠️ Moderate — Seek prompt medical attention from a specialist.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="severity-badge sev-mild">✅ Mild — Symptoms appear mild. Home care & monitoring recommended.</div>', unsafe_allow_html=True)

        # Predictions table: per your earlier request, we show ONLY a neat table (Condition + Specialist)
        st.markdown("### 🔍 Top Predictions")
        if preds:
            pred_table_rows = []
            for cond, prob in preds:
                pred_table_rows.append({
                    "Condition": cond,
                    "Specialist": DISEASE_TO_SPECIALTY.get(cond, "General Physician")
                })
            pred_df_ui = pd.DataFrame(pred_table_rows)
            # display as st.table (no separate 'why' tokens or prob lines)
            st.table(pred_df_ui)
        else:
            st.info("Model could not determine a likely condition. Consider updating symptoms with more details (duration, severity).")

        # Care advice for top condition
        top_condition = preds[0][0] if preds else "Unknown"
        care_text, meds = get_care_and_tablets(top_condition)
        st.markdown(f"### 🩺 Care advice — {top_condition}")
        st.write(care_text)
        if meds:
            # show as pills
            pills_html = " ".join([f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;background:#eef2ff;margin-right:6px;font-weight:800'>{m}</span>" for m in meds])
            st.markdown(pills_html, unsafe_allow_html=True)

        # Model confidence metric (separate small UI piece)
        if preds:
            top_confidence = preds[0][1]
        else:
            top_confidence = 0.0
        st.metric(label="Model confidence (top)", value=f"{top_confidence:.1%}")

        # Doctor recommendations section
        st.markdown("---")
        st.markdown(f"### 👩‍⚕️ Nearby doctors for **{DISEASE_TO_SPECIALTY.get(top_condition, 'General Physician')}** in {selected_city}")

        matches = predict_doctors_for_condition(top_condition, city=selected_city)
        if matches.empty:
            st.info("No exact matches found in this city. Showing same-specialty doctors across all seeded cities.")
            matches = predict_doctors_for_condition(top_condition, city=None)

        # Render each doctor as a styled card
        if matches.empty:
            st.write("No doctors of this specialty are available in the seeded database.")
        else:
            for i, row in matches.iterrows():
                # Render small doctor card UI
                name = row.get('name', row.get('Name', 'Unknown'))
                hosp = row.get('hospital', row.get('Hospital', ''))
                city = row.get('city', row.get('City', ''))
                phone = row.get('phone', row.get('Contact', ''))
                email = row.get('email', row.get('Email', ''))
                rating = row.get('rating', row.get('Rating', 'N/A'))

                st.markdown('<div class="doctor-card">', unsafe_allow_html=True)
                st.markdown(f"**{name}** — {hosp}  ")
                st.markdown(f"<div class='muted'>City: {city} • Rating: {rating}</div>", unsafe_allow_html=True)
                doc_cols = st.columns([1, 1, 1])
                with doc_cols[0]:
                    if phone:
                        st.markdown(f'<a class="btn btn-green" href="tel:{phone}">📞 Call</a>', unsafe_allow_html=True)
                    else:
                        st.write("No phone")
                with doc_cols[1]:
                    maps_q = quote_plus(f"{hosp} {city}")
                    st.markdown(f'<a class="btn btn-blue" target="_blank" href="https://www.google.com/maps/search/{maps_q}">🗺️ Map</a>', unsafe_allow_html=True)
                with doc_cols[2]:
                    if email:
                        st.markdown(f'<a class="btn btn-yellow" href="mailto:{email}">✉️ Email</a>', unsafe_allow_html=True)
                    else:
                        st.write("No email")
                st.markdown('</div>', unsafe_allow_html=True)

        # downloadable report
        st.markdown("---")
        st.markdown("### 📄 Download / Share")
        report_text = generate_text_report(st.session_state.get('patient_name', 'Patient'), selected_city, current_symptoms, preds, severity_label, notes=severity_notes)
        st.session_state['last_report'] = report_text
        report_bytes = generate_pdf_bytes(report_text)

        # Download button (if PDF available will be PDF bytes; otherwise it's text)
        if REPORTLAB_AVAILABLE:
            st.download_button("📄 Download PDF report", data=report_bytes, file_name="mediconnect_report.pdf", mime="application/pdf")
        else:
            st.download_button("📄 Download report (txt)", data=report_bytes, file_name="mediconnect_report.txt", mime="text/plain")

        # WhatsApp share
        wa_link = f"https://wa.me/?text={quote_plus(report_text)}"
        st.markdown(f'<a class="btn btn-green" target="_blank" href="{wa_link}">💬 Share via WhatsApp</a>', unsafe_allow_html=True)

        # Safety note displayed inline
        st.markdown("<div style='margin-top:10px; padding:12px; border-radius:10px; background:#fff3f3;'><strong>Safety:</strong> This tool is an AI-powered screening assistant and not a diagnostic system. If red-flag symptoms are present, call emergency services immediately.</div>", unsafe_allow_html=True)

        # If emergency, show extra prominent assist
        if severity_label == "Emergency":
            st.markdown("<div style='padding:12px; border-radius:10px; margin-top:8px; background: linear-gradient(90deg,#fee2e2,#fecaca);'>🔴 Detected emergency signs. Please call emergency numbers now.</div>", unsafe_allow_html=True)
            # Quick emergency call links (repeat visibly)
            st.markdown(f'<a class="btn btn-red" href="tel:115">🚑 Call 115 (Ambulance)</a> <a class="btn btn-red" href="tel:911">🚨 Call 911 (Emergency)</a>', unsafe_allow_html=True)

        # Option to show model internals (for dev) — hidden by default
        with st.expander("Developer: Model info (hidden)"):
            try:
                classes = model.named_steps['randomforestclassifier'].classes_
                vocab_size = len(model.named_steps['tfidfvectorizer'].get_feature_names_out())
                st.write("Model classes:", classes)
                st.write("Vectorizer vocab size:", vocab_size)
                st.write("Raw predictions (condition:prob):", preds)
            except Exception as e:
                st.write("Model info not available:", e)

# Sidebar: detailed controls, history, admin actions

with st.sidebar:
    st.markdown('<div style="padding:12px; border-radius:8px; background: linear-gradient(90deg,#0f172a,#0f172a);">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">👤 Patient & App Controls</div>', unsafe_allow_html=True)

    # Patient name
    patient_name = st.text_input("Your name", value=st.session_state.get('patient_name', 'Patient'))
    st.session_state['patient_name'] = patient_name

    # Recent checks history with small table
    st.markdown("---")
    st.markdown("### 🕘 Recent checks", unsafe_allow_html=True)
    hist = st.session_state.get('history', [])
    if hist:
        # show last 6 checks
        last_items = list(reversed(hist[-6:]))
        for item in last_items:
            ts = item['timestamp']
            txt = item['symptoms'] if len(item['symptoms']) <= 80 else item['symptoms'][:77] + "..."
            sev = item.get('severity', 'N/A')
            st.markdown(f"- **{ts}** — {txt} — **{sev}**")
    else:
        st.write("No checks yet")

    # Admin: upload doctors CSV to append to seeded DB
    st.markdown("---")
    st.markdown("### 🗂️ Doctor DB (admin)", unsafe_allow_html=True)
    st.write("Upload CSV to append doctors (columns: name,specialty,hospital,city,phone,email,rating)")

    uploaded = st.file_uploader("Upload doctor CSV", type=['csv'], accept_multiple_files=False)
    if uploaded is not None:
        try:
            newdoc = pd.read_csv(uploaded)
            if 'name' in newdoc.columns and 'specialty' in newdoc.columns:
                # append to existing
                st.session_state['doctors_df'] = pd.concat([st.session_state['doctors_df'], newdoc.fillna('')], ignore_index=True)
                st.success(f"Added {len(newdoc)} records — doctor DB updated for this session")
            else:
                st.error("CSV must contain 'name' and 'specialty' columns.")
        except Exception as e:
            st.error("Could not read CSV: " + str(e))

    # Small helper: view doctor sample
    if st.button("Show sample seeded doctors"):
        st.dataframe(st.session_state['doctors_df'].head(10))

    st.markdown("---")
    st.markdown("### ⚙️ App Info", unsafe_allow_html=True)
    st.write("Version: demo")
    st.caption("This is a prototype. Replace datasets & models for production.")

    # PDF status
    if REPORTLAB_AVAILABLE:
        st.success("PDF export available")
    else:
        st.info("PDF export not available (install reportlab for PDF)")

    st.markdown("---")
    st.markdown("### 📚 Resources (dev)", unsafe_allow_html=True)
    if st.button("Export history as JSON"):
        # prepare history download
        hist_bytes = json.dumps(st.session_state['history'], indent=2).encode('utf-8')
        st.download_button("Download history.json", data=hist_bytes, file_name="mediconnect_history.json", mime="application/json")

    st.markdown('</div>', unsafe_allow_html=True)

# Bottom: Additional helper sections and guidance

st.markdown("---")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("## 📘 Guidance & Notes", unsafe_allow_html=True)
st.markdown("""
- **Not medical advice:** This tool provides AI-powered screening suggestions only. It is *not* a diagnostic system.
- **Emergencies:** If someone has chest pain, difficulty breathing, sudden weakness, loss of consciousness, or other worrying signs call local emergency services immediately (115 / 911 / 1122 etc).
- **Model limitations:** This AI model is a small. For production use, collect labeled data, validate thoroughly, and consult clinicians.
- **Extending the app:** You can upload a doctor CSV in the sidebar to extend the seeded DB for local availability.
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Footer: developer note
st.markdown("<div style='margin-top:12px; color:#475569; font-size:0.9rem'>Built with ❤️ for Your help. Replace datasets & refine model prior to clinical use.</div>", unsafe_allow_html=True)

