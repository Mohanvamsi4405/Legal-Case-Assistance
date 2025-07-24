import streamlit as st
import fitz  # PyMuPDF
import easyocr
import numpy as np
import requests
import pandas as pd
import docx
import io
import zipfile
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import re
import base64
import hashlib
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.parser import ParserError
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import sqlite3
import bcrypt
import json
import secrets
from fuzzywuzzy import fuzz

# --- Database Setup ---
def init_db():
    """Initialize SQLite database and ensure all required columns exist"""
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL
                 )''')
    
    # Check and add full_name column if missing
    c.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in c.fetchall()]
    if 'full_name' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
        st.session_state.debug_log.append("Added full_name column to users table")
    
    # Check and add phone_number column if missing
    if 'phone_number' not in columns:
        c.execute("ALTER TABLE users ADD COLUMN phone_number TEXT")
        st.session_state.debug_log.append("Added phone_number column to users table")
    
    # Create other tables
    c.execute('''CREATE TABLE IF NOT EXISTS case_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    case_data TEXT NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                 )''')
    conn.commit()
    return conn

def add_user(username, password, email, full_name="", phone_number=""):
    """Add a new user to the database with hashed password"""
    conn = init_db()
    c = conn.cursor()
    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        c.execute("INSERT INTO users (username, password, email, full_name, phone_number) VALUES (?, ?, ?, ?, ?)",
                  (username, hashed_password, email, full_name, phone_number))
        conn.commit()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticate user and return user ID and details if successful"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT id, password, full_name, email, phone_number FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and bcrypt.checkpw(password.encode('utf-8'), result[1]):
        return True, "Login successful!", result[0], {
            "full_name": result[2] or "",
            "email": result[3],
            "phone_number": result[4] or ""
        }
    return False, "Invalid username or password.", None, None

def save_user_case_history(user_id, cases):
    """Save user's case history to the database"""
    conn = init_db()
    c = conn.cursor()
    
    serializable_cases = []
    for case in cases:
        case_copy = case.copy()
        case_copy['embeddings'] = [emb.tolist() for emb in case['embeddings']]
        case_copy['crime_scene_summaries'] = [
            {k: v if k != 'image' else None for k, v in img_data.items()}
            for img_data in case['crime_scene_summaries']
        ]
        case_copy['chat_history'] = [
            {k: base64.b64encode(v).decode() if k == 'audio' and v else v for k, v in msg.items()}
            for msg in case['chat_history']
        ]
        if case_copy['date']:
            case_copy['date'] = case_copy['date'].isoformat()
        serializable_cases.append(case_copy)
    
    case_data = json.dumps(serializable_cases)
    last_updated = datetime.now().isoformat()
    
    c.execute("INSERT OR REPLACE INTO case_history (user_id, case_data, last_updated) VALUES (?, ?, ?)",
              (user_id, case_data, last_updated))
    conn.commit()
    conn.close()
    st.session_state.debug_log.append(f"Saved case history for user_id {user_id}")

def load_user_case_history(user_id):
    """Load user's case history from the database"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT case_data FROM case_history WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        cases = json.loads(result[0])
        for case in cases:
            case['embeddings'] = [np.array(emb) for emb in case['embeddings']]
            case['chat_history'] = [
                {k: base64.b64decode(v) if k == 'audio' and v else v for k, v in msg.items()}
                for msg in case['chat_history']
            ]
            case['date'] = parser.parse(case['date']).date() if case['date'] else None
        st.session_state.debug_log.append(f"Loaded {len(cases)} cases for user_id {user_id}")
        return cases
    return []

def generate_reset_token(user_id, email):
    """Generate a password reset token and store it with 1-hour expiration"""
    conn = init_db()
    c = conn.cursor()
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=1)
    
    c.execute("INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
              (user_id, token, expires_at.isoformat()))
    conn.commit()
    conn.close()
    
    reset_link = f"http://localhost:8501/?reset_token={token}"
    st.session_state.debug_log.append(f"Generated reset token for user_id {user_id}")
    return reset_link

def validate_reset_token(token):
    """Validate a password reset token and return user_id if valid"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT user_id, expires_at FROM password_reset_tokens WHERE token = ?", (token,))
    result = c.fetchone()
    
    if result:
        user_id, expires_at = result
        expires_at = datetime.fromisoformat(expires_at)
        if expires_at > datetime.now():
            return True, user_id
        else:
            c.execute("DELETE FROM password_reset_tokens WHERE token = ?", (token,))
            conn.commit()
    conn.close()
    return False, None

def reset_password(user_id, new_password):
    """Update user's password"""
    conn = init_db()
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    c.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
    c.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    st.session_state.debug_log.append(f"Password reset for user_id {user_id}")

def update_user_details(user_id, full_name, email, phone_number):
    """Update user details in the database"""
    conn = init_db()
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET full_name = ?, email = ?, phone_number = ? WHERE id = ?",
                  (full_name, email, phone_number, user_id))
        conn.commit()
        return True, "User details updated successfully!"
    except sqlite3.IntegrityError:
        return False, "Email already exists."
    finally:
        conn.close()

def get_user_details(user_id):
    """Retrieve user details from the database"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT username, full_name, email, phone_number FROM users WHERE id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return {
            "username": result[0],
            "full_name": result[1] or "",
            "email": result[2],
            "phone_number": result[3] or ""
        }
    return None

# --- Global Configuration ---
st.set_page_config(
    page_title="⚖️ LegalEase AI – Multi-Case Legal Analysis",
    page_icon="⚖️",
    layout="wide"
)

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "cases": [],
        "active_case_index": 0,
        "processed_file_hash": None,
        "audio_uploader_key_counter": 0,
        "gemini_configured": False,
        "debug_log": [],
        "authenticated": False,
        "username": None,
        "user_id": None,
        "user_details": {},
        "reset_token": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Language Selection ---
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Marathi": "mr",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Gujarati": "gu"
}
# --- Sidebar Configuration ---
def display_auth_forms():
    """Display login, signup, forgot password, and reset password forms in the sidebar"""
    st.sidebar.title("🔐 Authentication")
    
    if "reset_token" in st.query_params and st.query_params["reset_token"]:
        st.session_state.reset_token = st.query_params["reset_token"]
        display_reset_password_form()
    else:
        auth_option = st.sidebar.radio("Select Action", ["Login", "Sign Up", "Forgot Password"])
        
        if auth_option == "Sign Up":
            st.sidebar.subheader("Create Account")
            username = st.sidebar.text_input("Username", key="signup_username")
            email = st.sidebar.text_input("Email", key="signup_email")
            full_name = st.sidebar.text_input("Full Name", key="signup_full_name")
            phone_number = st.sidebar.text_input("Phone Number", key="signup_phone_number")
            password = st.sidebar.text_input("Password", type="password", key="signup_password")
            confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="signup_confirm_password")
            
            if st.sidebar.button("Sign Up"):
                if not username or not email or not password or not confirm_password:
                    st.sidebar.error("All required fields (username, email, password) must be filled.")
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.sidebar.error("Invalid email format.")
                elif password != confirm_password:
                    st.sidebar.error("Passwords do not match.")
                elif len(password) < 8:
                    st.sidebar.error("Password must be at least 8 characters long.")
                else:
                    success, message = add_user(username, password, email, full_name, phone_number)
                    if success:
                        st.sidebar.success(message)
                        success, _, user_id, user_details = authenticate_user(username, password)
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_id = user_id
                        st.session_state.user_details = user_details
                        st.session_state.cases = load_user_case_history(user_id)
                        st.rerun()
                    else:
                        st.sidebar.error(message)
        
        elif auth_option == "Forgot Password":
            st.sidebar.subheader("Reset Password")
            email = st.sidebar.text_input("Enter your email", key="forgot_email")
            
            if st.sidebar.button("Send Reset Link"):
                if not email:
                    st.sidebar.error("Email is required.")
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.sidebar.error("Invalid email format.")
                else:
                    conn = init_db()
                    c = conn.cursor()
                    c.execute("SELECT id FROM users WHERE email = ?", (email,))
                    result = c.fetchone()
                    conn.close()
                    
                    if result:
                        user_id = result[0]
                        reset_link = generate_reset_token(user_id, email)
                        st.sidebar.success(f"Password reset link generated (simulated email): {reset_link}")
                        st.session_state.debug_log.append(f"Reset link displayed for email: {email}")
                    else:
                        st.sidebar.error("No account found with this email.")
        
        else:
            st.sidebar.subheader("Login")
            username = st.sidebar.text_input("Username", key="login_username")
            password = st.sidebar.text_input("Password", type="password", key="login_password")
            
            if st.sidebar.button("Login"):
                if not username or not password:
                    st.sidebar.error("Username and password are required.")
                else:
                    success, message, user_id, user_details = authenticate_user(username, password)
                    if success:
                        st.sidebar.success(message)
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_id = user_id
                        st.session_state.user_details = user_details
                        st.session_state.cases = load_user_case_history(user_id)
                        st.rerun()
                    else:
                        st.sidebar.error(message)

def display_reset_password_form():
    """Display the password reset form"""
    st.sidebar.subheader("Reset Password")
    valid, user_id = validate_reset_token(st.session_state.reset_token)
    
    if not valid:
        st.sidebar.error("Invalid or expired reset token.")
        return
    
    new_password = st.sidebar.text_input("New Password", type="password", key="reset_new_password")
    confirm_password = st.sidebar.text_input("Confirm New Password", type="password", key="reset_confirm_password")
    
    if st.sidebar.button("Reset Password"):
        if not new_password or not confirm_password:
            st.sidebar.error("Both password fields are required.")
        elif new_password != confirm_password:
            st.sidebar.error("Passwords do not match.")
        elif len(new_password) < 8:
            st.sidebar.error("Password must be at least 8 characters long.")
        else:
            reset_password(user_id, new_password)
            st.sidebar.success("Password reset successfully! Please log in.")
            st.session_state.reset_token = None
            st.query_params.clear()
            st.rerun()

def display_user_profile():
    """Display and allow editing of user profile details"""
    if not st.session_state.authenticated or not st.session_state.user_id:
        return
    
    st.sidebar.header("👤 User Profile")
    user_details = st.session_state.user_details
    
    with st.sidebar.expander("View/Edit Profile"):
        full_name = st.text_input("Full Name", value=user_details["full_name"], key="profile_full_name")
        email = st.text_input("Email", value=user_details["email"], key="profile_email")
        phone_number = st.text_input("Phone Number", value=user_details["phone_number"], key="profile_phone_number")
        
        if st.button("Update Profile"):
            if not email:
                st.sidebar.error("Email is required.")
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.sidebar.error("Invalid email format.")
            else:
                success, message = update_user_details(st.session_state.user_id, full_name, email, phone_number)
                if success:
                    st.sidebar.success(message)
                    st.session_state.user_details = {
                        "username": st.session_state.username,
                        "full_name": full_name,
                        "email": email,
                        "phone_number": phone_number
                    }
                else:
                    st.sidebar.error(message)

# --- Logout Function ---
def logout():
    """Log out the current user and save case history"""
    if st.session_state.user_id and st.session_state.cases:
        save_user_case_history(st.session_state.user_id, st.session_state.cases)
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.user_details = {}
    st.session_state.cases = []
    st.session_state.active_case_index = 0
    st.session_state.processed_file_hash = None
    st.session_state.audio_uploader_key_counter = 0
    st.session_state.debug_log = []
    st.session_state.reset_token = None
    st.query_params.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Logged out successfully!")
    st.rerun()

# --- Core Processing Functions ---
@st.cache_resource
def get_easyocr_reader():
    """Initialize and cache the EasyOCR reader"""
    return easyocr.Reader(['en'], gpu=False, verbose=False)

reader = get_easyocr_reader()

def transcribe_audio(audio_bytes, lang_code):
    """Convert audio to text using Google Speech Recognition"""
    r = sr.Recognizer()
    try:
        if audio_bytes.startswith(b'ID3') or audio_bytes[4:8] == b'ftyp':
            input_format = "mp3"
        elif audio_bytes.startswith(b'RIFF') and audio_bytes[8:12] == b'WAVE':
            input_format = "wav"
        elif audio_bytes.startswith(b'\x1aE\xdf\xa3'):
            input_format = "webm"
        else:
            input_format = "webm"

        with io.BytesIO(audio_bytes) as audio_file:
            audio = AudioSegment.from_file(audio_file, format=input_format)
            wav_file = io.BytesIO()
            audio.export(wav_file, format="wav")
            wav_file.seek(0)

            with sr.AudioFile(wav_file) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language=lang_code)
                return text
    except sr.UnknownValueError:
        return f"Could not understand audio in {lang_code}"
    except sr.RequestError as e:
        return f"Speech recognition service error: {e}"
    except Exception as e:
        return f"Error transcribing audio: {e}"

def generate_embeddings(text, api_key):
    """Generate embeddings using Gemini API"""
    if not api_key or not text.strip():
        st.session_state.debug_log.append(f"Embedding failed: Empty text or no API key")
        return np.zeros((768,))
    
    try:
        payload = {
            "model": "models/embedding-001",
            "content": {"parts": [{"text": text}]}
        }
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent",
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json=payload
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"]["values"])
    except Exception as e:
        st.session_state.debug_log.append(f"Embedding API error: {e}")
        return np.zeros((768,))

def llm_image_summary(image_bytes, api_key, model_name="gemini-1.5-flash"):
    """Generate summary of an image using Gemini"""
    if not api_key or not st.session_state.gemini_configured:
        st.session_state.debug_log.append("Image summary failed: Gemini API not configured")
        return "[Image Summary error]: Gemini API not configured."
    
    try:
        model = genai.GenerativeModel(model_name)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        max_size = 2048
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        prompt = """Analyze this image in detail for legal relevance. Focus on:
        - People, objects, documents visible
        - Any text content (even if partial)
        - Potential legal significance
        - Relationships between elements
        Be thorough but concise."""
        
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.GenerationConfig(max_output_tokens=500)
        )
        
        if response.candidates and response.candidates[0].content.parts:
            summary = response.candidates[0].content.parts[0].text
            st.session_state.debug_log.append(f"Image summary generated: {summary[:50]}...")
            return summary
        st.session_state.debug_log.append("Image summary failed: No valid response")
        return "[Image Summary error]: No valid response from model."
    except Exception as e:
        st.session_state.debug_log.append(f"Image summary error: {e}")
        return f"[Image Summary error]: {str(e)}"

def ocr_image(image_bytes):
    """Perform OCR on an image with enhanced validation"""
    try:
        img_io = io.BytesIO(image_bytes)
        image = Image.open(img_io)
        if image.format not in ['PNG', 'JPEG', 'JPG']:
            st.session_state.debug_log.append(f"Unsupported image format: {image.format}")
            return "", None
        image.verify()
        image = Image.open(img_io).convert("RGB")
        ocr_text = "\n".join(reader.readtext(np.array(image), detail=0))
        st.session_state.debug_log.append(f"OCR successful, extracted text length: {len(ocr_text)}")
        return ocr_text, image
    except Exception as e:
        st.session_state.debug_log.append(f"OCR failed: {e}")
        return "", None

def generate_text_completion(prompt, api_key, model_name="gemini-1.5-flash", max_tokens=500):
    """Generate text using Gemini"""
    if not api_key or not st.session_state.gemini_configured:
        st.session_state.debug_log.append("Text completion failed: Gemini API not configured")
        return "[LLM Error]: Gemini API not configured."
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens)
        )
        
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        st.session_state.debug_log.append("Text completion failed: No valid response")
        return "[LLM Error]: No valid response from model."
    except Exception as e:
        st.session_state.debug_log.append(f"Text completion error: {e}")
        return f"[LLM Error]: {str(e)}"

# --- Document Processing Functions ---
def extract_text_from_pdf(file_bytes):
    """Extract text and images from PDF"""
    text = ""
    images = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n"
            for img_info in page.get_images(full=True):
                base_image = doc.extract_image(img_info[0])
                if base_image and base_image["image"]:
                    try:
                        Image.open(io.BytesIO(base_image["image"])).verify()
                        images.append(base_image["image"])
                        st.session_state.debug_log.append(f"Extracted valid image from PDF page {page.number}")
                    except Exception as e:
                        st.session_state.debug_log.append(f"Invalid image in PDF page {page.number}: {e}")
        doc.close()
    except Exception as e:
        st.session_state.debug_log.append(f"PDF extraction error: {e}")
    return text, images

def extract_text_from_docx(file_bytes):
    """Extract text and images from DOCX with enhanced validation"""
    text = ""
    images = []
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref and rel.target_part and hasattr(rel.target_part, 'blob'):
                img_bytes = rel.target_part.blob
                try:
                    Image.open(io.BytesIO(img_bytes)).verify()
                    images.append(img_bytes)
                    st.session_state.debug_log.append(f"Extracted valid image from DOCX: {len(img_bytes)} bytes")
                except Exception as e:
                    st.session_state.debug_log.append(f"Invalid image in DOCX: {e}")
    except Exception as e:
        st.session_state.debug_log.append(f"DOCX extraction error: {e}")
    return text, images

def extract_text_from_xlsx(file_bytes):
    """Extract data from Excel files"""
    text = ""
    images = []
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
        text = df.to_string()
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            for name in z.namelist():
                if name.startswith("xl/media/"):
                    img_bytes = z.read(name)
                    try:
                        Image.open(io.BytesIO(img_bytes)).verify()
                        images.append(img_bytes)
                        st.session_state.debug_log.append(f"Extracted valid image from XLSX: {name}")
                    except Exception as e:
                        st.session_state.debug_log.append(f"Invalid image in XLSX: {e}")
    except Exception as e:
        st.session_state.debug_log.append(f"Excel extraction error: {e}")
    return text, images

def is_legal_content(text):
    """Determine if text contains legal-relevant content"""
    if not isinstance(text, str):
        return False
    
    legal_keywords = [
        "court", "judge", "petitioner", "respondent", "plaintiff", "defendant",
        "section", "act", "statute", "law", "legal", "case", "filing",
        "order", "judgment", "hearing", "trial", "evidence", "witness",
        "contract", "agreement", "clause", "party", "parties"
    ]
    return any(re.search(r"\b" + re.escape(k) + r"\b", text, re.IGNORECASE) for k in legal_keywords)

# --- Case Management Functions ---
def extract_all_cases(text_content, api_key):
    """Extract all cases from text using LLM"""
    if not st.session_state.gemini_configured or not text_content.strip():
        st.session_state.debug_log.append("Case extraction failed: API not configured or empty text")
        return []
    
    prompt = f"""Identify ALL distinct legal cases in the following text. For EACH case, provide:
    1. Case Type (e.g., Criminal, Civil, Family, Property)
    2. Case Number 
    3. Petitioner/Plaintiff
    4. Respondent/Defendant
    5. Date (YYYY-MM-DD format)
    6. Location (court/city/state)
    7. Relevant Sections/Acts (comma-separated)
    8. Brief Summary (1-2 sentences)

    Format each case EXACTLY like this:
    === Case Start ===
    Case Type: <value>
    Case Number: <value>
    Petitioner: <value>
    Respondent: <value>
    Date: <value>
    Location: <value>
    Sections: <value>
    Summary: <value>
    === Case End ===

    Text:
    {text_content[:20000]}"""

    response = generate_text_completion(prompt, api_key, max_tokens=2000)
    st.session_state.debug_log.append(f"Extracted cases: {response[:100]}...")
    return parse_multiple_cases(response)

def parse_multiple_cases(response):
    """Parse multiple cases from LLM response"""
    cases = []
    current_case = {}
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('=== Case Start ==='):
            current_case = {
                "type_of_case": "N/A",
                "case_number": "N/A",
                "petitioner": "N/A",
                "respondent": "N/A",
                "date": None,
                "location": "N/A",
                "sections": "N/A",
                "summary": "N/A",
                "chunks": [],
                "embeddings": [],
                "crime_scene_summaries": [],
                "chat_history": [],
                "manual_notes": "",
                "analysis": {
                    "summary": "N/A",
                    "key_entities": "N/A",
                    "legal_issues": "N/A",
                    "recommendations": "N/A"
                }
            }
        elif line.startswith('=== Case End ==='):
            if current_case:
                cases.append(current_case)
        elif ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == "Case Type":
                current_case["type_of_case"] = value
            elif key == "Case Number":
                current_case["case_number"] = value
            elif key == "Petitioner":
                current_case["petitioner"] = value
            elif key == "Respondent":
                current_case["respondent"] = value
            elif key == "Date":
                try:
                    current_case["date"] = parser.parse(value).date()
                except (ValueError, ParserError):
                    current_case["date"] = None
            elif key == "Location":
                current_case["location"] = value
            elif key == "Sections":
                current_case["sections"] = value
            elif key == "Summary":
                current_case["summary"] = value
    
    st.session_state.debug_log.append(f"Parsed {len(cases)} cases")
    return cases

def generate_case_analysis(case_text, case_info, api_key):
    """Generate comprehensive case analysis"""
    if not st.session_state.gemini_configured or not case_text.strip():
        st.session_state.debug_log.append("Case analysis failed: API not configured or empty text")
        return {
            "summary": "N/A (API not configured)",
            "key_entities": "N/A",
            "legal_issues": "N/A",
            "recommendations": "N/A"
        }
    
    prompt = f"""Analyze the following legal case content for case {case_info['case_number']} ONLY:

    Case Information:
    - Type: {case_info['type_of_case']}
    - Number: {case_info['case_number']}
    - Petitioner: {case_info['petitioner']}
    - Respondent: {case_info['respondent']}
    - Date: {case_info['date']}
    - Location: {case_info['location']}
    - Sections: {case_info['sections']}

    Content:
    {case_text[:15000]}

    Provide your analysis with these sections:
    1. Case Summary (3-5 sentences summarizing key facts and issues for {case_info['case_number']})
    2. Key Entities (list of people, organizations, locations specific to this case)
    3. Legal Issues (main legal questions/problems identified for this case)
    4. Recommendations (suggested next steps or actions for this case)

    Format your response with clear section headings and focus ONLY on {case_info['case_number']}."""
    
    response = generate_text_completion(prompt, api_key, max_tokens=1000)
    st.session_state.debug_log.append(f"Case analysis for {case_info['case_number']}: {response[:50]}...")
    return parse_analysis_response(response)

def parse_analysis_response(response):
    """Parse the LLM's analysis response into structured data"""
    sections = {
        "summary": "N/A",
        "key_entities": "N/A",
        "legal_issues": "N/A",
        "recommendations": "N/A"
    }
    
    current_section = None
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if "Case Summary" in line:
            current_section = "summary"
            sections[current_section] = ""
        elif "Key Entities" in line:
            current_section = "key_entities"
            sections[current_section] = ""
        elif "Legal Issues" in line:
            current_section = "legal_issues"
            sections[current_section] = ""
        elif "Recommendations" in line:
            current_section = "recommendations"
            sections[current_section] = ""
        elif current_section:
            sections[current_section] += line + "\n"
    
    for key in sections:
        sections[key] = sections[key].strip()
        if not sections[key]:
            sections[key] = "N/A"
    
    return sections

def generate_chat_response(query, context_chunks, case_info, language_code, api_key, use_external_llm=False):
    """Generate a response to a legal query using RAG or judge-mode LLM"""
    if not st.session_state.gemini_configured:
        st.session_state.debug_log.append("Chat response failed: Gemini API not configured")
        return "Gemini API is not configured. Please enter your API key."
    
    context = "\n\n".join(context_chunks[:3])
    context_relevant = any(query.lower() in chunk.lower() for chunk in context_chunks)
    is_judge_query = any(kw in query.lower() for kw in ["judge", "win", "guilty", "liable"])
    
    if use_external_llm and (not context_relevant or is_judge_query):
        prompt = f"""You are a legal expert acting as a judge in {language_code}. Answer the following query for case {case_info['case_number']} by reasoning through the evidence and applicable laws, providing a speculative judgment if specific details are missing. Explain your reasoning clearly, considering the case context and general legal principles.

Case Information:
- Type: {case_info['type_of_case']}
- Number: {case_info['case_number']}
- Petitioner: {case_info['petitioner']}
- Respondent: {case_info['respondent']}
- Date: {case_info['date']}
- Location: {case_info['location']}
- Sections: {case_info['sections']}

Relevant Context:
{context}

User Query: {query}

Response:"""
        st.session_state.debug_log.append(f"Using judge-mode LLM for query: {query}")
    else:
        prompt = f"""You are a legal assistant analyzing case {case_info['case_number']} in {language_code}. Provide a concise, accurate response to the user's query based SOLELY on the provided context. If the answer isn't in the context, say "I cannot answer that based on the provided documents."

Case Information:
- Type: {case_info['type_of_case']}
- Number: {case_info['case_number']}
- Petitioner: {case_info['petitioner']}
- Respondent: {case_info['respondent']}
- Date: {case_info['date']}
- Location: {case_info['location']}
- Sections: {case_info['sections']}

Relevant Context:
{context}

User Query: {query}

Response:"""
        st.session_state.debug_log.append("Using RAG with document context")
    
    response = generate_text_completion(prompt, api_key)
    st.session_state.debug_log.append(f"Chat response: {response[:50]}...")
    return response

def handle_legal_query(query, case_index, language_code, api_key, use_external_llm):
    """Process a legal query with RAG for a specific case"""
    if not st.session_state.cases or case_index >= len(st.session_state.cases):
        st.session_state.debug_log.append("Query failed: No cases or invalid index")
        return "No case documents have been processed. Please upload relevant files first."
    
    case = st.session_state.cases[case_index]
    if not case['chunks'] or not case['embeddings']:
        st.session_state.debug_log.append("Query failed: No chunks or embeddings")
        return "No case documents have been processed for this case."
    
    query_embedding = generate_embeddings(query, api_key)
    
    similarities = cosine_similarity([query_embedding], case['embeddings'])[0]
    top_indices = similarities.argsort()[-3:][::-1]
    context_chunks = [case['chunks'][i] for i in top_indices if case['case_number'] in case['chunks'][i]]
    
    response = generate_chat_response(query, context_chunks, case, language_code, api_key, use_external_llm)
    
    if st.session_state.user_id:
        save_user_case_history(st.session_state.user_id, st.session_state.cases)
    
    return response

# --- UI Components ---
def display_case_selection():
    """Display case selection dropdown"""
    if not st.session_state.cases:
        return
    
    case_options = [f"{case['case_number']} - {case['type_of_case']}" for case in st.session_state.cases]
    selected_case = st.selectbox(
        "🔍 Select Case to Analyze",
        options=range(len(case_options)),
        format_func=lambda x: case_options[x],
        index=st.session_state.active_case_index
    )
    
    if selected_case != st.session_state.active_case_index:
        st.session_state.active_case_index = selected_case
        if st.session_state.user_id:
            save_user_case_history(st.session_state.user_id, st.session_state.cases)
        st.rerun()

def display_case_info():
    """Display case information section for the active case"""
    if not st.session_state.cases:
        st.info("No cases available. Please upload legal documents first.")
        return
    
    case = st.session_state.cases[st.session_state.active_case_index]
    
    st.header(f"📋 Case Information - {case['case_number']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Case Type:** {case['type_of_case']}")
        st.markdown(f"**Case Number:** {case['case_number']}")
        st.markdown(f"**Petitioner/Plaintiff:** {case['petitioner']}")
        st.markdown(f"**Respondent/Defendant:** {case['respondent']}")
    
    with col2:
        st.markdown(f"**Date:** {case['date'] or 'N/A'}")
        st.markdown(f"**Location:** {case['location']}")
        st.markdown(f"**Relevant Sections/Acts:**")
        st.markdown(case['sections'])
        st.markdown(f"**Summary:** {case['summary']}")
    
    st.subheader("Additional Notes")
    case['manual_notes'] = st.text_area(
        "Add your notes here:",
        value=case['manual_notes'],
        height=150,
        key=f"notes_{st.session_state.active_case_index}"
    )
    
    if st.session_state.user_id:
        save_user_case_history(st.session_state.user_id, st.session_state.cases)

def display_case_analysis():
    """Display the case analysis section for the active case"""
    if not st.session_state.cases:
        st.info("No cases available. Please upload legal documents first.")
        return
    
    case = st.session_state.cases[st.session_state.active_case_index]
    
    st.header(f"📊 Case Analysis - {case['case_number']}")
    
    st.subheader("Case Summary")
    st.write(case['analysis']['summary'])
    
    b64_summary = base64.b64encode(f"""Case Summary for {case['case_number']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{case['analysis']['summary']}
""".encode()).decode()
    st.markdown(f'<a href="data:file/txt;base64,{b64_summary}" download="case_summary_{case["case_number"]}.txt">📥 Download Case Summary</a>', unsafe_allow_html=True)
    
    st.subheader("Key Entities")
    st.write(case['analysis']['key_entities'])
    
    st.subheader("Legal Issues")
    st.write(case['analysis']['legal_issues'])
    
    st.subheader("Recommendations")
    st.write(case['analysis']['recommendations'])

    analysis_text = f"""LegalEase AI Case Analysis Report
Case: {case['case_number']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Case Information:
- Type: {case['type_of_case']}
- Number: {case['case_number']}
- Petitioner: {case['petitioner']}
- Respondent: {case['respondent']}
- Date: {case['date']}
- Location: {case['location']}
- Sections: {case['sections']}
- Summary: {case['summary']}

Case Summary:
{case['analysis']['summary']}

Key Entities:
{case['analysis']['key_entities']}

Legal Issues:
{case['analysis']['legal_issues']}

Recommendations:
{case['analysis']['recommendations']}

Additional Notes:
{case['manual_notes']}
"""
    
    b64 = base64.b64encode(analysis_text.encode()).decode()
    st.markdown(f'<a href="data:file/txt;base64,{b64}" download="legal_case_analysis_{case["case_number"]}.txt">📥 Download Full Analysis Report</a>', unsafe_allow_html=True)

def display_image_analysis():
    """Display image analysis results for the active case"""
    if not st.session_state.cases:
        st.info("No cases available. Please upload legal documents first.")
        return
    
    case = st.session_state.cases[st.session_state.active_case_index]
    
    st.header(f"📸 Document Image Analysis - {case['case_number']}")
    
    if not case['crime_scene_summaries']:
        st.info("No images with legal relevance detected for this case.")
        return
    
    for idx, img_data in enumerate(case['crime_scene_summaries']):
        with st.expander(f"Image {idx+1} from {img_data['file_name']}"):
            st.subheader("OCR Extracted Text")
            st.text(img_data['ocr_text'] or "No text detected")
            
            st.subheader("AI Analysis")
            st.write(img_data['llm_summary'])
            
            image_summary_text = f"""Image Analysis for {case['case_number']}
Image: {img_data['file_name']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OCR Extracted Text:
{img_data['ocr_text'] or 'No text detected'}

AI Analysis:
{img_data['llm_summary']}
"""
            b64_img = base64.b64encode(image_summary_text.encode()).decode()
            st.markdown(f'<a href="data:file/txt;base64,{b64_img}" download="image_summary_{case['case_number']}_{idx+1}.txt">📥 Download Image Summary</a>', unsafe_allow_html=True)

def display_legal_chat(language_code, api_key, use_external_llm):
    """Display the legal chat interface for the active case"""
    if not st.session_state.cases:
        st.info("No cases available. Please upload legal documents first.")
        return
    
    case = st.session_state.cases[st.session_state.active_case_index]
    
    st.header(f"💬 Legal Assistant Chat - {case['case_number']} ({language_code})")
    
    chat_history_text = f"""LegalEase AI Chat History for {case['case_number']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    for message in case['chat_history']:
        role = "User" if message["role"] == "user" else "Assistant"
        chat_history_text += f"{role}: {message['content']}\n\n"
    
    b64_chat = base64.b64encode(chat_history_text.encode()).decode()
    st.markdown(f'<a href="data:file/txt;base64,{b64_chat}" download="chat_history_{case['case_number']}.txt">📥 Download Chat History</a>', unsafe_allow_html=True)
    
    for idx, message in enumerate(case['chat_history']):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
                b64_audio = base64.b64encode(message["audio"]).decode()
                st.markdown(f'<a href="data:audio/mp3;base64,{b64_audio}" download="chat_response_{case['case_number']}_{idx+1}.mp3">📥 Download Audio Response</a>', unsafe_allow_html=True)
    
    chat_input = st.chat_input(f"Ask about case {case['case_number']} in {language_code}...")
    
    audio_file = st.file_uploader(
        "Or upload audio question",
        type=["mp3", "wav", "ogg"],
        key=f"audio_upload_{st.session_state.active_case_index}_{st.session_state.audio_uploader_key_counter}"
    )
    
    if chat_input or audio_file:
        if audio_file:
            audio_bytes = audio_file.getvalue()
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            
            if audio_hash != case.get("last_audio_hash"):
                with st.spinner("Processing audio..."):
                    transcribed_text = transcribe_audio(audio_bytes, language_code)
                    case["last_audio_hash"] = audio_hash
                    st.session_state.audio_uploader_key_counter += 1
                    
                    if transcribed_text.startswith("Could not understand"):
                        st.error("Could not transcribe audio. Please try again or type your question.")
                        st.session_state.debug_log.append(f"Audio transcription failed: {transcribed_text}")
                        return
                    
                    case['chat_history'].append({
                        "role": "user",
                        "content": transcribed_text,
                        "audio": audio_bytes
                    })
                    
                    with st.spinner("Researching your question..."):
                        response = handle_legal_query(transcribed_text, st.session_state.active_case_index, language_code, api_key, use_external_llm)
                        
                        audio_response = io.BytesIO()
                        try:
                            tts = gTTS(text=response, lang=language_code)
                            tts.write_to_fp(audio_response)
                            audio_response.seek(0)
                        except Exception as e:
                            st.warning(f"Couldn't generate audio: {e}")
                            st.session_state.debug_log.append(f"Audio generation error: {e}")
                            audio_response = None
                        
                        case['chat_history'].append({
                            "role": "assistant",
                            "content": response,
                            "audio": audio_response.getvalue() if audio_response else None
                        })
                        
                        if st.session_state.user_id:
                            save_user_case_history(st.session_state.user_id, st.session_state.cases)
                        
                        st.rerun()
        elif chat_input:
            case['chat_history'].append({
                "role": "user",
                "content": chat_input
            })
            
            with st.spinner("Researching your question..."):
                response = handle_legal_query(chat_input, st.session_state.active_case_index, language_code, api_key, use_external_llm)
                
                audio_response = io.BytesIO()
                try:
                    tts = gTTS(text=response, lang=language_code)
                    tts.write_to_fp(audio_response)
                    audio_response.seek(0)
                except Exception as e:
                    st.warning(f"Couldn't generate audio: {e}")
                    st.session_state.debug_log.append(f"Audio generation error: {e}")
                    audio_response = None
                
                case['chat_history'].append({
                    "role": "assistant",
                    "content": response,
                    "audio": audio_response.getvalue() if audio_response else None
                })
                
                if st.session_state.user_id:
                    save_user_case_history(st.session_state.user_id, st.session_state.cases)
                
                st.rerun()

# --- Main Processing Function ---
def process_uploaded_files(uploaded_files, api_key):
    """Process all uploaded files and extract case information"""
    if not uploaded_files:
        st.session_state.debug_log.append("No files uploaded")
        return
    
    all_text = []
    all_images = []
    
    hasher = hashlib.md5()
    for file in uploaded_files:
        hasher.update(file.getvalue())
    files_hash = hasher.hexdigest()
    
    if st.session_state.processed_file_hash == files_hash:
        st.session_state.debug_log.append("Files already processed, skipping")
        return
    
    for file in uploaded_files:
        file_bytes = file.getvalue()
        ext = file.name.split('.')[-1].lower()
        
        text_content = ""
        images = []
        
        if ext == "pdf":
            text_content, images = extract_text_from_pdf(file_bytes)
        elif ext == "docx":
            text_content, images = extract_text_from_docx(file_bytes)
        elif ext == "xlsx":
            text_content, images = extract_text_from_xlsx(file_bytes)
        elif ext in ["png", "jpg", "jpeg"]:
            images = [file_bytes]
            text_content = f"Image file: {file.name}"
        elif ext == "txt":
            text_content = file_bytes.decode("utf-8", errors="ignore")
        
        if text_content.strip() and is_legal_content(text_content):
            all_text.append(text_content)
        
        for idx, img_bytes in enumerate(images):
            ocr_text, pil_image = ocr_image(img_bytes)
            img_summary = llm_image_summary(img_bytes, api_key) if st.session_state.gemini_configured else "Image analysis requires API key"
            
            image_info = {
                "file_name": f"{file.name}_image_{idx+1}",
                "ocr_text": ocr_text,
                "llm_summary": img_summary,
                "image": pil_image
            }
            
            all_images.append(image_info)
            st.session_state.debug_log.append(
                f"Processed image {image_info['file_name']}: OCR text length={len(ocr_text)}, Summary={img_summary[:50]}..."
            )
    
    full_text = "\n\n".join(all_text)
    cases = extract_all_cases(full_text, api_key)
    
    for case in cases:
        case_text = ""
        for text in all_text:
            if case['case_number'].lower() in text.lower():
                case_text += text + "\n\n"
        
        case['chunks'] = [chunk.strip() for chunk in re.split(r'\n\s*\n', case_text) if len(chunk.strip()) > 100]
        
        case['embeddings'] = []
        for chunk in case['chunks']:
            emb = generate_embeddings(chunk, api_key) if api_key else np.zeros((768,))
            case['embeddings'].append(emb)
        
        case['crime_scene_summaries'] = []
        case_keywords = [
            case['case_number'],
            case['petitioner'],
            case['respondent'],
            case['location'],
            case['summary'],
            case['sections'],
            case['type_of_case'].lower()
        ]
        case_keywords = [kw.lower() for kw in case_keywords if kw and kw != "N/A"]
        
        for img_data in all_images:
            img_content = (img_data['ocr_text'].lower() + " " + img_data['llm_summary'].lower())
            match_scores = [fuzz.partial_ratio(kw, img_content) for kw in case_keywords]
            if any(score > 70 for score in match_scores) or any(kw in img_content for kw in case_keywords):
                case['crime_scene_summaries'].append(img_data)
                st.session_state.debug_log.append(
                    f"Assigned image {img_data['file_name']} to case {case['case_number']} (scores: {match_scores})"
                )
        
        case['analysis'] = generate_case_analysis(case_text, case, api_key)
        case['chat_history'] = []
        case['manual_notes'] = ""
    
    st.session_state.cases = cases
    st.session_state.active_case_index = 0 if cases else 0
    st.session_state.processed_file_hash = files_hash
    st.session_state.debug_log.append(
        f"Processed {len(cases)} cases with {sum(len(c['crime_scene_summaries']) for c in cases)} images"
    )
    st.success(f"✅ Successfully processed {len(cases)} cases with {sum(len(c['crime_scene_summaries']) for c in cases)} images")
    
    if st.session_state.user_id:
        save_user_case_history(st.session_state.user_id, st.session_state.cases)

# --- Main Application Flow ---
def main():
    """Main application flow"""
    if not st.session_state.authenticated and not st.session_state.reset_token:
        st.title("⚖️ LegalEase AI – Multi-Case Legal Analysis")
        st.info("Please log in or sign up to access LegalEase AI.")
        display_auth_forms()
        return
    
    if st.session_state.reset_token:
        display_auth_forms()
        return
    
    st.title(f"⚖️ LegalEase AI – Welcome, {st.session_state.username}")
    st.sidebar.title("🛠️ Controls")
    
    display_user_profile()
    
    if st.sidebar.button("🚪 Logout", key="logout_button"):
        logout()
        return
    
    selected_language = st.sidebar.selectbox(
        "🌐 Select Language for Voice Input/Output",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,
        help="Choose the language for audio input (speech-to-text) and output (text-to-speech)."
    )
    language_code = LANGUAGE_OPTIONS[selected_language]
    
    api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password", help="Required for AI features")
    
    use_external_llm = st.sidebar.checkbox(
        "Allow External LLM Queries",
        value=True,
        help="Enable to allow general legal queries and speculative judgments."
    )
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.gemini_configured = True
        except Exception as e:
            st.sidebar.error(f"Failed to configure Gemini API: {e}")
            st.session_state.gemini_configured = False
    else:
        st.session_state.gemini_configured = False
    
    st.sidebar.header("📁 Upload Case Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload legal documents (PDF, DOCX, XLSX, TXT, Images)",
        type=["pdf", "docx", "xlsx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload all relevant case documents for analysis"
    )
    
    if uploaded_files and st.session_state.gemini_configured:
        with st.spinner("Processing documents..."):
            process_uploaded_files(uploaded_files, api_key)
    
    display_case_selection()
    
    if st.session_state.cases:
        case = st.session_state.cases[st.session_state.active_case_index]
        tab1, tab2, tab3, tab4 = st.tabs([
            f"📋 {case['case_number']} Info", 
            f"📊 {case['case_number']} Analysis", 
            f"📸 {case['case_number']} Images", 
            f"💬 {case['case_number']} Chat"
        ])
    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Case Info", 
            "📊 Case Analysis", 
            "📸 Image Analysis", 
            "💬 Legal Chat"
        ])
    
    with tab1:
        display_case_info()
    
    with tab2:
        display_case_analysis()
    
    with tab3:
        display_image_analysis()
    
    with tab4:
        display_legal_chat(language_code, api_key, use_external_llm)
    
    if st.sidebar.checkbox("Show debug information"):
        st.sidebar.write("### Debug Info")
        st.sidebar.json({
            "num_cases": len(st.session_state.cases),
            "active_case_index": st.session_state.active_case_index,
            "file_hash": st.session_state.processed_file_hash,
            "debug_log": st.session_state.get('debug_log', []),
            "authenticated": st.session_state.authenticated,
            "username": st.session_state.username,
            "user_id": st.session_state.user_id,
            "user_details": st.session_state.user_details
        })

if __name__ == "__main__":
    main()