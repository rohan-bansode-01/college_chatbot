from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import csv, json, os, re, tempfile
import openai

# 🔹 NLP ADDITION
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "super_secret_key_123"

# ---------------- CONFIG ----------------
USERS_FILE = "users.json"
CSV_FILE = "data.csv"
UNKNOWN_CSV = "unknown_questions.csv"
CHAT_FILE = "chats.json"

openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------- USERS ----------------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)

# ---------------- CHAT STORAGE ----------------
def load_chats():
    if not os.path.exists(CHAT_FILE):
        return {}
    with open(CHAT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chats(chats):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, indent=4)

# ---------------- NORMALIZE ----------------
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ---------------- NLP CSV ANSWER (ADDED) ----------------
def get_answer_from_csv(raw_question):
    if not os.path.exists(CSV_FILE):
        return None

    normalized_question = normalize(raw_question)

    # 🔹 Load CSV
    questions = []
    answers = []

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = normalize(row.get("question", ""))
            a = row.get("answer", "")
            if q:
                questions.append(q)
                answers.append(a)

    # 1️⃣ EXACT MATCH (OLD LOGIC – PRESERVED)
    if normalized_question in questions:
        return answers[questions.index(normalized_question)]

    # 2️⃣ NLP SIMILARITY MATCH (NEW)
    corpus = questions + [normalized_question]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])

    best_score = similarity.max()
    best_index = similarity.argmax()

    # 🔹 Threshold (important)
    if best_score >= 0.45:
        return answers[best_index]

    return None

# ---------------- SAVE UNKNOWN ----------------
def save_unknown_question(raw_question):
    normalized = normalize(raw_question)

    if not os.path.exists(UNKNOWN_CSV):
        with open(UNKNOWN_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question"])
            writer.writeheader()

    existing = set()
    with open(UNKNOWN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add(normalize(row["question"]))

    if normalized not in existing:
        with open(UNKNOWN_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question"])
            writer.writerow({"question": raw_question.strip()})

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return redirect(url_for("login"))

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("dashboard"))

    users = load_users()

    if request.method == "POST":
        identity = request.form.get("identity", "").lower().strip()
        password = request.form.get("password")

        for username, data in users.items():
            if (
                identity == username.lower()
                or identity == data.get("email", "").lower()
                or identity == data.get("phone", "")
            ):
                if check_password_hash(data["password"], password):
                    session.clear()
                    session["user"] = username
                    return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["POST"])
def register():
    users = load_users()

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").lower().strip()
    phone = request.form.get("phone", "").strip()
    password = request.form.get("password")
    confirm = request.form.get("confirm")

    if password != confirm:
        return render_template("login.html", error="Passwords do not match")

    if username in users:
        return render_template("login.html", error="Username already exists")

    for u in users.values():
        if u.get("email", "").lower() == email:
            return render_template("login.html", error="Email already registered")
        if phone and u.get("phone") == phone:
            return render_template("login.html", error="Phone number already registered")

    users[username] = {
        "email": email,
        "phone": phone,
        "password": generate_password_hash(password)
    }

    save_users(users)
    return redirect(url_for("login"))

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    chats = load_chats()
    history = chats.get(session["user"], [])
    return render_template("dashboard.html", user=session["user"], history=history)

# ---------------- CHAT ----------------
@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session:
        return jsonify({"reply": "Session expired"})

    raw_question = request.json.get("message", "")
    answer = get_answer_from_csv(raw_question)

    if not answer:
        answer = "I don't know this yet. Your question has been saved 😊"
        save_unknown_question(raw_question)

    chats = load_chats()
    user = session["user"]

    chats.setdefault(user, []).append({
        "question": raw_question,
        "answer": answer
    })

    save_chats(chats)
    return jsonify({"reply": answer})

# ---------------- VOICE ----------------
@app.route("/voice", methods=["POST"])
def voice():
    if "user" not in session:
        return jsonify({"reply": "Session expired"})

    audio = request.files.get("audio")
    if not audio:
        return jsonify({"reply": "No audio received"})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                file=audio_file,
                model="gpt-4o-transcribe"
            )

        spoken_text = transcript.text
        answer = get_answer_from_csv(spoken_text)

        if not answer:
            answer = "I don't know this yet. Your question has been saved 😊"
            save_unknown_question(spoken_text)

        chats = load_chats()
        user = session["user"]

        chats.setdefault(user, []).append({
            "question": spoken_text,
            "answer": answer
        })

        save_chats(chats)
        return jsonify({"text": spoken_text, "reply": answer})

    finally:
        os.remove(tmp_path)


@app.route("/change_password", methods=["GET", "POST"])
def change_password():
    users = load_users()

    if request.method == "POST":
        identity = request.form.get("username", "").strip().lower()
        old_password = request.form.get("old_password")
        new_password = request.form.get("new_password")

        for username, data in users.items():
            if (
                identity == username.lower()
                or identity == data.get("email", "").lower()
                or identity == data.get("phone", "")
            ):
                if check_password_hash(data["password"], old_password):
                    users[username]["password"] = generate_password_hash(new_password)
                    save_users(users)
                    session.clear()
                    return redirect(url_for("login"))

        return render_template("change_password.html", error="Invalid credentials")

    return render_template("change_password.html")

@app.route("/reset_password", methods=["POST"])
def reset_password():
    try:
        data = request.get_json()
        phone = data.get("phone")
        new_password = data.get("password")

        if not phone or not new_password:
            return jsonify({"success": False, "msg": "Invalid request data"})

        users = load_users()

        for username, u in users.items():
            if u.get("phone") == phone:
                users[username]["password"] = generate_password_hash(new_password)
                save_users(users)
                return jsonify({"success": True})

        return jsonify({"success": False, "msg": "Phone number not registered"})

    except Exception as e:
        print("RESET PASSWORD ERROR:", e)
        return jsonify({"success": False, "msg": "Internal server error"})

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
