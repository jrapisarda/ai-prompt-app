# app.py • Flask 3 + OpenAI chat completions + Web Search (search-preview model)

import os, datetime as dt
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, \
                        login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import os, datetime as dt
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  # NEW

# ── env vars ────────────────────────────────────────────────────────────
load_dotenv()                      # make sure .env is read *before* we look up the key

# ── Chroma  OpenAI embeddings ──────────────────────────────────────────
embed_fn = OpenAIEmbeddingFunction(          # OpenAI is the embedding model
    api_key=os.getenv("OPENAI_API_KEY"),     # set in your .env
    model_name="text-embedding-3-small"      # cheapest model that’s still good
)

# Initialize ChromaDB client (local/embedded)
#chroma_client = chromadb.Client(Settings(persist_directory="./chroma_data"        # disk path to keep vectors
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# Create (or fetch) the shared collection with the embedding function
collection = chroma_client.get_or_create_collection(
    "my_collection", embedding_function=embed_fn
)

import openai

# ── config ────────────────────────────────────────────────────────────────
load_dotenv()                                            # reads .env during dev

app = Flask(__name__)
app.config["SECRET_KEY"]              = os.getenv("FLASK_SECRET", "dev-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ── auth models ───────────────────────────────────────────────────────────
login_mgr = LoginManager(app)
login_mgr.login_view = "login"

class User(db.Model, UserMixin):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(256),               nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)

class QueryLog(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    user_id  = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    prompt   = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    ts       = db.Column(db.DateTime, default=dt.datetime.utcnow)

@login_mgr.user_loader
def load_user(uid: str):
    return User.query.get(int(uid))

# ── auth routes ───────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u = request.form["username"]
        e = request.form["email"]
        pw = request.form["password"]
        cpw = request.form["confirm"]

        if pw != cpw:
            flash("Passwords don’t match"); return redirect(url_for("register"))
        if User.query.filter_by(username=u).first():
            flash("Username already exists"); return redirect(url_for("register"))
        if User.query.filter_by(email=e).first():
            flash("Email already in use"); return redirect(url_for("register"))
        db.session.add(User(username=u, email=e, password=generate_password_hash(pw)))
        db.session.commit()
        flash("Account created — please log in"); return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user); return redirect(url_for("index"))
        flash("Invalid credentials")
    return render_template("login.html")

@app.get("/logout")
@login_required
def logout():
    logout_user(); return redirect(url_for("login"))

# ── core pages ────────────────────────────────────────────────────────────
@app.get("/")
@login_required
def index():
    return render_template("index.html")

@app.post("/ask")
@login_required
def ask():
    prompt = request.json.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt"}), 400

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ✔ pick a search-preview model
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-search-preview")

    try:
        # ✔ chat endpoint + web_search_options
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],  # Use "web_search" if production
            input=prompt
        )
        answer = response.output_text.strip() if hasattr(response, "output_text") else response.choices[0].message.content.strip()
        
                # ── NEW: save to Chroma ───────────────────────────────
        uid = f"{current_user.id}-{int(dt.datetime.utcnow().timestamp()*1000)}"
        collection.add(
            documents=[prompt + "\n\n" + answer],
            ids=[uid],
            metadatas=[{
                "user_id": current_user.id,
                "ts": dt.datetime.utcnow().isoformat()
            }]
        )
        #chroma_client.persist()           # flushes to ./chroma_data
        # ─────────────────────────────────────────────────────

        db.session.add(QueryLog(
            user_id=current_user.id, prompt=prompt, response=answer
        ))
        db.session.commit()

        return jsonify({"response": answer})

        db.session.add(QueryLog(
            user_id=current_user.id, prompt=prompt, response=answer
        ))
        db.session.commit()
        return jsonify({"response": answer})

    except Exception as e:
        app.logger.exception("OpenAI call failed")
        return jsonify({"error": str(e)}), 500

@app.get("/dashboard")
@login_required
def dashboard():
    rows = (db.session.query(QueryLog, User.username)
                     .join(User, User.id == QueryLog.user_id)
                     .order_by(QueryLog.id.desc()).limit(50).all())
    return render_template("dashboard.html", logs=rows)

# ── launch ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT",8080))
    app.run(debug=True, host="0.0.0.0", port=port)
