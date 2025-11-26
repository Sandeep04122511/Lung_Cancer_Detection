# app.py
import os
import uuid
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision.models import resnet18

# ---------------- APP SETUP ----------------
app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("model", exist_ok=True)  # ensure model folder exists if needed

# ---------------- LOGIN SETUP ----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
bcrypt = Bcrypt(app)

# ---------------- DATABASE SETUP ----------------
DB_NAME = "database.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Create users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    # Create history table (base)
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            result TEXT
        )
    """)

    # AUTO-MIGRATE: add created_at if missing
    c.execute("PRAGMA table_info(history)")
    columns = [col[1] for col in c.fetchall()]
    if "created_at" not in columns:
        c.execute("ALTER TABLE history ADD COLUMN created_at TEXT")
        print("✔ Auto-added missing column: created_at")

    conn.commit()
    conn.close()


init_db()

# ---------------- USER CLASS ----------------
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    conn.close()

    if user:
        return User(id=user[0], username=user[1], password=user[2])
    return None


# ---------------- MODEL LOADING ----------------
def load_model():
    """
    Load a resnet18 prepared for 2-class classification.
    If checkpoint has mismatched fc, attempt a safe partial load.
    """
    num_classes = 2
    model = resnet18(num_classes=num_classes)
    weight_path = "model/model_weights.pth"

    if not os.path.exists(weight_path):
        print("MODEL FILE MISSING:", weight_path)
        return None

    state = torch.load(weight_path, map_location="cpu")

    # Try exact load first
    try:
        model.load_state_dict(state)
        model.eval()
        print("✔ Model loaded (exact).")
        return model
    except Exception as e:
        print("⚠ Exact load failed, trying safe partial load:", e)

    # Safe partial load: only load matching keys/shapes
    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v

    model_state.update(filtered)
    model.load_state_dict(model_state)
    model.eval()
    print("✔ Model loaded (partial keys).")
    return model


model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ---------------- GRAD-CAM METHOD ----------------
def generate_gradcam(model, img_tensor, save_path):
    """
    img_tensor: (1,3,224,224) float tensor (0..1)
    Produces and saves a blended heatmap image -> save_path.
    Important: uses .detach() before converting to numpy to avoid the runtime error.
    """
    if model is None:
        raise RuntimeError("Model not loaded for Grad-CAM")

    model.eval()

    activations = []
    gradients = []

    # hooks
    def forward_hook(module, inp, out):
        # store detached outputs
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        # grad_out may be tuple
        gradients.append(grad_out[0].detach())

    # choose target convolutional layer (ResNet-18)
    target_layer = model.layer4[1].conv2
    fh = target_layer.register_forward_hook(forward_hook)
    # use full backward hook for modern PyTorch; fallback to register_backward_hook if not present
    try:
        bh = target_layer.register_full_backward_hook(backward_hook)
    except Exception:
        bh = target_layer.register_backward_hook(backward_hook)

    # prepare input for grad-cam
    x = img_tensor.clone().detach()
    x.requires_grad = True

    # forward
    out = model(x)

    # determine predicted class (works for 2-class softmax)
    if out.ndim == 1 or out.shape[1] == 1:
        # single-logit scenario (unlikely if you trained with num_classes=2)
        logits = out.view(-1)
        prob_pos = float(torch.sigmoid(logits)[0].item())
        pred_class = 1 if prob_pos >= 0.5 else 0
    else:
        probs = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
        pred_class = int(np.argmax(probs))

    # backward on predicted class score
    model.zero_grad()
    # ensure we backprop the scalar score
    score = out[0, pred_class]
    score.backward(retain_graph=False)

    # ensure we captured activations and gradients
    if len(activations) == 0 or len(gradients) == 0:
        fh.remove()
        bh.remove()
        raise RuntimeError("Grad-CAM hooks didn't capture activations/gradients.")

    # both activations[0] and gradients[0] are detached tensors
    fmap = activations[0].cpu().numpy()[0]   # (C, H, W)
    grad = gradients[0].cpu().numpy()[0]     # (C, H, W)

    # weights: global average pooling of gradients
    weights = np.mean(grad, axis=(1, 2))     # (C,)

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)  # (H, W)
    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)

    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max
    else:
        cam = np.zeros_like(cam)

    # Resize cam to input size (224x224)
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * cam_resized)
    heat_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Prepare original image: detach before converting to numpy
    img_np = img_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()  # (H,W,3)
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(img_bgr, 0.6, heat_color, 0.4, 0)

    # write blended image
    cv2.imwrite(save_path, blended)

    fh.remove()
    bh.remove()

    return save_path


# ---------------- ROUTES ----------------
@app.route("/")
@login_required
def home():
    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and bcrypt.check_password_hash(user[2], password):
            login_user(User(id=user[0], username=user[1], password=user[2]))
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username,password) VALUES (?,?)",
                      (username, password))
            conn.commit()
            flash("Registered Successfully!")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already taken.")
        finally:
            conn.close()

    return render_template("register.html")


@app.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    if request.method == "POST":

        if model is None:
            flash("Model not loaded!")
            return render_template("detect.html")

        file = request.files.get("image")
        if not file:
            flash("Please upload an image!")
            return render_template("detect.html")

        orig_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4().hex}_{orig_filename}"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        pil_img = Image.open(path).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0)  # (1,3,224,224)

        # Get probabilities for display (safe no-grad)
        with torch.no_grad():
            out = model(img_tensor)
            if out.ndim == 1 or (out.shape[1] == 1):
                prob_pos = float(torch.sigmoid(out.view(-1))[0].item())
                probs = np.array([1 - prob_pos, prob_pos])
                pred_idx = 1 if prob_pos >= 0.5 else 0
            else:
                probs = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
                pred_idx = int(np.argmax(probs))

        result = "Cancer Detected" if pred_idx == 1 else "Normal"

        # Probability graph
        graph_name = f"graph_{filename}.png"
        graph_path = os.path.join(UPLOAD_FOLDER, graph_name)
        plt.figure(figsize=(5, 4))
        plt.bar(["Normal", "Cancer"], probs)
        plt.title("Prediction Probability")
        plt.ylim(0, 1)
        plt.savefig(graph_path, bbox_inches="tight")
        plt.close()

        # Grad-CAM generation (runs its own forward/backward with grads enabled)
        heat_name = f"heatmap_{filename}.png"
        heat_path = os.path.join(UPLOAD_FOLDER, heat_name)
        try:
            generate_gradcam(model, img_tensor, heat_path)
        except Exception as e:
            print("Grad-CAM generation failed:", e)
            heat_name = None

        # Save history (with created_at)
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO history (user_id, filename, result, created_at) VALUES (?,?,?,?)",
                  (current_user.id, filename, result, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

        return render_template("detect.html",
                               result=result,
                               filename=filename,
                               graph_file=(f"uploads/{graph_name}" if graph_name else None),
                               heatmap_file=(f"uploads/{heat_name}" if heat_name else None),
                               description="Red/warmer regions indicate higher model activation (suspicion).")

    return render_template("detect.html")


@app.route("/history")
@login_required
def history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT filename, result, created_at FROM history WHERE user_id=? ORDER BY id DESC",
              (current_user.id,))
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", records=rows)

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
