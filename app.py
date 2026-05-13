import os
import json
import shutil
import uuid
import threading
import base64

import cv2
import numpy as np
import face_recognition
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, Response,
)
from werkzeug.utils import secure_filename

from utils import process_video, get_database_people, DATABASE_DIR
from camera import VideoCamera
from core.recognizer import (
    init_db,
    add_person,
    delete_person as db_delete_person,
    load_all_encodings,
    recognize,
)
from core.preprocessor import preprocess
from core.detector import detect_faces
from core.encoder import encode_faces
from core.landmark_extractor import extract_landmarks
from core.sketcher import landmarks_to_sketch, generate_sketch as core_generate_sketch
from core.restoration import enhance_face
from core.face_completion import complete_face

app = Flask(__name__)
app.secret_key = "cctv-pro-secret-key"

VIDEO_UPLOAD_DIR = os.path.join("static", "uploads", "videos")
RESULTS_DIR      = os.path.join("static", "results")
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}

for d in [VIDEO_UPLOAD_DIR, RESULTS_DIR, DATABASE_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Database bootstrap
# ---------------------------------------------------------------------------

init_db()


def _sync_db_from_filesystem():
    """
    On first run (or after a manual file copy), populate SQLite from the
    face-photo folders so the analyse endpoint can find existing people.
    """
    _, existing_names = load_all_encodings()
    already_known = set(existing_names)

    if not os.path.exists(DATABASE_DIR):
        return

    for person_name in os.listdir(DATABASE_DIR):
        if person_name in already_known:
            continue
        person_dir = os.path.join(DATABASE_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for fname in os.listdir(person_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            fpath = os.path.join(person_dir, fname)
            try:
                img  = face_recognition.load_image_file(fpath)
                encs = face_recognition.face_encodings(img)
                if encs:
                    add_person(person_name, encs[0],
                               f"uploads/database/{person_name}/{fname}")
            except Exception:
                pass


_sync_db_from_filesystem()

# ---------------------------------------------------------------------------
# Camera singleton
# ---------------------------------------------------------------------------

_camera      = None
_camera_lock = threading.Lock()


def get_camera():
    global _camera
    with _camera_lock:
        if _camera is None:
            _camera = VideoCamera()
    return _camera


def _generate_frames(camera):
    while True:
        frame, _ = camera.get_frame()
        if not frame:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------------------------------------------------------------------
# Dashboard & Core Modules
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    people = get_database_people()
    
    total_videos = 0
    total_detections = 0
    unknown_alerts = 0
    
    recent_videos = []
    if os.path.exists(RESULTS_DIR):
        dirs = sorted(os.listdir(RESULTS_DIR), reverse=True)
        total_videos = len(dirs)
        
        for vid in dirs:
            rfile = os.path.join(RESULTS_DIR, vid, "results.json")
            if os.path.exists(rfile):
                with open(rfile) as f:
                    data = json.load(f)
                known_c = len(data.get("known", []))
                unknown_c = len(data.get("unknown", []))
                
                total_detections += (known_c + unknown_c)
                unknown_alerts += unknown_c
                
                if len(recent_videos) < 5:
                    data["video_id"] = vid
                    recent_videos.append(data)
                    
    return render_template(
        "index.html",
        people_count=len(people),
        total_videos=total_videos,
        total_detections=total_detections,
        unknown_alerts=unknown_alerts,
        recent_videos=recent_videos,
        sys_status="ONLINE"
    )

@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/sketch")
def sketch():
    unknown_log = []
    if os.path.exists(RESULTS_DIR):
        for vid in sorted(os.listdir(RESULTS_DIR), reverse=True):
            rfile = os.path.join(RESULTS_DIR, vid, "results.json")
            if os.path.exists(rfile):
                with open(rfile) as f: rdata = json.load(f)
                for entry in rdata.get("unknown", []):
                    entry["video_id"] = vid
                    if entry.get("crop_path", "").startswith("unknown/"):
                        entry["crop_path"] = f"results/{vid}/{entry['crop_path']}"
                    unknown_log.append(entry)
    return render_template("sketch.html", recent_unknowns=unknown_log)

@app.route("/settings")
def settings():
    history = []
    if os.path.exists(RESULTS_DIR):
        for vid in sorted(os.listdir(RESULTS_DIR), reverse=True):
            rfile = os.path.join(RESULTS_DIR, vid, "results.json")
            if not os.path.exists(rfile):
                continue
            with open(rfile) as f:
                rdata = json.load(f)
            history.append({
                "video_id":       vid,
                "known_count":    len(rdata.get("known", [])),
                "unknown_count":  len(rdata.get("unknown", [])),
                "frames_analyzed": rdata.get("frames_analyzed", 0),
                "total_frames":   rdata.get("total_frames", 0),
            })
    return render_template("settings.html", history=history)


@app.route("/settings/history/delete/<video_id>", methods=["POST"])
def history_delete(video_id):
    target = os.path.join(RESULTS_DIR, video_id)
    if os.path.exists(target):
        shutil.rmtree(target)
        flash(f"Deleted result {video_id}.", "success")
    else:
        flash("Record not found.", "error")
    return redirect(url_for("settings"))


@app.route("/settings/history/delete_all", methods=["POST"])
def history_delete_all():
    if os.path.exists(RESULTS_DIR):
        for vid in os.listdir(RESULTS_DIR):
            shutil.rmtree(os.path.join(RESULTS_DIR, vid), ignore_errors=True)
    flash("All processing history cleared.", "success")
    return redirect(url_for("settings"))

@app.route("/recognition")
def recognition():
    known_log = []
    if os.path.exists(RESULTS_DIR):
        for vid in sorted(os.listdir(RESULTS_DIR), reverse=True):
            rfile = os.path.join(RESULTS_DIR, vid, "results.json")
            if os.path.exists(rfile):
                with open(rfile) as f: rdata = json.load(f)
                for entry in rdata.get("known", []):
                    entry["video_id"] = vid
                    if "crop_path" in entry and not entry["crop_path"].startswith("results/"):
                        entry["crop_path"] = f"results/{vid}/{entry['crop_path']}"
                    if "match_path" not in entry:
                        entry["match_path"] = entry["crop_path"]
                    known_log.append(entry)
    return render_template("recognition.html", logs=known_log)

@app.route("/unknowns")
def unknowns():
    unknown_log = []
    if os.path.exists(RESULTS_DIR):
        for vid in sorted(os.listdir(RESULTS_DIR), reverse=True):
            rfile = os.path.join(RESULTS_DIR, vid, "results.json")
            if os.path.exists(rfile):
                with open(rfile) as f: rdata = json.load(f)
                for entry in rdata.get("unknown", []):
                    entry["video_id"] = vid
                    if entry.get("crop_path", "").startswith("unknown/"):
                        entry["crop_path"] = f"results/{vid}/{entry['crop_path']}"
                    unknown_log.append(entry)
    return render_template("unknowns.html", logs=unknown_log)

# ---------------------------------------------------------------------------
# Database management
# ---------------------------------------------------------------------------

@app.route("/database")
def database():
    people = get_database_people()
    return render_template("database.html", people=people)


@app.route("/database/add", methods=["POST"])
def database_add():
    name  = request.form.get("name", "").strip()
    photo = request.files.get("photo")

    if not name:
        flash("Please enter a name.", "error")
        return redirect(url_for("database"))
    if not photo or photo.filename == "":
        flash("Please upload a photo.", "error")
        return redirect(url_for("database"))

    ext = os.path.splitext(photo.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        flash("Only JPG / PNG images are allowed.", "error")
        return redirect(url_for("database"))

    person_dir = os.path.join(DATABASE_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    filename  = secure_filename(photo.filename)
    save_path = os.path.join(person_dir, filename)
    photo.save(save_path)

    # Encode face & persist to SQLite so /api/analyze_frame can find it
    try:
        img  = face_recognition.load_image_file(save_path)
        encs = face_recognition.face_encodings(img)
        if encs:
            add_person(name, encs[0], f"uploads/database/{name}/{filename}")
            flash(f"Enrolled {name} successfully.", "success")
        else:
            flash(
                f"Photo saved for {name}, but no face was detected. "
                "Recognition may not work — use a clearer frontal photo.",
                "error",
            )
    except Exception as exc:
        flash(f"Photo saved but encoding failed: {exc}", "error")

    return redirect(url_for("database"))


@app.route("/database/delete/<name>", methods=["POST"])
def database_delete(name):
    person_dir = os.path.join(DATABASE_DIR, name)
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
    db_delete_person(name)          # remove from SQLite
    flash(f"Deleted {name} from database.", "success")
    return redirect(url_for("database"))


# ---------------------------------------------------------------------------
# API: analyse a single cropped / full frame
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    return jsonify({"error": traceback.format_exc()}), 500

@app.route("/api/analyze_frame", methods=["POST"])
def analyze_frame():
    """
    POST JSON: { "image": "<base64 or data-URI>" }

    Pipeline
    --------
    1. Decode image
    2. Preprocess (CLAHE + denoise) via core.preprocessor
    3. Detect faces (dlib HOG) via core.detector
    4. Generate 128-D encoding via core.encoder
    5. Compare with SQLite database via core.recognizer
    6a. Match  → return { status:'known', name }
    6b. No match → extract 68 landmarks, build sketch via core.sketcher
                → return { status:'unknown', sketch:<base64> }
    """
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "No image data provided."}), 400

    # ── 1. Decode ────────────────────────────────────────────────────────
    img_str = payload["image"]
    if img_str.startswith("data:"):
        img_str = img_str.split(",", 1)[1]
    try:
        arr   = np.frombuffer(base64.b64decode(img_str), np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        return jsonify({"error": f"Failed to decode image: {exc}"}), 400

    if image is None or image.size == 0:
        return jsonify({"error": "Invalid or empty image."}), 400

    # ── 2. Preprocess (CLAHE + fast denoise) ─────────────────────────────
    proc = preprocess(image)

    # ── 3. Face detection (dlib HOG) ─────────────────────────────────────
    h_img, w_img = proc.shape[:2]
    
    # If the image is relatively small, it's likely already a tightly cropped face 
    # from the video processing gallery. Running HOG again on a masked crop often 
    # yields a smaller false positive (e.g. detecting just the eyes), causing 
    # the shape predictor to squish the landmarks. We bypass detection for crops.
    if h_img <= 400 and w_img <= 400:
        locations = []
    else:
        locations = detect_faces(proc, model="hog")
        
    if not locations:
        # Fallback: Assume the image is a cropped face from utils.py which adds ~20px padding.
        # We strip about 15% padding from each side so the dlib shape predictor 
        # gets an accurate bounding box, preventing the landmarks from shifting!
        py = int(h_img * 0.15)
        px = int(w_img * 0.15)
        face_loc = (py, w_img - px, h_img - py, px)
    else:
        # Pick the largest face by bounding-box area
        face_loc = max(locations, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
        
    # ── 4. Generate 128-D encoding ───────────────────────────────────────
    # Use the ORIGINAL image (not proc or GFPGAN) for recognition
    encodings = encode_faces(image, [face_loc])
    
    # ── 5. Recognise against SQLite database ─────────────────────────────
    matched_name = recognize(encodings[0], tolerance=0.50) if encodings else None
    
    print("Recognized name:", matched_name)

    if matched_name:
        return jsonify({
            "status":  "known",
            "name":    matched_name,
            "message": f"Known Person: {matched_name}",
        })
    else:
        print("Unrecognized face -> proceeding to sketch generation")

    top, right, bottom, left = face_loc

    # ── 6b. Unknown – extract 68 landmarks & render sketch ───────────────
    h_img, w_img = proc.shape[:2]
    pad      = 10
    c_top    = max(0, top    - pad)
    c_bottom = min(h_img, bottom + pad)
    c_left   = max(0, left   - pad)
    c_right  = min(w_img, right  + pad)
    face_crop = proc[c_top:c_bottom, c_left:c_right]

    # Fix Image Distortion: Crop face as square, then resize to (256, 256)
    h, w = face_crop.shape[:2]
    size = min(h, w)
    c_y, c_x = h // 2, w // 2
    half_size = size // 2
    
    crop_start_y = c_top + c_y - half_size
    crop_start_x = c_left + c_x - half_size
    
    face_crop = face_crop[c_y - half_size : c_y + half_size, c_x - half_size : c_x + half_size]
    face_crop = cv2.resize(face_crop, (256, 256))

    # Disable GFPGAN for testing
    restored_crop = face_crop
    
    # Save restored face to base64 for UI
    _, r_buf = cv2.imencode(".jpg", restored_crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    restored_b64 = "data:image/jpeg;base64," + base64.b64encode(r_buf.tobytes()).decode()

    landmark_meta = {
        "is_masked": False,
        "mask_confidence": 0.0,
        "missing_landmarks": [],
    }

    try:
        # Use proc and face_loc so dlib reliably extracts landmarks
        landmark_results = extract_landmarks(proc, face_boxes=[face_loc], model="hog")
    except Exception:
        landmark_results = []

    if landmark_results:
        landmark_payload = landmark_results[0]
        
        # Always use all landmarks so we draw the complete facial marks
        active_landmarks = landmark_payload.get("landmarks") or {}
        
        # Transform landmarks to match the square 256x256 crop
        adjusted = {}
        for grp, pts in active_landmarks.items():
            if not pts: continue
            grp_pts = []
            for (x, y) in pts:
                nx = int((x - crop_start_x) * (256.0 / size))
                ny = int((y - crop_start_y) * (256.0 / size))
                grp_pts.append((nx, ny))
            adjusted[grp] = grp_pts
        
        is_masked = bool(landmark_payload.get("is_masked", False))
        reconstructed = False
        
        if is_masked:
            print("Masked face detected -> applying reconstruction")
            pts68_orig = landmark_payload.get("all_landmarks_68", [])
            adj_pts68 = []
            for (x, y) in pts68_orig:
                nx = int((x - crop_start_x) * (256.0 / size))
                ny = int((y - crop_start_y) * (256.0 / size))
                adj_pts68.append((nx, ny))
                
            if adj_pts68:
                try:
                    input_for_sketch = complete_face(restored_crop, adj_pts68)
                    reconstructed = True
                    
                    # Store result in static/results/reconstructed/
                    recon_dir = os.path.join(RESULTS_DIR, "reconstructed")
                    os.makedirs(recon_dir, exist_ok=True)
                    recon_filename = f"{uuid.uuid4().hex[:12]}_reconstructed.jpg"
                    cv2.imwrite(os.path.join(recon_dir, recon_filename), input_for_sketch)
                    
                    # Update base64 representation with the newly reconstructed crop
                    _, r_buf = cv2.imencode(".jpg", input_for_sketch, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    restored_b64 = "data:image/jpeg;base64," + base64.b64encode(r_buf.tobytes()).decode()
                except Exception as e:
                    print(f"Face completion error: {e}")
                    input_for_sketch = restored_crop
            else:
                input_for_sketch = restored_crop
        else:
            print("Unmasked face -> using original image")
            input_for_sketch = restored_crop

        if adjusted and is_masked:
            sketch = landmarks_to_sketch(input_for_sketch, adjusted)
        else:
            sketch = core_generate_sketch(input_for_sketch)

        landmark_meta = {
            "is_masked": is_masked,
            "mask_confidence": float(landmark_payload.get("mask_confidence", 0.0)),
            "missing_landmarks": landmark_payload.get("missing_landmarks", []),
            "reconstructed": reconstructed,
        }
    else:
        print("No landmarks found -> using original image")
        input_for_sketch = restored_crop
        sketch = core_generate_sketch(input_for_sketch)

    _, buf    = cv2.imencode(".jpg", sketch, [cv2.IMWRITE_JPEG_QUALITY, 88])
    sketch_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    return jsonify({
        "status":  "unknown",
        "message": "Unknown Person",
        "sketch":  sketch_b64,
        "restored": restored_b64,
        "landmark_meta": landmark_meta,
    })


# ---------------------------------------------------------------------------
# Video upload & processing
# ---------------------------------------------------------------------------

@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    video = request.files.get("video")
    if not video or video.filename == "":
        flash("Please select a video file.", "error")
        return redirect(url_for("upload"))

    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        flash("Unsupported video format. Use MP4, AVI, MOV, MKV, or WMV.", "error")
        return redirect(url_for("upload"))

    video_id = uuid.uuid4().hex[:12]
    video_filename = f"{video_id}{ext}"
    video_path = os.path.join(VIDEO_UPLOAD_DIR, video_filename)
    video.save(video_path)

    output_dir = os.path.join(RESULTS_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    # Process synchronously (for simplicity)
    frame_interval = int(request.form.get("frame_interval", 30))
    results = process_video(video_path, output_dir, frame_interval=frame_interval)

    if "error" in results:
        flash(results["error"], "error")
        return redirect(url_for("upload"))

    if results.get("unknown"):
        first_unknown = results["unknown"][0]
        image_url = url_for("static", filename=first_unknown["crop_path"])
        return redirect(url_for("sketch", image=image_url))

    return redirect(url_for("results", video_id=video_id))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@app.route("/results/<video_id>")
def results(video_id):
    results_file = os.path.join(RESULTS_DIR, video_id, "results.json")
    if not os.path.exists(results_file):
        flash("Results not found.", "error")
        return redirect(url_for("index"))

    with open(results_file) as f:
        data = json.load(f)

    # Backwards compatibility for old result files
    for k in data.get("known", []):
        if "crop_path" in k and not k["crop_path"].startswith("results/"):
            k["crop_path"] = f"results/{video_id}/{k['crop_path']}"
        if "match_path" not in k:
            k["match_path"] = k["crop_path"]
            
    for u in data.get("unknown", []):
        if "crop_path" in u and not u["crop_path"].startswith("results/"):
            u["crop_path"] = f"results/{video_id}/{u['crop_path']}"

    return render_template("results.html", video_id=video_id, data=data)


# ---------------------------------------------------------------------------
# Person detail
# ---------------------------------------------------------------------------

@app.route("/person/<name>")
def person(name):
    person_dir = os.path.join(DATABASE_DIR, name)
    if not os.path.exists(person_dir):
        flash("Person not found.", "error")
        return redirect(url_for("database"))

    images = [
        f"uploads/database/{name}/{f}"
        for f in os.listdir(person_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Gather all result appearances for this person across all processed videos
    appearances = []
    if os.path.exists(RESULTS_DIR):
        for vid in os.listdir(RESULTS_DIR):
            rfile = os.path.join(RESULTS_DIR, vid, "results.json")
            if not os.path.exists(rfile):
                continue
            with open(rfile) as f:
                rdata = json.load(f)
            for entry in rdata.get("known", []):
                if entry["name"] == name:
                    entry["video_id"] = vid
                    if "crop_path" in entry and not entry["crop_path"].startswith("results/"):
                        entry["crop_path"] = f"results/{vid}/{entry['crop_path']}"
                    if "match_path" not in entry:
                        entry["match_path"] = entry["crop_path"]
                    appearances.append(entry)

    return render_template(
        "person.html",
        name=name,
        images=images,
        appearances=appearances,
    )


# ---------------------------------------------------------------------------
# Live camera feed
# ---------------------------------------------------------------------------

@app.route("/video_feed")
def video_feed():
    camera = get_camera()
    return Response(
        _generate_frames(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detections")
def detections():
    camera = get_camera()
    return jsonify(camera.get_current_detections())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)


