import os
import json
import cv2
import numpy as np
import face_recognition

from core.preprocessor import preprocess as cv_preprocess
# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

DATABASE_DIR = os.path.join("static", "uploads", "database")


def load_known_faces():
    """
    Scan DATABASE_DIR/<person_name>/ folders.
    Return (encodings_list, names_list, paths_list) where each element
    corresponds to one encoded face photo.
    """
    encodings = []
    names = []
    paths = []

    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
        return encodings, names, paths

    for person_name in os.listdir(DATABASE_DIR):
        person_dir = os.path.join(DATABASE_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            filepath = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(filepath)
            face_encs = face_recognition.face_encodings(image)

            if face_encs:
                encodings.append(face_encs[0])
                names.append(person_name)
                paths.append(f"uploads/database/{person_name}/{filename}")

    return encodings, names, paths


def get_database_people():
    """Return a list of dicts: [{name, image_count, thumbnail}, ...]"""
    people = []
    if not os.path.exists(DATABASE_DIR):
        return people

    for person_name in sorted(os.listdir(DATABASE_DIR)):
        person_dir = os.path.join(DATABASE_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        images = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        thumbnail = None
        if images:
            thumbnail = f"uploads/database/{person_name}/{images[0]}"

        people.append({
            "name": person_name,
            "photo_count": len(images),
            "image_path": thumbnail,
        })

    return people


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(video_path, output_dir, frame_interval=30, tolerance=0.5):
    """
    Process an uploaded video:
      1. Load known face encodings from the database.
      2. Extract frames at *frame_interval* intervals.
      3. Preprocess each frame (CLAHE + denoise) via core.preprocessor.
      4. Detect & encode faces.
      5. Compare against known encodings.
      6. For unknown faces: extract 68 landmarks → landmark sketch via core.sketcher.
      7. Save results.json.
    """
    known_encodings, known_names, known_paths = load_known_faces()

    # Path prefix relative to static/ for use with url_for('static', ...)
    video_rel = os.path.relpath(output_dir, "static").replace("\\", "/")

    os.makedirs(os.path.join(output_dir, "known"),   exist_ok=True)
    os.makedirs(os.path.join(output_dir, "unknown"), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    known_results   = []
    unknown_results = []
    seen_known      = {}
    unknown_counter = 0
    analyzed_count  = 0
    frame_no        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % frame_interval != 0:
            frame_no += 1
            continue

        analyzed_count += 1
        timestamp = round(frame_no / fps, 2)

        # ── Preprocess (CLAHE + denoise) via core.preprocessor ───────────
        processed = cv_preprocess(frame)

        # ── Detect & encode (dlib HOG via face_recognition) ──────────────
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for face_loc, face_enc in zip(face_locations, face_encodings):
            top, right, bottom, left = face_loc

            h, w     = processed.shape[:2]
            pad      = 20
            c_top    = max(0, top    - pad)
            c_bottom = min(h, bottom + pad)
            c_left   = max(0, left   - pad)
            c_right  = min(w, right  + pad)
            face_crop = processed[c_top:c_bottom, c_left:c_right]

            if face_crop.size == 0:
                continue

            # ── Recognition ──────────────────────────────────────────────
            name = None
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, face_enc)
                best_idx  = int(np.argmin(distances))
                if distances[best_idx] <= tolerance:
                    name = known_names[best_idx]

            if name:
                crop_filename = f"{name}_frame{frame_no}.jpg"
                crop_path     = os.path.join(output_dir, "known", crop_filename)
                cv2.imwrite(crop_path, face_crop)
                entry = {
                    "name":       name,
                    "match_path": known_paths[best_idx],
                    "crop_path":  f"{video_rel}/known/{crop_filename}",
                    "frame_no":   frame_no,
                    "frame_idx":  frame_no,
                    "timestamp":  timestamp,
                    "distance":   float(distances[best_idx]),
                }
                known_results.append(entry)
                seen_known.setdefault(name, entry)
            else:
                unknown_counter += 1
                crop_filename   = f"unknown_{unknown_counter}_frame{frame_no}.jpg"
                crop_path       = os.path.join(output_dir, "unknown", crop_filename)

                cv2.imwrite(crop_path, face_crop)

                unknown_results.append({
                    "crop_path":   f"{video_rel}/unknown/{crop_filename}",
                    "frame_no":    frame_no,
                    "frame_idx":   frame_no,
                    "timestamp":   timestamp,
                })

        frame_no += 1

    cap.release()

    results = {
        "total_frames":    total_frames,
        "frames_analyzed": analyzed_count,
        "fps":             fps,
        "known":           known_results,
        "unknown":         unknown_results,
        "known_unique":    list(seen_known.keys()),
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Sketch generation
# ---------------------------------------------------------------------------

def generate_sketch(face_roi):
    """Generate a pencil-sketch representation of a face."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inv_blur = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inv_blur, scale=256.0)

    # Add slight stylisation
    sketch = cv2.adaptiveThreshold(
        sketch, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 9, 9,
    )
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return sketch_bgr
