import cv2
import numpy as np
import face_recognition
from utils import load_known_faces
from core.landmark_extractor import extract_landmarks
from core.sketcher import landmarks_to_sketch, generate_sketch

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.known_encodings, self.known_names, _ = load_known_faces()
        self.current_detections = []
        
    def __del__(self):
        if self.video:
            self.video.release()
            
    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
    def reduce_blur(self, image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
        
    def get_frame(self):
        if not self.video or not self.video.isOpened():
            return b'', []
            
        success, image = self.video.read()
        if not success or image is None:
            return b'', self.current_detections
            
        # 1. Blur Reduction
        image = self.reduce_blur(image)
        # 2. Contrast Enhancement
        image = self.enhance_contrast(image)
        
        # Convert BGR -> RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Face Detection & Recognition
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        detections_info = []

        for face_loc, face_enc in zip(face_locations, face_encodings):
            top, right, bottom, left = face_loc
            h, w = bottom - top, right - left

            name = "Unknown"

            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, face_enc)
                best_idx = int(np.argmin(distances))
                if distances[best_idx] <= 0.5:
                    name = self.known_names[best_idx]

            detections_info.append({"name": name, "bbox": [top, right, bottom, left]})

            if name == "Unknown":
                face_roi = image[top:bottom, left:right]
                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    try:
                        landmark_results = extract_landmarks(image, face_boxes=[face_loc], model="hog")
                    except Exception:
                        landmark_results = []

                    if landmark_results:
                        landmark_payload = landmark_results[0]
                        active_landmarks = landmark_payload.get("visible_landmarks") or landmark_payload.get("landmarks") or {}
                        adjusted = {
                            grp: [(x - left, y - top) for (x, y) in pts]
                            for grp, pts in active_landmarks.items()
                            if pts
                        }
                        if adjusted:
                            sketch = landmarks_to_sketch(face_roi, adjusted)
                        else:
                            sketch = generate_sketch(face_roi)
                    else:
                        sketch = generate_sketch(face_roi)
                    image[top:bottom, left:right] = sketch

                cv2.circle(image, (left + int(w*0.3), top + int(h*0.35)), 2, (0, 100, 255), -1)
                cv2.circle(image, (left + int(w*0.7), top + int(h*0.35)), 2, (0, 100, 255), -1)
                cv2.circle(image, (left + int(w*0.5), top + int(h*0.55)), 2, (0, 100, 255), -1)
                cv2.circle(image, (left + int(w*0.35), top + int(h*0.75)), 2, (0, 100, 255), -1)
                cv2.circle(image, (left + int(w*0.65), top + int(h*0.75)), 2, (0, 100, 255), -1)
                cv2.circle(image, (left + int(w*0.5), top + int(h*0.8)), 2, (0, 100, 255), -1)

                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(image, "Unknown", (left, bottom + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
            else:
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        self.current_detections = detections_info

        cv2.putText(image, f"Faces Detected: {len(face_locations)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            return b'', self.current_detections
            
        return jpeg.tobytes(), detections_info
        
    def get_current_detections(self):
        return self.current_detections
