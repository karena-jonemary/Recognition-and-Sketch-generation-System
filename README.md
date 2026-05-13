# CCTV Based Recognition and Sketch Generation System

An AI-powered intelligent surveillance system that performs real-time face detection, face recognition, unknown person identification, and sketch generation from CCTV video streams.

---

## 📌 Features

- 🎥 Live CCTV / Webcam Video Monitoring
- 🖼️ Video Upload & Frame Extraction
- 😀 Face Detection using dlib & OpenCV
- 🧠 Face Recognition using 128-D Encoding
- 🗂️ Facial Database Management with MongoDB
- 🚨 Unknown Person Identification
- 📍 Facial Landmark Extraction
- ✏️ AI-Based Sketch Generation
- 🔍 Face Restoration using GFPGAN

---

## 🛠️ Technologies Used

### Frontend
- HTML
- CSS
- JavaScript
- Bootstrap

### Backend
- Python
- Flask

### Database
- MongoDB

### Computer Vision & AI
- OpenCV
- dlib
- face_recognition
- NumPy
- GFPGAN

---

## 📂 Project Structure

```bash
CCTV_PROJECT/
│
├── app.py
├── requirements.txt
│
├── core/
│   ├── detector.py
│   ├── recognizer.py
│   ├── sketcher.py
│   ├── preprocessor.py
│   └── landmarks.py
│
├── static/
│   ├── uploads/
│   ├── results/
│   └── database/
│
├── templates/
│   ├── index.html
│   ├── upload.html
│   └── dashboard.html
│
└── models/
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/CCTV-Analytics-Platform.git
cd CCTV-Analytics-Platform
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate Virtual Environment:

#### Windows
```bash
venv\Scripts\activate
```

#### Linux / Mac
```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Start MongoDB

Make sure MongoDB is running on:

```bash
mongodb://localhost:27017/
```

---

### 5️⃣ Run Application

```bash
python app.py
```

---

## 🌐 Open in Browser

```bash
http://127.0.0.1:5000
```

---

## 📊 Modules

<img width="901" height="410" alt="image" src="https://github.com/user-attachments/assets/58d0317d-d7ce-415d-9ad1-401d8ef7ded7" />

---

## 🧠 Algorithms Used

- Histogram of Oriented Gradients (HOG)
- 128-D Face Encoding
- Euclidean Distance Algorithm
- Image Inversion & Dodge Blending
- Generative Adversarial Network (GFPGAN)

---

## 📈 Results

- Real-time face detection achieved high accuracy.
- Known individuals were recognized successfully.
- Unknown faces were processed for sketch generation.
- Sketch generation improved forensic investigation support.

---

## 🎯 Future Enhancements

- Multi-camera support
- Cloud database integration
- Criminal record integration
- Real-time alert notification system
- Advanced AI sketch enhancement

---

## 👨‍💻 Team Meambers

KARENA JONEMARY J
ABINAYA M
DHANUSUYA K

---

## 📜 License

This project is developed for educational and research purposes.
