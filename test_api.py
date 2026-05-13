import sys
import traceback

try:
    import base64
    import json
    import cv2
    import numpy as np
    import urllib.request

    urllib.request.urlretrieve("https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg", "test_face.jpg")

    from app import app
    client = app.test_client()

    img = cv2.imread("test_face.jpg")
    ret, buf = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buf.tobytes()).decode()
    data = json.dumps({"image": "data:image/jpeg;base64," + b64})

    response = client.post('/api/analyze_frame', data=data, content_type='application/json')
    print(f"Status Code: {response.status_code}")
    print(response.get_data(as_text=True)[:2000])

except Exception as e:
    traceback.print_exc(file=sys.stdout)
