import urllib.request
import urllib.error

with open('payload.json', 'rb') as f:
    data = f.read()

req = urllib.request.Request('http://127.0.0.1:5000/api/analyze_frame', data=data, headers={'Content-Type': 'application/json'})

try:
    resp = urllib.request.urlopen(req)
    print("STATUS:", resp.status)
    print(resp.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("STATUS:", e.code)
    print(e.read().decode('utf-8'))
except Exception as e:
    print("ERROR:", e)
