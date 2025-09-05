ESP32 MicroPython example (sketch)

This is a minimal example for an ESP32 running MicroPython. It will:
- connect to Wi-Fi
- send the current domain (hardcoded) to the Flask server at http://<PC_IP>:5000/predict
- blink the onboard LED (or a connected LED) if the model reports phishing

Python-like pseudocode for main.py on the ESP32:

```python
import network
import urequests
import time

SSID = 'YOUR_SSID'
PASSWORD = 'YOUR_PASSWORD'
SERVER = 'http://192.168.1.100:5000/predict'  # replace with your PC IP

led = Pin(2, Pin.OUT)

def connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)
    while not wlan.isconnected():
        time.sleep(1)

def check_domain(domain):
    payload = {'url': domain}
    try:
        r = urequests.post(SERVER, json=payload)
        j = r.json()
        r.close()
        return j
    except Exception as e:
        return {'error': str(e)}

connect()
while True:
    res = check_domain('http://example.com/login')
    if res.get('prediction') == 1:
        # blink fast for phishing
        for _ in range(10):
            led.on(); time.sleep(0.1); led.off(); time.sleep(0.1)
    else:
        # slow pulse for safe
        led.on(); time.sleep(0.5); led.off(); time.sleep(2)

```

Notes:
- Replace SERVER with the IP of the PC running the Flask model server.
- This example uses HTTP and exposes requests in plaintext — for deployment consider secure channels or local isolated network.
