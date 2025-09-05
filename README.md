Phishing detection prototype

What this provides
- Training script and small Flask server: `train_and_serve.py` trains RF/ET/DT/SVC on `123.csv`, picks best model and serves `/predict`.
- Simple UI: `ui.html` for quick local testing (served by Flask root if you open it with the server).
- ESP32 MicroPython example: `esp32_example.md` shows how an IoT device can call the server and blink an LED.

Quick start (Windows PowerShell)
1) Create and activate a venv (optional):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Train and run server (this will train models then start Flask on port 5000):

```powershell
python .\train_and_serve.py
```

3) Open the UI in a browser: http://127.0.0.1:5000/ui.html (or open `ui.html` file and change fetch URL to http://127.0.0.1:5000/predict)

4) Update `esp32_example.md` with your PC IP and flash `main.py` to your ESP32 with MicroPython.

Notes & next steps
- I assumed `status` is the label column with 1 = phishing. If different, update `train_and_serve.py`.
- The SVC model may be slow on large datasets; training uses small defaults.
- For production use, create authentication, validation, TLS, and a robust DNS-forwarder or browser extension.

Additional notes about authentication
- A minimal signup/login flow is included for local testing: open `signup.html` to create a user or `login.html` to sign in.
- User credentials are stored (hashed) in `users.json` in the project folder. When logged in, your checks are saved per-user and available via the `/history` API.
- This is intentionally minimal and not hardened for public deployment. Use only on a trusted local network.
