#!/usr/bin/env python3
"""
simulate_device.py

Lightweight simulator for IoT devices that POST to the model server's /predict endpoint.

Usage examples:
  python simulate_device.py --server http://127.0.0.1:5000/predict --devices 5 --requests 10

The script will try to use the 'requests' library if available and fall back to urllib.request.
It will also try to read example URLs from `123.csv` if that file exists and has a 'url' column.
"""
import argparse
import random
import threading
import time
import json
import sys
import os
from datetime import datetime


def load_domains_from_csv(path):
    try:
        import csv
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            urls = []
            for r in reader:
                if 'url' in r and r['url']:
                    urls.append(r['url'])
            return urls
    except Exception:
        return []


def make_post_function():
    # prefer 'requests' if available
    try:
        import requests

        session = requests.Session()

        def _post(url, payload, timeout=5):
            r = session.post(url, json=payload, timeout=timeout)
            try:
                return r.status_code, r.json()
            except Exception:
                return r.status_code, {'raw_text': r.text}

        return _post
    except Exception:
        import urllib.request
        import urllib.error

        def _post(url, payload, timeout=5):
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode('utf-8')
                    try:
                        j = json.loads(body)
                    except Exception:
                        j = {'raw_text': body}
                    return resp.getcode(), j
            except urllib.error.HTTPError as e:
                try:
                    return e.code, json.loads(e.read().decode('utf-8'))
                except Exception:
                    return e.code, {'error': str(e)}
            except Exception as e:
                return None, {'error': str(e)}

        return _post


def worker(device_id, server, post_func, domains, requests_per_device, min_delay, max_delay, results):
    for i in range(requests_per_device):
        url = random.choice(domains)
        payload = {'url': url}
        ts = datetime.utcnow().isoformat() + 'Z'
        status, body = post_func(server, payload)
        ok = (status == 200)
        pred = None
        score = None
        if isinstance(body, dict):
            pred = body.get('prediction')
            score = body.get('score')
        print(f"[{ts}] device={device_id} try={i+1}/{requests_per_device} url={url} status={status} pred={pred} score={score}")
        results.append({'device': device_id, 'url': url, 'status': status, 'prediction': pred, 'score': score, 'ts': ts})
        time.sleep(random.uniform(min_delay, max_delay))


def main():
    parser = argparse.ArgumentParser(description='Simulate IoT devices sending URL checks to /predict')
    parser.add_argument('--server', '-s', default=os.environ.get('SIM_SERVER', 'http://127.0.0.1:5000/predict'), help='Full /predict URL')
    parser.add_argument('--devices', '-d', type=int, default=3, help='Number of concurrent simulated devices')
    parser.add_argument('--requests', '-r', type=int, default=5, help='Requests per device')
    parser.add_argument('--min-delay', type=float, default=0.2, help='Minimum delay between requests (s)')
    parser.add_argument('--max-delay', type=float, default=1.0, help='Maximum delay between requests (s)')
    parser.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), '123.csv'), help='Optional dataset CSV to source URLs')
    args = parser.parse_args()

    # prepare domain list
    sample_domains = [
        'http://example.com/login',
        'https://google.com',
        'http://secure-payments.example.verify-login.com',
        'http://192.168.1.100/admin',
        'https://accounts.google.com/signin',
        'http://phishy-login.example.com',
        'http://bank.example.com/login?session=1',
        'https://github.com',
    ]

    domains = []
    if os.path.exists(args.csv):
        domains = load_domains_from_csv(args.csv)
    if not domains:
        domains = sample_domains

    post_func = make_post_function()

    threads = []
    results = []
    print(f"Starting simulation: server={args.server} devices={args.devices} requests/device={args.requests}")
    for d in range(args.devices):
        t = threading.Thread(target=worker, args=(d + 1, args.server, post_func, domains, args.requests, args.min_delay, args.max_delay, results), daemon=True)
        threads.append(t)
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print('Interrupted; exiting')

    # summary
    total = len(results)
    positives = sum(1 for r in results if r.get('prediction') == 1)
    errors = sum(1 for r in results if r.get('status') is None or (isinstance(r.get('status'), int) and r.get('status') >= 400))
    print('\nSimulation complete:')
    print(f'  total requests: {total}')
    print(f'  positives (prediction==1): {positives}')
    print(f'  errors (HTTP errors / connection failures): {errors}')


if __name__ == '__main__':
    main()
