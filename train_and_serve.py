import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import json
from flask import Flask, request, jsonify, send_from_directory, session, redirect
from werkzeug.security import generate_password_hash, check_password_hash
import re

DATA_PATH = os.path.join(os.path.dirname(__file__), '123.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.joblib')
IMPUTER_PATH = os.path.join(os.path.dirname(__file__), 'imputer.joblib')


WHITELIST = {'google.com', 'www.google.com', 'youtube.com', 'www.youtube.com', 'microsoft.com', 'www.microsoft.com', 'github.com', 'www.github.com'}
USERS_PATH = os.path.join(os.path.dirname(__file__), 'users.json')


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if 'status' not in df.columns:
        raise RuntimeError('No status column found in CSV')
    X = df.drop(columns=['url', 'status'], errors='ignore')
    # ensure numeric features where possible
    X = X.apply(pd.to_numeric, errors='coerce')
    y = df['status']
    return X, y


def train_models(X_train, y_train):
    models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'et': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'dt': DecisionTreeClassifier(random_state=42),
    }
    # fit tree-based models (they handle imputed numeric arrays)
    for name, m in models.items():
        print('Training', name)
        m.fit(X_train, y_train)

    # SVC needs scaling; wrap in a pipeline and train inside try/except to avoid crash
    try:
        svc_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, kernel='rbf'))
        ])
        print('Training svc')
        svc_pipeline.fit(X_train, y_train)
        models['svc'] = svc_pipeline
    except Exception as e:
        print('SVC training failed:', e)
    return models
    return models


def evaluate(models, X_test, y_test):
    results = {}
    for name, m in models.items():
        preds = m.predict(X_test)
        proba = None
        try:
            # some estimators (or pipelines) expose predict_proba
            proba = m.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            # fallback: try decision_function for score-like output
            try:
                scores = m.decision_function(X_test)
                auc = roc_auc_score(y_test, scores)
            except Exception:
                auc = None
        cr = classification_report(y_test, preds, output_dict=True)
        results[name] = {'report': cr, 'auc': auc}
        print(name, 'AUC:', auc)
        print(classification_report(y_test, preds))
    return results


def pick_best(models, results):
    # pick best by ROC AUC if available, else accuracy
    best_name = None
    best_score = -1
    for name, res in results.items():
        score = res['auc'] if res['auc'] is not None else res['report']['accuracy']
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, models[best_name]


def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print('Saved model to', path)


# session secret (in a real deployment use a persistent secret)
app = Flask(__name__)
# session secret (in a real deployment use a persistent secret)
app.secret_key = os.environ.get('FLASK_SECRET') or 'dev-secret-change-me'
model = None
imputer = None
feature_columns = None

def extract_features_from_url(url, cols):
    # lightweight heuristics to populate common URL-derived features used in dataset
    u = url or ''
    u_low = u.lower()
    res = {}
    for c in cols:
        lc = c.lower()
        # default
        val = 0
        try:
            if 'length' in lc:
                val = len(u)
            elif 'dot' in lc or 'period' in lc:
                val = u.count('.')
            elif 'slash' in lc:
                val = u.count('/')
            elif 'digit' in lc or 'numbers' in lc:
                val = sum(ch.isdigit() for ch in u)
            elif 'https' in lc:
                val = 1 if u_low.startswith('https') else 0
            elif 'http' in lc:
                val = 1 if u_low.startswith('http') else 0
            elif 'at' in lc or 'has_at' in lc:
                val = 1 if '@' in u else 0
            elif 'ip' in lc:
                # crude IP-in-domain check
                import re
                val = 1 if re.search(r"\\b\d{1,3}(?:\.\d{1,3}){3}\\b", u) else 0
            elif 'www' in lc:
                val = 1 if 'www.' in u_low else 0
            else:
                # fallback numeric heuristic: presence of dash, underscore, query params
                if '-' in u: val = 1
                if '?' in u: val = 1
        except Exception:
            val = 0
        res[c] = float(val)
    return res


def load_users():
    if not os.path.exists(USERS_PATH):
        return {}
    with open(USERS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_users(u):
    with open(USERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(u, f, indent=2)


def validate_password(pw):
    # basic strength checks
    errs = []
    if len(pw) < 8:
        errs.append('Password must be at least 8 characters')
    if not any(c.islower() for c in pw):
        errs.append('Include at least one lowercase letter')
    if not any(c.isupper() for c in pw):
        errs.append('Include at least one uppercase letter')
    if not any(c.isdigit() for c in pw):
        errs.append('Include at least one digit')
    if not any(c in '!@#$%^&*()-_=+[]{};:,.<>?/' for c in pw):
        errs.append('Include at least one special character')
    return errs


def compute_display_name(username):
    # if looks like an email, derive a human name from local part (john.doe -> John Doe)
    if '@' in username:
        local = username.split('@', 1)[0]
        parts = re.split(r'[._\-]+', local)
        cleaned = [p for p in parts if p]
        if not cleaned:
            return local
        return ' '.join(p.capitalize() for p in cleaned)
    # otherwise show username in ALL CAPS
    return username.upper()


def save_history_for_user(username, entry, limit=200):
    users = load_users()
    if username not in users:
        return
    history = users[username].setdefault('history', [])
    history.insert(0, entry)
    if len(history) > limit:
        history[:] = history[:limit]
    users[username]['history'] = history
    save_users(users)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model, feature_columns
    # allow simple GET-based checks for devices that can't send JSON
    if request.method == 'GET':
        data = {'url': request.args.get('url', '')}
    else:
        data = request.get_json() or {}
    # accept either {"url": "..."} or features dict
    if feature_columns is None:
        return jsonify({'error': 'model not ready'}), 500
    if 'url' in data:
        raw_url = (data.get('url') or '').strip()
        # normalize: if user provided bare domain like 'google.com', add a scheme so urlparse works
        if raw_url and not raw_url.startswith(('http://', 'https://')):
            url = 'http://' + raw_url
        else:
            url = raw_url
        feats = extract_features_from_url(url, feature_columns)
    else:
        feats = data
    # whitelist short-circuit
    try:
        from urllib.parse import urlparse
        # use normalized url when available, fall back to raw
        check_url = (data.get('url','') if isinstance(data, dict) else '')
        if check_url and not check_url.startswith(('http://', 'https://')):
            check_url = 'http://' + check_url
        host = urlparse(check_url).hostname
        if host and host.lower() in WHITELIST:
            # persist to history for logged-in users
            username = session.get('username')
            if username:
                save_history_for_user(username, {'url': data.get('url') or '', 'pred': 0, 'score': 0.1, 't': int(pd.Timestamp.now().timestamp() * 1000)})
            return jsonify({'prediction': 0, 'score': 0.1})
    except Exception:
        pass
    row = [feats.get(c, 0) for c in feature_columns]
    arr = np.array(row).reshape(1, -1)
    arr_before = arr.copy()
    arr_after = None
    # model is saved as a pipeline (imputer + estimator) so call predict/predict_proba on raw arr
    pred = model.predict(arr)[0]
    prob = None
    try:
        prob = model.predict_proba(arr)[0, 1]
    except Exception:
        prob = None
    # debug mode: return features/arrays for inspection
    debug = False
    # allow debug via JSON payload or query string
    if isinstance(data, dict) and data.get('debug'):
        debug = True
    if request.args.get('debug') in ('1', 'true', 'yes'):
        debug = True
    if debug:
        resp = {'prediction': int(pred), 'score': prob, 'features': feats}
        try:
            resp['arr_before_impute'] = arr_before.flatten().tolist()
        except Exception:
            resp['arr_before_impute'] = None
        try:
            resp['arr_after_impute'] = (arr_after.flatten().tolist() if arr_after is not None else None)
        except Exception:
            resp['arr_after_impute'] = None
        return jsonify(resp)
    # if user logged in, persist to their history
    username = session.get('username')
    if username:
        save_history_for_user(username, {'url': data.get('url') or '', 'pred': int(pred), 'score': float(prob) if prob is not None else None, 't': int(pd.Timestamp.now().timestamp() * 1000)})
    return jsonify({'prediction': int(pred), 'score': prob})


@app.route('/ui.html')
def ui():
    return send_from_directory('.', 'ui.html')


@app.route('/login.html')
def login_page():
    return send_from_directory('.', 'login.html')


@app.route('/signup.html')
def signup_page():
    return send_from_directory('.', 'signup.html')


@app.route('/signup', methods=['POST'])
def signup():
    users = load_users()
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'missing username or password'}), 400
    if username in users:
        return jsonify({'error': 'username exists'}), 400
    errs = validate_password(password)
    if errs:
        return jsonify({'error': 'weak_password', 'details': errs}), 400
    users[username] = {'pw': generate_password_hash(password), 'history': []}
    # store a display-friendly name
    users[username]['display'] = compute_display_name(username)
    save_users(users)
    session['username'] = username
    return jsonify({'ok': True, 'username': username})


@app.route('/login', methods=['POST'])
def login():
    users = load_users()
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'missing'}), 400
    if username not in users or not check_password_hash(users[username]['pw'], password):
        return jsonify({'error': 'invalid'}), 401
    session['username'] = username
    return jsonify({'ok': True, 'username': username})


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'ok': True})


@app.route('/whoami')
def whoami():
    username = session.get('username')
    if not username:
        return jsonify({'username': None})
    users = load_users()
    user = users.get(username, {})
    return jsonify({'username': username, 'display': user.get('display')})


@app.route('/history')
def history():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'not_logged_in'}), 401
    users = load_users()
    user = users.get(username, {})
    return jsonify({'history': user.get('history', [])})


@app.route('/clear_history', methods=['POST'])
def clear_history():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'not_logged_in'}), 401
    users = load_users()
    if username in users:
        users[username]['history'] = []
        save_users(users)
    return jsonify({'ok': True})

@app.route('/')
def root():
    return send_from_directory('.', 'ui.html')


def main_train_and_serve():
    global model, feature_columns
    X, y = load_data()
    feature_columns = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Impute missing numeric values (median) before training
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_columns, index=X_test.index)

    models = train_models(X_train_imp, y_train)
    results = evaluate(models, X_test_imp, y_test)
    best_name, best_model = pick_best(models, results)
    print('Best model:', best_name)
    model = best_model
    # create a pipeline that applies imputer then the model so predict uses identical preprocessing
    from sklearn.pipeline import make_pipeline
    full_pipeline = make_pipeline(imputer, model)
    joblib.dump(full_pipeline, MODEL_PATH)
    # serve model
    print('Starting Flask server on http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    # attempt to load existing model + imputer if present to avoid retraining each start
    try:
        if os.path.exists(MODEL_PATH):
            full_pipeline = joblib.load(MODEL_PATH)
            # pipeline's last step is the estimator; we keep a reference for predict_proba and predict
            model = full_pipeline
        # try to set feature_columns from training CSV if available
        if os.path.exists(DATA_PATH):
            Xtmp, ytmp = load_data()
            feature_columns = list(Xtmp.columns)
    except Exception as e:
        print('Startup load warning:', e)
    # if model missing or feature columns not known, run training flow
    if model is None or feature_columns is None:
        main_train_and_serve()
    else:
        print('Model pipeline loaded, starting server')
        app.run(host='0.0.0.0', port=5000)
