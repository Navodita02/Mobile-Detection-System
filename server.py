"""
╔══════════════════════════════════════════════════════════════════╗
║  Mobile Detection System — AI-Powered Python Backend            ║
║  Features: ML Classification · Anomaly Detection · Distance Est ║
╚══════════════════════════════════════════════════════════════════╝

SETUP:
    pip install flask flask-cors numpy scikit-learn

RUN:
    python server.py

Dashboard: http://localhost:5000
ESP32 POST: http://<YOUR_PC_IP>:5000/api/detect

Find your PC IP:
    Windows:   ipconfig
    Mac/Linux: ifconfig | grep inet
"""

import json
import math
import queue
import random
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
HOST        = "0.0.0.0"
PORT        = 5000
MAX_HISTORY = 1000

# ══════════════════════════════════════════════════════════════
#  AI / ML ENGINE
# ══════════════════════════════════════════════════════════════

class SignalClassifier:
    """
    Rule-based + statistical classifier.
    Classifies detections as: PHONE | DEVICE | NOISE | UNKNOWN
    Uses RSSI, signal type, name patterns, and temporal behaviour.
    No training data needed — works out of the box.
    """

    PHONE_KEYWORDS = [
        'iphone', 'galaxy', 'pixel', 'redmi', 'oneplus', 'realme',
        'poco', 'vivo', 'oppo', 'nokia', 'motorola', 'xiaomi',
        'samsung', 'huawei', 'nothing', 'asus', 'lg', 'sony',
        'mi ', 'phone', 'mobile', 'android', 'ios',
    ]

    ACCESSORY_KEYWORDS = [
        'buds', 'earphone', 'airpod', 'headphone', 'watch', 'band',
        'fit', 'gear', 'bt speaker', 'jbl', 'boat', 'galaxy buds',
        'tws', 'wearable', 'tracker',
    ]

    INFRA_KEYWORDS = [
        'router', 'tp-link', 'netgear', 'asus router', 'd-link',
        'tenda', 'jio', 'bsnl', 'airtel', 'broadband', 'fiber',
        'linksys', 'cisco', 'mikrotik', 'ubiquiti',
    ]

    def classify(self, dtype: str, name: str, rssi: int, extra: str = "") -> dict:
        name_lower = name.lower()
        confidence = 0.5
        label      = "UNKNOWN"
        reason     = ""

        if dtype == "RF":
            # LM358 circuit: raw RF signal
            if rssi > -40:
                label, confidence, reason = "PHONE", 0.82, "Strong RF — active call/data likely"
            elif rssi > -65:
                label, confidence, reason = "DEVICE", 0.65, "Moderate RF — device transmitting"
            else:
                label, confidence, reason = "NOISE", 0.70, "Weak RF — background noise or distant"

        elif dtype == "WiFi":
            if any(kw in name_lower for kw in self.PHONE_KEYWORDS):
                label, confidence, reason = "PHONE", 0.91, "Device name matches known phone brand"
            elif any(kw in name_lower for kw in self.INFRA_KEYWORDS):
                label, confidence, reason = "INFRASTRUCTURE", 0.88, "Known router/AP keyword"
            elif name_lower == "[hidden network]":
                label, confidence, reason = "DEVICE", 0.60, "Hidden SSID — possibly a phone hotspot"
            elif rssi > -55:
                label, confidence, reason = "DEVICE", 0.72, "Strong WiFi signal — close device"
            else:
                label, confidence, reason = "DEVICE", 0.55, "WiFi signal — type unclear"

        elif dtype == "Bluetooth":
            if any(kw in name_lower for kw in self.PHONE_KEYWORDS):
                label, confidence, reason = "PHONE", 0.93, "BT name matches known phone brand"
            elif any(kw in name_lower for kw in self.ACCESSORY_KEYWORDS):
                label, confidence, reason = "ACCESSORY", 0.88, "Bluetooth accessory (earbuds/watch)"
            elif name_lower == "[ble device]":
                label, confidence, reason = "DEVICE", 0.58, "Anonymous BLE beacon"
            else:
                label, confidence, reason = "DEVICE", 0.65, "Unknown BT device"

        # Boost confidence if very strong RSSI (device is very close)
        if rssi > -45 and label not in ("NOISE", "INFRASTRUCTURE"):
            confidence = min(0.97, confidence + 0.10)
            reason += " + Very strong signal"

        return {
            "label":      label,
            "confidence": round(confidence, 2),
            "reason":     reason,
            "is_phone":   label == "PHONE",
        }


class AnomalyDetector:
    """
    Z-score based anomaly detector on RSSI values per signal type.
    Flags sudden spikes or unusual patterns.
    """

    def __init__(self, window=30):
        self.window  = window
        self.history = {"WiFi": deque(maxlen=window),
                        "Bluetooth": deque(maxlen=window),
                        "RF": deque(maxlen=window)}

    def update_and_check(self, dtype: str, rssi: int) -> dict:
        buf = self.history.get(dtype, self.history["RF"])
        buf.append(rssi)

        if len(buf) < 5:
            return {"anomaly": False, "z_score": 0.0, "reason": "Collecting baseline"}

        arr    = np.array(buf)
        mean   = float(np.mean(arr))
        std    = float(np.std(arr)) or 1.0
        z      = (rssi - mean) / std
        is_anom = abs(z) > 2.5

        reason = ""
        if is_anom:
            if z > 0:
                reason = f"Signal spike ({rssi} dBm vs avg {mean:.0f} dBm)"
            else:
                reason = f"Signal drop ({rssi} dBm vs avg {mean:.0f} dBm)"

        return {
            "anomaly":  is_anom,
            "z_score":  round(z, 2),
            "mean":     round(mean, 1),
            "std":      round(std, 1),
            "reason":   reason,
        }


class DistanceEstimator:
    """
    Path-loss model for distance estimation from RSSI.
    d = 10 ^ ((TxPower - RSSI) / (10 * n))
    n = path-loss exponent (2.0 free space, 3.5 indoor)
    """

    TX_POWER = {
        "WiFi":      -40,  # dBm at 1m reference
        "Bluetooth": -59,
        "RF":        -35,
    }
    PATH_LOSS_N = {
        "WiFi":      2.7,
        "Bluetooth": 2.5,
        "RF":        3.0,
    }

    def estimate(self, dtype: str, rssi: int) -> dict:
        tx  = self.TX_POWER.get(dtype, -40)
        n   = self.PATH_LOSS_N.get(dtype, 2.7)
        exp = (tx - rssi) / (10 * n)
        dist = round(10 ** exp, 2)

        if dist < 1:
            zone, color = "CRITICAL", "red"
        elif dist < 3:
            zone, color = "CLOSE", "orange"
        elif dist < 8:
            zone, color = "NEARBY", "yellow"
        else:
            zone, color = "FAR", "green"

        return {
            "distance_m": dist,
            "zone":       zone,
            "zone_color": color,
        }


class TrendAnalyzer:
    """
    Tracks detection rate over time.
    Returns detections/minute and trend direction.
    """

    def __init__(self):
        self.timestamps = deque(maxlen=200)

    def record(self, ts: datetime):
        self.timestamps.append(ts)

    def analyze(self) -> dict:
        now = datetime.now()
        # Count in last 1 min and last 5 min
        last1  = sum(1 for t in self.timestamps if (now - t).seconds < 60)
        last5  = sum(1 for t in self.timestamps if (now - t).seconds < 300)
        rate1  = last1
        rate5  = round(last5 / 5, 1)

        if rate1 > rate5 * 1.5:
            trend = "RISING"
        elif rate1 < rate5 * 0.5:
            trend = "FALLING"
        else:
            trend = "STABLE"

        return {
            "per_min_now":  rate1,
            "per_min_avg5": rate5,
            "trend":        trend,
        }


# ══════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════
detections   = deque(maxlen=MAX_HISTORY)
sse_clients  = []
clients_lock = threading.Lock()

# AI engines
classifier = SignalClassifier()
anomaly_det = AnomalyDetector(window=30)
dist_est    = DistanceEstimator()
trend_an    = TrendAnalyzer()

stats = {
    "total":       0,
    "wifi":        0,
    "bluetooth":   0,
    "rf":          0,
    "alerts":      0,
    "phones":      0,
    "anomalies":   0,
    "started":     datetime.now().isoformat(),
}

# Trend history for chart (last 60 data points)
trend_history = deque(maxlen=60)

# ══════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder=".")
CORS(app)

# ── SSE Helpers ────────────────────────────────────────────────
def push_to_clients(payload: dict):
    msg = f"data: {json.dumps(payload)}\n\n"
    with clients_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)


def record_detection(dtype, name, device_id, rssi, extra="", esp_id="", raw_data=None):
    """Core detection recorder — runs all AI pipelines."""
    stats["total"] += 1
    key = dtype.lower()
    if key in stats:
        stats[key] += 1

    now = datetime.now()
    trend_an.record(now)

    # ── AI Pipeline ────────────────────────────────────────────
    ai_class  = classifier.classify(dtype, name, rssi, extra)
    ai_anomaly = anomaly_det.update_and_check(dtype, rssi)
    ai_dist   = dist_est.estimate(dtype, rssi)
    ai_trend  = trend_an.analyze()

    # Update phone count
    if ai_class["is_phone"]:
        stats["phones"] += 1

    # Alert logic: very close OR phone detected OR anomaly
    alert = (rssi > -45) or ai_class["is_phone"] or ai_anomaly["anomaly"]
    if alert:
        stats["alerts"] += 1

    if ai_anomaly["anomaly"]:
        stats["anomalies"] += 1

    # Trend snapshot (for chart)
    trend_history.append({
        "t":     now.strftime("%H:%M:%S"),
        "count": stats["total"],
        "rate":  ai_trend["per_min_now"],
    })

    entry = {
        "id":        stats["total"],
        "type":      dtype,
        "name":      name,
        "device_id": device_id,
        "rssi":      rssi,
        "extra":     extra,
        "esp_id":    esp_id,
        "alert":     alert,
        "timestamp": now.isoformat(),
        "raw":       raw_data or {},
        # AI results
        "ai": {
            "classification": ai_class,
            "anomaly":        ai_anomaly,
            "distance":       ai_dist,
            "trend":          ai_trend,
        },
    }

    detections.appendleft(entry)

    push_to_clients({
        "event":  "detection",
        "data":   entry,
        "stats":  dict(stats),
        "trend_history": list(trend_history)[-20:],
    })

    # Console log
    tag   = "🚨 ALERT " if alert else "       "
    anom  = "⚡ANOM" if ai_anomaly["anomaly"] else "     "
    print(
        f"[{now.strftime('%H:%M:%S')}] {tag}{anom}  "
        f"{dtype:<10} {rssi:4d} dBm  "
        f"{ai_dist['distance_m']:5.1f}m  "
        f"[{ai_class['label']:<15} {ai_class['confidence']:.0%}]  "
        f"{name[:30]}"
    )
    return entry


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if not request.is_json:
        return jsonify({"error": "JSON required"}), 400

    d = request.get_json()
    raw_data = {k: d.get(k) for k in ("raw_adc", "rf_avg", "rf_stddev") if d.get(k) is not None}

    entry = record_detection(
        dtype     = d.get("type",      "Unknown"),
        name      = d.get("name",      "Unknown"),
        device_id = d.get("device_id", "Unknown"),
        rssi      = int(d.get("rssi",  -99)),
        extra     = d.get("extra",     ""),
        esp_id    = d.get("esp_id",    ""),
        raw_data  = raw_data or None,
    )
    return jsonify({"ok": True, "id": entry["id"], "ai": entry["ai"]}), 200


@app.route("/api/status")
def api_status():
    return jsonify({
        "stats":         dict(stats),
        "recent":        list(detections)[:50],
        "trend_history": list(trend_history),
        "server":        "online",
    })


@app.route("/api/history")
def api_history():
    limit = int(request.args.get("limit", 100))
    return jsonify(list(detections)[:limit])


@app.route("/api/ai_summary")
def api_ai_summary():
    """AI analysis summary of recent detections."""
    recent = list(detections)[:100]
    if not recent:
        return jsonify({"message": "No data yet"})

    phones     = [d for d in recent if d["ai"]["classification"]["is_phone"]]
    anomalies  = [d for d in recent if d["ai"]["anomaly"]["anomaly"]]
    close_ones = [d for d in recent if d["ai"]["distance"]["zone"] in ("CRITICAL", "CLOSE")]
    rssi_vals  = [d["rssi"] for d in recent]

    return jsonify({
        "total_analyzed":   len(recent),
        "phones_detected":  len(phones),
        "anomalies":        len(anomalies),
        "close_devices":    len(close_ones),
        "avg_rssi":         round(float(np.mean(rssi_vals)), 1) if rssi_vals else 0,
        "min_rssi":         min(rssi_vals) if rssi_vals else 0,
        "max_rssi":         max(rssi_vals) if rssi_vals else 0,
        "trend":            trend_an.analyze(),
        "top_phones":       [d["name"] for d in phones[:5]],
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    detections.clear()
    trend_history.clear()
    for k in ("total", "wifi", "bluetooth", "rf", "alerts", "phones", "anomalies"):
        stats[k] = 0
    stats["started"] = datetime.now().isoformat()
    push_to_clients({"event": "cleared", "stats": dict(stats)})
    return jsonify({"ok": True})


@app.route("/stream")
def stream():
    q = queue.Queue(maxsize=100)
    with clients_lock:
        sse_clients.append(q)

    def generate():
        init = {
            "event":         "init",
            "stats":         dict(stats),
            "recent":        list(detections)[:50],
            "trend_history": list(trend_history),
        }
        yield f"data: {json.dumps(init)}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield ": ping\n\n"
        except GeneratorExit:
            pass
        finally:
            with clients_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ══════════════════════════════════════════════════════════════
#  DEMO MODE (remove when real ESP32 is connected)
# ══════════════════════════════════════════════════════════════
DEMO_DATA = [
    ("WiFi",      "iPhone-15-Pro",     "AA:BB:CC:11:22:33", -42, "ch6"),
    ("WiFi",      "Galaxy-S24-Ultra",  "DD:EE:FF:44:55:66", -58, "ch11"),
    ("WiFi",      "Redmi-Note13",      "11:22:33:AA:BB:CC", -71, "ch1"),
    ("WiFi",      "OnePlus-12R",       "22:33:44:BB:CC:DD", -63, "ch6"),
    ("WiFi",      "[Hidden Network]",  "CC:33:44:55:66:77", -48, "ch6"),
    ("WiFi",      "TP-Link_Router",    "FF:EE:DD:CC:BB:AA", -82, "ch9"),
    ("Bluetooth", "Galaxy Buds Pro",   "AA:11:22:33:44:55", -55, "BLE"),
    ("Bluetooth", "Nothing Phone 2",   "BB:22:33:44:55:66", -47, "BLE-MFR"),
    ("Bluetooth", "[BLE Device]",      "CC:33:44:55:FF:00", -78, "BLE"),
    ("RF",        "RF-Signal",         "LM358-Sensor",      -50, "adc=2800,avg=1200.0,std=400.0,anomaly=0"),
    ("RF",        "RF-Signal",         "LM358-Sensor",      -35, "adc=3900,avg=1200.0,std=400.0,anomaly=1"),
]

def demo_thread():
    time.sleep(2)
    print("\n" + "="*60)
    print("  [DEMO] Injecting sample detections")
    print("  [DEMO] Remove demo_thread() call for production")
    print("="*60 + "\n")
    while True:
        row  = random.choice(DEMO_DATA)
        rssi = row[3] + random.randint(-6, 6)
        record_detection(row[0], row[1], row[2], rssi, row[4], "DEMO-ESP32")
        time.sleep(random.uniform(1.5, 4.0))


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Mobile Detection System — AI-Powered Backend")
    print("=" * 60)
    print(f"  Dashboard:   http://localhost:{PORT}")
    print(f"  ESP32 POST:  http://<YOUR_PC_IP>:{PORT}/api/detect")
    print(f"  AI Summary:  http://localhost:{PORT}/api/ai_summary")
    print(f"  SSE Stream:  http://localhost:{PORT}/stream")
    print("=" * 60)
    print("\nAI Modules:")
    print("  ✓ Signal Classifier   (Phone / Device / Noise / Infrastructure)")
    print("  ✓ Anomaly Detector    (Z-score based spike detection)")
    print("  ✓ Distance Estimator  (Path-loss model → metres)")
    print("  ✓ Trend Analyzer      (Detections per minute + trend)")
    print()

    # Comment out demo when real ESP32 is connected:
    threading.Thread(target=demo_thread, daemon=True).start()

    app.run(host=HOST, port=PORT, debug=False, threaded=True)
