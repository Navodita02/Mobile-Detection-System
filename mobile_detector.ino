/*
  ╔══════════════════════════════════════════════════════════════╗
  ║   Mobile Detection System — ESP32 + LM358 Firmware          ║
  ║   Hardware: LM358 RF circuit output → ESP32 GPIO            ║
  ║   Sends HTTP POST to Python AI backend server               ║
  ╚══════════════════════════════════════════════════════════════╝

  WIRING (LM358 → ESP32):
  ─────────────────────────────────────────────────
    LM358 Pin 1 (Output) → ESP32 GPIO 34 (analog input)
    LM358 Pin 8 (+Vcc)   → ESP32 3.3V (or external 5V)
    LM358 Pin 4 (GND)    → ESP32 GND
    Buzzer/LED           → Already connected to LM358 Pin 1 (works independently)

  HOW IT WORKS:
    The LM358 comparator output swings HIGH when RF detected.
    ESP32 reads this analog voltage on GPIO34 (ADC1_CH6).
    ESP32 also does its own WiFi probe scan for richer data.
    All detections are POSTed to your PC backend via HTTP.

  LIBRARIES (install via Arduino Library Manager):
    - ArduinoJson  by Benoit Blanchon  (v6.x)
    - HTTPClient   built-in with ESP32 package
    - BLEDevice    built-in with ESP32 package

  BOARD: ESP32 Dev Module  |  Upload speed: 115200  |  Flash: 4MB
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "esp_wifi.h"
#include <BLEDevice.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>

// ══════════════════════════════════════════════════════════════
//  ★  CONFIGURE THESE THREE LINES  ★
// ══════════════════════════════════════════════════════════════
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* BACKEND_URL   = "http://192.168.1.100:5000/api/detect"; // Your PC's local IP
// ══════════════════════════════════════════════════════════════

// ── Hardware Pins ──────────────────────────────────────────────
const int LM358_OUT_PIN = 34;   // GPIO34 = ADC1_CH6 (input-only, safe for analog)
const int ESP_LED_PIN   = 2;    // Built-in blue LED on most ESP32 boards
const int EXT_LED_PIN   = 26;   // Optional: external LED (connect to GPIO26 + 220Ω + GND)
const int EXT_BUZZER    = 27;   // Optional: second buzzer on GPIO27 (PWM capable)

// ── Detection Tuning ───────────────────────────────────────────
const int   LM358_THRESHOLD    = 2000;  // ADC value (0-4095). Tune based on your circuit.
                                         // Start at 2000, lower if too sensitive, raise if missing detections.
const int   RSSI_THRESHOLD     = -90;   // Ignore WiFi networks weaker than this (dBm)
const int   WIFI_SCAN_INTERVAL = 4000;  // ms between WiFi scans
const int   BLE_SCAN_SECS      = 3;     // seconds for BLE scan
const bool  ENABLE_BLE         = true;  // enable Bluetooth Low Energy scan
const bool  ENABLE_LM358_READ  = true;  // enable reading LM358 analog output

// ── Globals ────────────────────────────────────────────────────
unsigned long lastWifiScan = 0;
unsigned long lastBleScan  = 0;
int detectionCount         = 0;
BLEScan* pBLEScan          = nullptr;

// Running stats for anomaly data
int   rfReadings[20];
int   rfReadIndex = 0;
float rfAvg       = 0;
float rfStdDev    = 0;

// ══════════════════════════════════════════════════════════════
//  SETUP
// ══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println(F("\n╔══════════════════════════════════════╗"));
  Serial.println(F("║  Mobile Detection System  v2.0       ║"));
  Serial.println(F("║  ESP32 + LM358 RF Circuit            ║"));
  Serial.println(F("╚══════════════════════════════════════╝"));

  // Pin setup
  pinMode(ESP_LED_PIN, OUTPUT);
  pinMode(EXT_LED_PIN, OUTPUT);
  ledcSetup(0, 2000, 8);           // PWM channel 0, 2kHz, 8-bit for buzzer
  ledcAttachPin(EXT_BUZZER, 0);

  analogReadResolution(12);        // ESP32 ADC = 12-bit (0-4095)
  analogSetAttenuation(ADC_11db); // Full 0-3.3V range on ADC

  // Init RF readings array
  for (int i = 0; i < 20; i++) rfReadings[i] = 0;

  // Connect WiFi
  connectWiFi();

  // Init BLE
  if (ENABLE_BLE) {
    BLEDevice::init("MobileDetector");
    pBLEScan = BLEDevice::getScan();
    pBLEScan->setActiveScan(true);
    pBLEScan->setInterval(100);
    pBLEScan->setWindow(99);
    Serial.println(F("[BLE] Scanner initialized"));
  }

  Serial.printf("[CONFIG] LM358 threshold: %d / 4095\n", LM358_THRESHOLD);
  Serial.printf("[CONFIG] Backend: %s\n", BACKEND_URL);
  Serial.println(F("[READY] Detection system active!\n"));

  alertBlink(3);
}

// ══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ══════════════════════════════════════════════════════════════
void loop() {
  // Reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println(F("[WiFi] Lost connection — reconnecting..."));
    connectWiFi();
    return;
  }

  unsigned long now = millis();

  // 1. Read LM358 analog output continuously (fastest check)
  if (ENABLE_LM358_READ) {
    readLM358Signal(now);
  }

  // 2. WiFi network scan
  if (now - lastWifiScan >= WIFI_SCAN_INTERVAL) {
    lastWifiScan = now;
    scanWiFiDevices();
  }

  // 3. BLE scan (every 10 seconds to avoid overlap)
  if (ENABLE_BLE && now - lastBleScan >= 10000) {
    lastBleScan = now;
    scanBLEDevices();
  }

  delay(50);
}

// ══════════════════════════════════════════════════════════════
//  LM358 RF SIGNAL READER
//  Reads the op-amp comparator output voltage on GPIO34
//  Sends alert when analog value exceeds threshold
// ══════════════════════════════════════════════════════════════
void readLM358Signal(unsigned long now) {
  static unsigned long lastRfPost = 0;
  static int           lastRfVal  = 0;

  int raw = analogRead(LM358_OUT_PIN);

  // Update rolling stats for anomaly detection
  rfReadings[rfReadIndex % 20] = raw;
  rfReadIndex++;
  if (rfReadIndex >= 20) {
    // Compute mean
    float sum = 0;
    for (int i = 0; i < 20; i++) sum += rfReadings[i];
    rfAvg = sum / 20.0;
    // Compute std dev
    float varSum = 0;
    for (int i = 0; i < 20; i++) varSum += pow(rfReadings[i] - rfAvg, 2);
    rfStdDev = sqrt(varSum / 20.0);
  }

  // Only post if: above threshold AND either value changed significantly OR 5s elapsed
  bool aboveThreshold = (raw > LM358_THRESHOLD);
  bool valueChanged   = (abs(raw - lastRfVal) > 100);
  bool timeoutPost    = (now - lastRfPost > 5000) && aboveThreshold;

  if (aboveThreshold && (valueChanged || timeoutPost)) {
    lastRfVal  = raw;
    lastRfPost = now;

    // Map ADC 0-4095 to dBm-like scale (-100 to 0)
    int rssiLike = map(raw, 0, 4095, -100, 0);

    // Is this anomalous? (spike above 2 std deviations)
    bool isAnomaly = (rfStdDev > 0) && ((raw - rfAvg) > (2.5 * rfStdDev));

    Serial.printf("[LM358] ADC: %d  →  ~%d dBm  %s\n",
                  raw, rssiLike, isAnomaly ? "⚡ ANOMALY" : "");

    // Alert LED + buzzer
    digitalWrite(EXT_LED_PIN, HIGH);
    if (rssiLike > -45) {
      ledcWriteTone(0, 1200);  // High pitched = very close
      delay(150);
      ledcWriteTone(0, 0);
    }
    delay(100);
    digitalWrite(EXT_LED_PIN, LOW);

    // Build extra metadata for AI backend
    char extra[64];
    snprintf(extra, sizeof(extra), "adc=%d,avg=%.0f,std=%.0f,anomaly=%d",
             raw, rfAvg, rfStdDev, isAnomaly ? 1 : 0);

    sendDetection("RF", "RF-Signal", "LM358-Sensor", rssiLike, extra);
  }
}

// ══════════════════════════════════════════════════════════════
//  WIFI SCANNER
//  Detects nearby phones by their WiFi probe requests
// ══════════════════════════════════════════════════════════════
void scanWiFiDevices() {
  Serial.println(F("\n[WiFi] Scanning for nearby devices..."));
  int n = WiFi.scanNetworks(false, true);  // blocking, show hidden

  if (n <= 0) {
    Serial.println(F("[WiFi] No networks found"));
    return;
  }

  Serial.printf("[WiFi] Found %d networks\n", n);

  for (int i = 0; i < n; i++) {
    int rssi = WiFi.RSSI(i);
    if (rssi < RSSI_THRESHOLD) continue;

    String ssid    = WiFi.SSID(i);
    String bssid   = WiFi.BSSIDstr(i);
    int    channel = WiFi.channel(i);

    if (ssid.length() == 0) ssid = "[Hidden Network]";

    Serial.printf("  %-28s  %s  ch%02d  %d dBm\n",
                  ssid.c_str(), bssid.c_str(), channel, rssi);

    if (rssi > -50) alertBlink(2);

    char extra[32];
    snprintf(extra, sizeof(extra), "ch%d", channel);
    sendDetection("WiFi", ssid.c_str(), bssid.c_str(), rssi, extra);
  }

  WiFi.scanDelete();
}

// ══════════════════════════════════════════════════════════════
//  BLE SCANNER
//  Detects phones, earbuds, smartwatches via Bluetooth LE
// ══════════════════════════════════════════════════════════════
void scanBLEDevices() {
  if (!pBLEScan) return;
  Serial.println(F("\n[BLE] Scanning for Bluetooth devices..."));

  BLEScanResults results = pBLEScan->start(BLE_SCAN_SECS, false);
  int count = results.getCount();
  Serial.printf("[BLE] Found %d devices\n", count);

  for (int i = 0; i < count; i++) {
    BLEAdvertisedDevice device = results.getDevice(i);
    int rssi = device.getRSSI();
    if (rssi < RSSI_THRESHOLD) continue;

    String name    = device.haveName() ? String(device.getName().c_str()) : "[BLE Device]";
    String address = String(device.getAddress().toString().c_str());

    // Detect device type from advertising data
    String extra = "BLE";
    if (device.haveServiceUUID()) extra = "BLE-SVC";
    if (device.haveManufacturerData()) extra = "BLE-MFR";

    Serial.printf("  %-28s  %s  %d dBm  %s\n",
                  name.c_str(), address.c_str(), rssi, extra.c_str());

    if (rssi > -50) alertBlink(3);

    sendDetection("Bluetooth", name.c_str(), address.c_str(), rssi, extra.c_str());
  }

  pBLEScan->clearResults();
}

// ══════════════════════════════════════════════════════════════
//  HTTP POST TO AI BACKEND
// ══════════════════════════════════════════════════════════════
void sendDetection(const char* type, const char* name,
                   const char* deviceId, int rssi, const char* extra) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(BACKEND_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);

  StaticJsonDocument<320> doc;
  doc["type"]      = type;
  doc["name"]      = name;
  doc["device_id"] = deviceId;
  doc["rssi"]      = rssi;
  doc["extra"]     = extra;
  doc["esp_id"]    = WiFi.macAddress();

  // Include raw LM358 ADC reading for AI processing
  if (strcmp(type, "RF") == 0) {
    doc["raw_adc"]   = analogRead(LM358_OUT_PIN);
    doc["rf_avg"]    = rfAvg;
    doc["rf_stddev"] = rfStdDev;
  }

  String body;
  serializeJson(doc, body);

  int code = http.POST(body);
  if (code == 200) {
    detectionCount++;
    Serial.printf("  → Posted OK (#%d)\n", detectionCount);
  } else {
    Serial.printf("  → HTTP error: %d (check BACKEND_URL)\n", code);
  }
  http.end();
}

// ══════════════════════════════════════════════════════════════
//  WIFI CONNECTION
// ══════════════════════════════════════════════════════════════
void connectWiFi() {
  Serial.printf("\n[WiFi] Connecting to '%s'", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 30) {
    delay(500);
    Serial.print(".");
    tries++;
    digitalWrite(ESP_LED_PIN, tries % 2);
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] ✓ Connected!  IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("[WiFi] MAC: %s\n", WiFi.macAddress().c_str());
    digitalWrite(ESP_LED_PIN, HIGH);
  } else {
    Serial.println(F("\n[WiFi] ✗ Failed — will retry in loop"));
    digitalWrite(ESP_LED_PIN, LOW);
  }
}

// ══════════════════════════════════════════════════════════════
//  ALERT LED BLINK
// ══════════════════════════════════════════════════════════════
void alertBlink(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(ESP_LED_PIN, HIGH);
    delay(80);
    digitalWrite(ESP_LED_PIN, LOW);
    delay(80);
  }
}
