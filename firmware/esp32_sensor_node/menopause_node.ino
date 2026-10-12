/*
 * menopause_node.ino — reference firmware for one body-site sensor node.
 *
 * Target: ESP32 (BLE). One node = one body site (wrist/sternum/axilla/neck/head).
 * The node samples its local sensors at the configured rate, runs light
 * on-device conditioning (drift removal), and streams compact readings over BLE
 * to the phone that carries the digital twin.
 *
 * NOT A MEDICAL DEVICE. Research reference only.
 */

#include <Wire.h>

// ---- configuration (must match config/sensors.yaml) ---------------------
static const char *NODE_ID = "sternum";
static const char *BODY_SITE = "sternum";

static const uint32_t TEMP_HZ = 4;   // skin temperature array
static const uint32_t EDA_HZ = 8;    // electrodermal activity
static const uint32_t PPG_HZ = 64;   // optical pulse (if present on node)
static const uint32_t IMU_HZ = 32;   // movement / sleep position

// ---- lightweight ring buffer for local baseline (drift) removal ---------
struct Ring {
  static const int N = 128;
  float v[N];
  int i = 0;
  int n = 0;
  float mean() {
    if (n == 0) return 0.0f;
    float s = 0.0f;
    for (int k = 0; k < n; k++) s += v[k];
    return s / n;
  }
  void push(float x) {
    v[i] = x;
    i = (i + 1) % N;
    if (n < N) n++;
  }
};

static Ring tempBaseline;
static Ring edaBaseline;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  initBle(NODE_ID);
  initSensors();
  Serial.printf("[node] %s @ %s ready\n", NODE_ID, BODY_SITE);
}

void loop() {
  // 4 Hz baseline loop; higher-rate sensors are task-driven on a real board.
  float tempC = readSkinTemp();
  float edaUs = readEda();

  tempBaseline.push(tempC);
  edaBaseline.push(edaUs);

  // drift-removed deltas (the sensitive, subtle signal)
  float tempDelta = tempC - tempBaseline.mean();
  float edaDelta = edaUs - edaBaseline.mean();

  // packet: site | temp_delta | eda_delta | optional hr/hrv
  char pkt[160];
  snprintf(pkt, sizeof(pkt),
           "{\"site\":\"%s\",\"temp_delta\":%.4f,\"eda_delta\":%.4f,\"temp_c\":%.4f}",
           BODY_SITE, tempDelta, edaDelta, tempC);
  bleNotify(pkt);

  delay(1000 / TEMP_HZ);
}

// ---- sensor stubs (replace with your driver) ----------------------------
void initSensors() { /* configure micro-thermistor array, EDA electrodes */ }
float readSkinTemp() { return 36.6f + (random(-10, 10) / 1000.0f); }
float readEda() { return 2.0f + (random(-50, 50) / 100.0f); }

// ---- BLE stubs (replace with your stack) --------------------------------
void initBle(const char *id) { (void)id; }
void bleNotify(const char *payload) { Serial.println(payload); }
