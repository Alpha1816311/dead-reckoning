# Android ↔ IDR Engine Interface Compatibility

## Status: COMPATIBLE — No schema changes required

The Android app (`MainActivity.kt` + `LiveSensorTransport.kt`) sends events to
`/ws/sensor` in a format that the production engine already accepts.

---

## IMU Event (Android → Engine)

Android sends (every 100 ms via `sendImu()`):
```json
{
  "type": "imu",
  "timestamp": 36123.456,
  "ax": 0.12, "ay": -0.05, "az": 9.81,
  "gx": 0.01, "gy": -0.02, "gz": 0.00,
  "mx": 25.0, "my": 5.0, "mz": -40.0
}
```

**Timestamp**: `SystemClock.elapsedRealtimeNanos() / 1e9` — boot-relative seconds.  
The engine's `TimestampNormalizer` accepts this domain.  
The backend monotonic clock (`_clock=time.monotonic`) is used for GNSS freshness,
not the Android device timestamp.

**Sensor fields**: `ax/ay/az` m/s², `gx/gy/gz` rad/s, `mx/my/mz` µT (optional).

**Canonical adapter** (for future use):
```python
from sensor_interface import android_imu_to_event, normalize_sensor_event
ev = android_imu_to_event(payload)   # returns IMUEvent
# or auto-detect:
ev = normalize_sensor_event(payload) # handles both Android + canonical format
```

---

## GNSS Event (Android → Engine)

Android sends on each `Location` callback:
```json
{
  "type": "gnss",
  "timestamp": 36123.456,
  "latitude": 51.5074,
  "longitude": -0.1278,
  "speed": 5.2,
  "accuracy": 4.5,
  "altitude": 23.0
}
```

**Timestamp**: `location.elapsedRealtimeNanos / 1e9` — boot-relative seconds.  
This is the **same domain** as the IMU timestamp. ✅

**Speed**: `location.speed` in m/s (Android default). ✅  
**Accuracy**: `location.accuracy` in metres. ✅

---

## WebSocket Transport

- Endpoint: `ws://<server_ip>:8000/ws/sensor`
- Protocol: acknowledged single-in-flight (one packet awaiting ACK at a time)
- IMU: newest sample wins (replaceable slot)
- GNSS: FIFO queue, capped at 4 fixes
- Reconnect: exponential backoff, 1–30 seconds

**Server response per packet**:
```json
{"ok": true, "data": { ...navigation_state... }}
```

---

## Navigation State Response (Engine → Android)

Every IMU or GNSS packet returns the full navigation state:
```json
{
  "mode": "GNSS_INS_FUSED",
  "gnss_state": "FUSED",
  "position": {"latitude": 51.5074, "longitude": -0.1278},
  "local_position_m": {"east": 123.4, "north": 567.8},
  "speed_mps": 5.2,
  "heading_deg": 270.0,
  "yaw_axis": 2,
  "alignment_confidence": 0.95,
  "uncertainty_m": 2.0,
  ...
}
```

When GNSS is lost, `gnss_state` = `"INS_DEAD_RECKONING"` and `mode` = `"DEAD_RECKONING"`.

---

## WebView Integration

The Android app embeds a WebView that loads `http://<server>:8000/`.  
The MVP demo dashboard is served at `http://<server>:8000/mvp`.

To show the MVP demo directly in the Android WebView:
```kotlin
webView.loadUrl("$backendUrl/mvp")
```

---

## Sensor Access (Verified)

| Sensor | Android API | Status |
|--------|-------------|--------|
| Accelerometer | `Sensor.TYPE_ACCELEROMETER` | ✅ Registered |
| Gyroscope | `Sensor.TYPE_GYROSCOPE` | ✅ Registered |
| Magnetometer | `Sensor.TYPE_MAGNETIC_FIELD` | ✅ Registered (optional) |
| GNSS | `LocationManager.GPS_PROVIDER` | ✅ Registered |

---

## Required Build Command

```bash
./gradlew assembleDebug
```

Requires: Android SDK, Java 17+, Gradle 8+.

**Currently blocked by**: Android SDK / Java availability in the build environment.

The source is complete. Build and install when tooling is available:
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## Real Device Test Procedure

1. Start engine: `python run_mvp.py --mode server`
2. Note LAN IP: `ipconfig` → IPv4 address (e.g. `192.168.1.42`)
3. Install APK on phone
4. Enter `http://192.168.1.42:8000` in app URL field, tap "Test Connection"
5. Tap "Start Collection" — sensors begin streaming
6. Drive/walk. Confirm GNSS updates in GNSS status display
7. Simulate GNSS blackout: call `POST /config` with `gnss_timeout_s=0.1`,  
   then stop sending GNSS (or block GPS provider in Android Settings)
8. Verify `mode = DEAD_RECKONING` in navigation state
9. Restore GNSS: re-enable GPS, watch `REACQUISITION → GNSS_INS_FUSED`
10. View live dashboard at `http://192.168.1.42:8000/mvp` in browser
