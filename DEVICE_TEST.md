# IDR MVP — Real Device Test Procedure

## Pre-Flight Checklist

- [ ] Python venv activated: `.venv\Scripts\activate`
- [ ] Server starts: `python run_mvp.py --mode server`
- [ ] Dashboard accessible: `http://localhost:8000/mvp`
- [ ] API health check passes: `curl http://localhost:8000/health`
- [ ] LAN IP known: `ipconfig` → IPv4 Address

---

## TEST A — STATIC PHONE (Connection Verification)

### Setup
1. Start navigation server: `python run_mvp.py --mode server`
2. Open dashboard: `http://<SERVER_IP>:8000/mvp`
3. On Android phone: open IDR app, enter `http://<SERVER_IP>:8000`, tap **Test Connection**
4. Tap **Start Collection**

### Expected Results
- [ ] Dashboard header shows **1 client** with green dot
- [ ] IMU rate shows **~10 Hz** (may vary)
- [ ] GNSS rate shows **~0.2–1 Hz** once outdoors
- [ ] Navigation mode changes from `WAITING` to `GNSS_AIDED` once outside
- [ ] Latitude/Longitude appear in dashboard
- [ ] Map marker appears at current location
- [ ] No crashes or errors in server terminal

### Verification Commands
```bash
curl http://localhost:8000/api/live_telemetry
# Should show: imu_rate_hz > 0, ws_clients_connected = 1
curl http://localhost:8000/navigation/state
# Should show: gnss_state, position
```

---

## TEST B — WALKING / HAND MOVEMENT

### Setup
Continue from Test A. Move around with the phone.

### Expected Results
- [ ] IMU events continue streaming
- [ ] Map marker updates position when GNSS available
- [ ] Mode displays `GNSS_INS_FUSED` when moving with good GNSS
- [ ] Speed value is non-zero when moving
- [ ] Heading changes when turning

### Check
```bash
curl http://localhost:8000/navigation/state | python -m json.tool
# Look for: speed_mps > 0, heading_deg changes
```

---

## TEST C — VEHICLE TEST (GNSS + INS Fusion)

### Setup
1. Mount phone securely in vehicle (dashboard or mount)
2. Drive with clear sky view

### Expected Results
- [ ] GNSS position follows road
- [ ] Speed matches vehicle speedometer approximately
- [ ] Mode shows `GNSS_INS_FUSED`
- [ ] Map track follows vehicle path
- [ ] Yaw calibration locks after a few turns (`yaw_calibration_locked: true`)
- [ ] Alignment confidence > 0.8 after driving

---

## TEST D — SIMULATED GNSS BLACKOUT (Live)

### Setup
Continue driving from Test C. Phone streaming live.

### Steps
1. In the dashboard, click **"Start GNSS Blackout"** button (yellow button in sidebar)
   — OR via API: `curl -X POST http://localhost:8000/api/blackout/start`

### Expected Results
- [ ] Dashboard shows **GNSS: BLACKOUT** in header (orange)
- [ ] Blackout duration counter increments
- [ ] Navigation mode transitions: `GNSS_INS_FUSED → DEAD_RECKONING`
- [ ] Mode Transitions panel shows the transition with timestamp
- [ ] Map vehicle marker **continues moving** (DR active)
- [ ] `GNSS suppressed` counter increments in dashboard

### Verify DR is active
```bash
curl http://localhost:8000/navigation/state | python -m json.tool
# Look for: "gnss_state": "INS_DEAD_RECKONING", "mode": "DEAD_RECKONING"
```

### Record
- Blackout start time: ___
- Last GNSS mode before blackout: ___
- DR start position: lat=___ lon=___

---

## TEST E — GNSS RESTORATION AND RECOVERY

### Setup
Continue from Test D (currently in DEAD_RECKONING).

### Steps
1. Click **"Restore GNSS"** button (green button in sidebar)
   — OR via API: `curl -X POST http://localhost:8000/api/blackout/stop`

### Expected Results
- [ ] Header changes from **GNSS: BLACKOUT** back to **GNSS: OK**
- [ ] Navigation mode transitions: `DEAD_RECKONING → REACQUISITION → GNSS_INS_FUSED`
- [ ] Both transitions visible in Mode Transitions panel
- [ ] Position correction is smooth (no large jump)
- [ ] Map marker snaps gradually toward true GNSS position
- [ ] Recovery does NOT freeze or reset to zero

### Record
```bash
curl http://localhost:8000/navigation/state | python -m json.tool
# After recovery: "gnss_state": "FUSED" or "GNSS_AIDED"
```

- DR end position: lat=___ lon=___
- GNSS position at recovery: lat=___ lon=___
- Position error at recovery: ___ m
- Time from recovery start to FUSED: ___ s

---

## TEST F — SESSION LOG VERIFICATION

### Check session log was written
```bash
# File auto-created at server start:
type Data\live_phone_session.jsonl
# Should contain JSON lines with imu/gnss events
```

### Replay the live session
```bash
# The session JSONL is not directly replayable (it uses engine output, not raw input)
# For raw replay: the demo_replay uses Data/S-S1.csv (IO-VNBD format)
# TODO: build live session replay adapter
```

---

## TEST G — REGRESSION VERIFICATION (Post-Test)

After device testing:

```bash
python -m pytest -q
# Expected: 54+ passed

python benchmark_iovnbd.py
# Expected: DR drift ~ 7.64%
```

---

## PERFORMANCE TARGETS

| Metric | Target | Typical |
|--------|--------|---------|
| Server update rate | ≥ 10 Hz | ~400 Hz (batch) |
| Android IMU rate | ~10 Hz | 10 Hz (scheduled) |
| Android GNSS rate | ~1 Hz | 0.2–1 Hz (GPS) |
| WS round-trip latency | < 100 ms | < 10 ms (LAN) |
| Mode transition delay | < gnss_timeout_s (3s) | 3–5 s |
| Recovery convergence | < 30 s | 2–5 s |
| DR drift (30s) | < 15% | ~7.6–10.4% |

---

## QUICK START

```bash
# 1. Start server
python run_mvp.py --mode server

# 2. Open dashboard in browser
# http://<IP>:8000/mvp

# 3. Android app: enter http://<IP>:8000, Start Collection

# 4. Test blackout
curl -X POST http://localhost:8000/api/blackout/start
# ... wait 10-30s ...
curl -X POST http://localhost:8000/api/blackout/stop

# 5. Check results
curl http://localhost:8000/api/live_telemetry
```

---

## KNOWN LIMITATIONS FOR MVP

- Android build requires Java 17+ and Android SDK — see `ANDROID_INTEGRATION.md`
- GNSS accuracy depends on phone hardware and sky visibility
- Indoor GNSS may be unavailable; DR will activate automatically
- Session JSONL logs engine output, not raw sensor input (replay adapter not yet built)
- Map tile loading requires internet connection on the dashboard
