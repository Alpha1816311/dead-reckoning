# IDR live integration guide

This repository now has two complementary paths:

- `main.py`, `ali3.py`, `ali4.py`, and `plot_results.py` are the supplied
  batch-analysis workflow. They were kept intact.
- `app.py` and `navigation_engine.py` are the incremental runtime used by an
  Android phone, an external IMU, or a replay client.

## 1. Start the backend in VS Code

Open the `IDR_Project` folder itself in VS Code. In the integrated terminal:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

On this Replit workspace the equivalent interpreter is
`.pythonlibs/bin/python`.

Verify from the laptop:

```bash
curl http://127.0.0.1:8000/health
```

Open `http://127.0.0.1:8000/` in a browser for the presentation dashboard.
It has working **Run full outage demo**, **Simulate outage**, **Restore GNSS**,
and **Reset engine** buttons. The buttons send packets through the live API;
they are not a pre-recorded animation. API metadata is available at
`/api` and interactive API documentation at `/docs`.

The backend is bound to `0.0.0.0` for LAN access. The phone must **not** use
`127.0.0.1` or `localhost`; it must use the laptop's Wi-Fi/LAN IPv4 address,
for example `http://192.168.1.42:8000`.

Find the address:

- Windows: `ipconfig`
- macOS/Linux: `ifconfig` or `ip addr`

Allow Python/port 8000 through the laptop firewall if the phone cannot reach
`/health`. Keep both devices on the same Wi-Fi network.

## 2. Test the live contract without a phone

```bash
curl -X POST http://127.0.0.1:8000/sensor/gnss \
  -H "Content-Type: application/json" \
  -d '{"timestamp":1.0,"latitude":12.0,"longitude":77.0,"speed":8.0,"accuracy":5.0}'

curl -X POST http://127.0.0.1:8000/sensor/imu \
  -H "Content-Type: application/json" \
  -d '{"timestamp":1.01,"ax":0,"ay":0,"az":9.80665,"gx":0,"gy":0,"gz":0}'
```

The response includes position, speed, heading, `gnss_state`, `mode`,
uncertainty, NHC state, map state, and accepted/rejected sample counters.
The low-latency alternative is `ws://<laptop-ip>:8000/ws/sensor`; each JSON
message has the same fields plus `"type": "imu"` or `"type": "gnss"`.

## 3. Configure phone mounting

The default assumes the phone's +Y points toward vehicle forward and +Z points
up. If the holder is different, send the measured unit vectors once:

```bash
curl -X POST http://127.0.0.1:8000/config/alignment \
  -H "Content-Type: application/json" \
  -d '{"forward_phone":[0,1,0],"up_phone":[0,0,1]}'
```

The phone frame, vehicle frame, and local east/north navigation frame are
documented in `ARCHITECTURE.md`.

## 4. Build and connect the Android collector

The supplied ZIP did not contain an Android source tree, so the collector is
provided under `android/`.

1. Open `IDR_Project/android/` in Android Studio.
2. Let Gradle sync and install the Android SDK/API 35 if prompted.
3. Connect the phone with USB debugging enabled, or use an emulator.
4. Build and run `app`.
5. Grant location permission.
6. In the app, replace the default URL with the laptop LAN URL, such as
   `http://192.168.1.42:8000`.
7. Start the backend first, verify the phone can open
   `http://<laptop-ip>:8000/health`, then press **START LIVE SENSORS**.

The app reads accelerometer, gyroscope, magnetometer when available, and GPS.
It sends an IMU packet every 100 ms and GNSS packets as Android delivers them.
The UI reports backend connection, GNSS status, navigation mode, speed,
heading, NHC, map status, and uncertainty.

## 5. Use the real supplied data replay

This exercises the same incremental engine as the API and simulates a
30-second GNSS outage:

```bash
python demo_replay.py \
  --input Data/S-S1.csv \
  --outage-start 30 \
  --outage-duration 30 \
  --output Data/live_replay.jsonl
```

To use the exported Random Forest after training:

```bash
python train_speed_model.py
# macOS/Linux:
IDR_SPEED_MODEL=models/speed_model.joblib python demo_replay.py
# Windows PowerShell:
$env:IDR_SPEED_MODEL="models/speed_model.joblib"; python demo_replay.py
```

When `models/speed_model.joblib` is present, the FastAPI app loads it
automatically. `IDR_SPEED_MODEL` is only needed to point at a different model.

`train_speed_model.py` uses an 80/20 chronological holdout and writes
`models/speed_model_metrics.json`. The included metrics are from the supplied
CSV only. IO-VNBD results remain pending until the actual dataset is supplied.

## 6. Offline map support

No road GeoJSON was included in the supplied project. When you have one,
start the server with:

```bash
IDR_ROADS_GEOJSON=/path/to/roads.geojson \
  python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The existing `map_matching.py` matcher is reused. If no map is configured,
the API honestly reports `map_status: "UNAVAILABLE"` rather than inventing a
match.

## 7. What is measured today

The supplied legacy batch run completed on 51,746 rows. Its simulated outage
reported 19.817 m maximum/final error over an 11.377 m reference displacement,
or 174.18% final drift. That is above the <10% target and is not presented as
success. Run the replay and your own phone/road data before claiming a
benchmark result.