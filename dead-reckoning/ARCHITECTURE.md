# IDR repository audit and live architecture

Audit performed on the supplied ZIP at commit `94cfdef` (branch `main`).
The ZIP contained a Git worktree with uncommitted changes; those changes were
preserved. The live work is on the `idr-live-integration` branch.

## Current state versus required state

| Capability | Verified in supplied repository | Live status after integration |
|---|---|---|
| CSV dataset loading/preprocessing | `main.py` | Preserved |
| Gravity removal | `main.py`, `ali4.py` | Reused and made incremental in `sensor_processing.py` |
| Orientation/alignment | `alignment.py`, `ali2.py`, `ali3.py` | Reused by `navigation_engine.py` |
| Random Forest speed workflow | Inline in `main.py` | Existing offline workflow preserved; live engine uses measured GNSS speed then bounded inertial propagation |
| GNSS outage simulation | Inline in `main.py` | Preserved; `demo_replay.py` provides a repeatable live-engine replay |
| GNSS/INS fusion | `gnss_ins_fusion.py` | Existing batch function preserved; live incremental fusion is in `navigation_engine.py` |
| NHC/map matching | `map_matching.py` | Optional offline map hook in live engine; no road file was present |
| FastAPI backend | Not present | `app.py` with HTTP + WebSocket sensor APIs |
| Android application | Not present in supplied ZIP | Added under `android/` |
| Tests | No test suite present | Added `tests/` for preprocessing, state transitions, API, and replay primitives |
| Model artifact/export | No model artifact or export script present | Dataset-backed export script added; results are only reported when run |
| IO-VNBD | Not present | Configurable evaluation entry point; no IO-VNBD claim is made |

## Runtime data flow

```text
Android / external IMU / replay
          |
          | timestamped JSON over HTTP or WebSocket
          v
      app.py
          v
TimestampNormalizer -> RobustIMUPreprocessor
          v
PhoneOrientation + configured phone->vehicle alignment
          v
bounded forward speed + heading propagation
          v
GNSS state machine / smooth GNSS correction
          v
optional NHC/map matching
          v
continuous NavigationState JSON
```

Timestamps are monotonic seconds. Android uses `elapsedRealtimeNanos / 1e9`
for both sensor and location events, avoiding wall-clock jumps and keeping the
streams comparable. The backend accepts seconds, milliseconds, and nanoseconds
for integrations that already emit those formats.

Coordinate conventions:

- Phone frame: Android sensor axes.
- Vehicle frame: X forward, Y left, Z up.
- Navigation frame: local east/north metres, with heading 0° = north and 90° = east.
- GNSS is converted to a local tangent-plane approximation around the first fix.

## Verified legacy baseline

The supplied `main.py` completed on the bundled `Data/S-S1.csv` (51,746 rows).
Its 30-second simulated outage reported 19.817 m maximum/final error over an
11.377 m reference displacement, or 174.18% final drift. This does **not**
meet the hackathon target and is recorded honestly rather than hidden.

The supplied repository did not contain an Android source tree, an offline road
GeoJSON file, a model artifact, or IO-VNBD data. Those remain configurable
inputs rather than fabricated results.