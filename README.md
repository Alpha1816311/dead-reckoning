# Intelligent Dead Reckoning (IDR)

**SIH 2026 — Intelligent Dead Reckoning Navigation System**

A smartphone-based navigation prototype that combines GNSS, IMU/INS, dead reckoning and navigation processing to provide continuous positioning and movement information.

---

## 1. Project Structure

```text
dead-reckoning/
│
├── android/                         # Android application
│   └── app/
│       └── src/main/
│           └── kotlin/
│               └── com/idr/
│                   └── navigation/ # Navigation engine
│
├── tests/                           # Python test suite
│
├── app.py                           # FastAPI backend
├── navigation_engine.py             # Python navigation engine
├── gnss_ins_fusion.py               # GNSS/INS fusion
├── sensor_processing.py             # Sensor processing
├── map_matching.py                  # Map matching
├── speed_model.py                   # Speed estimation/model
│
├── requirements.txt                 # Python dependencies
└── requirements-dev.txt             # Development/test dependencies
```

### Main Android navigation code

```text
android/app/src/main/kotlin/com/idr/navigation/IDRNavigationEngine.kt
```

---

## 2. Requirements

### For Android

* Android Studio
* Android SDK
* Physical Android phone recommended for GNSS/IMU testing
* Location/GPS enabled
* Required app permissions enabled
* USB debugging enabled if running directly from Android Studio

### For Python backend

* Python 3.10+
* PowerShell / Terminal

---

# 3. Important: APK Is Independent

The **APK is an independent Android application**.

Once the APK is built and installed on the phone, it can be run directly from the phone without opening the project in Android Studio.

The source code, Python development environment and Android Studio are required for **development/testing/building**, but they are not required for normal operation of the installed APK.

---

# 4. Before Starting the APK

For the navigation system to initialize correctly, follow these steps **before starting the app**:

### 1. Turn ON Location

Make sure the phone's **Location/GPS is turned ON**.

Also make sure the application has the required location permissions.

### 2. Go to an Open-Sky Area

Before starting the application, move outdoors or to an area with a clear view of the sky.

Avoid starting the app:

* Inside a building
* In a basement
* In a covered/underground area
* In an area with heavily obstructed sky view

### 3. Wait for the Current Location

Stay in the open-sky area for approximately **5–6 seconds** so the phone can acquire the current GNSS location.

### 4. Start the Application

After the initial location is available, open the APK and start navigation.

> **Recommended flow:**
> **Location ON → Open sky → Wait 5–6 seconds → Start APK**

This initial GNSS acquisition helps the application obtain the current starting position before navigation begins.

---

# 5. Run the Python Backend

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the backend:

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The backend can be checked at:

```text
http://127.0.0.1:8000/health
```

For a physical Android phone, use the computer's local network IP instead of `127.0.0.1`.

Example:

```text
http://192.168.1.42:8000
```

The phone and computer should be connected to the same network.

---

# 6. Run Tests

Install development dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Run the complete Python test suite:

```powershell
python -m pytest -q
```

The tests are located in:

```text
tests/
```

---

# 7. Open the Android Application

Open the following folder in **Android Studio**:

```text
dead-reckoning/android
```

Allow Gradle to sync and finish indexing.

Connect the Android phone through USB with USB debugging enabled.

Then select the `app` configuration and press:

**Run ▶**

The application will be installed directly on the connected device.

---

# 8. Build the APK

From Android Studio:

**Build → Build App Bundle(s) / APK(s) → Build APK(s)**

Or from PowerShell:

```powershell
cd android
.\gradlew assembleDebug
```

The generated debug APK will be located at:

```text
android/app/build/outputs/apk/debug/app-debug.apk
```

---

# 9. APK Download

The latest test/MVP APK will also be provided here:

**[Download Latest IDR APK](https://github.com/Alpha1816311/dead-reckoning/releases/tag/v1.0.0)**


---

# 10. Basic APK Testing Flow

1. Turn **Location/GPS ON**.
2. Grant the application the required permissions.
3. Go to an **open-sky/outdoor area**.
4. Wait approximately **5–6 seconds** for the current location to be acquired.
5. Open the APK.
6. Start navigation/data collection.
7. Keep the phone stationary and check the reported speed.
8. Walk/move with the phone and check position and speed.
9. Stop and verify that the speed returns toward zero.
10. If testing GNSS outage/dead reckoning, temporarily block GNSS and observe the navigation behavior.

---

# 11. Important Notes

* The APK is **independent** and can be operated directly after installation.
* For meaningful GNSS/IMU testing, use a **physical Android device**.
* **Location/GPS must be enabled** before starting the application.
* Perform the initial location acquisition in an **open-sky area**.
* Allow approximately **5–6 seconds** for the initial GNSS position.
* Do not commit generated Android build files.
* The Android application and Python backend are separate components.
* When modifying navigation logic, run the Python test suite before pushing changes.
* Test important navigation changes on an actual device before considering them validated.

---

# 12. Quick Commands

### Run backend

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Run tests

```powershell
python -m pytest -q
```

### Build APK

```powershell
cd android
.\gradlew assembleDebug
```

### APK location

```text
android/app/build/outputs/apk/debug/app-debug.apk
```
