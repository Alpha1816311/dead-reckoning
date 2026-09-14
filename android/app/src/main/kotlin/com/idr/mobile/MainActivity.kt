package com.idr.mobile

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.ViewGroup
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.FileProvider
import com.idr.navigation.IDRNavigationEngine
import com.idr.navigation.SessionRecorder
import com.idr.server.IDRLocalServer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * IDR Fusion — Standalone Android MVP
 *
 * Architecture:
 *  Real Android Sensors (Accel, Gyro, Mag, GNSS)
 *    ↓
 *  IDRNavigationEngine (on-device, no server)
 *    ↓
 *  IDRLocalServer (embedded HTTP on localhost:8080)
 *    ↓
 *  WebView loads advanced IDR Fusion UI from assets
 *    ↓
 *  UI polls /navigation/state at 600ms
 *
 * No USB, no laptop, no external server required.
 */
class MainActivity : Activity(), SensorEventListener, LocationListener {

    private val PERM_REQ = 42
    private val SERVER_PORT = 8080
    private val IMU_RATE = SensorManager.SENSOR_DELAY_GAME
    private val GNSS_MIN_MS = 500L

    // Core engine + server + recorder
    private val engine = IDRNavigationEngine()
    private lateinit var server: IDRLocalServer
    private lateinit var recorder: SessionRecorder

    // Android system services
    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager

    // Handler for periodic track updates
    private val handler = Handler(Looper.getMainLooper())

    // Sensor state (written by sensor callbacks, atomics for thread safety)
    private val accel = AtomicReference(FloatArray(3))
    private val gyro  = AtomicReference(FloatArray(3))
    private val mag   = AtomicReference<FloatArray?>(null)
    private val hasAccel = AtomicBoolean(false)
    private val hasGyro  = AtomicBoolean(false)
    private val hasMag   = AtomicBoolean(false)

    // Last GNSS (for session recording)
    @Volatile private var lastGnssLat: Double? = null
    @Volatile private var lastGnssLon: Double? = null
    @Volatile private var lastGnssSpeed: Double? = null
    @Volatile private var lastGnssAcc: Double? = null

    @Volatile private var navigationRunning = false
    @Volatile private var permissionPending = false

    // WebView — the advanced IDR Fusion UI
    private lateinit var webView: WebView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        recorder = SessionRecorder(this)

        // Start embedded HTTP server
        server = IDRLocalServer(SERVER_PORT, assets, engine, recorder)
        try {
            server.start()
        } catch (e: Exception) {
            Toast.makeText(this, "Server start failed: ${e.message}", Toast.LENGTH_LONG).show()
        }

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        setupWebView()
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView = WebView(this).apply {
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                allowFileAccess = false
                allowContentAccess = false
                cacheMode = WebSettings.LOAD_NO_CACHE
                // Allow mixed content (for loading OSM tiles over http from https pages)
                mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                setSupportZoom(false)
            }
            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) {
                    // Inject Android bridge for start/stop/export controls
                    injectAndroidBridge()
                    // Auto-request permissions if not yet granted
                    if (!hasLocationPermission()) {
                        evaluateJavascript(
                            "window._idrPermissionNeeded = true;", null
                        )
                    }
                }
            }
            // Load the advanced navigation UI from local server
            loadUrl("http://localhost:$SERVER_PORT/navigate")
        }
        setContentView(webView, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))
    }

    /**
     * Inject JavaScript bridge into the WebView so UI buttons can call
     * native Android functions (start nav, blackout, export, etc.)
     */
    private fun injectAndroidBridge() {
        val js = """
(function() {
  if (window._idrBridgeInjected) return;
  window._idrBridgeInjected = true;

  // Override fetch() calls to the local server — they should work natively
  // but we patch the base URL just in case the HTML has relative paths.
  const BASE = 'http://localhost:$SERVER_PORT';
  const _origFetch = window.fetch.bind(window);
  window.fetch = function(url, opts) {
    if (typeof url === 'string' && url.startsWith('/')) {
      url = BASE + url;
    }
    return _origFetch(url, opts);
  };

  // START NAVIGATION — called from any "Start" button in the UI
  window.idrStartNavigation = function() {
    Android.startNavigation();
  };

  // STOP NAVIGATION
  window.idrStopNavigation = function() {
    Android.stopNavigation();
  };

  // GNSS BLACKOUT
  window.idrStartBlackout = function() {
    fetch('/gnss/blackout', {method:'POST'});
  };

  // GNSS RESTORE
  window.idrRestoreGnss = function() {
    fetch('/gnss/restore', {method:'POST'});
  };

  // CALIBRATE
  window.idrCalibrate = function() {
    fetch('/calibrate/start', {method:'POST'});
  };

  // SESSION
  window.idrStartSession = function() {
    fetch('/session/start', {method:'POST'})
      .then(r => r.json())
      .then(d => { if (window._idrShowToast) _idrShowToast('Session started: '+d.name); });
  };

  window.idrStopSession = function() {
    fetch('/session/stop', {method:'POST'})
      .then(r => r.json())
      .then(d => { if (window._idrShowToast) _idrShowToast('Session saved: '+d.records+' records'); });
  };

  window.idrExportSession = function() {
    Android.exportSession();
  };

  // Toast helper
  window._idrShowToast = function(msg) {
    Android.showToast(msg);
  };

  // Auto-start navigation when user presses START in the UI
  // Watch for the simTunnelBtn which is the existing demo button
  var simBtn = document.getElementById('simTunnelBtn');
  if (simBtn) {
    // Replace the original demo handler with real GNSS blackout
    simBtn.onclick = function(e) {
      e.stopPropagation();
      e.preventDefault();
      window.idrStartBlackout();
    };
  }

  console.log('[IDR] Android bridge injected — standalone mode active');
})();
        """.trimIndent()
        webView.evaluateJavascript(js, null)

        // Also add the Android interface object
        webView.addJavascriptInterface(AndroidBridge(), "Android")
    }

    inner class AndroidBridge {
        @android.webkit.JavascriptInterface
        fun startNavigation() {
            runOnUiThread { onStartNavigation() }
        }

        @android.webkit.JavascriptInterface
        fun stopNavigation() {
            runOnUiThread { stopNav() }
        }

        @android.webkit.JavascriptInterface
        fun exportSession() {
            runOnUiThread { doExportSession() }
        }

        @android.webkit.JavascriptInterface
        fun showToast(msg: String) {
            runOnUiThread { Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show() }
        }
    }

    // ---------------------------------------------------------------
    // NAVIGATION START
    // ---------------------------------------------------------------
    private fun onStartNavigation() {
        permissionPending = true
        if (!hasLocationPermission()) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
                PERM_REQ
            )
            return
        }
        startNav()
    }

    @SuppressLint("MissingPermission")
    private fun startNav() {
        if (navigationRunning) return
        engine.reset()
        server.clearTrack()

        val accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroSensor  = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        if (accelSensor == null || gyroSensor == null) {
            Toast.makeText(this, "Required IMU sensors not available on this device", Toast.LENGTH_LONG).show()
            return
        }

        sensorManager.registerListener(this, accelSensor, IMU_RATE)
        sensorManager.registerListener(this, gyroSensor, IMU_RATE)
        sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_UI)
        }

        try {
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, GNSS_MIN_MS, 0f, this)
        } catch (e: Exception) {
            Toast.makeText(this, "GPS: ${e.message}", Toast.LENGTH_SHORT).show()
        }
        try {
            if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, GNSS_MIN_MS * 3, 0f, this)
            }
        } catch (_: Exception) {}

        navigationRunning = true
        handler.post(trackUpdateRunnable)
        Toast.makeText(this, "Navigation started — waiting for GNSS fix", Toast.LENGTH_SHORT).show()
    }

    private fun stopNav() {
        navigationRunning = false
        handler.removeCallbacks(trackUpdateRunnable)
        sensorManager.unregisterListener(this)
        try { locationManager.removeUpdates(this) } catch (_: Exception) {}
    }

    // ---------------------------------------------------------------
    // PERIODIC TRACK UPDATE — appends current position to track history
    // so the UI map can draw the path
    // ---------------------------------------------------------------
    private val trackUpdateRunnable = object : Runnable {
        override fun run() {
            if (!navigationRunning) return
            val state = engine.getState()
            if (state.latitude != 0.0 && state.longitude != 0.0) {
                server.appendTrack(state.latitude, state.longitude)
            }
            handler.postDelayed(this, 1000) // add a track point every second
        }
    }

    // ---------------------------------------------------------------
    // SESSION EXPORT
    // ---------------------------------------------------------------
    private fun doExportSession() {
        val file = recorder.getSessionFile() ?: run {
            Toast.makeText(this, "No session file to export", Toast.LENGTH_SHORT).show()
            return
        }
        if (!file.exists()) {
            Toast.makeText(this, "Session file not found", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            val uri = FileProvider.getUriForFile(this, "${packageName}.provider", file)
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "application/json"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(intent, "Export IDR Session"))
        } catch (e: Exception) {
            Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // ---------------------------------------------------------------
    // SENSOR CALLBACKS — real Android hardware
    // ---------------------------------------------------------------
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                val v = event.values.copyOf()
                accel.set(v)
                hasAccel.set(true)
                if (!hasGyro.get()) return
                val g = gyro.get()
                val ts = SystemClock.elapsedRealtimeNanos() / 1_000_000_000.0
                engine.processImu(
                    ax = v[0].toDouble(), ay = v[1].toDouble(), az = v[2].toDouble(),
                    gx = g[0].toDouble(), gy = g[1].toDouble(), gz = g[2].toDouble(),
                    timestamp = ts
                )
                // Session recording
                if (recorder.isRecording) {
                    val state = engine.getState()
                    val m = if (hasMag.get()) mag.get() else null
                    recorder.record(SessionRecorder.SessionRecord(
                        wallMs = System.currentTimeMillis(), timestamp = ts,
                        ax = v[0].toDouble(), ay = v[1].toDouble(), az = v[2].toDouble(),
                        gx = g[0].toDouble(), gy = g[1].toDouble(), gz = g[2].toDouble(),
                        mx = m?.get(0)?.toDouble(), my = m?.get(1)?.toDouble(), mz = m?.get(2)?.toDouble(),
                        gnssLat = lastGnssLat, gnssLon = lastGnssLon,
                        gnssSpeedMps = lastGnssSpeed, gnssAccuracyM = lastGnssAcc,
                        navLat = state.latitude, navLon = state.longitude,
                        navSpeedMps = state.speedMps, navHeadingDeg = state.headingDeg,
                        navMode = state.mode.name,
                        gnssBlackout = state.gnssBlackout, blackoutSeconds = state.blackoutSeconds
                    ))
                }
            }
            Sensor.TYPE_GYROSCOPE -> { gyro.set(event.values.copyOf()); hasGyro.set(true) }
            Sensor.TYPE_MAGNETIC_FIELD -> { mag.set(event.values.copyOf()); hasMag.set(true) }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    // ---------------------------------------------------------------
    // GNSS CALLBACKS
    // ---------------------------------------------------------------
    override fun onLocationChanged(location: Location) {
        lastGnssLat   = location.latitude
        lastGnssLon   = location.longitude
        lastGnssSpeed = if (location.hasSpeed()) location.speed.toDouble() else null
        lastGnssAcc   = location.accuracy.toDouble()
        engine.processGnss(
            latitude    = location.latitude,
            longitude   = location.longitude,
            speedMpsRaw = if (location.hasSpeed()) location.speed.toDouble() else null,
            accuracyM   = location.accuracy.toDouble(),
            timestamp   = location.elapsedRealtimeNanos / 1_000_000_000.0
        )
    }

    override fun onProviderEnabled(provider: String)  = Unit
    override fun onProviderDisabled(provider: String) = Unit
    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(p: String?, s: Int, e: Bundle?) = Unit

    // ---------------------------------------------------------------
    // PERMISSIONS
    // ---------------------------------------------------------------
    private fun hasLocationPermission() =
        ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(req: Int, perms: Array<out String>, grants: IntArray) {
        super.onRequestPermissionsResult(req, perms, grants)
        if (req != PERM_REQ) return
        if (hasLocationPermission()) startNav()
        else Toast.makeText(this, "Location permission required for GNSS navigation", Toast.LENGTH_LONG).show()
    }

    // ---------------------------------------------------------------
    // LIFECYCLE
    // ---------------------------------------------------------------
    override fun onResume() {
        super.onResume()
        // Auto-start navigation on first launch if permission already granted
        if (!navigationRunning && hasLocationPermission() && permissionPending) {
            startNav()
        }
    }

    override fun onDestroy() {
        navigationRunning = false
        handler.removeCallbacksAndMessages(null)
        sensorManager.unregisterListener(this)
        try { locationManager.removeUpdates(this) } catch (_: Exception) {}
        if (recorder.isRecording) recorder.stopSession()
        server.stop()
        webView.destroy()
        super.onDestroy()
    }

    override fun onBackPressed() {
        if (webView.canGoBack()) webView.goBack() else super.onBackPressed()
    }
}
