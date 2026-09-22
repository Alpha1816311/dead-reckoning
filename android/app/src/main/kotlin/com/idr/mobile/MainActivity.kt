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
import android.net.Uri
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
 *  IDRNavigationEngine (on-device, no external server)
 *    ↓
 *  IDRLocalServer (embedded HTTP on localhost:8080)
 *    ↓
 *  WebView loads IDR Fusion UI from assets
 *    ↓
 *  UI polls /navigation/state at 200ms (5 Hz)
 *
 * Sensors start immediately on launch — no user interaction required to begin.
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
    private val hasAccel = AtomicBoolean(false)
    private val hasGyro  = AtomicBoolean(false)
    private val hasMag   = AtomicBoolean(false)

    // Last GNSS (for session recording)
    @Volatile private var lastGnssLat: Double? = null
    @Volatile private var lastGnssLon: Double? = null
    @Volatile private var lastGnssSpeed: Double? = null
    @Volatile private var lastGnssAcc: Double? = null

    @Volatile private var navigationRunning = false

    // WebView — the IDR Fusion UI
    private lateinit var webView: WebView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        recorder = SessionRecorder(this)

        // Start embedded HTTP server before anything else.
        // Pass onExportRequest so the /session/export HTTP endpoint can
        // trigger the native Android share sheet (HTTP fallback path).
        server = IDRLocalServer(SERVER_PORT, assets, engine, recorder) {
            runOnUiThread { doExportSession() }
        }
        try {
            server.start()
        } catch (e: Exception) {
            Toast.makeText(this, "Server start failed: ${e.message}", Toast.LENGTH_LONG).show()
        }

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        setupWebView()

        // Auto-start sensors immediately — IMU does NOT need location permission.
        // GNSS will be requested after we check/request the permission.
        startImuSensors()
        requestLocationPermissionAndStartGnss()

        // Handle Android App Link deep-link that launched the app cold.
        // https://rkshaan.github.io/idr  →  navigate screen (already the default)
        handleDeepLink(intent)
    }

    /**
     * Called when the app is already running (singleTop) and a new
     * Android App Link intent arrives — e.g. user taps URL while app
     * is in the foreground or background.
     */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleDeepLink(intent)
    }

    /**
     * Route an incoming App Link intent to the correct screen.
     *
     * Current mapping:
     *   https://rkshaan.github.io/idr        -> /navigate  (navigate screen)
     *   https://rkshaan.github.io/idr/{any}  -> /navigate  (any sub-path -> same screen)
     *
     * The navigation engine, sensors, GNSS, DR, and calibration are all
     * on-device.  This routing only affects which WebView page is shown;
     * it does NOT require a network connection.
     */
    private fun handleDeepLink(intent: Intent) {
        if (intent.action != Intent.ACTION_VIEW) return
        val uri: Uri = intent.data ?: return
        if (uri.host != "rkshaan.github.io") return
        val path = uri.path ?: return
        if (!path.startsWith("/idr")) return

        // The app already loads /navigate on launch — if it is already
        // showing the navigate page, do nothing.  Otherwise reload it so
        // a background → foreground tap always lands on the correct screen.
        val currentUrl = webView.url ?: ""
        if (!currentUrl.contains("/navigate")) {
            webView.loadUrl("http://localhost:$SERVER_PORT/navigate")
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView = WebView(this).apply {
            // Register the Android bridge BEFORE the page loads so the
            // JavaScript object is available synchronously on first access.
            addJavascriptInterface(AndroidBridge(), "Android")
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                allowFileAccess = false
                allowContentAccess = false
                cacheMode = WebSettings.LOAD_NO_CACHE
                // Allow mixed content (OSM tiles over http from https-origin pages)
                mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                // Disable WebView page zoom entirely. Leaflet manages its own
                // pinch-zoom internally and does not rely on WebView zoom.
                setSupportZoom(false)
                builtInZoomControls = false
                displayZoomControls = false
            }
            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) {
                    // Inject helper JS aliases after the page is ready
                    injectAndroidBridge()
                }
            }
            // Load the navigation UI from the embedded server
            loadUrl("http://localhost:$SERVER_PORT/navigate")
        }
        setContentView(webView, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))
    }

    /**
     * Inject a minimal JavaScript bridge so the export button can call
     * native Android functions.  The main navigation state flows through
     * the embedded HTTP server — no bridge needed for that.
     *
     * addJavascriptInterface must be called BEFORE the page loads so the
     * object is available synchronously.  We call it during setupWebView()
     * so by the time onPageFinished fires the object already exists.
     */
    private fun injectAndroidBridge() {
        val js = """
(function() {
  if (window._idrBridgeInjected) return;
  window._idrBridgeInjected = true;

  window.idrExportSession = function() {
    if (typeof Android !== 'undefined') Android.exportSession();
  };

  window._idrShowToast = function(msg) {
    if (typeof Android !== 'undefined') Android.showToast(msg);
  };

  console.log('[IDR] Android bridge injected — standalone on-device mode');
})();
        """.trimIndent()
        webView.evaluateJavascript(js, null)
    }

    inner class AndroidBridge {
        @android.webkit.JavascriptInterface
        fun exportSession() {
            // Must not run on the WebView JS thread — dispatch to UI thread
            runOnUiThread { doExportSession() }
        }

        @android.webkit.JavascriptInterface
        fun showToast(msg: String) {
            runOnUiThread { Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show() }
        }
    }

    // ---------------------------------------------------------------
    // IMU SENSOR START — does not require location permission
    // ---------------------------------------------------------------
    private fun startImuSensors() {
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
    }

    // ---------------------------------------------------------------
    // GNSS — request permission then start
    // ---------------------------------------------------------------
    private fun requestLocationPermissionAndStartGnss() {
        if (hasLocationPermission()) {
            startGnss()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
                PERM_REQ
            )
        }
    }

    @SuppressLint("MissingPermission")
    private fun startGnss() {
        if (navigationRunning) return
        navigationRunning = true
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
        handler.post(trackUpdateRunnable)
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
        // Check if any session data exists (recording or already stopped)
        if (recorder.recordCount == 0 && recorder.getSessionFile() == null) {
            Toast.makeText(this, "NO SESSION DATA AVAILABLE TO EXPORT", Toast.LENGTH_LONG).show()
            webView.evaluateJavascript(
                "if(typeof showToast==='function') showToast('warning','NO SESSION DATA AVAILABLE TO EXPORT');", null
            )
            return
        }

        // Convert NDJSON → CSV on-device (no fake data, only real recorded telemetry)
        val csvFile = recorder.exportAsCsv()
        if (csvFile == null || !csvFile.exists()) {
            Toast.makeText(this, "NO SESSION DATA AVAILABLE TO EXPORT", Toast.LENGTH_LONG).show()
            webView.evaluateJavascript(
                "if(typeof showToast==='function') showToast('error','EXPORT FAILED — NO RECORDED DATA');", null
            )
            return
        }

        try {
            val uri = FileProvider.getUriForFile(this, "${packageName}.provider", csvFile)
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, csvFile.name)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(intent, "Export IDR Session Log"))
            webView.evaluateJavascript(
                "if(typeof showToast==='function') showToast('check_circle','SESSION EXPORTED SUCCESSFULLY');", null
            )
        } catch (e: Exception) {
            android.util.Log.e("IDR", "Export failed", e)
            Toast.makeText(this, "EXPORT FAILED: ${e.message}", Toast.LENGTH_LONG).show()
            webView.evaluateJavascript(
                "if(typeof showToast==='function') showToast('error','EXPORT FAILED: ${e.message?.replace("'", "\\'")}');", null
            )
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
                    val mx = if (hasMag.get()) event.values.getOrNull(0)?.toDouble() else null
                    recorder.record(SessionRecorder.SessionRecord(
                        wallMs = System.currentTimeMillis(), timestamp = ts,
                        ax = v[0].toDouble(), ay = v[1].toDouble(), az = v[2].toDouble(),
                        gx = g[0].toDouble(), gy = g[1].toDouble(), gz = g[2].toDouble(),
                        mx = null, my = null, mz = null,
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
            Sensor.TYPE_MAGNETIC_FIELD -> {
                hasMag.set(true)
                engine.processMag(event.values[0], event.values[1], event.values[2])
            }
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
        if (hasLocationPermission()) {
            startGnss()
        } else {
            Toast.makeText(this, "Location permission required for GNSS navigation", Toast.LENGTH_LONG).show()
        }
    }

    // ---------------------------------------------------------------
    // LIFECYCLE
    // ---------------------------------------------------------------
    override fun onResume() {
        super.onResume()
        // If GNSS not yet running but permission was granted (e.g. via Settings), start now
        if (!navigationRunning && hasLocationPermission()) {
            startGnss()
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

    @Deprecated("Deprecated in Java")
    @Suppress("DEPRECATION")
    override fun onBackPressed() {
        if (webView.canGoBack()) webView.goBack() else super.onBackPressed()
    }
}
