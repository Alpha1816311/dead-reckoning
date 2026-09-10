package com.idr.mobile

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.SharedPreferences
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
import android.os.SystemClock
import android.text.InputType
import android.view.Gravity
import android.view.ViewGroup
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.app.ActivityCompat
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

class MainActivity : Activity(), SensorEventListener, LocationListener {
    private val permissionRequest = 42
    private val preferencesName = "idr_connection"
    private val serverUrlKey = "server_url"
    // 10.0.2.2 is the Android emulator loopback to the host.
    // On a physical device enter the laptop LAN IP, e.g. http://192.168.1.42:8000
    private val defaultBackendUrl = "http://10.0.2.2:8000"

    private lateinit var preferences: SharedPreferences
    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private lateinit var webView: WebView
    private lateinit var serverUrlInput: EditText
    private lateinit var connectionStatusView: TextView
    private lateinit var collectionStatusView: TextView
    private lateinit var telemetryView: TextView
    private lateinit var sensorDebugView: TextView
    private lateinit var gnssStatusView: TextView
    private lateinit var errorView: TextView
    private lateinit var transport: LiveSensorTransport

    private val scheduler = Executors.newSingleThreadScheduledExecutor()
    private val healthExecutor = ThreadPoolExecutor(
        2, 2, 0L, TimeUnit.MILLISECONDS, ArrayBlockingQueue<Runnable>(8), ThreadPoolExecutor.AbortPolicy()
    )
    private var sender: ScheduledFuture<*>? = null

    // ---------------------------------------------------------------
    // Sensor state — written by the SensorManager callback thread,
    // read by the scheduler thread.  Use AtomicReference / AtomicBoolean
    // so the scheduler thread always sees the latest values (no CPU-cache
    // visibility hazard).
    // ---------------------------------------------------------------
    private val accelRef = AtomicReference(FloatArray(3))
    private val gyroRef  = AtomicReference(FloatArray(3))
    private val magRef   = AtomicReference(FloatArray(3))
    private val hasAccel = AtomicBoolean(false)
    private val hasGyro  = AtomicBoolean(false)
    private val hasMag   = AtomicBoolean(false)

    // Stage-by-stage diagnostic counters
    private val accelCallbacks   = AtomicLong(0)   // Stage 1: raw sensor events
    private val gyroCallbacks    = AtomicLong(0)   // Stage 1: raw sensor events
    private val gnssCallbacks    = AtomicLong(0)   // Stage 2: raw GNSS events
    private val imuPacketsProduced = AtomicLong(0) // Stage 3: packets assembled
    private val imuPacketsSent   = AtomicLong(0)   // Stage 5: backend ACK received
    private val gnssPacketsSent  = AtomicLong(0)   // Stage 5: backend ACK received
    private val imuPacketsDropped = AtomicLong(0)  // dropped (superseded)
    private val gnssPacketsDropped = AtomicLong(0) // dropped (buffer full)
    private val failedPackets    = AtomicLong(0)   // Stage 6: transport failures

    @Volatile private var running = false
    @Volatile private var collectionRequested = false
    @Volatile private var backendUrl = defaultBackendUrl

    private var connectionState = "DISCONNECTED"
    private var latestError = "None"
    private var lastSuccessfulTransmission: Long? = null
    private var lastGnssEventTime: Long? = null
    private var lastGnssLat: Double? = null
    private var lastGnssLon: Double? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        preferences = getSharedPreferences(preferencesName, Context.MODE_PRIVATE)
        backendUrl = loadSavedBackendUrl()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        setupUi()
        transport = LiveSensorTransport(object : LiveSensorTransport.Listener {
            override fun onConnectionState(state: String, message: String) {
                setConnectionState(state, message)
            }
            override fun onPacketAcknowledged(type: LiveSensorTransport.PacketType) {
                if (type == LiveSensorTransport.PacketType.IMU) imuPacketsSent.incrementAndGet()
                else gnssPacketsSent.incrementAndGet()
                recordSuccessfulTransmission()
            }
            override fun onPacketDropped(type: LiveSensorTransport.PacketType, message: String) {
                if (type == LiveSensorTransport.PacketType.IMU) imuPacketsDropped.incrementAndGet()
                else gnssPacketsDropped.incrementAndGet()
                updateTelemetry()
            }
            override fun onPacketFailure(message: String) {
                recordNetworkFailure(message)
            }
        })
        updateTelemetry()
        updateSensorDebug()
        updateCollectionStatus("STOPPED")
        setConnectionState("DISCONNECTED", "Enter the LAN IP and tap Test Connection.")
    }

    private fun loadSavedBackendUrl(): String {
        val saved = preferences.getString(serverUrlKey, defaultBackendUrl) ?: defaultBackendUrl
        return validateBackendUrl(saved) ?: defaultBackendUrl
    }

    private fun validateBackendUrl(value: String): String? {
        val raw = value.trim()
        if (raw.isEmpty() || raw.any { it.isWhitespace() }) return null
        val uri = try { Uri.parse(raw) } catch (_: Exception) { return null }
        if (uri.scheme !in setOf("http", "https") || uri.host.isNullOrBlank()) return null
        if (uri.port !in -1..65535 || uri.port == 0) return null
        if (!uri.userInfo.isNullOrBlank() || !uri.query.isNullOrBlank() || !uri.fragment.isNullOrBlank()) return null
        return uri.buildUpon().path(null).query(null).fragment(null).build().toString().trimEnd('/')
    }

    private fun saveBackendUrlFromInput(): String? {
        val normalized = validateBackendUrl(serverUrlInput.text.toString())
        if (normalized == null) {
            setConnectionState("ERROR", "Enter a valid http:// or https:// URL with a port.")
            return null
        }
        backendUrl = normalized
        preferences.edit().putString(serverUrlKey, normalized).apply()
        serverUrlInput.setText(normalized)
        webView.loadUrl("$normalized/")
        return normalized
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }

        // URL row
        val serverRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        serverUrlInput = EditText(this).apply {
            hint = "e.g. http://192.168.1.42:8000  (laptop LAN IP)"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setSingleLine(true)
            setText(backendUrl)
        }
        serverRow.addView(serverUrlInput, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        serverRow.addView(button("Test") { testConnection() })
        root.addView(serverRow)

        connectionStatusView = statusText()
        collectionStatusView = statusText()
        root.addView(connectionStatusView)
        root.addView(collectionStatusView)

        // Control buttons
        val controls = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        controls.addView(button("▶ Start") { requestStartCollection() }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        controls.addView(button("■ Stop")  { stopCollectionByUser()   }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(controls)

        // Stage-by-stage diagnostics
        sensorDebugView = statusText()
        telemetryView   = statusText()
        gnssStatusView  = statusText()
        errorView       = statusText()
        root.addView(sensorDebugView)
        root.addView(telemetryView)
        root.addView(gnssStatusView)
        root.addView(errorView)

        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            settings.allowContentAccess = true
            webViewClient = WebViewClient()
            webChromeClient = WebChromeClient()
            loadUrl("$backendUrl/")
        }
        root.addView(webView, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)
    }

    private fun button(label: String, action: () -> Unit): Button = Button(this).apply {
        text = label
        setOnClickListener { action() }
    }

    private fun statusText(): TextView = TextView(this).apply {
        textSize = 12f
        setPadding(0, dp(2), 0, dp(2))
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    // ---------------------------------------------------------------
    // TEST CONNECTION
    // ---------------------------------------------------------------
    private fun testConnection() {
        val url = saveBackendUrlFromInput() ?: return
        setConnectionState("CONNECTING", "Testing $url/health …")
        try {
            healthExecutor.execute {
                try {
                    val conn = openConnection("$url/health", "GET")
                    val status = conn.responseCode
                    conn.disconnect()
                    if (status in 200..299) {
                        recordSuccessfulTransmission()
                        setConnectionState("CONNECTED", "Health OK (HTTP $status) — ready to stream.")
                    } else {
                        recordNetworkFailure("Health returned HTTP $status")
                    }
                } catch (e: Exception) {
                    recordNetworkFailure("Health failed: ${e.message ?: e.javaClass.simpleName}")
                }
            }
        } catch (_: RejectedExecutionException) {
            recordNetworkFailure("Network worker busy; retry.")
        }
    }

    // ---------------------------------------------------------------
    // START / STOP COLLECTION
    // ---------------------------------------------------------------
    private fun requestStartCollection() {
        collectionRequested = true
        if (saveBackendUrlFromInput() == null) {
            collectionRequested = false
            return
        }
        if (!hasFineLocationPermission()) {
            updateCollectionStatus("WAITING FOR LOCATION PERMISSION")
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
                permissionRequest
            )
            return
        }
        startSensors()
    }

    private fun hasFineLocationPermission(): Boolean =
        ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED

    @SuppressLint("MissingPermission")
    private fun startSensors() {
        if (running || !hasFineLocationPermission()) return

        // Register sensors — report registration result on-screen
        val accelOk = registerSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroOk  = registerSensor(Sensor.TYPE_GYROSCOPE)
        registerSensor(Sensor.TYPE_MAGNETIC_FIELD) // optional

        if (!accelOk || !gyroOk) {
            updateCollectionStatus("ERROR — required IMU sensor missing")
            return
        }

        // GPS location updates
        try {
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 200L, 0f, this)
        } catch (e: Exception) {
            // Not fatal — GNSS will just be absent
            recordNetworkFailure("GPS registration: ${e.message ?: e.javaClass.simpleName}")
        }

        running = true
        transport.start(backendUrl)

        // Schedule IMU packets at 10 Hz (every 100 ms)
        sender = scheduler.scheduleAtFixedRate({ sendImu() }, 50L, 100L, TimeUnit.MILLISECONDS)

        updateCollectionStatus("RUNNING — url=$backendUrl")
        updateSensorDebug()
    }

    /** Returns true if sensor was successfully registered. */
    private fun registerSensor(type: Int): Boolean {
        val sensor = sensorManager.getDefaultSensor(type) ?: return false
        return sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME)
    }

    private fun stopCollectionByUser() {
        collectionRequested = false
        stopSensors("STOPPED")
    }

    private fun stopSensors(status: String) {
        running = false
        sender?.cancel(false)
        sender = null
        if (::transport.isInitialized) transport.stop()
        sensorManager.unregisterListener(this)
        try { locationManager.removeUpdates(this) } catch (_: Exception) { }
        updateCollectionStatus(status)
    }

    // ---------------------------------------------------------------
    // SEND IMU — called from scheduler thread at 10 Hz
    // ---------------------------------------------------------------
    private fun sendImu() {
        if (!running) return
        if (!hasAccel.get() || !hasGyro.get()) return  // wait for first sensor events

        val a = accelRef.get()
        val g = gyroRef.get()
        val m = magRef.get()

        // Use SystemClock.elapsedRealtimeNanos for monotonic boot-relative time in seconds.
        // The backend TimestampNormalizer accepts any finite monotonic value.
        val timestamp = SystemClock.elapsedRealtimeNanos() / 1_000_000_000.0

        val magJson = if (hasMag.get()) ",\"mx\":${m[0]},\"my\":${m[1]},\"mz\":${m[2]}" else ""
        val body = "{\"type\":\"imu\",\"timestamp\":$timestamp," +
                   "\"ax\":${a[0]},\"ay\":${a[1]},\"az\":${a[2]}," +
                   "\"gx\":${g[0]},\"gy\":${g[1]},\"gz\":${g[2]}$magJson}"

        imuPacketsProduced.incrementAndGet()
        transport.submitImu(body)
    }

    // ---------------------------------------------------------------
    // GNSS CALLBACK
    // ---------------------------------------------------------------
    override fun onLocationChanged(location: Location) {
        if (!running) return
        gnssCallbacks.incrementAndGet()
        lastGnssEventTime = System.currentTimeMillis()
        lastGnssLat = location.latitude
        lastGnssLon = location.longitude

        val timestamp = location.elapsedRealtimeNanos / 1_000_000_000.0
        val speed    = if (location.hasSpeed())    location.speed.toString()    else "null"
        val altitude = if (location.hasAltitude()) location.altitude.toString() else "null"
        val body = "{\"type\":\"gnss\",\"timestamp\":$timestamp," +
                   "\"latitude\":${location.latitude},\"longitude\":${location.longitude}," +
                   "\"speed\":$speed,\"accuracy\":${location.accuracy},\"altitude\":$altitude}"

        transport.submitGnss(body)
        updateGnssStatus()
    }

    // ---------------------------------------------------------------
    // SENSOR CALLBACKS
    // ---------------------------------------------------------------
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accelRef.set(event.values.copyOf())
                hasAccel.set(true)
                accelCallbacks.incrementAndGet()
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroRef.set(event.values.copyOf())
                hasGyro.set(true)
                gyroCallbacks.incrementAndGet()
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                magRef.set(event.values.copyOf())
                hasMag.set(true)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    override fun onProviderEnabled(provider: String) = Unit
    override fun onProviderDisabled(provider: String) = Unit

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit

    // ---------------------------------------------------------------
    // NETWORK
    // ---------------------------------------------------------------
    private fun openConnection(url: String, method: String): HttpURLConnection =
        (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 4_000
            readTimeout = 4_000
            setRequestProperty("Accept", "application/json")
        }

    private fun recordSuccessfulTransmission() {
        lastSuccessfulTransmission = System.currentTimeMillis()
        updateTelemetry()
    }

    private fun recordNetworkFailure(message: String) {
        failedPackets.incrementAndGet()
        setConnectionState("ERROR", message)
    }

    // ---------------------------------------------------------------
    // UI UPDATES
    // ---------------------------------------------------------------
    private fun setConnectionState(state: String, message: String) {
        connectionState = state
        latestError = if (state == "ERROR") message else "None"
        runOnUiThread {
            connectionStatusView.text = "Transport: $connectionState — $message"
            errorView.text = "Last error: $latestError"
            updateTelemetry()
        }
    }

    private fun updateCollectionStatus(status: String) {
        if (!::collectionStatusView.isInitialized) return
        runOnUiThread { collectionStatusView.text = "Collection: $status" }
    }

    private fun updateSensorDebug() {
        if (!::sensorDebugView.isInitialized) return
        runOnUiThread {
            sensorDebugView.text =
                "S1 Accel cb:${accelCallbacks.get()}  Gyro cb:${gyroCallbacks.get()}  GNSS cb:${gnssCallbacks.get()}"
        }
    }

    private fun updateTelemetry() {
        if (!::telemetryView.isInitialized) return
        runOnUiThread {
            val lastTx = lastSuccessfulTransmission?.let {
                android.text.format.DateFormat.format("HH:mm:ss", it).toString()
            } ?: "never"
            // Update sensor debug counters at the same time
            sensorDebugView.text =
                "S1 Accel:${accelCallbacks.get()} Gyro:${gyroCallbacks.get()} GNSS:${gnssCallbacks.get()}"
            telemetryView.text =
                "S3 produced:${imuPacketsProduced.get()}  " +
                "S5 imu_ack:${imuPacketsSent.get()} gnss_ack:${gnssPacketsSent.get()}  " +
                "drop:${imuPacketsDropped.get()+gnssPacketsDropped.get()}  " +
                "S6 fail:${failedPackets.get()}  lastTx:$lastTx"
        }
    }

    private fun updateGnssStatus() {
        if (!::gnssStatusView.isInitialized) return
        runOnUiThread {
            val lat = lastGnssLat
            val lon = lastGnssLon
            val t = lastGnssEventTime?.let {
                android.text.format.DateFormat.format("HH:mm:ss", it).toString()
            } ?: "never"
            gnssStatusView.text = if (lat != null && lon != null)
                "S2 GNSS: ${String.format("%.5f", lat)}, ${String.format("%.5f", lon)} @$t"
            else
                "S2 GNSS: waiting for fix (outdoor + clear sky needed)"
        }
    }

    // ---------------------------------------------------------------
    // PERMISSIONS
    // ---------------------------------------------------------------
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != permissionRequest || !collectionRequested) return
        if (hasFineLocationPermission()) {
            startSensors()
        } else {
            collectionRequested = false
            updateCollectionStatus("STOPPED — PRECISE LOCATION REQUIRED")
            setConnectionState("ERROR", "Precise location permission is required for GNSS.")
        }
    }

    override fun onPause() {
        super.onPause()
        // Do NOT stop sensors on pause — screen-off must not kill the live stream.
        // User must explicitly press Stop to halt collection.
    }

    override fun onResume() {
        super.onResume()
        if (collectionRequested && !running && hasFineLocationPermission()) startSensors()
    }

    override fun onDestroy() {
        collectionRequested = false
        stopSensors("STOPPED")
        scheduler.shutdownNow()
        healthExecutor.shutdownNow()
        if (::transport.isInitialized) transport.shutdown()
        webView.destroy()
        super.onDestroy()
    }
}
