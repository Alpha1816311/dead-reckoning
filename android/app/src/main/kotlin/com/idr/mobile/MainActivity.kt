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
import android.view.View
import android.view.ViewGroup
import android.webkit.WebChromeClient
import android.webkit.WebSettings
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
import java.util.concurrent.atomic.AtomicLong

class MainActivity : Activity(), SensorEventListener, LocationListener {
    private val permissionRequest = 42
    private val preferencesName = "idr_connection"
    private val serverUrlKey = "server_url"
    private val defaultBackendUrl = "http://10.0.2.2:8000"

    private lateinit var preferences: SharedPreferences
    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private lateinit var webView: WebView
    private lateinit var serverUrlInput: EditText
    private lateinit var connectionStatusView: TextView
    private lateinit var collectionStatusView: TextView
    private lateinit var telemetryView: TextView
    private lateinit var errorView: TextView
    private lateinit var transport: LiveSensorTransport

    private val scheduler = Executors.newSingleThreadScheduledExecutor()
    private val healthExecutor = ThreadPoolExecutor(
        2, 2, 0L, TimeUnit.MILLISECONDS, ArrayBlockingQueue<Runnable>(8), ThreadPoolExecutor.AbortPolicy()
    )
    private var sender: ScheduledFuture<*>? = null

    private var accelerometer = FloatArray(3)
    private var gyroscope = FloatArray(3)
    private var magnetometer = FloatArray(3)
    private var hasAccelerometer = false
    private var hasGyroscope = false
    private var hasMagnetometer = false
    private var running = false
    private var collectionRequested = false
    private var backendUrl = defaultBackendUrl

    private val imuPacketsProduced = AtomicLong(0)
    private val imuPacketsSent = AtomicLong(0)
    private val gnssPacketsSent = AtomicLong(0)
    private val failedPackets = AtomicLong(0)
    private var connectionState = "DISCONNECTED"
    private var latestError = "None"
    private var lastSuccessfulTransmission: Long? = null

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

            override fun onPacketFailure(message: String) {
                recordNetworkFailure(message)
            }
        })
        updateTelemetry()
        updateCollectionStatus("STOPPED")
        setConnectionState("DISCONNECTED", "Set the server URL and test the connection.")
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
            setConnectionState("ERROR", "Enter a valid http:// or https:// server URL.")
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
        val serverRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        serverUrlInput = EditText(this).apply {
            hint = "Backend URL (e.g. http://192.168.1.42:8000)"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setSingleLine(true)
            setText(backendUrl)
        }
        serverRow.addView(serverUrlInput, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        serverRow.addView(button("Test Connection") { testConnection() })
        root.addView(serverRow)

        connectionStatusView = statusText()
        collectionStatusView = statusText()
        telemetryView = statusText()
        errorView = statusText()
        root.addView(connectionStatusView)
        root.addView(collectionStatusView)

        val controls = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        controls.addView(button("Start Collection") { requestStartCollection() }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        controls.addView(button("Stop Collection") { stopCollectionByUser() }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(controls)
        root.addView(telemetryView)
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
        textSize = 13f
        setPadding(0, dp(2), 0, dp(2))
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun testConnection() {
        val url = saveBackendUrlFromInput() ?: return
        setConnectionState("CONNECTING", "Testing $url/health")
        try {
            healthExecutor.execute {
                try {
                    val connection = openConnection("$url/health", "GET")
                    val status = connection.responseCode
                    connection.disconnect()
                    if (status in 200..299) {
                        recordSuccessfulTransmission()
                        setConnectionState("CONNECTED", "Backend health check succeeded.")
                    } else {
                        recordNetworkFailure("Health check returned HTTP $status")
                    }
                } catch (error: Exception) {
                    recordNetworkFailure("Health check failed: ${error.message ?: error.javaClass.simpleName}")
                }
            }
        } catch (_: RejectedExecutionException) {
            recordNetworkFailure("Network worker is busy; try again.")
        }
    }

    private fun requestStartCollection() {
        collectionRequested = true
        if (saveBackendUrlFromInput() == null) {
            collectionRequested = false
            return
        }
        if (!hasFineLocationPermission()) {
            updateCollectionStatus("WAITING FOR PRECISE LOCATION PERMISSION")
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION), permissionRequest)
            return
        }
        startSensors()
    }

    private fun hasFineLocationPermission(): Boolean =
        ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED

    @SuppressLint("MissingPermission")
    private fun startSensors() {
        if (running || !hasFineLocationPermission()) return
        registerSensor(Sensor.TYPE_ACCELEROMETER)
        registerSensor(Sensor.TYPE_GYROSCOPE)
        registerSensor(Sensor.TYPE_MAGNETIC_FIELD)
        try {
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 100L, 0f, this)
        } catch (error: Exception) {
            recordNetworkFailure("GPS registration failed: ${error.message ?: error.javaClass.simpleName}")
        }
        running = true
        transport.start(backendUrl)
        sender = scheduler.scheduleAtFixedRate({ sendImu() }, 0L, 100L, TimeUnit.MILLISECONDS)
        updateCollectionStatus("RUNNING")
    }

    private fun registerSensor(type: Int) {
        val sensor = sensorManager.getDefaultSensor(type)
        if (sensor == null) {
            if (type == Sensor.TYPE_ACCELEROMETER || type == Sensor.TYPE_GYROSCOPE) {
                recordNetworkFailure("Required ${sensorName(type)} is unavailable on this device.")
            }
            return
        }
        sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME)
    }

    private fun sensorName(type: Int): String = when (type) {
        Sensor.TYPE_ACCELEROMETER -> "accelerometer"
        Sensor.TYPE_GYROSCOPE -> "gyroscope"
        Sensor.TYPE_MAGNETIC_FIELD -> "magnetometer"
        else -> "sensor"
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

    private fun sendImu() {
        if (!running || !hasAccelerometer || !hasGyroscope) return
        val a = accelerometer.copyOf()
        val g = gyroscope.copyOf()
        val m = magnetometer.copyOf()
        val timestamp = SystemClock.elapsedRealtimeNanos() / 1_000_000_000.0
        val magnetometerJson = if (hasMagnetometer) ",\"mx\":${m[0]},\"my\":${m[1]},\"mz\":${m[2]}" else ""
        val body = "{\"type\":\"imu\",\"timestamp\":$timestamp,\"ax\":${a[0]},\"ay\":${a[1]},\"az\":${a[2]},\"gx\":${g[0]},\"gy\":${g[1]},\"gz\":${g[2]}$magnetometerJson}"
        imuPacketsProduced.incrementAndGet()
        submitUpload("/sensor/imu", body, true)
    }

    override fun onLocationChanged(location: Location) {
        if (!running) return
        val timestamp = location.elapsedRealtimeNanos / 1_000_000_000.0
        val speed = if (location.hasSpeed()) location.speed.toString() else "null"
        val altitude = if (location.hasAltitude()) location.altitude.toString() else "null"
        val body = "{\"timestamp\":$timestamp,\"latitude\":${location.latitude},\"longitude\":${location.longitude},\"speed\":$speed,\"accuracy\":${location.accuracy},\"altitude\":$altitude}"
        submitUpload("/sensor/gnss", body, false)
    }

    private fun submitUpload(path: String, body: String, isImu: Boolean) {
        if (isImu) transport.submitImu(body) else transport.submitGnss(body)
    }

    private fun openConnection(url: String, method: String): HttpURLConnection =
        (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 3_000
            readTimeout = 3_000
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

    private fun setConnectionState(state: String, message: String) {
        connectionState = state
        latestError = if (state == "ERROR") message else "None"
        runOnUiThread {
            connectionStatusView.text = "Connection: $connectionState — $message"
            errorView.text = "Last error: $latestError"
            updateTelemetry()
        }
    }

    private fun updateCollectionStatus(status: String) {
        if (!::collectionStatusView.isInitialized) return
        runOnUiThread { collectionStatusView.text = "Collection: $status" }
    }

    private fun updateTelemetry() {
        if (!::telemetryView.isInitialized) return
        runOnUiThread {
            val lastSuccess = lastSuccessfulTransmission?.let {
                android.text.format.DateFormat.format("HH:mm:ss", it).toString()
            } ?: "never"
            telemetryView.text = "IMU produced/sent: ${imuPacketsProduced.get()}/${imuPacketsSent.get()}   GNSS sent: ${gnssPacketsSent.get()}   Failed: ${failedPackets.get()}   Last success: $lastSuccess"
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> { accelerometer = event.values.copyOf(); hasAccelerometer = true }
            Sensor.TYPE_GYROSCOPE -> { gyroscope = event.values.copyOf(); hasGyroscope = true }
            Sensor.TYPE_MAGNETIC_FIELD -> { magnetometer = event.values.copyOf(); hasMagnetometer = true }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    override fun onProviderEnabled(provider: String) = Unit
    override fun onProviderDisabled(provider: String) = Unit

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != permissionRequest || !collectionRequested) return
        if (hasFineLocationPermission()) {
            startSensors()
        } else {
            collectionRequested = false
            updateCollectionStatus("STOPPED — PRECISE LOCATION REQUIRED")
            setConnectionState("ERROR", "Precise location permission is required to collect GNSS.")
        }
    }

    override fun onPause() {
        super.onPause()
        if (running) stopSensors("PAUSED")
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
