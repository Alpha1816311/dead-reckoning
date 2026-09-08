package com.idr.mobile

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.core.app.ActivityCompat
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

/**
 * Minimal dependency-free Android collector.
 *
 * The phone sends one IMU packet every 100 ms and GNSS fixes as they arrive.
 * All timestamps use elapsedRealtimeNanos, so sensor and location events share
 * a monotonic clock. Set the laptop LAN URL before pressing Start.
 */
class MainActivity : Activity(), SensorEventListener, LocationListener {
    private val permissionRequest = 42
    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private val network = Executors.newSingleThreadExecutor()
    private var sender: ScheduledFuture<*>? = null

    private var accelerometer = FloatArray(3)
    private var gyroscope = FloatArray(3)
    private var magnetometer = FloatArray(3)
    private var hasAccelerometer = false
    private var hasGyroscope = false
    private var hasMagnetometer = false
    private var running = false
    private var backendUrl = ""

    private lateinit var urlInput: EditText
    private lateinit var status: TextView
    private lateinit var metrics: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        buildUi()
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(28, 24, 28, 24)
            setBackgroundColor(0xFF0B1220.toInt())
        }
        val title = TextView(this).apply {
            text = "INTELLIGENT DEAD RECKONING"
            textSize = 22f
            setTextColor(0xFFE8F0FF.toInt())
            gravity = Gravity.CENTER
        }
        urlInput = EditText(this).apply {
            hint = "http://192.168.1.42:8000"
            setText("")
            setSingleLine(true)
            setTextColor(0xFFFFFFFF.toInt())
            setHintTextColor(0xFF9AA8BD.toInt())
        }
        val start = Button(this).apply {
            text = "START LIVE SENSORS"
            setOnClickListener { if (running) stopSensors() else startSensors() }
        }
        status = TextView(this).apply {
            text = "BACKEND: NOT STARTED"
            textSize = 16f
            setTextColor(0xFF9AA8BD.toInt())
            setPadding(0, 24, 0, 12)
        }
        metrics = TextView(this).apply {
            text = "GNSS: WAITING\nMODE: WAITING FOR GNSS\nIMU: IDLE\nNHC: BACKEND CONTROLLED\nMAP: BACKEND CONTROLLED"
            textSize = 16f
            setTextColor(0xFFE8F0FF.toInt())
        }
        root.addView(title)
        root.addView(urlInput)
        root.addView(start)
        root.addView(status)
        root.addView(metrics)
        val scroll = ScrollView(this)
        scroll.addView(root)
        setContentView(scroll)
    }

    private fun startSensors() {
        backendUrl = urlInput.text.toString().trim().trimEnd('/')
        if (backendUrl.isBlank() || backendUrl.contains("localhost") || backendUrl.contains("127.0.0.1")) {
            status.text = "ERROR: use the laptop LAN IP, not localhost"
            return
        }
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
                permissionRequest
            )
            return
        }
        val delay = 100L
        register(Sensor.TYPE_ACCELEROMETER) { hasAccelerometer = true }
        register(Sensor.TYPE_GYROSCOPE) { hasGyroscope = true }
        register(Sensor.TYPE_MAGNETIC_FIELD) { hasMagnetometer = true }
        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 100L, 0f, this)
        sender = network.scheduleAtFixedRate({ sendImu() }, 0L, delay, TimeUnit.MILLISECONDS)
        running = true
        status.text = "BACKEND: CONNECTING — $backendUrl"
    }

    private fun register(type: Int, onAvailable: () -> Unit) {
        sensorManager.getDefaultSensor(type)?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            onAvailable()
        }
    }

    private fun stopSensors() {
        running = false
        sender?.cancel(true)
        sender = null
        sensorManager.unregisterListener(this)
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED
        ) {
            locationManager.removeUpdates(this)
        }
        status.text = "BACKEND: STOPPED"
    }

    private fun sendImu() {
        if (!hasAccelerometer || !hasGyroscope) return
        val a = accelerometer.copyOf()
        val g = gyroscope.copyOf()
        val m = magnetometer.copyOf()
        val timestamp = SystemClock.elapsedRealtimeNanos() / 1_000_000_000.0
        val magJson = if (hasMagnetometer) {
            ",\"mx\":${m[0]},\"my\":${m[1]},\"mz\":${m[2]}"
        } else ""
        val body = """{"type":"imu","timestamp":$timestamp,"ax":${a[0]},"ay":${a[1]},"az":${a[2]},"gx":${g[0]},"gy":${g[1]},"gz":${g[2]}$magJson}"""
        postJson("/sensor/imu", body)
    }

    override fun onLocationChanged(location: Location) {
        val timestamp = location.elapsedRealtimeNanos / 1_000_000_000.0
        val speed = if (location.hasSpeed()) location.speed else null
        val altitude = if (location.hasAltitude()) location.altitude else null
        val speedJson = speed?.toString() ?: "null"
        val altitudeJson = altitude?.toString() ?: "null"
        val body = """{"timestamp":$timestamp,"latitude":${location.latitude},"longitude":${location.longitude},"speed":$speedJson,"accuracy":${location.accuracy},"altitude":$altitudeJson}"""
        postJson("/sensor/gnss", body)
    }

    private fun postJson(path: String, body: String) {
        network.execute {
            try {
                val connection = URL("$backendUrl$path").openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.connectTimeout = 1500
                connection.readTimeout = 1500
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json")
                connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
                val response = connection.inputStream.bufferedReader().use { it.readText() }
                runOnUiThread { updateFromBackend(response) }
                connection.disconnect()
            } catch (error: Exception) {
                runOnUiThread { status.text = "BACKEND: ERROR — ${error.javaClass.simpleName}" }
            }
        }
    }

    private fun updateFromBackend(json: String) {
        status.text = "BACKEND: CONNECTED"
        fun value(key: String): String? {
            val marker = "\"$key\":"
            val start = json.indexOf(marker)
            if (start < 0) return null
            val end = json.indexOf(',', start).let { if (it < 0) json.indexOf('}', start) else it }
            return json.substring(start + marker.length, end).trim().trim('"')
        }
        val speed = value("speed_kmh")?.toDoubleOrNull()?.roundToInt()?.toString() ?: "—"
        val heading = value("heading_deg")?.toDoubleOrNull()?.roundToInt()?.toString() ?: "—"
        metrics.text = "GNSS: ${value("gnss_status") ?: "—"}\n" +
            "MODE: ${value("mode") ?: "—"}\n" +
            "SPEED: $speed km/h\nHEADING: $heading°\n" +
            "IMU: ${value("imu_status") ?: "—"}\n" +
            "NHC: ${value("nhc_status") ?: "—"}\n" +
            "MAP: ${value("map_status") ?: "—"}\n" +
            "ERROR: ${value("uncertainty_m") ?: "—"} m"
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accelerometer = event.values.copyOf()
                hasAccelerometer = true
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroscope = event.values.copyOf()
                hasGyroscope = true
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                magnetometer = event.values.copyOf()
                hasMagnetometer = true
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    override fun onProviderEnabled(provider: String) = Unit
    override fun onProviderDisabled(provider: String) = Unit
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionRequest && grantResults.any { it == PackageManager.PERMISSION_GRANTED }) {
            startSensors()
        } else {
            status.text = "ERROR: location permission is required for GNSS"
        }
    }

    override fun onDestroy() {
        stopSensors()
        network.shutdownNow()
        super.onDestroy()
    }
}