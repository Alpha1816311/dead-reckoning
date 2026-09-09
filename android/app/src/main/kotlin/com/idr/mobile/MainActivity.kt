package com.idr.mobile

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.os.Bundle
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.ViewGroup
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

class MainActivity : Activity(), SensorEventListener {

    private val permissionRequest = 42

    // DEPLOYED BACKEND
    private val backendUrl = "https://dead-reckoning-ten.vercel.app"

    private lateinit var sensorManager: SensorManager
    private lateinit var webView: WebView
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    private val network = Executors.newSingleThreadScheduledExecutor()
    private var sender: ScheduledFuture<*>? = null

    private var accelerometer = FloatArray(3)
    private var gyroscope = FloatArray(3)
    private var magnetometer = FloatArray(3)

    private var hasAccelerometer = false
    private var hasGyroscope = false
    private var hasMagnetometer = false

    private var running = false

    private val locationRequest =
        LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            1000L
        )
            .setMinUpdateIntervalMillis(500L)
            .setWaitForAccurateLocation(false)
            .build()

    private val locationCallback =
        object : LocationCallback() {

            override fun onLocationResult(
                locationResult: LocationResult
            ) {

                for (location in locationResult.locations) {
                    handleRealLocation(location)
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d("IDR_GPS", "MainActivity started")

        sensorManager =
            getSystemService(Context.SENSOR_SERVICE) as SensorManager

        fusedLocationClient =
            LocationServices.getFusedLocationProviderClient(this)

        setupWebView()

        requestRequiredPermissions()
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {

        webView = WebView(this)

        val settings: WebSettings = webView.settings

        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.allowFileAccess = true
        settings.allowContentAccess = true

        webView.webViewClient = WebViewClient()
        webView.webChromeClient = WebChromeClient()

        setContentView(
            webView,
            ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )

        webView.loadUrl("$backendUrl/")
    }

    private fun requestRequiredPermissions() {

        val permissions = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )

        val fineGranted =
            ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED

        val coarseGranted =
            ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED

        Log.d(
            "IDR_GPS",
            "Permission status → FINE=$fineGranted, COARSE=$coarseGranted"
        )

        if (!fineGranted && !coarseGranted) {

            Log.d(
                "IDR_GPS",
                "Requesting location permissions"
            )

            ActivityCompat.requestPermissions(
                this,
                permissions,
                permissionRequest
            )

        } else {

            Log.d(
                "IDR_GPS",
                "Location permission already available"
            )

            startSensors()
        }
    }

    @SuppressLint("MissingPermission")
    private fun startSensors() {

        Log.d(
            "IDR_GPS",
            "startSensors() called"
        )

        if (running) {
            Log.d(
                "IDR_GPS",
                "Sensors already running"
            )
            return
        }

        val fineGranted =
            ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED

        val coarseGranted =
            ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED

        Log.d(
            "IDR_GPS",
            "Location permission → FINE=$fineGranted, COARSE=$coarseGranted"
        )

        if (!fineGranted && !coarseGranted) {

            Log.e(
                "IDR_GPS",
                "No location permission. Cannot start GPS."
            )

            requestRequiredPermissions()
            return
        }

        // -------------------------
        // IMU SENSORS
        // -------------------------

        registerSensor(Sensor.TYPE_ACCELEROMETER)
        registerSensor(Sensor.TYPE_GYROSCOPE)
        registerSensor(Sensor.TYPE_MAGNETIC_FIELD)

        // -------------------------
        // FUSED LOCATION
        // -------------------------

        try {

            fusedLocationClient.requestLocationUpdates(
                locationRequest,
                locationCallback,
                Looper.getMainLooper()
            )

            Log.d(
                "IDR_GPS",
                "Fused Location updates registered successfully"
            )

        } catch (e: Exception) {

            Log.e(
                "IDR_GPS",
                "Fused Location registration failed",
                e
            )
        }

        // Try to get the most recent location immediately.
        try {

            fusedLocationClient.lastLocation
                .addOnSuccessListener { location ->

                    if (location != null) {

                        Log.d(
                            "IDR_GPS",
                            "LAST FUSED GPS → " +
                                    "lat=${location.latitude}, " +
                                    "lon=${location.longitude}, " +
                                    "accuracy=${location.accuracy}"
                        )

                        handleRealLocation(location)

                    } else {

                        Log.d(
                            "IDR_GPS",
                            "LAST FUSED GPS → null"
                        )
                    }
                }
                .addOnFailureListener { error ->

                    Log.e(
                        "IDR_GPS",
                        "LAST FUSED GPS failed",
                        error
                    )
                }

        } catch (e: Exception) {

            Log.e(
                "IDR_GPS",
                "Could not request last fused location",
                e
            )
        }

        sender = network.scheduleAtFixedRate(
            {
                sendImu()
            },
            0L,
            100L,
            TimeUnit.MILLISECONDS
        )

        running = true

        Log.d(
            "IDR_GPS",
            "Sensors/network marked as RUNNING"
        )
    }

    private fun registerSensor(type: Int) {

        val sensor =
            sensorManager.getDefaultSensor(type)

        if (sensor != null) {

            val registered =
                sensorManager.registerListener(
                    this,
                    sensor,
                    SensorManager.SENSOR_DELAY_GAME
                )

            Log.d(
                "IDR_SENSOR",
                "Sensor type $type registered=$registered"
            )

        } else {

            Log.e(
                "IDR_SENSOR",
                "Sensor type $type NOT available"
            )
        }
    }

    private fun stopSensors() {

        Log.d(
            "IDR_GPS",
            "Stopping sensors"
        )

        running = false

        sender?.cancel(true)
        sender = null

        sensorManager.unregisterListener(this)

        try {

            fusedLocationClient.removeLocationUpdates(
                locationCallback
            )

            Log.d(
                "IDR_GPS",
                "Fused Location updates removed"
            )

        } catch (e: Exception) {

            Log.e(
                "IDR_GPS",
                "Error removing location updates",
                e
            )
        }
    }

    private fun sendImu() {

        if (!running) return

        if (!hasAccelerometer || !hasGyroscope) return

        val a = accelerometer.copyOf()
        val g = gyroscope.copyOf()
        val m = magnetometer.copyOf()

        val timestamp =
            SystemClock.elapsedRealtimeNanos() /
                    1_000_000_000.0

        val magnetometerJson =
            if (hasMagnetometer) {

                """
                ,"mx":${m[0]},
                "my":${m[1]},
                "mz":${m[2]}
                """.trimIndent()

            } else {
                ""
            }

        val body = """
            {
                "type":"imu",
                "timestamp":$timestamp,
                "ax":${a[0]},
                "ay":${a[1]},
                "az":${a[2]},
                "gx":${g[0]},
                "gy":${g[1]},
                "gz":${g[2]}
                $magnetometerJson
            }
        """.trimIndent()

        postJson(
            "/sensor/imu",
            body
        )
    }

    private fun handleRealLocation(
        location: Location
    ) {

        Log.d(
            "IDR_GPS",
            "REAL GPS → " +
                    "lat=${location.latitude}, " +
                    "lon=${location.longitude}, " +
                    "speed=${location.speed}, " +
                    "accuracy=${location.accuracy}"
        )

        if (!running) {
            Log.d(
                "IDR_GPS",
                "Ignoring location because app is not running"
            )
            return
        }

        val timestamp =
            if (location.elapsedRealtimeNanos > 0) {

                location.elapsedRealtimeNanos /
                        1_000_000_000.0

            } else {

                SystemClock.elapsedRealtimeNanos() /
                        1_000_000_000.0
            }

        val speed =
            if (location.hasSpeed()) {
                location.speed
            } else {
                null
            }

        val altitude =
            if (location.hasAltitude()) {
                location.altitude
            } else {
                null
            }

        val speedJson =
            speed?.toString() ?: "null"

        val altitudeJson =
            altitude?.toString() ?: "null"

        val body = """
            {
                "timestamp":$timestamp,
                "latitude":${location.latitude},
                "longitude":${location.longitude},
                "speed":$speedJson,
                "accuracy":${location.accuracy},
                "altitude":$altitudeJson
            }
        """.trimIndent()

        Log.d(
            "IDR_GPS",
            "Sending GNSS → $body"
        )

        // Send real GPS to backend
        postJson(
            "/sensor/gnss",
            body
        )

        // Update existing WebView UI immediately
        runOnUiThread {

            val latitude =
                location.latitude

            val longitude =
                location.longitude

            val speedMps =
                if (location.hasSpeed()) {
                    location.speed
                } else {
                    0f
                }

            val javascript =
                "window.updateNativeGPS(" +
                        "$latitude," +
                        "$longitude," +
                        "$speedMps" +
                        ")"

            webView.evaluateJavascript(
                javascript,
                null
            )
        }
    }

    private fun postJson(
        path: String,
        body: String
    ) {

        network.execute {

            try {

                val connection =
                    URL("$backendUrl$path")
                        .openConnection() as HttpURLConnection

                connection.requestMethod = "POST"

                connection.connectTimeout = 5000
                connection.readTimeout = 5000

                connection.doOutput = true

                connection.setRequestProperty(
                    "Content-Type",
                    "application/json"
                )

                connection.outputStream.use {

                    it.write(
                        body.toByteArray(
                            Charsets.UTF_8
                        )
                    )
                }

                val responseCode =
                    connection.responseCode

                Log.d(
                    "IDR_NETWORK",
                    "$path → HTTP $responseCode"
                )

                try {
                    connection.inputStream.close()
                } catch (_: Exception) {
                }

                connection.disconnect()

            } catch (error: Exception) {

                Log.e(
                    "IDR_NETWORK",
                    "POST failed: $path",
                    error
                )
            }
        }
    }

    override fun onSensorChanged(
        event: SensorEvent
    ) {

        when (event.sensor.type) {

            Sensor.TYPE_ACCELEROMETER -> {

                accelerometer =
                    event.values.copyOf()

                hasAccelerometer = true
            }

            Sensor.TYPE_GYROSCOPE -> {

                gyroscope =
                    event.values.copyOf()

                hasGyroscope = true
            }

            Sensor.TYPE_MAGNETIC_FIELD -> {

                magnetometer =
                    event.values.copyOf()

                hasMagnetometer = true
            }
        }
    }

    override fun onAccuracyChanged(
        sensor: Sensor?,
        accuracy: Int
    ) {
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {

        super.onRequestPermissionsResult(
            requestCode,
            permissions,
            grantResults
        )

        if (requestCode == permissionRequest) {

            val fineGranted =
                ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_FINE_LOCATION
                ) == PackageManager.PERMISSION_GRANTED

            val coarseGranted =
                ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ) == PackageManager.PERMISSION_GRANTED

            Log.d(
                "IDR_GPS",
                "Permission result → FINE=$fineGranted, COARSE=$coarseGranted"
            )

            if (fineGranted || coarseGranted) {

                startSensors()

            } else {

                Log.e(
                    "IDR_GPS",
                    "Location permission DENIED"
                )
            }
        }
    }

    override fun onDestroy() {

        stopSensors()

        network.shutdownNow()

        webView.destroy()

        super.onDestroy()
    }
}