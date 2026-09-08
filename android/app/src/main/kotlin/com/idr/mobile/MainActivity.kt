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
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.SystemClock
import android.view.ViewGroup
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.core.app.ActivityCompat
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

class MainActivity : Activity(), SensorEventListener, LocationListener {

    private val permissionRequest = 42

    /*
     * CHANGE THIS TO YOUR LAPTOP'S LAN IP
     *
     * Example:
     * http://192.168.1.5:8000
     */
    private val backendUrl = "http://10.44.226.8:8000"

    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager
    private lateinit var webView: WebView

    private val network = Executors.newSingleThreadScheduledExecutor()
    private var sender: ScheduledFuture<*>? = null

    private var accelerometer = FloatArray(3)
    private var gyroscope = FloatArray(3)
    private var magnetometer = FloatArray(3)

    private var hasAccelerometer = false
    private var hasGyroscope = false
    private var hasMagnetometer = false

    private var running = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        sensorManager =
            getSystemService(Context.SENSOR_SERVICE) as SensorManager

        locationManager =
            getSystemService(Context.LOCATION_SERVICE) as LocationManager

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

        /*
         * LOAD THE SAME LIVE UI THAT WORKS IN VS CODE
         */
        webView.loadUrl("$backendUrl/")
    }

    private fun requestRequiredPermissions() {

        val permissions = arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )

        val missingPermissions = permissions.any {
            ActivityCompat.checkSelfPermission(
                this,
                it
            ) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions) {

            ActivityCompat.requestPermissions(
                this,
                permissions,
                permissionRequest
            )

        } else {

            startSensors()
        }
    }

    private fun startSensors() {

        if (running) return

        if (
            ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestRequiredPermissions()
            return
        }

        registerSensor(Sensor.TYPE_ACCELEROMETER)

        registerSensor(Sensor.TYPE_GYROSCOPE)

        registerSensor(Sensor.TYPE_MAGNETIC_FIELD)

        try {

            locationManager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                100L,
                0f,
                this
            )

        } catch (e: Exception) {
            e.printStackTrace()
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
    }

    private fun registerSensor(type: Int) {

        val sensor = sensorManager.getDefaultSensor(type)

        if (sensor != null) {

            sensorManager.registerListener(
                this,
                sensor,
                SensorManager.SENSOR_DELAY_GAME
            )
        }
    }

    private fun stopSensors() {

        running = false

        sender?.cancel(true)
        sender = null

        sensorManager.unregisterListener(this)

        try {
            locationManager.removeUpdates(this)
        } catch (e: Exception) {
            e.printStackTrace()
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

    override fun onLocationChanged(location: Location) {

        if (!running) return

        val timestamp =
            location.elapsedRealtimeNanos /
                    1_000_000_000.0

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

        postJson(
            "/sensor/gnss",
            body
        )
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

                connection.connectTimeout = 3000
                connection.readTimeout = 3000

                connection.doOutput = true

                connection.setRequestProperty(
                    "Content-Type",
                    "application/json"
                )

                connection.outputStream.use {

                    it.write(
                        body.toByteArray(Charsets.UTF_8)
                    )
                }

                try {
                    connection.inputStream.close()
                } catch (_: Exception) {
                }

                connection.disconnect()

            } catch (error: Exception) {

                error.printStackTrace()
            }
        }
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

    override fun onAccuracyChanged(
        sensor: Sensor?,
        accuracy: Int
    ) {
    }

    override fun onProviderEnabled(provider: String) {
    }

    override fun onProviderDisabled(provider: String) {
    }

    @Deprecated("Deprecated in Java")
    override fun onStatusChanged(
        provider: String?,
        status: Int,
        extras: Bundle?
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

        if (
            requestCode == permissionRequest &&
            grantResults.any {
                it == PackageManager.PERMISSION_GRANTED
            }
        ) {

            startSensors()
        }
    }

    override fun onDestroy() {

        stopSensors()

        network.shutdownNow()

        webView.destroy()

        super.onDestroy()
    }
}