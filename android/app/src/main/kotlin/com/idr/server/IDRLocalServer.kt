package com.idr.server

import android.content.res.AssetManager
import com.idr.navigation.IDRNavigationEngine
import com.idr.navigation.SessionRecorder
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.InputStream

/**
 * Embedded NanoHTTPD-based local HTTP server.
 * Serves the advanced IDR Fusion UI from Android assets and provides
 * a REST API that the UI JavaScript polls.
 *
 * All navigation state comes from IDRNavigationEngine — no demo/fake values.
 */
class IDRLocalServer(
    port: Int,
    private val assets: AssetManager,
    private val engine: IDRNavigationEngine,
    private val recorder: SessionRecorder
) : NanoHTTPD(port) {

    // Track colour history for the map path (sent to UI as `state.track`)
    private val trackHistory = mutableListOf<Array<Double>>()
    private val trackLock = Any()
    private val MAX_TRACK = 600

    // Blackout control called from UI button
    @Volatile var blackoutActive = false

    // Calibration state
    @Volatile var calibrationState = "IDLE" // IDLE, CALIBRATING, DONE

    // Motion mode
    @Volatile var motionMode = "VEHICLE" // VEHICLE or PEDESTRIAN

    fun appendTrack(lat: Double, lon: Double) {
        if (lat == 0.0 && lon == 0.0) return
        synchronized(trackLock) {
            trackHistory.add(arrayOf(lat, lon))
            if (trackHistory.size > MAX_TRACK) trackHistory.removeAt(0)
        }
    }

    fun clearTrack() {
        synchronized(trackLock) { trackHistory.clear() }
    }

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri.trimEnd('/')
        val method = session.method

        return when {
            // ---------------------------------------------------------------
            // NAVIGATION STATE — polled by UI every 600ms
            // ---------------------------------------------------------------
            uri == "/navigation/state" && method == Method.GET -> {
                val state = engine.getState()
                val track = synchronized(trackLock) { trackHistory.map { "[${it[0]},${it[1]}]" } }
                val json = buildStateJson(state, track)
                newFixedLengthResponse(Response.Status.OK, "application/json", json)
            }

            // ---------------------------------------------------------------
            // GNSS OUTAGE CONTROL
            // ---------------------------------------------------------------
            uri == "/gnss/blackout" && method == Method.POST -> {
                val ts = android.os.SystemClock.elapsedRealtimeNanos() / 1e9
                engine.startGnssBlackout(ts)
                blackoutActive = true
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"mode":"BLACKOUT"}""")
            }
            uri == "/gnss/restore" && method == Method.POST -> {
                engine.stopGnssBlackout()
                blackoutActive = false
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"mode":"FUSED"}""")
            }

            // ---------------------------------------------------------------
            // CALIBRATION CONTROL
            // ---------------------------------------------------------------
            uri == "/calibrate/start" && method == Method.POST -> {
                calibrationState = "CALIBRATING"
                engine.startCalibration()
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"state":"CALIBRATING"}""")
            }
            uri == "/calibrate/state" && method == Method.GET -> {
                val s = engine.getCalibrationState()
                newFixedLengthResponse(Response.Status.OK, "application/json",
                    """{"state":"${s.status}","accel_bias":[${s.accelBias.joinToString(",")}],"gyro_bias":[${s.gyroBias.joinToString(",")}]}""")
            }

            // ---------------------------------------------------------------
            // SESSION RECORDING CONTROL
            // ---------------------------------------------------------------
            uri == "/session/start" && method == Method.POST -> {
                val name = recorder.startSession()
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"name":"$name"}""")
            }
            uri == "/session/stop" && method == Method.POST -> {
                recorder.stopSession()
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"records":${recorder.recordCount}}""")
            }
            uri == "/session/status" && method == Method.GET -> {
                newFixedLengthResponse(Response.Status.OK, "application/json",
                    """{"recording":${recorder.isRecording},"records":${recorder.recordCount}}""")
            }

            // ---------------------------------------------------------------
            // MOTION MODE
            // ---------------------------------------------------------------
            uri == "/mode/vehicle" && method == Method.POST -> {
                motionMode = "VEHICLE"
                engine.setMotionMode(IDRNavigationEngine.MotionMode.VEHICLE)
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"mode":"VEHICLE"}""")
            }
            uri == "/mode/pedestrian" && method == Method.POST -> {
                motionMode = "PEDESTRIAN"
                engine.setMotionMode(IDRNavigationEngine.MotionMode.PEDESTRIAN)
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"mode":"PEDESTRIAN"}""")
            }

            // ---------------------------------------------------------------
            // DEMO/RESET (keep compatibility with advanced UI's demo button)
            // ---------------------------------------------------------------
            uri == "/demo/reset" && method == Method.POST -> {
                engine.reset()
                clearTrack()
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true}""")
            }

            // ---------------------------------------------------------------
            // HEALTH
            // ---------------------------------------------------------------
            uri == "/health" -> {
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"source":"on-device"}""")
            }

            // ---------------------------------------------------------------
            // STATIC ASSET SERVING — HTML pages
            // ---------------------------------------------------------------
            uri == "" || uri == "/" || uri == "/navigate" -> serveAsset("navigate.html", "text/html")
            uri == "/outage"    -> serveAsset("gnssoutage.html",   "text/html")
            uri == "/calibrate" -> serveAsset("calibration.html",  "text/html")
            uri == "/pipeline"  -> serveAsset("pipeline.html",     "text/html")
            uri == "/sensors"   -> serveAsset("sensors.html",      "text/html")

            // Serve any other asset by path
            else -> {
                val path = uri.removePrefix("/")
                serveAsset(path, mimeForPath(path))
            }
        }
    }

    private fun serveAsset(name: String, mime: String): Response {
        return try {
            val stream: InputStream = assets.open(name)
            newChunkedResponse(Response.Status.OK, mime, stream)
        } catch (_: Exception) {
            newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "Asset not found: $name")
        }
    }

    private fun buildStateJson(state: IDRNavigationEngine.NavigationState, track: List<String>): String {
        val modeLabel = when (state.mode) {
            IDRNavigationEngine.NavMode.GNSS_INS_FUSED -> "GNSS + INS FUSION"
            IDRNavigationEngine.NavMode.DEAD_RECKONING -> "IDR / DEAD RECKONING"
            IDRNavigationEngine.NavMode.REACQUISITION  -> "GNSS REACQUISITION"
            IDRNavigationEngine.NavMode.WAITING_FOR_FIX -> "WAITING FOR GNSS FIX"
            IDRNavigationEngine.NavMode.INITIALIZING   -> "INITIALIZING"
            IDRNavigationEngine.NavMode.ERROR          -> "ERROR"
        }
        val gnssStatus = if (state.gnssBlackout) "SIMULATED OUTAGE" else when (state.mode) {
            IDRNavigationEngine.NavMode.GNSS_INS_FUSED, IDRNavigationEngine.NavMode.REACQUISITION -> "LOCKED"
            IDRNavigationEngine.NavMode.DEAD_RECKONING -> "BLACKOUT"
            else -> "SEARCHING"
        }
        val posLat = if (state.latitude != 0.0) state.latitude else null
        val posLon = if (state.longitude != 0.0) state.longitude else null
        val posJson = if (posLat != null && posLon != null)
            """{"latitude":$posLat,"longitude":$posLon}""" else "null"
        val trackJson = "[${track.joinToString(",")}]"
        val blackoutTimer = if (state.gnssBlackout) state.blackoutSeconds.toInt() else 0
        return """{
  "mode":"$modeLabel",
  "gnss_status":"$gnssStatus",
  "speed_kmh":${String.format("%.1f", state.speedMps * 3.6)},
  "heading_deg":${String.format("%.1f", state.headingDeg)},
  "position":$posJson,
  "uncertainty_m":${if (state.gnssAccuracyM > 0) String.format("%.1f", state.gnssAccuracyM) else "null"},
  "imu_hz":${String.format("%.1f", state.imuHz)},
  "gnss_hz":${String.format("%.2f", state.gnssHz)},
  "gnss_blackout":${state.gnssBlackout},
  "blackout_seconds":$blackoutTimer,
  "dr_dist_m":${String.format("%.1f", state.drDriftM)},
  "calibration":"${state.calibrationStatus}",
  "motion_mode":"$motionMode",
  "accel_x":${String.format("%.3f", state.lastAccelX)},
  "accel_y":${String.format("%.3f", state.lastAccelY)},
  "accel_z":${String.format("%.3f", state.lastAccelZ)},
  "gyro_x":${String.format("%.4f", state.lastGyroX)},
  "gyro_y":${String.format("%.4f", state.lastGyroY)},
  "gyro_z":${String.format("%.4f", state.lastGyroZ)},
  "last_imu_ms":${state.lastImuTs},
  "last_gnss_ms":${state.lastGnssTs},
  "has_gnss_fix":${posLat != null},
  "recording":${recorder.isRecording},
  "record_count":${recorder.recordCount},
  "track":$trackJson
}"""
    }

    private fun mimeForPath(path: String): String = when {
        path.endsWith(".html") -> "text/html"
        path.endsWith(".css")  -> "text/css"
        path.endsWith(".js")   -> "application/javascript"
        path.endsWith(".json") -> "application/json"
        path.endsWith(".png")  -> "image/png"
        path.endsWith(".svg")  -> "image/svg+xml"
        else -> "application/octet-stream"
    }
}
