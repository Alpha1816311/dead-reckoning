package com.idr.server

import android.content.res.AssetManager
import com.idr.navigation.IDRNavigationEngine
import com.idr.navigation.SessionRecorder
import org.json.JSONObject
import java.io.InputStream

/**
 * Embedded NanoHTTPD-based local HTTP server.
 * Serves the IDR Fusion UI from Android assets and provides
 * a REST API that the UI JavaScript polls.
 *
 * All navigation state comes from IDRNavigationEngine — no fake values.
 */
class IDRLocalServer(
    port: Int,
    private val assets: AssetManager,
    private val engine: IDRNavigationEngine,
    private val recorder: SessionRecorder,
    private val onExportRequest: (() -> Unit)? = null
) : NanoHTTPD(port) {

    // Track history for the map path (sent to UI as `state.track`)
    private val trackHistory = mutableListOf<Array<Double>>()
    private val trackLock = Any()
    private val MAX_TRACK = 600

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
            // NAVIGATION STATE — polled by UI
            // ---------------------------------------------------------------
            uri == "/navigation/state" && method == Method.GET -> {
                val state = engine.getState()
                val track = synchronized(trackLock) { trackHistory.map { "[${it[0]},${it[1]}]" } }
                newFixedLengthResponse(Response.Status.OK, "application/json", buildStateJson(state, track))
            }

            // ---------------------------------------------------------------
            // GNSS OUTAGE CONTROL
            // ---------------------------------------------------------------
            uri == "/gnss/blackout" && method == Method.POST -> {
                val ts = android.os.SystemClock.elapsedRealtimeNanos() / 1e9
                engine.startGnssBlackout(ts)
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"mode":"BLACKOUT"}""")
            }
            uri == "/gnss/restore" && method == Method.POST -> {
                engine.stopGnssBlackout()
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"mode":"FUSED"}""")
            }

            // ---------------------------------------------------------------
            // CALIBRATION CONTROL
            // ---------------------------------------------------------------
            uri == "/calibrate/start" && method == Method.POST -> {
                engine.startCalibration()
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"state":"CALIBRATING"}""")
            }
            uri == "/calibrate/state" && method == Method.GET -> {
                val s = engine.getCalibrationState()
                newFixedLengthResponse(Response.Status.OK, "application/json",
                    """{"state":"${s.status}","accel_bias":[${s.accelBias.joinToString(",")}],"gyro_bias":[${s.gyroBias.joinToString(",")}]}""")
            }

            // ---------------------------------------------------------------
            // ALIGNMENT CONFIGURATION
            // POST /config/alignment
            // Body: { "forward_phone": [x,y,z], "up_phone": [x,y,z] }
            // ---------------------------------------------------------------
            uri == "/config/alignment" && method == Method.POST -> {
                val body = readBody(session)
                try {
                    val json = JSONObject(body)
                    val fwd = json.getJSONArray("forward_phone")
                    val up  = json.getJSONArray("up_phone")
                    engine.setAlignment(
                        doubleArrayOf(fwd.getDouble(0), fwd.getDouble(1), fwd.getDouble(2)),
                        doubleArrayOf(up.getDouble(0),  up.getDouble(1),  up.getDouble(2))
                    )
                    val st = engine.getState()
                    newFixedLengthResponse(Response.Status.OK, "application/json",
                        """{"ok":true,"mounting_calibrated":${st.mountingCalibrated}}""")
                } catch (e: Exception) {
                    newFixedLengthResponse(Response.Status.BAD_REQUEST, "application/json",
                        """{"ok":false,"error":"${e.message}"}""")
                }
            }

            // ---------------------------------------------------------------
            // RUNTIME CONFIGURATION — PATCH /config
            // Body: { "filter_mode": "strict"|"balanced"|"raw", "nhc_enabled": true|false, "profile": "phone"|"edge" }
            // ---------------------------------------------------------------
            uri == "/config" && method == Method.PATCH -> {
                val body = readBody(session)
                try {
                    val json = JSONObject(body)
                    if (json.has("filter_mode")) {
                        engine.filterMode = when (json.getString("filter_mode").lowercase()) {
                            "strict" -> IDRNavigationEngine.FilterMode.STRICT
                            "raw"    -> IDRNavigationEngine.FilterMode.RAW
                            else     -> IDRNavigationEngine.FilterMode.BALANCED
                        }
                    }
                    if (json.has("nhc_enabled")) {
                        engine.nhcEnabled = json.getBoolean("nhc_enabled")
                    }
                    if (json.has("profile")) {
                        val profile = json.getString("profile")
                        if (profile == "vehicle") engine.setMotionMode(IDRNavigationEngine.MotionMode.VEHICLE)
                        else if (profile == "pedestrian") engine.setMotionMode(IDRNavigationEngine.MotionMode.PEDESTRIAN)
                    }
                    val st = engine.getState()
                    val filterLabel = st.filterMode.name.lowercase()
                    newFixedLengthResponse(Response.Status.OK, "application/json",
                        """{"ok":true,"configuration":{"filter_mode":"$filterLabel","nhc_enabled":${st.nhcEnabled}}}""")
                } catch (e: Exception) {
                    newFixedLengthResponse(Response.Status.BAD_REQUEST, "application/json",
                        """{"ok":false,"error":"${e.message}"}""")
                }
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
            // SESSION EXPORT — triggers native Android share sheet via callback
            // This is the HTTP fallback path; the primary path is Android.exportSession()
            // ---------------------------------------------------------------
            uri == "/session/export" && method == Method.POST -> {
                if (recorder.recordCount == 0 && recorder.getSessionFile() == null) {
                    newFixedLengthResponse(Response.Status.OK, "application/json",
                        """{"ok":false,"error":"NO SESSION DATA AVAILABLE TO EXPORT"}""")
                } else {
                    onExportRequest?.invoke()
                    newFixedLengthResponse(Response.Status.OK, "application/json",
                        """{"ok":true,"records":${recorder.recordCount}}""")
                }
            }

            // ---------------------------------------------------------------
            // MOTION MODE
            // ---------------------------------------------------------------
            uri == "/mode/vehicle" && method == Method.POST -> {
                engine.setMotionMode(IDRNavigationEngine.MotionMode.VEHICLE)
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"mode":"VEHICLE"}""")
            }
            uri == "/mode/pedestrian" && method == Method.POST -> {
                engine.setMotionMode(IDRNavigationEngine.MotionMode.PEDESTRIAN)
                newFixedLengthResponse(Response.Status.OK, "application/json", """{"ok":true,"mode":"PEDESTRIAN"}""")
            }

            // ---------------------------------------------------------------
            // RESET (clear track + reset engine)
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
            // STATIC ASSET SERVING — map routes to asset filenames
            // ---------------------------------------------------------------
            uri == "" || uri == "/" || uri == "/navigate" -> serveAsset("navigate.html", "text/html")
            uri == "/outage"     -> serveAsset("outage.html",      "text/html")
            uri == "/calibration"-> serveAsset("calibration.html", "text/html")
            uri == "/pipeline"   -> serveAsset("pipeline.html",    "text/html")
            uri == "/settings"   -> serveAsset("settings.html",    "text/html")

            // Serve any other asset by path
            else -> {
                val path = uri.removePrefix("/")
                serveAsset(path, mimeForPath(path))
            }
        }
    }

    private fun readBody(session: IHTTPSession): String {
        return try {
            session.getBody()
        } catch (_: Exception) { "" }
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
            IDRNavigationEngine.NavMode.GNSS_INS_FUSED   -> "GNSS + INS FUSION"
            IDRNavigationEngine.NavMode.DEAD_RECKONING   -> "IDR / DEAD RECKONING"
            IDRNavigationEngine.NavMode.REACQUISITION    -> "GNSS REACQUISITION"
            IDRNavigationEngine.NavMode.WAITING_FOR_FIX -> "WAITING FOR GNSS FIX"
            IDRNavigationEngine.NavMode.INITIALIZING    -> "INITIALIZING"
            IDRNavigationEngine.NavMode.ERROR           -> "ERROR"
        }
        val inBlackout = state.gnssBlackout
        val gnssStatus = if (inBlackout) "SIMULATED OUTAGE" else when (state.mode) {
            IDRNavigationEngine.NavMode.GNSS_INS_FUSED, IDRNavigationEngine.NavMode.REACQUISITION -> "LOCKED"
            IDRNavigationEngine.NavMode.DEAD_RECKONING -> "BLACKOUT"
            IDRNavigationEngine.NavMode.WAITING_FOR_FIX -> "SEARCHING"
            else -> "SEARCHING"
        }
        val hasPosition = state.latitude != 0.0 && state.longitude != 0.0
        val posJson = if (hasPosition)
            """{"latitude":${state.latitude},"longitude":${state.longitude}}"""
        else "null"

        val trackJson = "[${track.joinToString(",")}]"
        val blackoutTimer = if (inBlackout) state.blackoutSeconds.toInt() else 0
        val filterLabel  = state.filterMode.name.lowercase()
        val imuStatus    = if (state.imuCount > 0) "ACTIVE" else "WAITING"
        val orientInit   = state.mountingCalibrated || state.imuCount > 50

        // Magnetometer
        val magStatus = if (state.hasMag) "ACTIVE" else "UNAVAILABLE"

        // Speed: emit null when UNAVAILABLE so the UI can show "—" rather than a fake number.
        // For STATIONARY always emit 0.  For GNSS/INERTIAL emit the real value.
        val speedSourceLabel = state.speedSource.name  // "GNSS", "INERTIAL", "STATIONARY", "UNAVAILABLE"
        val isUnavailable = state.speedSource == IDRNavigationEngine.SpeedSource.UNAVAILABLE
        val speedKmhJson  = if (isUnavailable) "null" else String.format("%.1f", state.speedMps * 3.6)
        val speedMpsJson  = if (isUnavailable) "null" else String.format("%.2f", state.speedMps)

        return """{
  "mode":"$modeLabel",
  "gnss_status":"$gnssStatus",
  "gnss_state":"${if (inBlackout) "SIMULATED_OUTAGE_LOST" else gnssStatus}",
  "speed_kmh":$speedKmhJson,
  "speed_mps":$speedMpsJson,
  "speed_source":"$speedSourceLabel",
  "heading_deg":${String.format("%.1f", state.headingDeg)},
  "position":$posJson,
  "latitude":${if (hasPosition) state.latitude else "null"},
  "longitude":${if (hasPosition) state.longitude else "null"},
  "uncertainty_m":${if (state.gnssAccuracyM > 0) String.format("%.1f", state.gnssAccuracyM) else "null"},
  "imu_hz":${String.format("%.1f", state.imuHz)},
  "gnss_hz":${String.format("%.2f", state.gnssHz)},
  "gnss_blackout":${state.gnssBlackout},
  "blackout_seconds":$blackoutTimer,
  "dr_dist_m":${String.format("%.1f", state.drDriftM)},
  "calibration":"${state.calibrationStatus}",
  "accel_x":${String.format("%.3f", state.lastAccelX)},
  "accel_y":${String.format("%.3f", state.lastAccelY)},
  "accel_z":${String.format("%.3f", state.lastAccelZ)},
  "gyro_x":${String.format("%.4f", state.lastGyroX)},
  "gyro_y":${String.format("%.4f", state.lastGyroY)},
  "gyro_z":${String.format("%.4f", state.lastGyroZ)},
  "has_mag":${state.hasMag},
  "mag_status":"$magStatus",
  "mag_x":${if (state.hasMag) String.format("%.1f", state.lastMagX) else "null"},
  "mag_y":${if (state.hasMag) String.format("%.1f", state.lastMagY) else "null"},
  "mag_z":${if (state.hasMag) String.format("%.1f", state.lastMagZ) else "null"},
  "last_imu_ms":${state.lastImuTs},
  "last_gnss_ms":${state.lastGnssTs},
  "has_gnss_fix":$hasPosition,
  "accepted_imu":${state.imuCount},
  "accepted_gnss":${state.gnssCount},
  "rejected_samples":0,
  "imu_status":"$imuStatus",
  "nhc_status":"${if (state.nhcEnabled) "ACTIVE" else "DISABLED"}",
  "nhc_enabled":${state.nhcEnabled},
  "ai_model_status":"NOT LOADED",
  "map_status":"MAP DISPLAY ACTIVE",
  "filter_mode":"$filterLabel",
  "mounting_calibrated":${state.mountingCalibrated},
  "orientation_initialized":$orientInit,
  "is_stationary":${state.isStationary},
  "recording":${recorder.isRecording},
  "record_count":${recorder.recordCount},
  "track":$trackJson,
  "timestamp":${String.format("%.2f", state.timestamp)},
  "configuration":{"filter_mode":"$filterLabel","nhc_enabled":${state.nhcEnabled}}
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
