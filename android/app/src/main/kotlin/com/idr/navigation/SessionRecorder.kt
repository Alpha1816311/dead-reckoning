package com.idr.navigation

import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Session recorder — writes NDJSON (newline-delimited JSON) to local storage.
 * Each line is one timestamped record containing raw sensor data and nav output.
 */
class SessionRecorder(private val context: Context) {

    data class SessionRecord(
        val wallMs: Long,
        val timestamp: Double,
        val ax: Double, val ay: Double, val az: Double,
        val gx: Double, val gy: Double, val gz: Double,
        val mx: Double?, val my: Double?, val mz: Double?,
        val gnssLat: Double?, val gnssLon: Double?,
        val gnssSpeedMps: Double?, val gnssAccuracyM: Double?,
        val navLat: Double, val navLon: Double,
        val navSpeedMps: Double, val navHeadingDeg: Double,
        val navMode: String,
        val gnssBlackout: Boolean,
        val blackoutSeconds: Double
    )

    private val recording = AtomicBoolean(false)
    private var writer: BufferedWriter? = null
    private var sessionFile: File? = null
    private var sessionStart: Long = 0L
    private val lock = Any()

    val isRecording: Boolean get() = recording.get()
    var recordCount: Int = 0
        private set

    fun startSession(): String {
        synchronized(lock) {
            if (recording.get()) return sessionFile?.name ?: ""
            val sdf = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
            val name = "idr_session_${sdf.format(Date())}.ndjson"
            val dir = context.getExternalFilesDir(null) ?: context.filesDir
            dir.mkdirs()
            sessionFile = File(dir, name)
            writer = BufferedWriter(FileWriter(sessionFile!!, false))
            sessionStart = System.currentTimeMillis()
            recordCount = 0
            recording.set(true)
            return name
        }
    }

    fun record(r: SessionRecord) {
        if (!recording.get()) return
        val line = buildJsonLine(r)
        synchronized(lock) {
            try {
                writer?.write(line)
                writer?.newLine()
                recordCount++
                // Flush every 10 records to avoid data loss without excessive I/O
                if (recordCount % 10 == 0) writer?.flush()
            } catch (_: Exception) {}
        }
    }

    fun stopSession(): File? {
        synchronized(lock) {
            if (!recording.get()) return null
            recording.set(false)
            try {
                writer?.flush()
                writer?.close()
            } catch (_: Exception) {}
            writer = null
            return sessionFile
        }
    }

    fun getSessionFile(): File? = sessionFile

    private fun buildJsonLine(r: SessionRecord): String {
        val obj = JSONObject()
        obj.put("wall_ms", r.wallMs)
        obj.put("timestamp", r.timestamp)
        obj.put("ax", r.ax); obj.put("ay", r.ay); obj.put("az", r.az)
        obj.put("gx", r.gx); obj.put("gy", r.gy); obj.put("gz", r.gz)
        r.mx?.let { obj.put("mx", it) }
        r.my?.let { obj.put("my", it) }
        r.mz?.let { obj.put("mz", it) }
        r.gnssLat?.let { obj.put("gnss_lat", it) }
        r.gnssLon?.let { obj.put("gnss_lon", it) }
        r.gnssSpeedMps?.let { obj.put("gnss_speed_mps", it) }
        r.gnssAccuracyM?.let { obj.put("gnss_accuracy_m", it) }
        obj.put("nav_lat", r.navLat)
        obj.put("nav_lon", r.navLon)
        obj.put("nav_speed_mps", r.navSpeedMps)
        obj.put("nav_heading_deg", r.navHeadingDeg)
        obj.put("nav_mode", r.navMode)
        obj.put("gnss_blackout", r.gnssBlackout)
        obj.put("blackout_seconds", r.blackoutSeconds)
        return obj.toString()
    }
}
