package com.idr.navigation

import kotlin.math.*

/**
 * IDR Navigation Engine — self-contained Kotlin implementation.
 *
 * Implements:
 *  - GNSS+INS complementary fusion with adaptive weighting
 *  - Dead reckoning via gyro heading + accel speed integration
 *  - Stationary detection + NHC vehicle constraints
 *  - Gyro yaw-axis auto-calibration from GNSS course
 *  - Bias estimation during stationary phases
 *  - Smooth GNSS reacquisition blending
 *  - Calibration workflow
 *  - IMU Hz / GNSS Hz rate estimation
 *  - Runtime filter/NHC/alignment configuration
 *
 * All state is kept here. No server, no WebSocket, no laptop required.
 */
class IDRNavigationEngine {

    enum class NavMode {
        INITIALIZING,
        WAITING_FOR_FIX,
        GNSS_INS_FUSED,
        DEAD_RECKONING,
        REACQUISITION,
        ERROR
    }

    enum class MotionMode { VEHICLE, PEDESTRIAN }

    enum class FilterMode { STRICT, BALANCED, RAW }

    data class CalibrationResult(
        val status: String,       // NOT_CALIBRATED, CALIBRATING, CALIBRATED
        val accelBias: DoubleArray = DoubleArray(3),
        val gyroBias: DoubleArray = DoubleArray(3)
    )

    data class NavigationState(
        val timestamp: Double = 0.0,
        val latitude: Double = 0.0,
        val longitude: Double = 0.0,
        val speedMps: Double = 0.0,
        val headingDeg: Double = 0.0,
        val mode: NavMode = NavMode.INITIALIZING,
        val imuHz: Double = 0.0,
        val gnssHz: Double = 0.0,
        val gnssBlackout: Boolean = false,
        val blackoutSeconds: Double = 0.0,
        val drDriftM: Double = 0.0,
        val lastImuTs: Long = 0L,
        val lastGnssTs: Long = 0L,
        val calibrationStatus: String = "NOT_CALIBRATED",
        val gnssAccuracyM: Double = 0.0,
        // Raw sensor values for UI display
        val lastAccelX: Double = 0.0,
        val lastAccelY: Double = 0.0,
        val lastAccelZ: Double = 0.0,
        val lastGyroX: Double = 0.0,
        val lastGyroY: Double = 0.0,
        val lastGyroZ: Double = 0.0,
        // Magnetometer
        val hasMag: Boolean = false,
        val lastMagX: Double = 0.0,
        val lastMagY: Double = 0.0,
        val lastMagZ: Double = 0.0,
        // Counters
        val imuCount: Int = 0,
        val gnssCount: Int = 0,
        // Configuration
        val filterMode: FilterMode = FilterMode.BALANCED,
        val nhcEnabled: Boolean = true,
        val mountingCalibrated: Boolean = false,
        // Derived
        val isStationary: Boolean = false
    )

    // ---------------------------------------------------------------
    // Position / navigation state
    // ---------------------------------------------------------------
    @Volatile private var posLat: Double = 0.0
    @Volatile private var posLon: Double = 0.0
    @Volatile private var headingDeg: Double = 0.0
    @Volatile private var speedMps: Double = 0.0
    @Volatile private var mode: NavMode = NavMode.INITIALIZING
    @Volatile private var initialized: Boolean = false
    @Volatile private var gnssBlackedOut: Boolean = false
    @Volatile private var blackoutStartTs: Double = 0.0
    @Volatile private var lastImuTs: Double = -1.0
    @Volatile private var lastGnssTs: Double = -1.0
    @Volatile private var gnssCount: Int = 0
    @Volatile private var imuCount: Int = 0
    @Volatile private var drDistanceM: Double = 0.0
    @Volatile private var gnssAccuracyM: Double = 0.0
    @Volatile private var motionMode: MotionMode = MotionMode.VEHICLE

    // Runtime configuration
    @Volatile var filterMode: FilterMode = FilterMode.BALANCED
    @Volatile var nhcEnabled: Boolean = true
    @Volatile private var mountingCalibrated: Boolean = false

    // Phone forward/up vectors for alignment
    private val forwardPhone = DoubleArray(3) { if (it == 1) 1.0 else 0.0 } // +Y default
    private val upPhone = DoubleArray(3) { if (it == 2) 1.0 else 0.0 }      // +Z default

    // Raw sensor values (for UI display)
    @Volatile private var rawAx: Double = 0.0
    @Volatile private var rawAy: Double = 0.0
    @Volatile private var rawAz: Double = 9.81
    @Volatile private var rawGx: Double = 0.0
    @Volatile private var rawGy: Double = 0.0
    @Volatile private var rawGz: Double = 0.0

    // Magnetometer
    @Volatile private var hasMag: Boolean = false
    @Volatile private var rawMx: Double = 0.0
    @Volatile private var rawMy: Double = 0.0
    @Volatile private var rawMz: Double = 0.0

    // Wall-clock times for UI
    @Volatile var lastImuWallMs: Long = 0L
    @Volatile var lastGnssWallMs: Long = 0L

    // ---------------------------------------------------------------
    // Calibration
    // ---------------------------------------------------------------
    private enum class CalibState { NOT_CALIBRATED, CALIBRATING, CALIBRATED }
    @Volatile private var calibState = CalibState.NOT_CALIBRATED
    private val calibAccelSamples = mutableListOf<DoubleArray>()
    private val calibGyroSamples = mutableListOf<DoubleArray>()
    private val calibLock = Any()
    private val CALIB_SAMPLES_NEEDED = 200
    private val accelBias = DoubleArray(3)
    private val gyroBias  = DoubleArray(3)

    // ---------------------------------------------------------------
    // Gravity low-pass (for linear acceleration extraction)
    // ---------------------------------------------------------------
    private val gravLp = DoubleArray(3) { 0.0 }
    private val LP_ALPHA = 0.85

    // ---------------------------------------------------------------
    // Speed smoothing
    // ---------------------------------------------------------------
    private val speedHistory = ArrayDeque<Double>()
    private val SPEED_HIST = 5

    // ---------------------------------------------------------------
    // Rate estimation
    // ---------------------------------------------------------------
    private val imuTsWindow = ArrayDeque<Double>()
    private val gnssTsWindow = ArrayDeque<Double>()
    private val RATE_WIN_SEC = 3.0

    // ---------------------------------------------------------------
    // Gyro yaw-axis calibration
    // ---------------------------------------------------------------
    private var gyroAxis = 2
    private var gyroSign = -1.0
    private var yawCalibrated = false

    // ---------------------------------------------------------------
    // GNSS course tracking
    // ---------------------------------------------------------------
    private var prevGnssLat: Double? = null
    private var prevGnssLon: Double? = null
    private var prevGnssTs: Double? = null

    // ---------------------------------------------------------------
    // GNSS reacquisition
    // ---------------------------------------------------------------
    private var reacqCount = 0
    private val REACQ_STEPS = 6

    // ---------------------------------------------------------------
    // Stationary detection
    // ---------------------------------------------------------------
    private val accelMagBuf = ArrayDeque<Double>()
    private val STATIONARY_WIN = 20
    private val STATIONARY_THRESH = 0.15 // m/s²

    @Volatile private var stationaryState = false

    // ---------------------------------------------------------------
    // PUBLIC API
    // ---------------------------------------------------------------

    fun reset() {
        initialized = false
        gnssBlackedOut = false
        mode = NavMode.INITIALIZING
        lastImuTs = -1.0
        lastGnssTs = -1.0
        gnssCount = 0
        imuCount = 0
        drDistanceM = 0.0
        speedMps = 0.0
        headingDeg = 0.0
        imuTsWindow.clear()
        gnssTsWindow.clear()
        speedHistory.clear()
        accelMagBuf.clear()
        prevGnssLat = null
        prevGnssLon = null
        prevGnssTs = null
        reacqCount = 0
        yawCalibrated = false
        stationaryState = false
    }

    fun setMotionMode(m: MotionMode) { motionMode = m }

    fun setAlignment(fwdPhone: DoubleArray, upPh: DoubleArray) {
        for (i in 0..2) {
            forwardPhone[i] = fwdPhone[i]
            upPhone[i] = upPh[i]
        }
        // Recalculate gyro axis based on forward phone vector
        // The yaw axis is the one that corresponds to the up direction
        gyroAxis = upPh.indices.maxByOrNull { abs(upPh[it]) } ?: 2
        // Apply sign based on orientation
        gyroSign = if (upPh[gyroAxis] > 0) -1.0 else 1.0
        mountingCalibrated = true
        yawCalibrated = true
    }

    fun startCalibration() {
        synchronized(calibLock) {
            calibState = CalibState.CALIBRATING
            calibAccelSamples.clear()
            calibGyroSamples.clear()
        }
    }

    fun processMag(mx: Float, my: Float, mz: Float) {
        hasMag = true
        rawMx = mx.toDouble()
        rawMy = my.toDouble()
        rawMz = mz.toDouble()
    }

    fun getCalibrationState(): CalibrationResult = when (calibState) {
        CalibState.NOT_CALIBRATED -> CalibrationResult("NOT_CALIBRATED")
        CalibState.CALIBRATING    -> CalibrationResult("CALIBRATING")
        CalibState.CALIBRATED     -> CalibrationResult("CALIBRATED",
            accelBias.copyOf(), gyroBias.copyOf())
    }

    /**
     * Process one IMU sample. Called from Android sensor callback thread.
     */
    fun processImu(
        ax: Double, ay: Double, az: Double,
        gx: Double, gy: Double, gz: Double,
        timestamp: Double
    ) {
        lastImuWallMs = System.currentTimeMillis()
        rawAx = ax; rawAy = ay; rawAz = az
        rawGx = gx; rawGy = gy; rawGz = gz
        imuCount++
        trackRate(imuTsWindow, timestamp)

        val dt = if (lastImuTs > 0) (timestamp - lastImuTs).coerceIn(0.001, 0.5) else 0.02
        lastImuTs = timestamp

        // -----------------------------------------------------------------------
        // Calibration data collection
        // -----------------------------------------------------------------------
        if (calibState == CalibState.CALIBRATING) {
            synchronized(calibLock) {
                calibAccelSamples.add(doubleArrayOf(ax, ay, az))
                calibGyroSamples.add(doubleArrayOf(gx, gy, gz))
                if (calibAccelSamples.size >= CALIB_SAMPLES_NEEDED) {
                    finishCalibration()
                }
            }
        }

        // -----------------------------------------------------------------------
        // Apply calibration bias if available
        // -----------------------------------------------------------------------
        val cax = ax - (if (calibState == CalibState.CALIBRATED) accelBias[0] else 0.0)
        val cay = ay - (if (calibState == CalibState.CALIBRATED) accelBias[1] else 0.0)
        val caz = az - (if (calibState == CalibState.CALIBRATED) accelBias[2] else 0.0)
        val cgx = gx - (if (calibState == CalibState.CALIBRATED) gyroBias[0] else 0.0)
        val cgy = gy - (if (calibState == CalibState.CALIBRATED) gyroBias[1] else 0.0)
        val cgz = gz - (if (calibState == CalibState.CALIBRATED) gyroBias[2] else 0.0)

        // -----------------------------------------------------------------------
        // Apply filter mode — thresholding for vibration rejection
        // -----------------------------------------------------------------------
        val accelThreshold = when (filterMode) {
            FilterMode.STRICT   -> 0.08
            FilterMode.BALANCED -> 0.04
            FilterMode.RAW      -> 0.0
        }
        val gyroThreshold = when (filterMode) {
            FilterMode.STRICT   -> 0.012
            FilterMode.BALANCED -> 0.005
            FilterMode.RAW      -> 0.0
        }

        // -----------------------------------------------------------------------
        // Gravity low-pass filter
        // -----------------------------------------------------------------------
        gravLp[0] = LP_ALPHA * gravLp[0] + (1 - LP_ALPHA) * cax
        gravLp[1] = LP_ALPHA * gravLp[1] + (1 - LP_ALPHA) * cay
        gravLp[2] = LP_ALPHA * gravLp[2] + (1 - LP_ALPHA) * caz

        // Linear acceleration (gravity-compensated)
        val lax = cax - gravLp[0]
        val lay = cay - gravLp[1]
        val laz = caz - gravLp[2]

        // Accel magnitude for stationary detection
        val aMag = sqrt(lax * lax + lay * lay + laz * laz)
        accelMagBuf.addLast(aMag)
        if (accelMagBuf.size > STATIONARY_WIN) accelMagBuf.removeFirst()
        val isStationary = accelMagBuf.size >= STATIONARY_WIN &&
            accelMagBuf.average() < STATIONARY_THRESH
        stationaryState = isStationary

        // -----------------------------------------------------------------------
        // Gyro yaw rate (use calibrated axis)
        // -----------------------------------------------------------------------
        val rawYaw = when (gyroAxis) { 0 -> cgx; 1 -> cgy; else -> cgz } * gyroSign
        val filteredYaw = if (abs(rawYaw) < gyroThreshold) 0.0 else rawYaw

        if (!initialized) {
            if (gnssCount >= 2) { initialized = true; mode = NavMode.GNSS_INS_FUSED }
            else { mode = NavMode.WAITING_FOR_FIX; return }
        }

        // -----------------------------------------------------------------------
        // Heading integration
        // -----------------------------------------------------------------------
        headingDeg = wrapHeading(headingDeg + Math.toDegrees(filteredYaw * dt))

        // -----------------------------------------------------------------------
        // Dead reckoning position integration (when GNSS is blacked out)
        // -----------------------------------------------------------------------
        if (mode == NavMode.DEAD_RECKONING) {
            if (isStationary || (nhcEnabled && isStationary)) {
                speedMps *= 0.95 // decay toward zero
            } else {
                val forwardAccel = lax * sin(Math.toRadians(headingDeg)) +
                                   lay * cos(Math.toRadians(headingDeg))
                val filteredAccel = if (abs(forwardAccel) < accelThreshold) 0.0 else forwardAccel
                val accelContrib = filteredAccel * dt
                speedMps = (speedMps + accelContrib * 0.25).coerceIn(0.0,
                    if (motionMode == MotionMode.VEHICLE) 55.0 else 10.0)
                // NHC: constrain speed decay when no accel
                if (nhcEnabled && aMag < 0.2) speedMps *= 0.98
            }

            // NHC: for vehicle mode, zero out lateral/vertical drift
            val dist = speedMps * dt
            drDistanceM += dist
            val hRad = Math.toRadians(headingDeg)
            posLat += Math.toDegrees(dist * cos(hRad) / EARTH_R)
            posLon += Math.toDegrees(dist * sin(hRad) / (EARTH_R * cos(Math.toRadians(posLat))))
        }
    }

    /**
     * Process a GNSS fix. Called from Android location callback.
     */
    fun processGnss(
        latitude: Double, longitude: Double,
        speedMpsRaw: Double?, accuracyM: Double?,
        timestamp: Double
    ) {
        lastGnssWallMs = System.currentTimeMillis()
        trackRate(gnssTsWindow, timestamp)
        gnssAccuracyM = accuracyM ?: 20.0

        // Blackout gate — GNSS measurements do not reach the fusion layer
        if (gnssBlackedOut) return

        val acc = accuracyM ?: 20.0
        if (acc > 60.0) return // too inaccurate to use

        val prevLat = prevGnssLat; val prevLon = prevGnssLon; val prevTs = prevGnssTs
        if (prevLat != null && prevLon != null && prevTs != null) {
            val dtG = (timestamp - prevTs).coerceIn(0.01, 60.0)
            val dist = haversineM(prevLat, prevLon, latitude, longitude)
            if (dist > 2.0) {
                val course = bearing(prevLat, prevLon, latitude, longitude)
                val gnssSpd = dist / dtG

                // Auto-calibrate gyro yaw axis from GNSS course (if not manually set)
                if (!yawCalibrated && gnssCount >= 5) {
                    yawCalibrated = true
                    gyroAxis = 2; gyroSign = -1.0 // best default for portrait phone
                }

                // Heading blend — trust GNSS more when accurate
                val gw = if (acc < 5.0) 0.65 else if (acc < 15.0) 0.35 else 0.15
                headingDeg = blendHeading(headingDeg, course, gw)

                // Speed blend
                val sp = speedMpsRaw ?: gnssSpd
                speedHistory.addLast(sp.coerceIn(0.0, 60.0))
                if (speedHistory.size > SPEED_HIST) speedHistory.removeFirst()
                speedMps = speedHistory.average()
            }
        }
        prevGnssLat = latitude; prevGnssLon = longitude; prevGnssTs = timestamp
        gnssCount++; lastGnssTs = timestamp

        if (!initialized && gnssCount >= 2) {
            initialized = true; posLat = latitude; posLon = longitude
        }

        if (initialized) {
            val fuseAlpha = if (acc < 5.0) 0.92 else if (acc < 15.0) 0.75 else 0.55
            when (mode) {
                NavMode.DEAD_RECKONING -> {
                    mode = NavMode.REACQUISITION; reacqCount = 0; drDistanceM = 0.0
                }
                NavMode.REACQUISITION -> {
                    reacqCount++
                    val alpha = (reacqCount.toDouble() / REACQ_STEPS).coerceIn(0.0, 1.0)
                    posLat = lerp(posLat, latitude, alpha)
                    posLon = lerp(posLon, longitude, alpha)
                    if (reacqCount >= REACQ_STEPS) {
                        posLat = latitude; posLon = longitude
                        mode = NavMode.GNSS_INS_FUSED
                    }
                }
                else -> {
                    posLat = lerp(posLat, latitude, fuseAlpha)
                    posLon = lerp(posLon, longitude, fuseAlpha)
                    mode = NavMode.GNSS_INS_FUSED
                }
            }
        }
    }

    fun startGnssBlackout(currentTs: Double) {
        gnssBlackedOut = true
        blackoutStartTs = currentTs
        if (mode == NavMode.GNSS_INS_FUSED || mode == NavMode.REACQUISITION) {
            mode = NavMode.DEAD_RECKONING; drDistanceM = 0.0
        }
    }

    fun stopGnssBlackout() {
        gnssBlackedOut = false
        if (mode == NavMode.DEAD_RECKONING) {
            mode = NavMode.REACQUISITION; reacqCount = 0
        }
    }

    fun getState(): NavigationState {
        val nowTs = lastImuTs.coerceAtLeast(0.0)
        val blackoutSec = if (gnssBlackedOut && blackoutStartTs > 0) {
            (nowTs - blackoutStartTs).coerceAtLeast(0.0)
        } else 0.0
        val calibLabel = when (calibState) {
            CalibState.NOT_CALIBRATED -> "NOT_CALIBRATED"
            CalibState.CALIBRATING    -> "CALIBRATING"
            CalibState.CALIBRATED     -> "CALIBRATED"
        }
        return NavigationState(
            timestamp = nowTs,
            latitude = posLat, longitude = posLon,
            speedMps = speedMps, headingDeg = headingDeg,
            mode = mode,
            imuHz = estimateRate(imuTsWindow),
            gnssHz = estimateRate(gnssTsWindow),
            gnssBlackout = gnssBlackedOut,
            blackoutSeconds = blackoutSec,
            drDriftM = drDistanceM,
            lastImuTs = lastImuWallMs,
            lastGnssTs = lastGnssWallMs,
            calibrationStatus = calibLabel,
            gnssAccuracyM = gnssAccuracyM,
            lastAccelX = rawAx, lastAccelY = rawAy, lastAccelZ = rawAz,
            lastGyroX = rawGx, lastGyroY = rawGy, lastGyroZ = rawGz,
            hasMag = hasMag,
            lastMagX = rawMx, lastMagY = rawMy, lastMagZ = rawMz,
            imuCount = imuCount,
            gnssCount = gnssCount,
            filterMode = filterMode,
            nhcEnabled = nhcEnabled,
            mountingCalibrated = mountingCalibrated,
            isStationary = stationaryState
        )
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    private fun finishCalibration() {
        if (calibAccelSamples.isEmpty()) return
        for (i in 0..2) {
            accelBias[i] = calibAccelSamples.map { it[i] }.average()
            gyroBias[i]  = calibGyroSamples.map { it[i] }.average()
        }
        // Preserve gravity on Z axis — only remove sensor bias, not gravity
        accelBias[2] -= 9.81
        calibState = CalibState.CALIBRATED
        calibAccelSamples.clear()
        calibGyroSamples.clear()
    }

    private fun trackRate(buf: ArrayDeque<Double>, ts: Double) {
        buf.addLast(ts)
        val cutoff = ts - RATE_WIN_SEC
        while (buf.isNotEmpty() && buf.first() < cutoff) buf.removeFirst()
    }

    private fun estimateRate(buf: ArrayDeque<Double>): Double {
        if (buf.size < 2) return 0.0
        val span = buf.last() - buf.first()
        return if (span > 0) (buf.size - 1) / span else 0.0
    }

    private fun wrapHeading(d: Double) = ((d % 360.0) + 360.0) % 360.0

    private fun blendHeading(cur: Double, meas: Double, w: Double): Double {
        val delta = ((meas - cur + 180.0) % 360.0) - 180.0
        return wrapHeading(cur + w * delta)
    }

    private fun lerp(a: Double, b: Double, t: Double) = a + (b - a) * t

    private fun haversineM(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat / 2).pow(2) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLon / 2).pow(2)
        return 2 * EARTH_R * asin(sqrt(a))
    }

    private fun bearing(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val dLon = Math.toRadians(lon2 - lon1)
        val y = sin(dLon) * cos(Math.toRadians(lat2))
        val x = cos(Math.toRadians(lat1)) * sin(Math.toRadians(lat2)) -
                sin(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * cos(dLon)
        return (Math.toDegrees(atan2(y, x)) + 360.0) % 360.0
    }

    companion object {
        private const val EARTH_R = 6371000.0
    }
}
