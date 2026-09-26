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

    /**
     * Where the current speed estimate came from.
     * GNSS       — fresh Android Location.speed or haversine over two GNSS fixes
     * INERTIAL   — dead-reckoning via IMU integration (only if genuine IMU velocity exists)
     * STATIONARY — device is confidently stationary (speed = 0.0)
     * UNAVAILABLE — no trustworthy speed estimate yet
     */
    enum class SpeedSource { GNSS, INERTIAL, STATIONARY, UNAVAILABLE }

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
        val isStationary: Boolean = false,
        // Speed provenance — always set so UI can trust/display source
        val speedSource: SpeedSource = SpeedSource.UNAVAILABLE,
        // P12: Diagnostic rejection counters
        val rejectedGnssJitter: Int = 0,
        val rejectedSpeedOutlier: Int = 0,
        val rejectedPositionJump: Int = 0,
        val stationaryPositionHeld: Int = 0
    )

    // ---------------------------------------------------------------
    // Position / navigation state
    // ---------------------------------------------------------------
    @Volatile private var posLat: Double = 0.0
    @Volatile private var posLon: Double = 0.0
    @Volatile private var headingDeg: Double = 0.0
    @Volatile private var speedMps: Double = 0.0
    @Volatile private var speedSource: SpeedSource = SpeedSource.UNAVAILABLE
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

    // GNSS availability tracking — separate from fusion quality.
    // gnssReceived = at least one GNSS callback has arrived (regardless of accuracy).
    // This prevents the engine from ever being "fooled" by zero-timestamp into DR.
    @Volatile private var gnssReceived: Boolean = false
    // Number of GNSS callbacks received (regardless of accuracy)
    @Volatile private var gnssCallbackCount: Int = 0
    // Accuracy threshold for position fusion (not for availability/initialization)
    private val GNSS_FUSION_MAX_ACCURACY_M = 120.0
    // Accuracy threshold for counting as a "good" fix for initialization
    private val GNSS_INIT_MAX_ACCURACY_M = 200.0

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
    private var gravLpInitialized = false

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
    @Volatile private var reacqCount = 0
    private val REACQ_STEPS = 12      // ~12 GNSS epochs ≈ 12 s of smooth convergence

    // Saved DR position at the moment reacquisition begins.
    // The lerp runs FROM this fixed reference so that IMU updates during
    // reacquisition do not change the "from" end of the blend each step.
    @Volatile private var reacqStartLat: Double = 0.0
    @Volatile private var reacqStartLon: Double = 0.0

    // Running GNSS anchor used during reacquisition.
    // We require GNSS_REACQ_STABLE_FIXES consecutive good fixes whose
    // pairwise displacement is < GNSS_REACQ_MAX_JUMP_M before we trust
    // the fix enough to start blending.  This guards against accepting
    // a stale/old cached GNSS position from before the blackout.
    private val GNSS_REACQ_STABLE_FIXES = 2
    private val GNSS_REACQ_MAX_JUMP_M   = 80.0   // reject implausibly large first fix
    @Volatile private var reacqStableFixes = 0
    @Volatile private var reacqAnchorLat: Double = 0.0
    @Volatile private var reacqAnchorLon: Double = 0.0

    // ---------------------------------------------------------------
    // Stationary detection
    // ---------------------------------------------------------------
    private val accelMagBuf = ArrayDeque<Double>()
    // Window: 40 samples at ~50 Hz = ~0.8 s.
    // Previously 20 (0.4 s) was too short — a single road bump or sensor
    // glitch for 0.4 s would drop stationaryState and re-open the GNSS
    // fusion path, allowing 2 m/step of jitter via FUSION_MIN_STEP_M.
    // 0.8 s gives the filter enough inertia to survive brief vibration.
    private val STATIONARY_WIN = 40
    private val STATIONARY_THRESH = 0.15 // m/s²

    @Volatile private var stationaryState = false

    // ---------------------------------------------------------------
    // Stationary position anchor
    // When the device is detected as stationary, the navigation position
    // is locked to the last anchor position to prevent GNSS jitter from
    // creating a wandering trajectory.
    // ---------------------------------------------------------------
    @Volatile private var stationaryAnchorLat: Double = 0.0
    @Volatile private var stationaryAnchorLon: Double = 0.0
    @Volatile private var hasStationaryAnchor: Boolean = false

    // Minimum displacement to consider the device has actually moved
    // away from the stationary anchor before releasing the lock.
    // Chosen to exceed typical GNSS noise (~20–30 m accuracy) but remain
    // below the smallest intentional movement we care about.
    private val STATIONARY_RELEASE_DIST_M = 15.0

    // Consecutive non-stationary IMU windows needed before the position
    // anchor is released.  Prevents a single bump/glitch from unlocking.
    // 3 windows x 40 samples x ~20 ms = ~2.4 s sustained non-stationary.
    private val STATIONARY_RELEASE_IMU_COUNT = 3
    @Volatile private var nonStationaryConsecutive: Int = 0
    // Hysteresis latch: true when the navigation position is frozen at
    // the stationary anchor.  Set/cleared by processGnss() based on both
    // IMU stationaryState AND Doppler evidence.
    @Volatile private var positionFrozen: Boolean = false

    // ---------------------------------------------------------------
    // Speed outlier rejection
    // ---------------------------------------------------------------
    // Physical max for car (55 m/s ≈ 198 km/h) plus margin for short
    // acceleration burst. Used as hard cap for Doppler input validation.
    private val SPEED_PHYSICAL_MAX_MPS = 55.0
    // Consecutive speed history used to cross-check an incoming reading.
    // A single sample that is N× larger than the recent mean is an outlier.
    private val SPEED_OUTLIER_RATIO = 4.0   // 4× recent average = suspect

    // ---------------------------------------------------------------
    // Position jump protection
    // ---------------------------------------------------------------
    // Max implied speed (m/s) between two consecutive authoritative
    // positions before the position is flagged as a jump.
    // 60 m/s = 216 km/h — anything beyond this is physically implausible
    // for normal ground operation and must be validated by multiple samples.
    private val POS_JUMP_MAX_IMPLIED_MPS = 60.0

    // ---------------------------------------------------------------
    // GNSS fusion correction rate-limiter (P0 fix)
    // ---------------------------------------------------------------
    // Track the NAV position and IMU wall-clock timestamp at the moment
    // of the last accepted GNSS fusion correction.  This lets us compute
    // how far the NAV position has moved between corrections, and compare
    // that against the trusted speed to detect impossible lerp jumps.
    //
    // The bug: lerp(posLat, gnssLat, 0.92) with acc=2 m can move posLat
    // by 9+ metres in a single GNSS callback (14 ms after the previous
    // IMU sample), implying 650+ m/s.  P5 did not catch this because it
    // compared GNSS fix-to-fix distance, not NAV-position-to-proposed-correction.
    @Volatile private var lastFusedLat: Double = 0.0
    @Volatile private var lastFusedLon: Double = 0.0
    @Volatile private var lastFusedWallMs: Long = 0L
    @Volatile private var hasFusedPosition: Boolean = false

    // Maximum correction distance = trusted speed × elapsed time × safety margin.
    // We allow up to 3× the trusted speed so a sudden gentle acceleration is not
    // blocked, but a 9-metre jump while doing 0.75 m/s is still rejected.
    private val FUSION_RATE_MARGIN = 3.0
    // Absolute minimum correction distance allowed per fusion step regardless
    // of speed — ensures the filter converges even from complete standstill.
    // Set to 2× typical GNSS accuracy floor so small legitimate corrections pass.
    private val FUSION_MIN_STEP_M = 2.0

    // ---------------------------------------------------------------
    // Diagnostic rejection counters (P12)
    // ---------------------------------------------------------------
    @Volatile var rejectedGnssJitter: Int = 0
        private set
    @Volatile var rejectedSpeedOutlier: Int = 0
        private set
    @Volatile var rejectedPositionJump: Int = 0
        private set
    @Volatile var stationaryPositionHeld: Int = 0
        private set

    // ---------------------------------------------------------------
    // PUBLIC API
    // ---------------------------------------------------------------

    fun reset() {
        initialized = false
        gnssBlackedOut = false
        gnssReceived = false
        gnssCallbackCount = 0
        mode = NavMode.INITIALIZING
        lastImuTs = -1.0
        lastGnssTs = -1.0
        gnssCount = 0
        imuCount = 0
        drDistanceM = 0.0
        speedMps = 0.0
        speedSource = SpeedSource.UNAVAILABLE
        headingDeg = 0.0
        gravLp[0] = 0.0; gravLp[1] = 0.0; gravLp[2] = 0.0
        gravLpInitialized = false
        imuTsWindow.clear()
        gnssTsWindow.clear()
        speedHistory.clear()
        accelMagBuf.clear()
        prevGnssLat = null
        prevGnssLon = null
        prevGnssTs = null
        reacqCount = 0
        reacqStartLat = 0.0
        reacqStartLon = 0.0
        reacqStableFixes = 0
        reacqAnchorLat = 0.0
        reacqAnchorLon = 0.0
        yawCalibrated = false
        stationaryState = false
        // Reset stationary position anchor
        stationaryAnchorLat = 0.0
        stationaryAnchorLon = 0.0
        hasStationaryAnchor = false
        nonStationaryConsecutive = 0
        positionFrozen = false
        // Reset fusion rate-limiter
        lastFusedLat = 0.0
        lastFusedLon = 0.0
        lastFusedWallMs = 0L
        hasFusedPosition = false
        // Reset diagnostic counters
        rejectedGnssJitter = 0
        rejectedSpeedOutlier = 0
        rejectedPositionJump = 0
        stationaryPositionHeld = 0
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
        // On the very first sample, seed the filter with the actual reading so that
        // linear-acceleration extraction is immediately near-correct instead of
        // treating the whole gravity vector as a transient spike for ~20 samples.
        // -----------------------------------------------------------------------
        if (!gravLpInitialized) {
            gravLp[0] = cax; gravLp[1] = cay; gravLp[2] = caz
            gravLpInitialized = true
        } else {
            gravLp[0] = LP_ALPHA * gravLp[0] + (1 - LP_ALPHA) * cax
            gravLp[1] = LP_ALPHA * gravLp[1] + (1 - LP_ALPHA) * cay
            gravLp[2] = LP_ALPHA * gravLp[2] + (1 - LP_ALPHA) * caz
        }

        // Linear acceleration (gravity-compensated)
        val lax = cax - gravLp[0]
        val lay = cay - gravLp[1]
        val laz = caz - gravLp[2]

        // Accel magnitude for stationary detection
        val aMag = sqrt(lax * lax + lay * lay + laz * laz)
        accelMagBuf.addLast(aMag)
        if (accelMagBuf.size > STATIONARY_WIN) accelMagBuf.removeFirst()
        val imuSaysStationary = accelMagBuf.size >= STATIONARY_WIN &&
            accelMagBuf.average() < STATIONARY_THRESH
        // Hysteresis: track consecutive non-stationary IMU windows so that
        // a single bump/vibration burst does not immediately release the anchor.
        if (imuSaysStationary) {
            nonStationaryConsecutive = 0
        } else {
            nonStationaryConsecutive++
        }
        // stationaryState: true only when IMU is confident.
        // For release hysteresis, use nonStationaryConsecutive in processGnss.
        val isStationary = imuSaysStationary
        stationaryState = isStationary

        // -----------------------------------------------------------------------
        // Gyro yaw rate (use calibrated axis)
        // -----------------------------------------------------------------------
        val rawYaw = when (gyroAxis) { 0 -> cgx; 1 -> cgy; else -> cgz } * gyroSign
        val filteredYaw = if (abs(rawYaw) < gyroThreshold) 0.0 else rawYaw

        // -----------------------------------------------------------------------
        // Initialization gate — wait for a real GNSS fix before navigating.
        // IMPORTANT: we stay in WAITING_FOR_FIX (not DEAD_RECKONING) until we
        // have a valid starting position from GNSS.  We never auto-transition to
        // DEAD_RECKONING from here — that only happens via startGnssBlackout().
        // -----------------------------------------------------------------------
        if (!initialized) {
            if (gnssCount >= 2) {
                initialized = true
                mode = NavMode.GNSS_INS_FUSED
            } else {
                mode = NavMode.WAITING_FOR_FIX
                return
            }
        }

        // -----------------------------------------------------------------------
        // Heading integration
        // -----------------------------------------------------------------------
        headingDeg = wrapHeading(headingDeg + Math.toDegrees(filteredYaw * dt))

        // -----------------------------------------------------------------------
        // Dead reckoning position integration (when GNSS is blacked out)
        // -----------------------------------------------------------------------
        if (mode == NavMode.DEAD_RECKONING) {
            if (isStationary) {
                // Device is confidently stationary — zero speed, don't preserve stale DR speed
                speedMps = 0.0
                speedSource = SpeedSource.STATIONARY
            } else {
                val forwardAccel = lax * sin(Math.toRadians(headingDeg)) +
                                   lay * cos(Math.toRadians(headingDeg))
                val filteredAccel = if (abs(forwardAccel) < accelThreshold) 0.0 else forwardAccel
                val accelContrib = filteredAccel * dt
                speedMps = (speedMps + accelContrib * 0.25).coerceIn(0.0,
                    if (motionMode == MotionMode.VEHICLE) 55.0 else 10.0)
                // NHC: constrain speed decay when no accel
                if (nhcEnabled && aMag < 0.2) speedMps *= 0.98
                // Speed source is inertial during dead reckoning (only if we have a real speed)
                speedSource = if (speedMps > 0.0) SpeedSource.INERTIAL else SpeedSource.STATIONARY
            }

            // NHC: for vehicle mode, zero out lateral/vertical drift
            val dist = speedMps * dt
            drDistanceM += dist
            val hRad = Math.toRadians(headingDeg)
            posLat += Math.toDegrees(dist * cos(hRad) / EARTH_R)
            posLon += Math.toDegrees(dist * sin(hRad) / (EARTH_R * cos(Math.toRadians(posLat))))
        } else {
            // GNSS_INS_FUSED, REACQUISITION, INITIALIZING — no IMU position integration.
            // If device is stationary, clamp speed to zero regardless of stale DR state.
            // This is critical for REACQUISITION: speedMps may carry a stale DR value that
            // was valid during the blackout but must not persist after GNSS returns.
            if (isStationary) {
                speedMps = 0.0
                speedSource = SpeedSource.STATIONARY
            } else if (!hasStationaryAnchor && (posLat != 0.0 || posLon != 0.0)) {
                // IMU detects movement before GNSS anchor established — seed anchor lazily
                stationaryAnchorLat = posLat
                stationaryAnchorLon = posLon
                hasStationaryAnchor = true
            }
        }
    }

    /**
     * Process a GNSS fix. Called from Android location callback.
     *
     * Key separation of concerns:
     *  1. GNSS received  — always track (gnssCallbackCount, gnssAccuracyM, lastGnssWallMs)
     *  2. GNSS available — mark gnssReceived even if inaccurate (not an outage)
     *  3. GNSS usable    — only update gnssCount / fuse position when acc ≤ GNSS_FUSION_MAX_ACCURACY_M
     *  4. GNSS outage    — only declared via startGnssBlackout(); accuracy alone ≠ outage
     *
     * A fix with accuracy > GNSS_FUSION_MAX_ACCURACY_M is still a GNSS fix —
     * it just doesn't contribute to the fusion position or velocity estimate.
     *
     * Hardening (P1–P6):
     *  - Stationary: position locked to anchor; GNSS jitter does NOT update posLat/posLon
     *  - Speed outlier: Doppler reading above SPEED_PHYSICAL_MAX_MPS or 4× recent average rejected
     *  - Position jump: implied speed from coordinate delta validated against physical limits
     *  - Timestamp: dtG clamped to [0.1, 60.0]; stale fix (very large dt) is detected and
     *    haversine speed not used if the implied speed would be physically implausible
     */
    fun processGnss(
        latitude: Double, longitude: Double,
        speedMpsRaw: Double?, accuracyM: Double?,
        timestamp: Double
    ) {
        lastGnssWallMs = System.currentTimeMillis()
        trackRate(gnssTsWindow, timestamp)
        val acc = accuracyM ?: 20.0
        gnssAccuracyM = acc
        gnssCallbackCount++    // count every callback — proof that GNSS hardware is alive
        gnssReceived = true    // GNSS is alive regardless of accuracy

        // Blackout gate — when simulated outage is active, GNSS measurements do not
        // reach the fusion layer.  We still update gnssCallbackCount/gnssReceived above
        // so the UI correctly shows "GNSS hardware alive but signal suppressed".
        if (gnssBlackedOut) return

        // -----------------------------------------------------------------------
        // Accuracy gate for POSITION FUSION.
        // A fix worse than GNSS_FUSION_MAX_ACCURACY_M is:
        //   - NOT used to update the fused position
        //   - NOT used to update speed from haversine
        //   - NOT used to update heading
        // BUT it IS still used to:
        //   - prove GNSS is not outage
        //   - drive initial position if no better fix has been seen (via init path below)
        //   - increment gnssCount so the engine can leave WAITING_FOR_FIX
        // -----------------------------------------------------------------------
        val usableForFusion = acc <= GNSS_FUSION_MAX_ACCURACY_M

        // Always advance gnssCount for any fix within init accuracy range.
        // This ensures that even a phone with 100–120 m accuracy in an urban canyon
        // can still initialize rather than staying in WAITING_FOR_FIX forever.
        val usableForInit = acc <= GNSS_INIT_MAX_ACCURACY_M

        if (usableForInit) {
            val prevLat = prevGnssLat
            val prevLon = prevGnssLon
            val prevTs  = prevGnssTs

            if (usableForFusion && prevLat != null && prevLon != null && prevTs != null) {
                // ---------------------------------------------------------------
                // P6: Timestamp validation
                // dtG minimum is 0.1s (not 0.01s) to prevent div-by-near-zero speed spikes.
                // If dtG > 30s the fix is very stale; haversine-derived speed is unreliable
                // but Doppler is still valid.
                // ---------------------------------------------------------------
                val rawDtG = timestamp - prevTs
                val dtG = rawDtG.coerceIn(0.1, 60.0)
                val dtStale = rawDtG > 30.0  // flag: don't trust haversine speed for stale pair

                val dist = haversineM(prevLat, prevLon, latitude, longitude)

                // ---------------------------------------------------------------
                // P5: Position jump protection
                // Compute implied speed. If it exceeds POS_JUMP_MAX_IMPLIED_MPS AND
                // the current measured speed (Doppler or engine state) does NOT
                // independently support rapid movement, reject the position update.
                // ---------------------------------------------------------------
                val impliedSpeedMps = if (dtG > 0.0) dist / dtG else 0.0
                val currentKnownSpeedMps = speedMps  // authoritative engine speed before this update
                val posJump = impliedSpeedMps > POS_JUMP_MAX_IMPLIED_MPS &&
                              currentKnownSpeedMps < (POS_JUMP_MAX_IMPLIED_MPS * 0.5)
                if (posJump) {
                    // Position is implausible given current motion state — reject position
                    // but still update speed from Doppler if available.
                    rejectedPositionJump++
                    android.util.Log.w("IDRNav",
                        "P5 pos jump rejected: dist=${dist.toInt()}m dt=${dtG.toInt()}s " +
                        "implied=${impliedSpeedMps.toInt()}m/s knownSpeed=${currentKnownSpeedMps.toInt()}m/s")
                    // Still update Doppler speed (position-independent)
                    processGnssDopplerSpeed(speedMpsRaw)
                    // Do not update prevGnss* so the next fix is compared to the last good one
                } else {
                    if (dist > 2.0 && !dtStale) {
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

                        // ---------------------------------------------------------------
                        // P4: Speed outlier rejection
                        // Accept Doppler first (it's position-independent).
                        // If Doppler unavailable, use haversine speed, but validate it
                        // against the current speed history to catch isolated spikes.
                        // The coerceIn upper bound is SPEED_PHYSICAL_MAX_MPS (not 60 m/s=216 km/h).
                        // ---------------------------------------------------------------
                        if (speedMpsRaw != null && speedMpsRaw < 0.15) {
                            speedMps = 0.0
                            speedHistory.clear()
                            speedSource = SpeedSource.STATIONARY
                        } else if (speedMpsRaw != null) {
                            // Doppler available: validate against physical max and history
                            val validated = validateSpeed(speedMpsRaw)
                            if (validated != null) {
                                speedHistory.addLast(validated)
                                if (speedHistory.size > SPEED_HIST) speedHistory.removeFirst()
                                speedMps = speedHistory.average()
                                speedSource = SpeedSource.GNSS
                            }
                            // else: outlier rejected, keep existing speed estimate
                        } else {
                            // No Doppler — use haversine speed, but only if plausible
                            val validated = validateSpeed(gnssSpd)
                            if (validated != null) {
                                speedHistory.addLast(validated)
                                if (speedHistory.size > SPEED_HIST) speedHistory.removeFirst()
                                speedMps = speedHistory.average()
                                speedSource = SpeedSource.GNSS
                            }
                        }
                    } else if (dist > 2.0 && dtStale) {
                        // Stale fix pair — course may still be usable but haversine speed is not
                        val course = bearing(prevLat, prevLon, latitude, longitude)
                        if (!yawCalibrated && gnssCount >= 5) {
                            yawCalibrated = true; gyroAxis = 2; gyroSign = -1.0
                        }
                        val gw = if (acc < 5.0) 0.65 else if (acc < 15.0) 0.35 else 0.15
                        headingDeg = blendHeading(headingDeg, course, gw)
                        // Use Doppler speed if available but not haversine
                        processGnssDopplerSpeed(speedMpsRaw)
                    } else {
                        // dist ≤ 2.0 — minimal movement
                        processGnssDopplerSpeed(speedMpsRaw)
                    }

                    prevGnssLat = latitude; prevGnssLon = longitude; prevGnssTs = timestamp
                }
            } else {
                // First fix or no prior fix to compare: accept Doppler speed only
                processGnssDopplerSpeed(speedMpsRaw)
                prevGnssLat = latitude; prevGnssLon = longitude; prevGnssTs = timestamp
            }

            gnssCount++
            lastGnssTs = timestamp

            // -----------------------------------------------------------------------
            // Initialization: set starting position on first two valid fixes.
            // For initialization we accept any fix within GNSS_INIT_MAX_ACCURACY_M.
            // -----------------------------------------------------------------------
            if (!initialized && gnssCount >= 2) {
                initialized = true
                posLat = latitude
                posLon = longitude
                // Seed the stationary anchor on initialization
                stationaryAnchorLat = latitude
                stationaryAnchorLon = longitude
                hasStationaryAnchor = true
            }

            // -----------------------------------------------------------------------
            // Fusion: update position from GNSS when we have a good fix
            // -----------------------------------------------------------------------
            if (initialized) {
                when (mode) {
                    NavMode.DEAD_RECKONING -> {
                        // Only exit DR when the returning fix is reasonably accurate
                        if (usableForFusion) {
                             // Save the current DR position as the fixed blend-from anchor.
                            // This covers the case where GNSS returns naturally (no explicit
                            // stopGnssBlackout call — e.g. hardware-level GNSS loss).
                            reacqStartLat = posLat
                            reacqStartLon = posLon
                            reacqCount = 0
                            reacqStableFixes = 0
                            reacqAnchorLat = 0.0
                            reacqAnchorLon = 0.0
                            drDistanceM = 0.0
                            mode = NavMode.REACQUISITION
                        }
                        // else: poor accuracy fix during outage — stay in DR, don't jump
                    }
                    NavMode.REACQUISITION -> {
                        if (usableForFusion) {
                            // --------------------------------------------------
                            // Reacquisition phase (unchanged from previous fix):
                            // --------------------------------------------------
                            if (reacqStableFixes == 0) {
                                reacqAnchorLat = latitude
                                reacqAnchorLon = longitude
                                reacqStableFixes = 1
                            } else {
                                val jumpM = haversineM(reacqAnchorLat, reacqAnchorLon, latitude, longitude)
                                if (jumpM < GNSS_REACQ_MAX_JUMP_M) {
                                    reacqStableFixes++
                                    reacqAnchorLat = latitude
                                    reacqAnchorLon = longitude
                                } else {
                                    reacqStableFixes = 1
                                    reacqAnchorLat = latitude
                                    reacqAnchorLon = longitude
                                }
                            }

                            if (reacqStableFixes >= GNSS_REACQ_STABLE_FIXES) {
                                reacqCount++
                                val alpha = (reacqCount.toDouble() / REACQ_STEPS).coerceIn(0.0, 1.0)
                                posLat = lerp(reacqStartLat, latitude, alpha)
                                posLon = lerp(reacqStartLon, longitude, alpha)
                                if (reacqCount >= REACQ_STEPS) {
                                    posLat = latitude; posLon = longitude
                                    mode = NavMode.GNSS_INS_FUSED
                                    // Reset stationary anchor on reacquisition completion
                                    stationaryAnchorLat = latitude
                                    stationaryAnchorLon = longitude
                                    hasStationaryAnchor = true
                                }
                            }
                        }
                    }
                    else -> {
                        if (usableForFusion) {
                            // -------------------------------------------------------
                            // P1 + P2: Stationary position freeze with hysteresis
                            //
                            // positionFrozen = true:  NAV is locked to stationary anchor
                            // positionFrozen = false: NAV is fusing with GNSS normally
                            //
                            // ENTERING freeze:
                            //   stationaryState (IMU) becomes true
                            //
                            // RELEASING freeze (requires BOTH):
                            //   a) nonStationaryConsecutive >= STATIONARY_RELEASE_IMU_COUNT
                            //      (IMU has been non-stationary for N consecutive windows —
                            //       prevents a single road-bump from releasing the anchor)
                            //   b) validated Doppler speed > 1.0 m/s
                            //      (two independent sensors must agree on movement)
                            //
                            // The accuracy-scaled jitter radius still gates whether a GNSS
                            // coordinate change is "real displacement" or measurement noise.
                            // -------------------------------------------------------

                            // Update positionFrozen latch:
                            if (stationaryState) {
                                // IMU says stationary — engage freeze (if not already frozen,
                                // set anchor to current fused position)
                                if (!positionFrozen) {
                                    if (hasStationaryAnchor) {
                                        // Re-anchor at current fused position
                                        stationaryAnchorLat = posLat
                                        stationaryAnchorLon = posLon
                                    }
                                    positionFrozen = true
                                }
                                // Reset hysteresis counter while IMU is stationary
                                nonStationaryConsecutive = 0
                            } else if (positionFrozen) {
                                // IMU is no longer stationary — check hysteresis before releasing
                                val dopplerConfirmsMovement = speedMpsRaw != null && speedMpsRaw > 1.0
                                val imuSustainedMovement = nonStationaryConsecutive >= STATIONARY_RELEASE_IMU_COUNT
                                if (imuSustainedMovement && dopplerConfirmsMovement) {
                                    positionFrozen = false
                                    android.util.Log.d("IDRNav",
                                        "P1 stationary anchor released: " +
                                        "imuWindows=$nonStationaryConsecutive Doppler=${speedMpsRaw} m/s")
                                }
                                // else: keep frozen until both conditions are met
                            }

                            if (positionFrozen && hasStationaryAnchor) {
                                // Device is stationary (or hysteresis lock still active).
                                // Hold NAV position at anchor — GNSS jitter must not move it.
                                val jitterDist = haversineM(stationaryAnchorLat, stationaryAnchorLon, latitude, longitude)
                                // Release threshold: at least STATIONARY_RELEASE_DIST_M,
                                // but scales with GNSS accuracy to cover the noise envelope.
                                val releaseThreshM = maxOf(STATIONARY_RELEASE_DIST_M, acc * 1.5)
                                if (jitterDist <= releaseThreshM) {
                                    // Within jitter radius: silently hold
                                    posLat = stationaryAnchorLat
                                    posLon = stationaryAnchorLon
                                    stationaryPositionHeld++
                                } else {
                                    // Beyond jitter radius but position is frozen.
                                    // This can happen with very poor accuracy (>30m).
                                    // Still hold — do not allow random GNSS wander.
                                    posLat = stationaryAnchorLat
                                    posLon = stationaryAnchorLon
                                    rejectedGnssJitter++
                                    android.util.Log.d("IDRNav",
                                        "P2 GNSS jitter rejected (frozen): " +
                                        "drift=${jitterDist.toInt()}m acc=${acc.toInt()}m " +
                                        "threshold=${releaseThreshM.toInt()}m")
                                }
                                mode = NavMode.GNSS_INS_FUSED
                            } else {
                                // Device is moving (or no anchor yet) — fuse with rate-limit gate.
                                val accepted = fusePositionRateLimited(latitude, longitude, acc)
                                if (!accepted) {
                                    // Position correction was rate-limited: the position jumped
                                    // further than the trusted speed allows. Reset speed history
                                    // to the current Doppler reading (if valid) so that a stale
                                    // inflated speed estimate cannot perpetuate across multiple
                                    // GNSS epochs.
                                    if (speedMpsRaw != null && speedMpsRaw.isFinite() && speedMpsRaw >= 0.0) {
                                        speedHistory.clear()
                                        val doppler = speedMpsRaw.coerceIn(0.0, SPEED_PHYSICAL_MAX_MPS)
                                        speedMps = doppler
                                        speedSource = if (doppler < 0.15) SpeedSource.STATIONARY else SpeedSource.GNSS
                                    }
                                }
                                // Update stationary anchor to current fused position so that
                                // if the device stops, the anchor is at the last known position.
                                stationaryAnchorLat = posLat
                                stationaryAnchorLon = posLon
                                hasStationaryAnchor = true
                                mode = NavMode.GNSS_INS_FUSED
                            }
                        } else {
                            // Not usable for fusion — keep current fused position
                            mode = NavMode.GNSS_INS_FUSED
                        }
                    }
                }
            }
        } else if (!usableForInit && speedMpsRaw != null) {
            // Extremely poor accuracy but Doppler speed still available — use Doppler only
            processGnssDopplerSpeed(speedMpsRaw)
        }
        // If acc > GNSS_INIT_MAX_ACCURACY_M: fix is extremely poor.
        // Still counts as "GNSS received" (gnssReceived=true, gnssCallbackCount++) but
        // does NOT advance gnssCount/initialization/fusion.
    }

    /**
     * P0 fix: Rate-limited GNSS position fusion.
     *
     * Problem: lerp(posLat, gnssLat, 0.92) with acc=2 m can instantly move the
     * authoritative NAV position by many metres in a single GNSS callback that
     * arrives 14 ms after the previous IMU sample.  P5 did not catch this because
     * it checked GNSS-fix-to-GNSS-fix distance, not NAV-position-to-proposed-correction.
     *
     * Solution: before applying the lerp, verify the proposed correction distance
     * is physically achievable in the elapsed time since the last accepted fusion
     * correction, given the currently trusted speed.
     *   maxCorrM = max(FUSION_MIN_STEP_M, speedMps × elapsed × FUSION_RATE_MARGIN)
     *
     * If the proposed correction exceeds the limit: move only maxCorrM towards GNSS
     * (partial step). The filter converges over successive fixes. Legitimate vehicle
     * acceleration is never blocked because maxCorrM scales with speedMps.
     *
     * Returns true if accepted in full, false if rate-limited.
     * Caller resets speed history to Doppler on false to prevent stale inflated speed.
     */
    private fun fusePositionRateLimited(gnssLat: Double, gnssLon: Double, acc: Double): Boolean {
        val fuseAlpha = if (acc < 5.0) 0.92 else if (acc < 15.0) 0.75
                        else if (acc < 30.0) 0.55 else 0.30

        // Compute the proposed new position after the lerp
        val proposedLat = lerp(posLat, gnssLat, fuseAlpha)
        val proposedLon = lerp(posLon, gnssLon, fuseAlpha)

        if (!hasFusedPosition) {
            // First fusion: accept unconditionally to establish the initial state
            posLat = proposedLat
            posLon = proposedLon
            lastFusedLat = posLat
            lastFusedLon = posLon
            lastFusedWallMs = lastGnssWallMs
            hasFusedPosition = true
            return true
        }

        // How far would the proposed lerp move the NAV position?
        val correctionDist = haversineM(posLat, posLon, proposedLat, proposedLon)

        // Elapsed time since the last accepted fusion correction (wall clock, ms→s)
        val elapsedMs = (lastGnssWallMs - lastFusedWallMs).coerceAtLeast(0L)
        val elapsedSec = (elapsedMs / 1000.0).coerceIn(0.01, 10.0)

        // Maximum correction allowed this step.
        // trustedSpeed = current engine speed (Doppler or smoothed history).
        // Multiply by elapsed time and safety margin so that legitimate vehicle
        // acceleration is never blocked, but a 9 m jump at 0.75 m/s is rejected.
        val maxCorrM = maxOf(FUSION_MIN_STEP_M, speedMps * elapsedSec * FUSION_RATE_MARGIN)

        if (correctionDist <= maxCorrM) {
            // Correction is physically plausible — accept the full lerp
            posLat = proposedLat
            posLon = proposedLon
            lastFusedLat = posLat
            lastFusedLon = posLon
            lastFusedWallMs = lastGnssWallMs
            return true
        } else {
            // Correction exceeds the rate limit: move only maxCorrM metres toward GNSS.
            // This is a partial step — the filter will converge over successive fixes.
            rejectedPositionJump++
            android.util.Log.w("IDRNav",
                "P0 fusion rate-limited: proposed=${correctionDist.toInt()}m " +
                "max=${maxCorrM.toInt()}m speed=${speedMps}m/s elapsed=${elapsedSec}s acc=${acc}m")
            if (correctionDist > 0.0) {
                val fraction = maxCorrM / correctionDist
                posLat = lerp(posLat, proposedLat, fraction)
                posLon = lerp(posLon, proposedLon, fraction)
            }
            // Update lastFusedWallMs on the partial step so the budget resets.
            // Previously we did NOT update this, intending budget to "accumulate"
            // across steps.  In practice this caused elapsedSec to hit the 10.0s
            // cap and the next allowed correction to be 10× larger than needed,
            // creating visible jumps after a short period of rate-limiting.
            // By resetting the clock each partial step, each GNSS epoch gets a
            // fresh budget = max(FUSION_MIN_STEP_M, speed × dt × MARGIN), which
            // naturally converges without the unbounded accumulation.
            lastFusedWallMs = lastGnssWallMs
            return false
        }
    }

    /**
     * Process Doppler (Location.speed) input with physical plausibility validation.
     * Extracted as a helper to avoid repeating the outlier-rejection logic.
     * Stationary threshold (< 0.15 m/s) clears speed history and marks STATIONARY.
     */
    private fun processGnssDopplerSpeed(speedMpsRaw: Double?) {
        if (speedMpsRaw == null) return
        if (speedMpsRaw < 0.15) {
            speedMps = 0.0
            speedHistory.clear()
            speedSource = SpeedSource.STATIONARY
        } else {
            val validated = validateSpeed(speedMpsRaw)
            if (validated != null) {
                speedHistory.addLast(validated)
                if (speedHistory.size > SPEED_HIST) speedHistory.removeFirst()
                speedMps = speedHistory.average()
                speedSource = SpeedSource.GNSS
            }
        }
    }

    /**
     * P4: Speed outlier rejection.
     * Returns the validated speed value, or null if the measurement is rejected.
     *
     * Rejects if:
     *  1. Speed exceeds absolute physical maximum (SPEED_PHYSICAL_MAX_MPS)
     *  2. Speed is NaN or Infinite
     *  3. Speed is more than SPEED_OUTLIER_RATIO × the recent speed history average,
     *     AND the history is large enough to be meaningful (≥ 3 samples)
     *     — this catches isolated spikes without permanently suppressing acceleration
     *
     * Note: ratio gate is NOT applied when history is small (< 3 samples) to allow
     * initial acceleration from standstill.
     */
    private fun validateSpeed(rawMps: Double): Double? {
        if (!rawMps.isFinite() || rawMps < 0.0) {
            rejectedSpeedOutlier++
            android.util.Log.w("IDRNav", "P4 speed rejected (non-finite/negative): $rawMps m/s")
            return null
        }
        if (rawMps > SPEED_PHYSICAL_MAX_MPS) {
            rejectedSpeedOutlier++
            android.util.Log.w("IDRNav", "P4 speed rejected (>${SPEED_PHYSICAL_MAX_MPS} m/s): ${rawMps} m/s = ${rawMps * 3.6} km/h")
            return null
        }
        // Ratio-based outlier check (requires established history).
        // Previous threshold: recentAvg > 0.5 m/s (1.8 km/h).
        // Fixed threshold: recentAvg > 0.1 m/s — catches spikes from near-
        // stationary state (e.g. recent history = 0.3 m/s, spike = 60 m/s
        // was previously PASSING because 0.3 < 0.5).
        // The 0.1 lower bound still allows clean acceleration from standstill
        // because the ratio gate only fires when rawMps > recentAvg * 4.0.
        // With recentAvg = 0.0, 0.0 * 4.0 = 0.0 so any non-zero speed passes
        // (correct — device just started moving). With recentAvg = 0.15 m/s,
        // only rawMps > 0.6 m/s triggers the check, which still allows normal
        // initial acceleration.
        if (speedHistory.size >= 3) {
            val recentAvg = speedHistory.average()
            if (recentAvg > 0.1 && rawMps > recentAvg * SPEED_OUTLIER_RATIO) {
                rejectedSpeedOutlier++
                android.util.Log.w("IDRNav",
                    "P4 speed rejected (outlier): ${rawMps} m/s = ${rawMps * 3.6} km/h, recent avg=${recentAvg} m/s")
                return null
            }
        }
        // Additional stationary-state guard: if the IMU says stationary AND
        // the incoming raw speed is above a brisk walking pace, the sensor is
        // lying.  Reject without touching speedHistory so the next real Doppler
        // reading can re-seed the history cleanly.
        if (stationaryState && rawMps > 1.5) {
            rejectedSpeedOutlier++
            android.util.Log.w("IDRNav",
                "P4 speed rejected (IMU stationary but Doppler=${rawMps} m/s = ${rawMps * 3.6} km/h)")
            return null
        }
        return rawMps.coerceIn(0.0, SPEED_PHYSICAL_MAX_MPS)
    }

    fun startGnssBlackout(currentTs: Double) {
        gnssBlackedOut = true
        blackoutStartTs = currentTs
        if (mode == NavMode.GNSS_INS_FUSED || mode == NavMode.REACQUISITION) {
            mode = NavMode.DEAD_RECKONING; drDistanceM = 0.0
            // When entering DR: if we have a GNSS-derived speed keep it as the
            // initial inertial estimate; if speed is unknown mark UNAVAILABLE
            // so the UI doesn't display a stale or invented number.
            if (speedSource == SpeedSource.UNAVAILABLE) {
                speedMps = 0.0
            } else if (speedSource == SpeedSource.GNSS || speedSource == SpeedSource.INERTIAL) {
                speedSource = SpeedSource.INERTIAL
            }
            // Reset reacquisition state so the next recovery starts clean
            reacqCount = 0
            reacqStableFixes = 0
            reacqAnchorLat = 0.0
            reacqAnchorLon = 0.0
            // Invalidate GNSS history so that the first reacquisition fix is not
            // compared to a pre-blackout fix via haversine, which would produce a
            // spurious high speed from (large displacement) / (small stale dtG).
            prevGnssLat = null
            prevGnssLon = null
            prevGnssTs = null
            speedHistory.clear()
            // Reset fusion rate-limiter so that on reacquisition completion,
            // the first post-DR fused position is not rate-limited against a
            // stale pre-blackout lastFusedWallMs.
            hasFusedPosition = false
            lastFusedWallMs = 0L
            // Release the stationary position lock when entering DR.
            // During dead reckoning we integrate IMU velocity, so the freeze is
            // irrelevant — and we must not carry it into the post-blackout fused state.
            positionFrozen = false
            nonStationaryConsecutive = 0
        }
    }

    fun stopGnssBlackout() {
        gnssBlackedOut = false
        if (mode == NavMode.DEAD_RECKONING) {
            // Save the current DR position as the fixed "from" anchor for the
            // upcoming reacquisition lerp.  The IMU will keep updating posLat/posLon
            // during reacquisition, so we must snapshot the DR endpoint NOW.
            reacqStartLat = posLat
            reacqStartLon = posLon
            reacqCount = 0
            reacqStableFixes = 0
            reacqAnchorLat = 0.0
            reacqAnchorLon = 0.0
            mode = NavMode.REACQUISITION
            // Invalidate GNSS history so the first post-blackout fix does not
            // produce a spurious haversine speed against the last pre-blackout fix.
            prevGnssLat = null
            prevGnssLon = null
            prevGnssTs = null
            speedHistory.clear()
            // Reset fusion rate-limiter.
            hasFusedPosition = false
            lastFusedWallMs = 0L
            // Release the stationary freeze so REACQUISITION lerp proceeds normally.
            positionFrozen = false
            nonStationaryConsecutive = 0
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
        // Determine effective speed source for this snapshot:
        // If we are not yet initialized (no GNSS fix yet), speed is always UNAVAILABLE.
        // If stationary is detected outside DR mode, reflect STATIONARY.
        val effectiveSource = when {
            !initialized -> SpeedSource.UNAVAILABLE
            stationaryState && speedMps < 0.05 -> SpeedSource.STATIONARY
            else -> speedSource
        }
        val effectiveSpeedMps = when (effectiveSource) {
            SpeedSource.UNAVAILABLE -> null
            SpeedSource.STATIONARY  -> 0.0
            else -> speedMps
        }
        return NavigationState(
            timestamp = nowTs,
            latitude = posLat, longitude = posLon,
            speedMps = effectiveSpeedMps ?: 0.0,
            headingDeg = headingDeg,
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
            isStationary = stationaryState,
            speedSource = effectiveSource,
            rejectedGnssJitter = rejectedGnssJitter,
            rejectedSpeedOutlier = rejectedSpeedOutlier,
            rejectedPositionJump = rejectedPositionJump,
            stationaryPositionHeld = stationaryPositionHeld
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
