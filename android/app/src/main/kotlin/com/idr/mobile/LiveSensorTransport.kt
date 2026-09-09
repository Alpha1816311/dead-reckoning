package com.idr.mobile

import android.net.Uri
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.ArrayDeque
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

/**
 * Acknowledged, bounded live transport for the existing /ws/sensor API.
 *
 * Only one packet is awaiting an acknowledgement at a time. IMU samples are
 * replaceable (the newest sample wins); GNSS samples are FIFO and capped to
 * avoid replaying stale positions after an outage. REST is a serialized
 * fallback when a WebSocket cannot be established.
 */
class LiveSensorTransport(private val listener: Listener) {
    interface Listener {
        fun onConnectionState(state: String, message: String)
        fun onPacketAcknowledged(type: PacketType)
        fun onPacketFailure(message: String)
    }

    enum class PacketType { IMU, GNSS }

    private data class Packet(val type: PacketType, val json: String)

    private val lock = Any()
    private val client = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    private val worker = Executors.newSingleThreadExecutor()
    private val reconnectScheduler = Executors.newSingleThreadScheduledExecutor()
    private val gnssQueue = ArrayDeque<Packet>()

    private var backendUrl = ""
    private var webSocket: WebSocket? = null
    private var connecting = false
    private var active = false
    private var inFlight: Packet? = null
    private var latestImu: Packet? = null
    private var reconnectAttempt = 0
    private var reconnectFuture: ScheduledFuture<*>? = null

    fun start(url: String) {
        synchronized(lock) {
            backendUrl = url
            active = true
            reconnectAttempt = 0
        }
        connect()
    }

    fun stop() {
        val socket: WebSocket?
        synchronized(lock) {
            active = false
            connecting = false
            reconnectFuture?.cancel(false)
            reconnectFuture = null
            latestImu = null
            gnssQueue.clear()
            inFlight = null
            socket = webSocket
            webSocket = null
        }
        socket?.close(1000, "Collection stopped")
        listener.onConnectionState("DISCONNECTED", "Collection stopped.")
    }

    fun shutdown() {
        stop()
        worker.shutdownNow()
        reconnectScheduler.shutdownNow()
        client.dispatcher.executorService.shutdown()
        client.connectionPool.evictAll()
    }

    fun submitImu(json: String) {
        synchronized(lock) {
            if (!active) return
            latestImu = Packet(PacketType.IMU, json)
        }
        pump()
    }

    fun submitGnss(json: String) {
        synchronized(lock) {
            if (!active) return
            if (gnssQueue.size >= MAX_GNSS_BUFFER) {
                listener.onPacketFailure("GNSS buffer is full; newest fix was dropped.")
                return
            }
            gnssQueue.addLast(Packet(PacketType.GNSS, json))
        }
        pump()
    }

    private fun connect() {
        val request: Request
        synchronized(lock) {
            if (!active || connecting || webSocket != null || backendUrl.isBlank()) return
            connecting = true
            request = Request.Builder().url(toWebSocketUrl(backendUrl)).build()
        }
        listener.onConnectionState("CONNECTING", "Opening live sensor connection.")
        client.newWebSocket(request, socketListener)
    }

    private val socketListener = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            synchronized(lock) {
                if (!active) {
                    webSocket.close(1000, "Collection stopped")
                    return
                }
                this@LiveSensorTransport.webSocket = webSocket
                connecting = false
                reconnectAttempt = 0
            }
            listener.onConnectionState("CONNECTED", "WebSocket connected.")
            pump()
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            val packet: Packet?
            val accepted: Boolean
            try {
                accepted = JSONObject(text).optBoolean("ok", false)
            } catch (_: Exception) {
                acknowledgeFailure("Malformed acknowledgement from backend.")
                return
            }

            synchronized(lock) {
                packet = inFlight
                inFlight = null
            }
            if (packet == null) return
            if (accepted) listener.onPacketAcknowledged(packet.type)
            else listener.onPacketFailure("Backend rejected ${packet.type.name.lowercase()} packet.")
            pump()
        }

        override fun onFailure(webSocket: WebSocket, throwable: Throwable, response: Response?) {
            handleSocketUnavailable("WebSocket failed: ${throwable.message ?: throwable.javaClass.simpleName}")
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            handleSocketUnavailable("WebSocket closed: $code ${reason.ifBlank { "no reason" }}")
        }
    }

    private fun handleSocketUnavailable(message: String) {
        val shouldReconnect: Boolean
        synchronized(lock) {
            webSocket = null
            connecting = false
            // Delivery is ambiguous after a socket failure. Do not replay an
            // in-flight sample: the backend may already have processed it,
            // and replaying it could violate its strictly increasing timestamp
            // contract. Newer buffered samples remain available.
            inFlight = null
            shouldReconnect = active
        }
        if (!shouldReconnect) return
        listener.onConnectionState("ERROR", "$message; using REST fallback while reconnecting.")
        listener.onPacketFailure(message)
        scheduleReconnect()
        pump()
    }

    private fun scheduleReconnect() {
        val delaySeconds: Long
        synchronized(lock) {
            if (!active || reconnectFuture?.isDone == false) return
            delaySeconds = minOf(MAX_RECONNECT_SECONDS, 1L shl reconnectAttempt.coerceAtMost(5))
            reconnectAttempt += 1
            reconnectFuture = reconnectScheduler.schedule({ connect() }, delaySeconds, TimeUnit.SECONDS)
        }
    }

    private fun pump() {
        val packet: Packet
        val socket: WebSocket?
        synchronized(lock) {
            if (!active || inFlight != null || connecting) return
            packet = nextPacketLocked() ?: return
            inFlight = packet
            socket = webSocket
        }

        if (socket != null) {
            if (!socket.send(packet.json)) {
                acknowledgeFailure("WebSocket send failed.")
                handleSocketUnavailable("WebSocket send failed")
            }
            return
        }

        worker.execute { postRestFallback(packet) }
    }

    private fun postRestFallback(packet: Packet) {
        try {
            val endpoint = if (packet.type == PacketType.IMU) "/sensor/imu" else "/sensor/gnss"
            val connection = (URL("$backendUrl$endpoint").openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 3_000
                readTimeout = 3_000
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
            }
            connection.outputStream.use { it.write(packet.json.toByteArray(Charsets.UTF_8)) }
            val status = connection.responseCode
            connection.disconnect()
            if (status in 200..299) acknowledgeSuccess(packet.type)
            else acknowledgeFailure("REST fallback returned HTTP $status")
        } catch (error: Exception) {
            acknowledgeFailure("REST fallback failed: ${error.message ?: error.javaClass.simpleName}")
        }
    }

    private fun acknowledgeSuccess(type: PacketType) {
        synchronized(lock) { inFlight = null }
        listener.onPacketAcknowledged(type)
        pump()
    }

    private fun acknowledgeFailure(message: String) {
        synchronized(lock) { inFlight = null }
        listener.onPacketFailure(message)
        pump()
    }

    private fun nextPacketLocked(): Packet? {
        if (gnssQueue.isNotEmpty()) return gnssQueue.removeFirst()
        val packet = latestImu
        latestImu = null
        return packet
    }

    private fun toWebSocketUrl(httpUrl: String): String {
        val source = Uri.parse(httpUrl)
        val scheme = if (source.scheme == "https") "wss" else "ws"
        return Uri.Builder()
            .scheme(scheme)
            .encodedAuthority(source.encodedAuthority)
            .path("/ws/sensor")
            .build()
            .toString()
    }

    private companion object {
        const val MAX_GNSS_BUFFER = 4
        const val MAX_RECONNECT_SECONDS = 30L
    }
}
