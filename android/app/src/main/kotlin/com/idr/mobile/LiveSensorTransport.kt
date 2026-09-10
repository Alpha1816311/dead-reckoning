package com.idr.mobile

import android.net.Uri
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.util.ArrayDeque
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

/**
 * Acknowledged, bounded WebSocket transport for the existing /ws/sensor API.
 *
 * Only one packet is awaiting an acknowledgement at a time. IMU samples are
 * replaceable (the newest sample wins); GNSS samples are FIFO and capped to
 * avoid replaying stale positions after an outage. Sensor callbacks only
 * replace or append to bounded in-memory slots; they never perform I/O.
 */
class LiveSensorTransport(private val listener: Listener) {
    interface Listener {
        fun onConnectionState(state: String, message: String)
        fun onPacketAcknowledged(type: PacketType)
        fun onPacketDropped(type: PacketType, message: String)
        fun onPacketFailure(message: String)
    }

    enum class PacketType { IMU, GNSS }

    private data class Packet(val type: PacketType, val json: String)

    private val lock = Any()
    private val client = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
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
    private var acknowledgementTimeout: ScheduledFuture<*>? = null
    private var connectionGeneration = 0L

    fun start(url: String) {
        synchronized(lock) {
            backendUrl = url
            active = true
            reconnectAttempt = 0
            connectionGeneration += 1
        }
        connect()
    }

    fun stop() {
        val socket: WebSocket?
        synchronized(lock) {
            active = false
            connecting = false
            connectionGeneration += 1
            reconnectFuture?.cancel(false)
            reconnectFuture = null
            acknowledgementTimeout?.cancel(false)
            acknowledgementTimeout = null
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
        reconnectScheduler.shutdownNow()
        client.dispatcher.executorService.shutdown()
        client.connectionPool.evictAll()
    }

    fun submitImu(json: String) {
        var replaced = false
        synchronized(lock) {
            if (!active) return
            replaced = latestImu != null
            latestImu = Packet(PacketType.IMU, json)
        }
        if (replaced) listener.onPacketDropped(PacketType.IMU, "IMU sample superseded by a newer sample.")
        pump()
    }

    fun submitGnss(json: String) {
        synchronized(lock) {
            if (!active) return
            if (gnssQueue.size >= MAX_GNSS_BUFFER) {
                listener.onPacketDropped(PacketType.GNSS, "GNSS buffer is full; newest fix was dropped.")
                return
            }
            gnssQueue.addLast(Packet(PacketType.GNSS, json))
        }
        pump()
    }

    private fun connect() {
        val request: Request
        val generation: Long
        synchronized(lock) {
            if (!active || connecting || webSocket != null || backendUrl.isBlank()) return
            connecting = true
            request = Request.Builder().url(toWebSocketUrl(backendUrl)).build()
            generation = connectionGeneration
        }
        listener.onConnectionState("CONNECTING", "Opening live sensor connection.")
        client.newWebSocket(request, socketListener(generation))
    }

    private fun socketListener(generation: Long) = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            synchronized(lock) {
                if (!active || generation != connectionGeneration) {
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
                if (generation != connectionGeneration || this@LiveSensorTransport.webSocket !== webSocket) return
                packet = inFlight
                inFlight = null
                acknowledgementTimeout?.cancel(false)
                acknowledgementTimeout = null
            }
            if (packet == null) return
            if (accepted) listener.onPacketAcknowledged(packet.type)
            else listener.onPacketFailure("Backend rejected ${packet.type.name.lowercase()} packet.")
            pump()
        }

        override fun onFailure(webSocket: WebSocket, throwable: Throwable, response: Response?) {
            handleSocketUnavailable(webSocket, generation, "WebSocket failed: ${throwable.message ?: throwable.javaClass.simpleName}")
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            handleSocketUnavailable(webSocket, generation, "WebSocket closed: $code ${reason.ifBlank { "no reason" }}")
        }
    }

    private fun handleSocketUnavailable(socket: WebSocket?, generation: Long, message: String) {
        val shouldReconnect: Boolean
        synchronized(lock) {
            // A failed connection can fail before onOpen, when webSocket is
            // still null. Ignore only callbacks from a different live socket.
            if (generation != connectionGeneration || (webSocket != null && socket != null && webSocket !== socket)) return
            webSocket = null
            connecting = false
            acknowledgementTimeout?.cancel(false)
            acknowledgementTimeout = null
            // Delivery is ambiguous after a socket failure. Do not replay an
            // in-flight sample: the backend may already have processed it,
            // and replaying it could violate its strictly increasing timestamp
            // contract. Newer buffered samples remain available.
            inFlight = null
            shouldReconnect = active
        }
        if (!shouldReconnect) return
        listener.onConnectionState("ERROR", "$message; reconnecting live sensor connection.")
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
                handleSocketUnavailable(socket, connectionGeneration, "WebSocket send failed")
            } else {
                scheduleAcknowledgementTimeout()
            }
            return
        }
        scheduleReconnect()
    }

    private fun acknowledgeFailure(message: String) {
        synchronized(lock) {
            inFlight = null
            acknowledgementTimeout?.cancel(false)
            acknowledgementTimeout = null
        }
        listener.onPacketFailure(message)
        pump()
    }

    private fun scheduleAcknowledgementTimeout() {
        synchronized(lock) {
            acknowledgementTimeout?.cancel(false)
            acknowledgementTimeout = reconnectScheduler.schedule({
                var socket: WebSocket? = null
                synchronized(lock) {
                    if (!active || inFlight == null) return@schedule
                    socket = webSocket
                }
                socket?.cancel()
            }, ACKNOWLEDGEMENT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        }
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
        const val ACKNOWLEDGEMENT_TIMEOUT_SECONDS = 5L
    }
}
