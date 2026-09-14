package com.idr.server

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.net.URLDecoder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Minimal embedded HTTP/1.1 server — no external dependency.
 * Handles GET and POST, serves chunked and fixed-length responses.
 * Thread pool: 8 concurrent workers (enough for WebView + API polling).
 */
abstract class NanoHTTPD(private val port: Int) {

    enum class Method { GET, POST, PUT, DELETE, OPTIONS, HEAD }

    interface IHTTPSession {
        val uri: String
        val method: Method
        val headers: Map<String, String>
        val queryParams: Map<String, String>
        fun getBody(): String
    }

    class Response(val status: Status, val mimeType: String, val data: Any) {
        enum class Status(val code: Int, val description: String) {
            OK(200, "OK"),
            NOT_FOUND(404, "Not Found"),
            METHOD_NOT_ALLOWED(405, "Method Not Allowed"),
            INTERNAL_ERROR(500, "Internal Server Error")
        }
        var chunked = false
    }

    companion object {
        fun newFixedLengthResponse(status: Response.Status, mime: String, body: String): Response =
            Response(status, mime, body.toByteArray(Charsets.UTF_8))

        fun newChunkedResponse(status: Response.Status, mime: String, stream: InputStream): Response =
            Response(status, mime, stream).also { it.chunked = true }
    }

    abstract fun serve(session: IHTTPSession): Response

    private lateinit var serverSocket: ServerSocket
    private lateinit var pool: ExecutorService
    @Volatile private var running = false

    fun start() {
        serverSocket = ServerSocket(port)
        serverSocket.reuseAddress = true
        pool = Executors.newFixedThreadPool(8)
        running = true
        val acceptThread = Thread {
            while (running) {
                try {
                    val sock = serverSocket.accept()
                    pool.execute { handleConnection(sock) }
                } catch (_: SocketException) {
                    // server stopped
                }
            }
        }
        acceptThread.isDaemon = true
        acceptThread.start()
    }

    fun stop() {
        running = false
        try { serverSocket.close() } catch (_: Exception) {}
        pool.shutdownNow()
    }

    private fun handleConnection(socket: Socket) {
        try {
            socket.use { sock ->
                val input = sock.getInputStream()
                val output = sock.getOutputStream()
                val reader = BufferedReader(InputStreamReader(input))

                // Read request line
                val requestLine = reader.readLine() ?: return
                val parts = requestLine.split(" ")
                if (parts.size < 2) return

                val methodStr = parts[0].uppercase()
                val fullUri = parts[1]

                // Parse URI and query string
                val (uri, queryString) = if ('?' in fullUri) {
                    val idx = fullUri.indexOf('?')
                    fullUri.substring(0, idx) to fullUri.substring(idx + 1)
                } else fullUri to ""

                val queryParams = parseQuery(queryString)
                val method = Method.values().find { it.name == methodStr } ?: Method.GET

                // Read headers
                val headers = mutableMapOf<String, String>()
                var line = reader.readLine()
                while (line != null && line.isNotEmpty()) {
                    val colon = line.indexOf(':')
                    if (colon > 0) {
                        headers[line.substring(0, colon).trim().lowercase()] = line.substring(colon + 1).trim()
                    }
                    line = reader.readLine()
                }

                // Read body if Content-Length present
                val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
                val bodyBuilder = StringBuilder()
                if (contentLength > 0) {
                    val buf = CharArray(contentLength)
                    var totalRead = 0
                    while (totalRead < contentLength) {
                        val n = reader.read(buf, totalRead, contentLength - totalRead)
                        if (n < 0) break
                        totalRead += n
                    }
                    bodyBuilder.append(buf, 0, totalRead)
                }
                val bodyStr = bodyBuilder.toString()

                val session = object : IHTTPSession {
                    override val uri = URLDecoder.decode(uri, "UTF-8")
                    override val method = method
                    override val headers = headers
                    override val queryParams = queryParams
                    override fun getBody() = bodyStr
                }

                val response = try {
                    serve(session)
                } catch (e: Exception) {
                    Response(Response.Status.INTERNAL_ERROR, "text/plain",
                        "Server error: ${e.message}".toByteArray())
                }

                writeResponse(output, response)
            }
        } catch (_: Exception) {}
    }

    private fun writeResponse(output: OutputStream, response: Response) {
        val statusLine = "HTTP/1.1 ${response.status.code} ${response.status.description}\r\n"
        output.write(statusLine.toByteArray())
        output.write("Access-Control-Allow-Origin: *\r\n".toByteArray())
        output.write("Connection: close\r\n".toByteArray())
        output.write("Content-Type: ${response.mimeType}; charset=utf-8\r\n".toByteArray())

        when (val data = response.data) {
            is ByteArray -> {
                output.write("Content-Length: ${data.size}\r\n\r\n".toByteArray())
                output.write(data)
            }
            is InputStream -> {
                output.write("Transfer-Encoding: chunked\r\n\r\n".toByteArray())
                val buf = ByteArray(8192)
                var n: Int
                try {
                    while (data.read(buf).also { n = it } != -1) {
                        output.write("${n.toString(16)}\r\n".toByteArray())
                        output.write(buf, 0, n)
                        output.write("\r\n".toByteArray())
                    }
                    output.write("0\r\n\r\n".toByteArray())
                } finally {
                    data.close()
                }
            }
        }
        output.flush()
    }

    private fun parseQuery(query: String): Map<String, String> {
        if (query.isBlank()) return emptyMap()
        return query.split("&").mapNotNull { param ->
            val idx = param.indexOf('=')
            if (idx < 0) null else {
                URLDecoder.decode(param.substring(0, idx), "UTF-8") to
                URLDecoder.decode(param.substring(idx + 1), "UTF-8")
            }
        }.toMap()
    }
}
