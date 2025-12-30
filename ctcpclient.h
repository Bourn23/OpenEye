#ifndef CTCPCLIENT_H
#define CTCPCLIENT_H

#if defined(_WIN32)
// Must include winsock2.h before windows.h
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <string>
#include <thread>
#include <atomic>
#include <functional>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close
#endif

class CTcpClient
{
public:
    using MessageCallback = std::function<void(const std::string&)>;

    CTcpClient();
    ~CTcpClient();

    bool Connect(const std::string& host, int port);
    void Disconnect();
    bool IsConnected() const { return m_bConnected; }

    // Send data to server
    bool Send(const std::string& data);

    void SetMessageCallback(MessageCallback callback) { m_callback = callback; }
    void StartReceiveThread();
    void StopReceiveThread();

private:
    void ReceiveThreadFunc();
    void ProcessBuffer();

    SOCKET m_socket;
    std::atomic<bool> m_bConnected;
    std::atomic<bool> m_bRunning;
    std::thread* m_pReceiveThread;
    MessageCallback m_callback;
    std::string m_receiveBuffer;

    std::string m_host;
    int m_port;

#if defined(_WIN32)
    static bool s_bWsaInitialized;
    static void InitWsa();
    static void CleanupWsa();
#endif
};

#endif // CTCPCLIENT_H
