#include "ctcpclient.h"
#include "driverlog.h"
#include <cstring>

#if defined(_WIN32)
bool CTcpClient::s_bWsaInitialized = false;

void CTcpClient::InitWsa()
{
    if (!s_bWsaInitialized)
    {
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) == 0)
        {
            s_bWsaInitialized = true;
        }
    }
}

void CTcpClient::CleanupWsa()
{
    if (s_bWsaInitialized)
    {
        WSACleanup();
        s_bWsaInitialized = false;
    }
}
#endif

CTcpClient::CTcpClient()
    : m_socket(INVALID_SOCKET)
    , m_bConnected(false)
    , m_bRunning(false)
    , m_pReceiveThread(nullptr)
    , m_callback(nullptr)
    , m_port(0)
{
#if defined(_WIN32)
    InitWsa();
#endif
}

CTcpClient::~CTcpClient()
{
    StopReceiveThread();
    Disconnect();
}

bool CTcpClient::Connect(const std::string& host, int port)
{
    if (m_bConnected)
    {
        Disconnect();
    }

    m_host = host;
    m_port = port;

    m_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (m_socket == INVALID_SOCKET)
    {
        DriverLog("CTcpClient: Failed to create socket\n");
        return false;
    }

    sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(static_cast<u_short>(port));

    if (inet_pton(AF_INET, host.c_str(), &serverAddr.sin_addr) <= 0)
    {
        DriverLog("CTcpClient: Invalid address: %s\n", host.c_str());
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
        return false;
    }

    if (connect(m_socket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR)
    {
        DriverLog("CTcpClient: Failed to connect to %s:%d\n", host.c_str(), port);
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
        return false;
    }

    m_bConnected = true;
    DriverLog("CTcpClient: Connected to %s:%d\n", host.c_str(), port);
    return true;
}

void CTcpClient::Disconnect()
{
    if (m_socket != INVALID_SOCKET)
    {
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
    }
    m_bConnected = false;
    m_receiveBuffer.clear();
}

void CTcpClient::StartReceiveThread()
{
    if (m_pReceiveThread != nullptr)
    {
        return;
    }

    m_bRunning = true;
    m_pReceiveThread = new std::thread(&CTcpClient::ReceiveThreadFunc, this);
}

void CTcpClient::StopReceiveThread()
{
    m_bRunning = false;
    
    if (m_pReceiveThread != nullptr)
    {
        // Close socket to unblock recv()
        if (m_socket != INVALID_SOCKET)
        {
            closesocket(m_socket);
            m_socket = INVALID_SOCKET;
        }
        
        if (m_pReceiveThread->joinable())
        {
            m_pReceiveThread->join();
        }
        delete m_pReceiveThread;
        m_pReceiveThread = nullptr;
    }
    m_bConnected = false;
}

void CTcpClient::ReceiveThreadFunc()
{
    char buffer[4096];

    while (m_bRunning && m_bConnected)
    {
        int bytesReceived = recv(m_socket, buffer, sizeof(buffer) - 1, 0);
        
        if (bytesReceived > 0)
        {
            buffer[bytesReceived] = '\0';
            m_receiveBuffer += buffer;
            ProcessBuffer();
        }
        else if (bytesReceived == 0)
        {
            // Connection closed
            DriverLog("CTcpClient: Connection closed by server\n");
            m_bConnected = false;
            break;
        }
        else
        {
            // Error
            if (m_bRunning)
            {
                DriverLog("CTcpClient: Receive error\n");
                m_bConnected = false;
            }
            break;
        }
    }
}

void CTcpClient::ProcessBuffer()
{
    // Process complete lines (JSON messages are newline-delimited)
    size_t pos;
    while ((pos = m_receiveBuffer.find('\n')) != std::string::npos)
    {
        std::string line = m_receiveBuffer.substr(0, pos);
        m_receiveBuffer.erase(0, pos + 1);

        // Trim carriage return if present
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        if (!line.empty() && m_callback)
        {
            m_callback(line);
        }
    }
}

bool CTcpClient::Send(const std::string& data)
{
    if (!m_bConnected || m_socket == INVALID_SOCKET)
    {
        return false;
    }

    int totalSent = 0;
    int dataLen = static_cast<int>(data.length());

    while (totalSent < dataLen)
    {
        int sent = send(m_socket, data.c_str() + totalSent, dataLen - totalSent, 0);
        if (sent == SOCKET_ERROR)
        {
            DriverLog("CTcpClient: Send failed\n");
            return false;
        }
        totalSent += sent;
    }

    return true;
}
