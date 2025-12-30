#ifndef CVISIONSERVER_H
#define CVISIONSERVER_H

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>
#include "cframecapture.h"

// Vision request types
enum class VisionRequestType
{
    CaptureFrame,      // Single frame capture
    CaptureVideo,      // Multi-frame video capture
    GetStatus          // Get capture status
};

// Vision response sent back to Python
struct VisionResponse
{
    std::string type;           // "frame", "video", "status", "error"
    std::vector<std::string> frames;  // Base64 encoded JPEG frames
    int width;
    int height;
    int frameCount;
    std::string message;
};

class CVisionServer
{
public:
    CVisionServer();
    ~CVisionServer();

    // Initialize with frame capture settings
    bool Initialize(int windowX, int windowY, int windowWidth, int windowHeight);
    void Shutdown();

    // Start listening for vision requests on the pose data connection
    void SetSocket(SOCKET socket);
    
    // Process incoming vision request (called from TCP receive thread)
    bool ProcessRequest(const std::string& jsonRequest, std::string& jsonResponse);

    // Check if currently capturing
    bool IsCapturing() const { return m_bCapturing; }

private:
    bool HandleCaptureFrame(std::string& jsonResponse);
    bool HandleCaptureVideo(float duration, int fps, std::string& jsonResponse);
    bool HandleGetStatus(std::string& jsonResponse);

    std::string Base64Encode(const std::vector<uint8_t>& data);
    std::string BuildJsonResponse(const VisionResponse& response);

    CFrameCapture m_frameCapture;
    std::atomic<bool> m_bCapturing;
    std::atomic<bool> m_bInitialized;
    std::mutex m_captureMutex;
};

// Global instance
extern CVisionServer* g_pVisionServer;

#endif // CVISIONSERVER_H
