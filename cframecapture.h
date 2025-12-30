#ifndef CFRAMECAPTURE_H
#define CFRAMECAPTURE_H

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>

// Frame data structure
struct FrameData
{
    std::vector<uint8_t> jpegData;
    int width;
    int height;
    uint64_t timestamp;
};

// Callback for when capture is complete
using CaptureCallback = std::function<void(const std::vector<FrameData>&)>;

class CFrameCapture
{
public:
    CFrameCapture();
    ~CFrameCapture();

    // Initialize with window dimensions (fallback if mirror window not found)
    bool Initialize(int windowX, int windowY, int windowWidth, int windowHeight);
    void Shutdown();

    // Capture a single frame (returns JPEG data)
    bool CaptureFrame(FrameData& outFrame);

    // Capture video for specified duration (async)
    void StartVideoCapture(float durationSeconds, int fps, CaptureCallback callback);
    void StopVideoCapture();
    bool IsCapturing() const { return m_bCapturing; }

    // Get last captured frames
    std::vector<FrameData> GetCapturedFrames();

    // Try to find and use SteamVR mirror window
    bool FindSteamVRMirrorWindow();

private:
    void VideoCaptureThread(float durationSeconds, int fps, CaptureCallback callback);
    bool CaptureScreenRegion(std::vector<uint8_t>& outRgbData, int& outWidth, int& outHeight);
    bool CaptureWindow(HWND hwnd, std::vector<uint8_t>& outRgbData, int& outWidth, int& outHeight);
    bool EncodeJpeg(const std::vector<uint8_t>& rgbData, int width, int height, std::vector<uint8_t>& outJpeg, int quality = 80);

    int m_windowX;
    int m_windowY;
    int m_windowWidth;
    int m_windowHeight;
    bool m_bInitialized;
    bool m_bUseMirrorWindow;

    std::atomic<bool> m_bCapturing;
    std::thread* m_pCaptureThread;
    std::mutex m_framesMutex;
    std::vector<FrameData> m_capturedFrames;

#if defined(_WIN32)
    HDC m_hdcScreen;
    HDC m_hdcMem;
    HBITMAP m_hBitmap;
    HWND m_hwndMirror;  // SteamVR mirror window handle
#endif
};

// Global instance
extern CFrameCapture* g_pFrameCapture;

#endif // CFRAMECAPTURE_H
