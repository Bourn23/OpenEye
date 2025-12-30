#include "cframecapture.h"
#include "driverlog.h"
#include <chrono>
#include <cstring>

// Simple JPEG encoder (minimal implementation)
#include "jpge.h"

CFrameCapture* g_pFrameCapture = nullptr;

#if defined(_WIN32)
// Callback for EnumWindows to find SteamVR mirror window
struct EnumWindowsData
{
    HWND hwndFound;
    const char* targetTitle;
};

static BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam)
{
    EnumWindowsData* data = reinterpret_cast<EnumWindowsData*>(lParam);
    char title[256];
    char className[256];
    
    if (GetWindowTextA(hwnd, title, sizeof(title)) > 0)
    {
        GetClassNameA(hwnd, className, sizeof(className));
        
        // Check for specific window types
        bool isHeadsetWindow = (strcmp(title, "Headset Window") == 0);
        bool isVRView = (strstr(title, "VR View") != nullptr);

        if (!IsWindowVisible(hwnd)) return TRUE;

       // Skip tiny windows
        RECT rect;
        GetWindowRect(hwnd, &rect);
        if ((rect.right - rect.left) < 100) return TRUE;

        // --- NEW LOGIC FOR NULL DRIVER ---
        
        // 1. PRIORITY: The Null Driver's "Headset Window"
        // Since you verified this window has visuals, we grab it first.
        if (isHeadsetWindow)
        {
            DriverLog("CFrameCapture: Found Headset Window (Null Driver Target) - %s\n", title);
            data->hwndFound = hwnd;
            return FALSE; // STOP searching, we found the best one.
        }

        // 2. BACKUP: The "VR View" mirror
        // Only use this if we haven't found the headset window yet.
        if (isVRView && data->hwndFound == nullptr)
        {
            DriverLog("CFrameCapture: Found VR View (Backup) - %s\n", title);
            data->hwndFound = hwnd;
            // Don't stop yet, keep looking for the Headset Window just in case
        }
    }
    return TRUE; // Continue enumeration
}
#endif

CFrameCapture::CFrameCapture()
    : m_windowX(0)
    , m_windowY(0)
    , m_windowWidth(0)
    , m_windowHeight(0)
    , m_bInitialized(false)
    , m_bUseMirrorWindow(false)
    , m_bCapturing(false)
    , m_pCaptureThread(nullptr)
#if defined(_WIN32)
    , m_hdcScreen(nullptr)
    , m_hdcMem(nullptr)
    , m_hBitmap(nullptr)
    , m_hwndMirror(nullptr)
#endif
{
}

CFrameCapture::~CFrameCapture()
{
    Shutdown();
}

bool CFrameCapture::FindSteamVRMirrorWindow()
{
#if defined(_WIN32)
    EnumWindowsData data;
    data.hwndFound = nullptr;
    data.targetTitle = nullptr;

    EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&data));

    if (data.hwndFound)
    {
        m_hwndMirror = data.hwndFound;
        m_bUseMirrorWindow = true;

        // Get window dimensions
        RECT rect;
        if (GetClientRect(m_hwndMirror, &rect))
        {
            m_windowWidth = rect.right - rect.left;
            m_windowHeight = rect.bottom - rect.top;
        }

        char title[256];
        GetWindowTextA(m_hwndMirror, title, sizeof(title));
        DriverLog("CFrameCapture: Using SteamVR mirror window: '%s' (%dx%d)\n", 
                  title, m_windowWidth, m_windowHeight);
        return true;
    }

    DriverLog("CFrameCapture: SteamVR mirror window not found, using screen coordinates\n");
#endif
    return false;
}

bool CFrameCapture::Initialize(int windowX, int windowY, int windowWidth, int windowHeight)
{
    if (m_bInitialized)
    {
        Shutdown();
    }

    m_windowX = windowX;
    m_windowY = windowY;
    m_windowWidth = windowWidth;
    m_windowHeight = windowHeight;
    m_bUseMirrorWindow = false;
    m_hwndMirror = nullptr;

#if defined(_WIN32)
    // First, try to find the SteamVR mirror window
    FindSteamVRMirrorWindow();

    m_hdcScreen = GetDC(nullptr);
    if (!m_hdcScreen)
    {
        DriverLog("CFrameCapture: Failed to get screen DC\n");
        return false;
    }

    m_hdcMem = CreateCompatibleDC(m_hdcScreen);
    if (!m_hdcMem)
    {
        ReleaseDC(nullptr, m_hdcScreen);
        DriverLog("CFrameCapture: Failed to create compatible DC\n");
        return false;
    }

    m_hBitmap = CreateCompatibleBitmap(m_hdcScreen, m_windowWidth, m_windowHeight);
    if (!m_hBitmap)
    {
        DeleteDC(m_hdcMem);
        ReleaseDC(nullptr, m_hdcScreen);
        DriverLog("CFrameCapture: Failed to create bitmap\n");
        return false;
    }

    SelectObject(m_hdcMem, m_hBitmap);
#endif

    m_bInitialized = true;
    
    if (m_bUseMirrorWindow)
    {
        DriverLog("CFrameCapture: Initialized with mirror window capture (%dx%d)\n", 
            m_windowWidth, m_windowHeight);
    }
    else
    {
        DriverLog("CFrameCapture: Initialized for region (%d,%d) %dx%d\n", 
            m_windowX, m_windowY, m_windowWidth, m_windowHeight);
    }
    return true;
}

void CFrameCapture::Shutdown()
{
    StopVideoCapture();

#if defined(_WIN32)
    if (m_hBitmap)
    {
        DeleteObject(m_hBitmap);
        m_hBitmap = nullptr;
    }
    if (m_hdcMem)
    {
        DeleteDC(m_hdcMem);
        m_hdcMem = nullptr;
    }
    if (m_hdcScreen)
    {
        ReleaseDC(nullptr, m_hdcScreen);
        m_hdcScreen = nullptr;
    }
    m_hwndMirror = nullptr;
#endif

    m_bInitialized = false;
    m_bUseMirrorWindow = false;
}

bool CFrameCapture::CaptureWindow(HWND hwnd, std::vector<uint8_t>& outRgbData, int& outWidth, int& outHeight)
{
#if defined(_WIN32)
    if (!hwnd || !IsWindow(hwnd))
    {
        DriverLog("CFrameCapture: Invalid window handle\n");
        return false;
    }

    // Get window dimensions
    RECT clientRect;
    if (!GetClientRect(hwnd, &clientRect))
    {
        DriverLog("CFrameCapture: Failed to get client rect\n");
        return false;
    }

    int width = clientRect.right - clientRect.left;
    int height = clientRect.bottom - clientRect.top;

    if (width <= 0 || height <= 0)
    {
        DriverLog("CFrameCapture: Invalid window dimensions %dx%d\n", width, height);
        return false;
    }

    // Recreate bitmap if size changed
    if (width != m_windowWidth || height != m_windowHeight)
    {
        m_windowWidth = width;
        m_windowHeight = height;

        if (m_hBitmap)
        {
            DeleteObject(m_hBitmap);
        }
        m_hBitmap = CreateCompatibleBitmap(m_hdcScreen, m_windowWidth, m_windowHeight);
        if (!m_hBitmap)
        {
            DriverLog("CFrameCapture: Failed to recreate bitmap\n");
            return false;
        }
        SelectObject(m_hdcMem, m_hBitmap);
    }

    // Get window DC
    HDC hdcWindow = GetDC(hwnd);
    if (!hdcWindow)
    {
        DriverLog("CFrameCapture: Failed to get window DC\n");
        return false;
    }

    // Use PrintWindow for better compatibility with hardware-accelerated windows
    // PW_RENDERFULLCONTENT (0x00000002) captures DirectX content on Windows 8.1+
    BOOL result = PrintWindow(hwnd, m_hdcMem, 0x00000002);
    
    if (!result)
    {
        // Fallback to BitBlt from window DC
        result = BitBlt(m_hdcMem, 0, 0, width, height, hdcWindow, 0, 0, SRCCOPY);
        if (!result)
        {
            DriverLog("CFrameCapture: Both PrintWindow and BitBlt failed\n");
            ReleaseDC(hwnd, hdcWindow);
            return false;
        }
    }

    ReleaseDC(hwnd, hdcWindow);

    // Get bitmap data
    BITMAPINFOHEADER bi;
    memset(&bi, 0, sizeof(bi));
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height; // Negative for top-down
    bi.biPlanes = 1;
    bi.biBitCount = 24;
    bi.biCompression = BI_RGB;

    int rowSize = ((width * 3 + 3) & ~3);
    int dataSize = rowSize * height;

    std::vector<uint8_t> bmpData(dataSize);
    
    if (!GetDIBits(m_hdcMem, m_hBitmap, 0, height, 
                   bmpData.data(), (BITMAPINFO*)&bi, DIB_RGB_COLORS))
    {
        DriverLog("CFrameCapture: GetDIBits failed\n");
        return false;
    }

    // Convert BGR to RGB
    outRgbData.resize(width * height * 3);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int srcIdx = y * rowSize + x * 3;
            int dstIdx = (y * width + x) * 3;
            outRgbData[dstIdx + 0] = bmpData[srcIdx + 2]; // R
            outRgbData[dstIdx + 1] = bmpData[srcIdx + 1]; // G
            outRgbData[dstIdx + 2] = bmpData[srcIdx + 0]; // B
        }
    }

    outWidth = width;
    outHeight = height;
    return true;
#else
    return false;
#endif
}

bool CFrameCapture::CaptureScreenRegion(std::vector<uint8_t>& outRgbData, int& outWidth, int& outHeight)
{
#if defined(_WIN32)
    if (!m_bInitialized)
    {
        return false;
    }

    // ---------------------------------------------------------
    // JUST-IN-TIME DISCOVERY (The Fix)
    // ---------------------------------------------------------
    // If we are currently in "Fallback Mode" (no mirror window), 
    // try to find the window NOW, before capturing.
    if (!m_bUseMirrorWindow || !m_hwndMirror || !IsWindow(m_hwndMirror))
    {
        if (FindSteamVRMirrorWindow()) 
        {
            DriverLog("CFrameCapture: Late discovery! Found VR View window.\n");
        }
    }
    // ---------------------------------------------------------

    // 1. Try to capture the specific Window
    if (m_bUseMirrorWindow && m_hwndMirror)
    {
        if (IsWindow(m_hwndMirror))
        {
            if (CaptureWindow(m_hwndMirror, outRgbData, outWidth, outHeight))
            {
                return true;
            }
            DriverLog("CFrameCapture: CaptureWindow failed, will try fallback.\n");
        }
        else
        {
             DriverLog("CFrameCapture: Mirror window handle became invalid.\n");
             m_bUseMirrorWindow = false; 
             m_hwndMirror = nullptr;
        }
    }

    // 2. Fallback: Absolute Screen Coordinates (Desktop Capture)
    // This runs if the window wasn't found OR if capturing the window failed.
    if (!BitBlt(m_hdcMem, 0, 0, m_windowWidth, m_windowHeight,
                m_hdcScreen, m_windowX, m_windowY, SRCCOPY))
    {
        DriverLog("CFrameCapture: BitBlt fallback failed\n");
        return false;
    }

    // Get bitmap info
    BITMAPINFOHEADER bi;
    memset(&bi, 0, sizeof(bi));
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = m_windowWidth;
    bi.biHeight = -m_windowHeight; // Negative for top-down
    bi.biPlanes = 1;
    bi.biBitCount = 24;
    bi.biCompression = BI_RGB;

    // Calculate row size (must be DWORD aligned)
    int rowSize = ((m_windowWidth * 3 + 3) & ~3);
    int dataSize = rowSize * m_windowHeight;

    std::vector<uint8_t> bmpData(dataSize);
    
    if (!GetDIBits(m_hdcMem, m_hBitmap, 0, m_windowHeight, 
                   bmpData.data(), (BITMAPINFO*)&bi, DIB_RGB_COLORS))
    {
        DriverLog("CFrameCapture: GetDIBits failed\n");
        return false;
    }

    // Convert BGR to RGB
    outRgbData.resize(m_windowWidth * m_windowHeight * 3);
    for (int y = 0; y < m_windowHeight; y++)
    {
        for (int x = 0; x < m_windowWidth; x++)
        {
            int srcIdx = y * rowSize + x * 3;
            int dstIdx = (y * m_windowWidth + x) * 3;
            outRgbData[dstIdx + 0] = bmpData[srcIdx + 2]; // R
            outRgbData[dstIdx + 1] = bmpData[srcIdx + 1]; // G
            outRgbData[dstIdx + 2] = bmpData[srcIdx + 0]; // B
        }
    }

    outWidth = m_windowWidth;
    outHeight = m_windowHeight;
    return true;
#else
    // Linux: Would need X11/XCB capture
    DriverLog("CFrameCapture: Screen capture not implemented on this platform\n");
    return false;
#endif
}

bool CFrameCapture::EncodeJpeg(const std::vector<uint8_t>& rgbData, int width, int height, 
                               std::vector<uint8_t>& outJpeg, int quality)
{
    // Use jpge for JPEG encoding
    int bufSize = width * height * 3 + 1024; // Generous buffer
    outJpeg.resize(bufSize);

    jpge::params params;
    params.m_quality = quality;
    params.m_subsampling = jpge::H2V2; // 4:2:0 subsampling

    int actualSize = bufSize;
    if (!jpge::compress_image_to_jpeg_file_in_memory(
            outJpeg.data(), actualSize, width, height, 3, rgbData.data(), params))
    {
        DriverLog("CFrameCapture: JPEG encoding failed\n");
        return false;
    }

    outJpeg.resize(actualSize);
    return true;
}

bool CFrameCapture::CaptureFrame(FrameData& outFrame)
{
    std::vector<uint8_t> rgbData;
    int width, height;

    if (!CaptureScreenRegion(rgbData, width, height))
    {
        return false;
    }

    if (!EncodeJpeg(rgbData, width, height, outFrame.jpegData))
    {
        return false;
    }

    outFrame.width = width;
    outFrame.height = height;
    outFrame.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    return true;
}

void CFrameCapture::StartVideoCapture(float durationSeconds, int fps, CaptureCallback callback)
{
    if (m_bCapturing)
    {
        DriverLog("CFrameCapture: Already capturing\n");
        return;
    }

    m_bCapturing = true;
    m_pCaptureThread = new std::thread(&CFrameCapture::VideoCaptureThread, this, 
                                        durationSeconds, fps, callback);
}

void CFrameCapture::StopVideoCapture()
{
    m_bCapturing = false;
    
    if (m_pCaptureThread)
    {
        if (m_pCaptureThread->joinable())
        {
            m_pCaptureThread->join();
        }
        delete m_pCaptureThread;
        m_pCaptureThread = nullptr;
    }
}

void CFrameCapture::VideoCaptureThread(float durationSeconds, int fps, CaptureCallback callback)
{
    DriverLog("CFrameCapture: Starting video capture for %.1f seconds at %d fps\n", 
              durationSeconds, fps);

    std::vector<FrameData> frames;
    int frameInterval = 1000 / fps; // ms between frames
    int totalFrames = static_cast<int>(durationSeconds * fps);

    auto startTime = std::chrono::steady_clock::now();

    for (int i = 0; i < totalFrames && m_bCapturing; i++)
    {
        auto frameStart = std::chrono::steady_clock::now();

        FrameData frame;
        if (CaptureFrame(frame))
        {
            frames.push_back(std::move(frame));
        }

        // Wait for next frame time
        auto frameEnd = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();
        
        if (elapsed < frameInterval)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(frameInterval - elapsed));
        }
    }

    DriverLog("CFrameCapture: Captured %zu frames\n", frames.size());

    // Store frames
    {
        std::lock_guard<std::mutex> lock(m_framesMutex);
        m_capturedFrames = std::move(frames);
    }

    m_bCapturing = false;

    // Call callback with captured frames
    if (callback)
    {
        std::lock_guard<std::mutex> lock(m_framesMutex);
        callback(m_capturedFrames);
    }
}

std::vector<FrameData> CFrameCapture::GetCapturedFrames()
{
    std::lock_guard<std::mutex> lock(m_framesMutex);
    return m_capturedFrames;
}
