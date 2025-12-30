#include "cvisionserver.h"
#include "driverlog.h"
#include <sstream>
#include <cstring>

CVisionServer* g_pVisionServer = nullptr;

// Base64 encoding table
static const char* s_base64_chars = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

CVisionServer::CVisionServer()
    : m_bCapturing(false)
    , m_bInitialized(false)
{
}

CVisionServer::~CVisionServer()
{
    Shutdown();
}

bool CVisionServer::Initialize(int windowX, int windowY, int windowWidth, int windowHeight)
{
    if (m_bInitialized)
    {
        Shutdown();
    }

    if (!m_frameCapture.Initialize(windowX, windowY, windowWidth, windowHeight))
    {
        DriverLog("CVisionServer: Failed to initialize frame capture\n");
        return false;
    }

    m_bInitialized = true;
    DriverLog("CVisionServer: Initialized for window region (%d,%d) %dx%d\n",
        windowX, windowY, windowWidth, windowHeight);
    return true;
}

void CVisionServer::Shutdown()
{
    m_frameCapture.Shutdown();
    m_bInitialized = false;
}

std::string CVisionServer::Base64Encode(const std::vector<uint8_t>& data)
{
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i < data.size())
    {
        uint32_t octet_a = i < data.size() ? data[i++] : 0;
        uint32_t octet_b = i < data.size() ? data[i++] : 0;
        uint32_t octet_c = i < data.size() ? data[i++] : 0;

        uint32_t triple = (octet_a << 16) + (octet_b << 8) + octet_c;

        result += s_base64_chars[(triple >> 18) & 0x3F];
        result += s_base64_chars[(triple >> 12) & 0x3F];
        result += s_base64_chars[(triple >> 6) & 0x3F];
        result += s_base64_chars[triple & 0x3F];
    }

    // Add padding
    size_t mod = data.size() % 3;
    if (mod == 1)
    {
        result[result.size() - 1] = '=';
        result[result.size() - 2] = '=';
    }
    else if (mod == 2)
    {
        result[result.size() - 1] = '=';
    }

    return result;
}

std::string CVisionServer::BuildJsonResponse(const VisionResponse& response)
{
    std::ostringstream json;
    json << "{\"type\":\"" << response.type << "\"";
    
    if (!response.message.empty())
    {
        json << ",\"message\":\"" << response.message << "\"";
    }
    
    json << ",\"width\":" << response.width;
    json << ",\"height\":" << response.height;
    json << ",\"frameCount\":" << response.frameCount;
    
    if (!response.frames.empty())
    {
        json << ",\"frames\":[";
        for (size_t i = 0; i < response.frames.size(); i++)
        {
            if (i > 0) json << ",";
            json << "\"" << response.frames[i] << "\"";
        }
        json << "]";
    }
    
    json << "}";
    return json.str();
}

bool CVisionServer::HandleCaptureFrame(std::string& jsonResponse)
{
    std::lock_guard<std::mutex> lock(m_captureMutex);
    
    FrameData frame;
    if (!m_frameCapture.CaptureFrame(frame))
    {
        VisionResponse resp;
        resp.type = "error";
        resp.message = "Failed to capture frame";
        resp.width = 0; resp.height = 0; resp.frameCount = 0;
        jsonResponse = BuildJsonResponse(resp);
        return false;
    }

    VisionResponse resp;
    resp.type = "frame";
    resp.width = frame.width;
    resp.height = frame.height;
    resp.frameCount = 1;
    resp.frames.push_back(Base64Encode(frame.jpegData));
    
    jsonResponse = BuildJsonResponse(resp);
    DriverLog("CVisionServer: Captured frame %dx%d, %zu bytes\n", 
        frame.width, frame.height, frame.jpegData.size());
    return true;
}

bool CVisionServer::HandleCaptureVideo(float duration, int fps, std::string& jsonResponse)
{
    std::lock_guard<std::mutex> lock(m_captureMutex);
    
    if (m_bCapturing)
    {
        VisionResponse resp;
        resp.type = "error";
        resp.message = "Already capturing";
        resp.width = 0; resp.height = 0; resp.frameCount = 0;
        jsonResponse = BuildJsonResponse(resp);
        return false;
    }

    m_bCapturing = true;
    DriverLog("CVisionServer: Starting video capture for %.1f seconds at %d fps\n", duration, fps);

    // Capture synchronously (blocking) for simplicity
    std::vector<FrameData> frames;
    int frameInterval = 1000 / fps;
    int totalFrames = static_cast<int>(duration * fps);

    for (int i = 0; i < totalFrames; i++)
    {
        auto frameStart = std::chrono::steady_clock::now();
        
        FrameData frame;
        if (m_frameCapture.CaptureFrame(frame))
        {
            frames.push_back(std::move(frame));
        }

        auto frameEnd = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();
        
        if (elapsed < frameInterval)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(frameInterval - elapsed));
        }
    }

    m_bCapturing = false;

    VisionResponse resp;
    resp.type = "video";
    resp.frameCount = static_cast<int>(frames.size());
    resp.width = frames.empty() ? 0 : frames[0].width;
    resp.height = frames.empty() ? 0 : frames[0].height;
    
    for (const auto& frame : frames)
    {
        resp.frames.push_back(Base64Encode(frame.jpegData));
    }

    jsonResponse = BuildJsonResponse(resp);
    DriverLog("CVisionServer: Captured %zu frames\n", frames.size());
    return true;
}

bool CVisionServer::HandleGetStatus(std::string& jsonResponse)
{
    VisionResponse resp;
    resp.type = "status";
    resp.message = m_bCapturing ? "capturing" : "ready";
    resp.width = 0;
    resp.height = 0;
    resp.frameCount = 0;
    jsonResponse = BuildJsonResponse(resp);
    return true;
}

bool CVisionServer::ProcessRequest(const std::string& jsonRequest, std::string& jsonResponse)
{
    if (!m_bInitialized)
    {
        VisionResponse resp;
        resp.type = "error";
        resp.message = "Vision server not initialized";
        resp.width = 0; resp.height = 0; resp.frameCount = 0;
        jsonResponse = BuildJsonResponse(resp);
        return false;
    }

    DriverLog("CVisionServer: Processing request: %s\n", jsonRequest.c_str());

    // Parse request type from JSON
    // Format: {"type":"vision_request","action":"capture_frame"} or
    //         {"type":"vision_request","action":"capture_video","duration":3.0,"fps":10}
    
    size_t actionPos = jsonRequest.find("\"action\"");
    if (actionPos == std::string::npos)
    {
        VisionResponse resp;
        resp.type = "error";
        resp.message = "Missing action field";
        resp.width = 0; resp.height = 0; resp.frameCount = 0;
        jsonResponse = BuildJsonResponse(resp);
        return false;
    }

    if (jsonRequest.find("capture_frame") != std::string::npos)
    {
        return HandleCaptureFrame(jsonResponse);
    }
    else if (jsonRequest.find("capture_video") != std::string::npos)
    {
        // Parse duration and fps
        float duration = 3.0f;
        int fps = 10;

        size_t durPos = jsonRequest.find("\"duration\"");
        if (durPos != std::string::npos)
        {
            size_t colonPos = jsonRequest.find(':', durPos);
            if (colonPos != std::string::npos)
            {
                duration = static_cast<float>(std::atof(jsonRequest.c_str() + colonPos + 1));
            }
        }

        size_t fpsPos = jsonRequest.find("\"fps\"");
        if (fpsPos != std::string::npos)
        {
            size_t colonPos = jsonRequest.find(':', fpsPos);
            if (colonPos != std::string::npos)
            {
                fps = std::atoi(jsonRequest.c_str() + colonPos + 1);
            }
        }

        return HandleCaptureVideo(duration, fps, jsonResponse);
    }
    else if (jsonRequest.find("get_status") != std::string::npos)
    {
        return HandleGetStatus(jsonResponse);
    }

    VisionResponse resp;
    resp.type = "error";
    resp.message = "Unknown action";
    resp.width = 0; resp.height = 0; resp.frameCount = 0;
    jsonResponse = BuildJsonResponse(resp);
    return false;
}
