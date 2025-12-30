#include "cserverdriver_sample.h"
#include "basics.h"
#include "driverlog.h"
#include "cvisionserver.h"

using namespace vr;

EVRInitError CServerDriver_Sample::Init(vr::IVRDriverContext *pDriverContext)
{
    VR_INIT_SERVER_DRIVER_CONTEXT(pDriverContext);
    InitDriverLog(vr::VRDriverLog());

    DriverLog("CServerDriver_Sample::Init() called\n");

    // Initialize TCP pose receiver if enabled
    bool tcpEnabled = vr::VRSettings()->GetBool(k_pch_Sample_Section, k_pch_Sample_TcpEnabled_Bool);
    DriverLog("TCP enabled: %s\n", tcpEnabled ? "true" : "false");
    if (tcpEnabled)
    {
        char hostBuf[256];
        vr::VRSettings()->GetString(k_pch_Sample_Section, k_pch_Sample_TcpHost_String, hostBuf, sizeof(hostBuf));
        int port = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_TcpPort_Int32);

        DriverLog("TCP config: host=%s, port=%d\n", hostBuf, port);

        g_pPoseDataReceiver = new CPoseDataReceiver();
        if (g_pPoseDataReceiver->Start(hostBuf, port))
        {
            DriverLog("TCP pose receiver started successfully\n");
        }
        else
        {
            DriverLog("TCP pose receiver failed to start\n");
        }
    }

    m_pNullHmdLatest = new CSampleDeviceDriver();
    vr::VRServerDriverHost()->TrackedDeviceAdded(m_pNullHmdLatest->GetSerialNumber().c_str(), vr::TrackedDeviceClass_HMD, m_pNullHmdLatest);

    m_pController = new CSampleControllerDriver();
    m_pController->SetControllerIndex(1);
    vr::VRServerDriverHost()->TrackedDeviceAdded(m_pController->GetSerialNumber().c_str(), vr::TrackedDeviceClass_Controller, m_pController);

    m_pController2 = new CSampleControllerDriver();
    m_pController2->SetControllerIndex(2);
    vr::VRServerDriverHost()->TrackedDeviceAdded(m_pController2->GetSerialNumber().c_str(), vr::TrackedDeviceClass_Controller, m_pController2);

    // Initialize vision server for frame capture
    bool visionEnabled = vr::VRSettings()->GetBool(k_pch_Sample_Section, k_pch_Sample_VisionEnabled_Bool);
    DriverLog("Vision enabled: %s\n", visionEnabled ? "true" : "false");
    if (visionEnabled && tcpEnabled)
    {
        int windowX = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_WindowX_Int32);
        int windowY = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_WindowY_Int32);
        int windowWidth = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_WindowWidth_Int32);
        int windowHeight = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_WindowHeight_Int32);

        g_pVisionServer = new CVisionServer();
        if (g_pVisionServer->Initialize(windowX, windowY, windowWidth, windowHeight))
        {
            DriverLog("Vision server initialized for window (%d,%d) %dx%d\n", 
                windowX, windowY, windowWidth, windowHeight);

            // Set up send callback for vision responses
            if (g_pPoseDataReceiver)
            {
                g_pPoseDataReceiver->SetSendCallback([](const std::string& response) {
                    if (g_pPoseDataReceiver)
                    {
                        g_pPoseDataReceiver->GetTcpClient().Send(response);
                        DriverLog("Vision response sent: %zu bytes\n", response.size());
                    }
                });
            }
        }
        else
        {
            DriverLog("Vision server failed to initialize\n");
            delete g_pVisionServer;
            g_pVisionServer = nullptr;
        }
    }

    return VRInitError_None;
}

void CServerDriver_Sample::Cleanup()
{
    DriverLog("CServerDriver_Sample::Cleanup() called\n");

    if (g_pVisionServer)
    {
        g_pVisionServer->Shutdown();
        delete g_pVisionServer;
        g_pVisionServer = nullptr;
    }

    if (g_pPoseDataReceiver)
    {
        g_pPoseDataReceiver->Stop();
        delete g_pPoseDataReceiver;
        g_pPoseDataReceiver = nullptr;
    }

    delete m_pNullHmdLatest;
    m_pNullHmdLatest = NULL;
    delete m_pController;
    m_pController = NULL;
    delete m_pController2;
    m_pController2 = NULL;
}

void CServerDriver_Sample::RunFrame()
{
    if (m_pNullHmdLatest) {
        m_pNullHmdLatest->RunFrame();
    }
    if (m_pController) {
        m_pController->RunFrame();
    }
    if (m_pController2) {
        m_pController2->RunFrame();
    }

#if 0
    vr::VREvent_t vrEvent;
    while ( vr::VRServerDriverHost()->PollNextEvent( &vrEvent, sizeof( vrEvent ) ) )
    {
        if ( m_pController )
        {
            m_pController->ProcessEvent(vrEvent);
        }
        if ( m_pController2)
        {
            m_pController2->ProcessEvent(vrEvent);
        }
    }
#endif
}
