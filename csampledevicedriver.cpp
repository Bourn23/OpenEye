#include "csampledevicedriver.h"

#include "basics.h"
#include <cmath>
#include "driverlog.h"
#include "vrmath.h"
#include "cposedatareceiver.h"

using namespace vr;

// Head tracking vars
static double angleX = 0, angleY = 0, angleZ = 0;
static double pX = 0, pY = 0, pZ = 0;

struct Quaternion {
    double w, x, y, z;
};

// Function to convert rotation angles to quaternion using global/extrinsic rotation order
// Rotations are applied in the global coordinate system: first Z (yaw), then Y (pitch), then X (roll)
// This is equivalent to ZYX extrinsic order, which means rotations are relative to the fixed world axes
Quaternion anglesToQuaternion(double angleX, double angleY, double angleZ) {
    // Convert angles from degrees to radians
    double radX = DEG_TO_RAD(angleX);  // Roll  - rotation around global X axis
    double radY = DEG_TO_RAD(angleY);  // Pitch - rotation around global Y axis
    double radZ = DEG_TO_RAD(angleZ);  // Yaw   - rotation around global Z axis

    // Calculate half-angle trig values
    double cX = cos(radX * 0.5);
    double sX = sin(radX * 0.5);
    double cY = cos(radY * 0.5);
    double sY = sin(radY * 0.5);
    double cZ = cos(radZ * 0.5);
    double sZ = sin(radZ * 0.5);

    // ZYX extrinsic rotation order (equivalent to XYZ intrinsic reversed)
    // This applies rotations relative to the fixed global coordinate system
    Quaternion q;
    q.w = cZ * cY * cX + sZ * sY * sX;
    q.x = cZ * cY * sX - sZ * sY * cX;
    q.y = cZ * sY * cX + sZ * cY * sX;
    q.z = sZ * cY * cX - cZ * sY * sX;

    return q;
}

// Function to normalize a quaternion
Quaternion normalizeQuaternion(const Quaternion& q) {
    double length = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    Quaternion normalized;
    normalized.w = q.w / length;
    normalized.x = q.x / length;
    normalized.y = q.y / length;
    normalized.z = q.z / length;
    return normalized;
}

CSampleDeviceDriver::CSampleDeviceDriver() {
    m_unObjectId = vr::k_unTrackedDeviceIndexInvalid;
    m_ulPropertyContainer = vr::k_ulInvalidPropertyContainer;

    //DriverLog( "Using settings values\n" );
    m_flIPD = vr::VRSettings()->GetFloat(k_pch_SteamVR_Section, k_pch_SteamVR_IPD_Float);

    char buf[1024];
    vr::VRSettings()->GetString(k_pch_Sample_Section, k_pch_Sample_SerialNumber_String, buf, sizeof(buf));
    m_sSerialNumber = buf;

    vr::VRSettings()->GetString(k_pch_Sample_Section, k_pch_Sample_ModelNumber_String, buf, sizeof(buf));
    m_sModelNumber = buf;

    m_nWindowX = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_WindowX_Int32);
    m_nWindowY = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_WindowY_Int32);
    m_nWindowWidth = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_WindowWidth_Int32);
    m_nWindowHeight = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_WindowHeight_Int32);
    m_nRenderWidth = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_RenderWidth_Int32);
    m_nRenderHeight = vr::VRSettings()->GetInt32(k_pch_Sample_Section, k_pch_Sample_RenderHeight_Int32);
    m_flSecondsFromVsyncToPhotons = vr::VRSettings()->GetFloat(k_pch_Sample_Section, k_pch_Sample_SecondsFromVsyncToPhotons_Float);
    m_flDisplayFrequency = vr::VRSettings()->GetFloat(k_pch_Sample_Section, k_pch_Sample_DisplayFrequency_Float);
}

CSampleDeviceDriver::~CSampleDeviceDriver() {}

EVRInitError CSampleDeviceDriver::Activate(TrackedDeviceIndex_t unObjectId) {
    m_unObjectId = unObjectId;
    m_ulPropertyContainer = vr::VRProperties()->TrackedDeviceToPropertyContainer(m_unObjectId);

    vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, Prop_ModelNumber_String, m_sModelNumber.c_str());
    vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, Prop_RenderModelName_String, m_sModelNumber.c_str());
    vr::VRProperties()->SetFloatProperty(m_ulPropertyContainer, Prop_UserIpdMeters_Float, m_flIPD);
    vr::VRProperties()->SetFloatProperty(m_ulPropertyContainer, Prop_UserHeadToEyeDepthMeters_Float, 0.f);
    vr::VRProperties()->SetFloatProperty(m_ulPropertyContainer, Prop_DisplayFrequency_Float, m_flDisplayFrequency);
    vr::VRProperties()->SetFloatProperty(m_ulPropertyContainer, Prop_SecondsFromVsyncToPhotons_Float, m_flSecondsFromVsyncToPhotons);

    vr::VRProperties()->SetUint64Property(m_ulPropertyContainer, Prop_CurrentUniverseId_Uint64, 2);
    vr::VRProperties()->SetBoolProperty(m_ulPropertyContainer, Prop_IsOnDesktop_Bool, false);
    vr::VRProperties()->SetBoolProperty(m_ulPropertyContainer, Prop_DisplayDebugMode_Bool, true);

    bool bSetupIconUsingExternalResourceFile = true;
    if (!bSetupIconUsingExternalResourceFile) {
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_NamedIconPathDeviceOff_String, "{null}/icons/headset_sample_status_off.png");
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_NamedIconPathDeviceSearching_String, "{null}/icons/headset_sample_status_searching.gif");
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_NamedIconPathDeviceSearchingAlert_String, "{null}/icons/headset_sample_status_searching_alert.gif");
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_NamedIconPathDeviceReady_String, "{null}/icons/headset_sample_status_ready.png");
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_NamedIconPathDeviceReadyAlert_String, "{null}/icons/headset_sample_status_ready_alert.png");
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_NamedIconPathDeviceNotReady_String, "{null}/icons/headset_sample_status_error.png");
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_NamedIconPathDeviceStandby_String, "{null}/icons/headset_sample_status_standby.png");
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_NamedIconPathDeviceAlertLow_String, "{null}/icons/headset_sample_status_ready_low.png");
    }

    return VRInitError_None;
}

void CSampleDeviceDriver::Deactivate() {
    m_unObjectId = vr::k_unTrackedDeviceIndexInvalid;
}

void CSampleDeviceDriver::EnterStandby() {}

void* CSampleDeviceDriver::GetComponent(const char* pchComponentNameAndVersion) {
    if (!_stricmp(pchComponentNameAndVersion, vr::IVRDisplayComponent_Version)) {
        return (vr::IVRDisplayComponent*)this;
    }
    return NULL;
}

void CSampleDeviceDriver::PowerOff() {}

void CSampleDeviceDriver::DebugRequest(const char* pchRequest, char* pchResponseBuffer, uint32_t unResponseBufferSize) {
    if (unResponseBufferSize >= 1) {
        pchResponseBuffer[0] = 0;
    }
}

void CSampleDeviceDriver::GetWindowBounds(int32_t* pnX, int32_t* pnY, uint32_t* pnWidth, uint32_t* pnHeight) {
    *pnX = m_nWindowX;
    *pnY = m_nWindowY;
    *pnWidth = m_nWindowWidth;
    *pnHeight = m_nWindowHeight;
}

bool CSampleDeviceDriver::IsDisplayOnDesktop() {
    return true;
}

bool CSampleDeviceDriver::IsDisplayRealDisplay() {
    return false;
}

void CSampleDeviceDriver::GetRecommendedRenderTargetSize(uint32_t* pnWidth, uint32_t* pnHeight) {
    *pnWidth = m_nRenderWidth;
    *pnHeight = m_nRenderHeight;
}

void CSampleDeviceDriver::GetEyeOutputViewport(EVREye eEye, uint32_t* pnX, uint32_t* pnY, uint32_t* pnWidth, uint32_t* pnHeight) {
    // Single screen mode: both eyes see the full window (no stereo split)
    // This gives a single unified view covering both eyes' field of vision
    *pnX = 0;
    *pnY = 0;
    *pnWidth = m_nWindowWidth;
    *pnHeight = m_nWindowHeight;
}

void CSampleDeviceDriver::GetProjectionRaw(EVREye eEye, float* pfLeft, float* pfRight, float* pfTop, float* pfBottom) {
    *pfLeft = -1.0;
    *pfRight = 1.0;
    *pfTop = -1.0;
    *pfBottom = 1.0;
}

DistortionCoordinates_t CSampleDeviceDriver::ComputeDistortion(EVREye eEye, float fU, float fV) {
    DistortionCoordinates_t coordinates;
    coordinates.rfBlue[0] = fU;
    coordinates.rfBlue[1] = fV;
    coordinates.rfGreen[0] = fU;
    coordinates.rfGreen[1] = fV;
    coordinates.rfRed[0] = fU;
    coordinates.rfRed[1] = fV;
    return coordinates;
}

vr::DriverPose_t CSampleDeviceDriver::GetPose() {
    static vr::DriverPose_t pose = { 0 };
    pose.poseIsValid = true;
    pose.result = vr::TrackingResult_Running_OK;
    pose.deviceIsConnected = true;
    pose.qWorldFromDriverRotation = HmdQuaternion_Init(1, 0, 0, 0);
    pose.qDriverFromHeadRotation = HmdQuaternion_Init(1, 0, 0, 0);

    // Get pose data from TCP receiver
    if (g_pPoseDataReceiver && g_pPoseDataReceiver->IsConnected())
    {
        PoseData tcpPose = g_pPoseDataReceiver->GetHeadsetPose();
        if (tcpPose.updated)
        {
            DriverLog("CSampleDeviceDriver: Got headset pose update: pos=[%.2f,%.2f,%.2f], rot=[%.2f,%.2f,%.2f]\n",
                tcpPose.posX, tcpPose.posY, tcpPose.posZ, tcpPose.rotX, tcpPose.rotY, tcpPose.rotZ);
            pX = tcpPose.posX;
            pY = tcpPose.posY;
            pZ = tcpPose.posZ;
            angleX = tcpPose.rotX;
            angleY = tcpPose.rotY;
            angleZ = tcpPose.rotZ;
        }
    }

    // Update pose position
    pose.vecPosition[0] = pX;
    pose.vecPosition[1] = pY;
    pose.vecPosition[2] = pZ;

    // Convert rotation angles to quaternion
    Quaternion currentRotation = anglesToQuaternion(angleX, angleY, angleZ);

    // Normalize the quaternion to avoid drift
    currentRotation = normalizeQuaternion(currentRotation);

    // Update the pose with the new quaternion
    pose.qRotation.w = currentRotation.w;
    pose.qRotation.x = currentRotation.x;
    pose.qRotation.y = currentRotation.y;
    pose.qRotation.z = currentRotation.z;

    return pose;
}

void CSampleDeviceDriver::RunFrame() {
    // In a real driver, this should happen from some pose tracking thread.
    // The RunFrame interval is unspecified and can be very irregular if some other
    // driver blocks it for some periodic task.
    if (m_unObjectId != vr::k_unTrackedDeviceIndexInvalid) {
        vr::VRServerDriverHost()->TrackedDevicePoseUpdated(m_unObjectId, GetPose(), sizeof(DriverPose_t));
    }
}
