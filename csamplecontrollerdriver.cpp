#include "csamplecontrollerdriver.h"
#include "basics.h"
#include <cmath>
#include "driverlog.h"
#include "vrmath.h"
#include "cposedatareceiver.h"

using namespace vr;

// Controller tracking vars
static double cAngleX = 0, cAngleY = 0, cAngleZ = 0;
static double c2AngleX = 0, c2AngleY = 0, c2AngleZ = 0;

static double cpX = 0, cpY = 0, cpZ = 0;
static double c2pX = 0, c2pY = 0, c2pZ = 0;

// Controller input state from TCP
static ControllerInput ctrl1Input;
static ControllerInput ctrl2Input;

// Keyboard input toggle - disabled by default, enabled with Shift+F10
static bool g_keyboardInputEnabled = false;
static bool g_toggleKeyWasPressed = false;

struct Quaternion {
    double w, x, y, z;
};

// Function to convert rotation angles to quaternion using global/extrinsic rotation order
// Rotations are applied in the global coordinate system: first Z (yaw), then Y (pitch), then X (roll)
// This is equivalent to ZYX extrinsic order, which means rotations are relative to the fixed world axes
Quaternion anglesToQuaternion1(double angleX, double angleY, double angleZ) {
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
Quaternion normalizeQuaternion1(const Quaternion &q) {
    double length = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    Quaternion normalized;
    normalized.w = q.w / length;
    normalized.x = q.x / length;
    normalized.y = q.y / length;
    normalized.z = q.z / length;
    return normalized;
}

CSampleControllerDriver::CSampleControllerDriver() {
    m_unObjectId = vr::k_unTrackedDeviceIndexInvalid;
    m_ulPropertyContainer = vr::k_ulInvalidPropertyContainer;
}

void CSampleControllerDriver::SetControllerIndex(int32_t CtrlIndex) {
    ControllerIndex = CtrlIndex;
}

CSampleControllerDriver::~CSampleControllerDriver() {}

vr::EVRInitError CSampleControllerDriver::Activate(vr::TrackedDeviceIndex_t unObjectId) {
    m_unObjectId = unObjectId;
    m_ulPropertyContainer = vr::VRProperties()->TrackedDeviceToPropertyContainer(m_unObjectId);

    vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_ControllerType_String, "vive_controller");
    vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_LegacyInputProfile_String, "vive_controller");

    vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_ModelNumber_String, "ViveMV");
    vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_ManufacturerName_String, "HTC");
    vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, vr::Prop_RenderModelName_String, "vr_controller_vive_1_5");

    vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, Prop_TrackingSystemName_String, "VR Controller");
    vr::VRProperties()->SetInt32Property(m_ulPropertyContainer, Prop_DeviceClass_Int32, TrackedDeviceClass_Controller);

    switch (ControllerIndex) {
    case 1:
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, Prop_SerialNumber_String, "CTRL1Serial");
        break;
    case 2:
        vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, Prop_SerialNumber_String, "CTRL2Serial");
        break;
    }

    uint64_t supportedButtons = 0xFFFFFFFFFFFFFFFFULL;
    vr::VRProperties()->SetUint64Property(m_ulPropertyContainer, vr::Prop_SupportedButtons_Uint64, supportedButtons);

    switch (ControllerIndex) {
    case 1:
        vr::VRProperties()->SetInt32Property(m_ulPropertyContainer, Prop_ControllerRoleHint_Int32, TrackedControllerRole_LeftHand);
        break;
    case 2:
        vr::VRProperties()->SetInt32Property(m_ulPropertyContainer, Prop_ControllerRoleHint_Int32, TrackedControllerRole_RightHand);
        break;
    }

    vr::VRProperties()->SetStringProperty(m_ulPropertyContainer, Prop_InputProfilePath_String, "{null}/input/mycontroller_profile.json");

    // Button handles
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/system/click", &HButtons[0]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/application_menu/click", &HButtons[1]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/grip/click", &HButtons[2]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/dpad_left/click", &HButtons[3]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/dpad_up/click", &HButtons[4]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/dpad_right/click", &HButtons[5]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/dpad_down/click", &HButtons[6]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/a/click", &HButtons[7]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/b/click", &HButtons[8]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/x/click", &HButtons[9]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/y/click", &HButtons[10]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/trigger/click", &HButtons[11]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/trigger/value", &HButtons[12]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/trackpad/click", &HButtons[13]);
    vr::VRDriverInput()->CreateBooleanComponent(m_ulPropertyContainer, "/input/trackpad/touch", &HButtons[14]);

    // Analog handles
    vr::VRDriverInput()->CreateScalarComponent(
        m_ulPropertyContainer, "/input/trackpad/x", &HAnalog[0],
        vr::EVRScalarType::VRScalarType_Absolute, vr::EVRScalarUnits::VRScalarUnits_NormalizedTwoSided
    );
    vr::VRDriverInput()->CreateScalarComponent(
        m_ulPropertyContainer, "/input/trackpad/y", &HAnalog[1],
        vr::EVRScalarType::VRScalarType_Absolute, vr::EVRScalarUnits::VRScalarUnits_NormalizedTwoSided
    );
    vr::VRDriverInput()->CreateScalarComponent(
        m_ulPropertyContainer, "/input/trigger/value", &HAnalog[2],
        vr::EVRScalarType::VRScalarType_Absolute, vr::EVRScalarUnits::VRScalarUnits_NormalizedOneSided
    );

    vr::VRProperties()->SetInt32Property(m_ulPropertyContainer, vr::Prop_Axis0Type_Int32, vr::k_eControllerAxis_TrackPad);

    // Create our haptic component
    vr::VRDriverInput()->CreateHapticComponent(m_ulPropertyContainer, "/output/haptic", &m_compHaptic);

    return VRInitError_None;
}

void CSampleControllerDriver::Deactivate() {
    m_unObjectId = vr::k_unTrackedDeviceIndexInvalid;
}

void CSampleControllerDriver::EnterStandby() {}

void *CSampleControllerDriver::GetComponent(const char *pchComponentNameAndVersion) {
    return NULL;
}

void CSampleControllerDriver::PowerOff() {}

void CSampleControllerDriver::DebugRequest(const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize) {
    if (unResponseBufferSize >= 1) {
        pchResponseBuffer[0] = 0;
    }
}

vr::DriverPose_t CSampleControllerDriver::GetPose() {
    vr::DriverPose_t pose = {0};
    pose.poseIsValid = true;
    pose.result = vr::TrackingResult_Running_OK;
    pose.deviceIsConnected = true;
    pose.qWorldFromDriverRotation = HmdQuaternion_Init(1, 0, 0, 0);
    pose.qDriverFromHeadRotation = HmdQuaternion_Init(1, 0, 0, 0);

    // Get pose data from TCP receiver
    if (g_pPoseDataReceiver && g_pPoseDataReceiver->IsConnected())
    {
        PoseData tcpPose;
        if (ControllerIndex == 1)
        {
            tcpPose = g_pPoseDataReceiver->GetController1Pose();
            if (tcpPose.updated)
            {
                DriverLog("CSampleControllerDriver: Got controller1 pose update: pos=[%.2f,%.2f,%.2f]\n",
                    tcpPose.posX, tcpPose.posY, tcpPose.posZ);
                cpX = tcpPose.posX;
                cpY = tcpPose.posY;
                cpZ = tcpPose.posZ;
                cAngleX = tcpPose.rotX;
                cAngleY = tcpPose.rotY;
                cAngleZ = tcpPose.rotZ;
            }
            // Update input state if received
            if (tcpPose.input.inputUpdated)
            {
                ctrl1Input = tcpPose.input;
            }
        }
        else
        {
            tcpPose = g_pPoseDataReceiver->GetController2Pose();
            if (tcpPose.updated)
            {
                DriverLog("CSampleControllerDriver: Got controller2 pose update: pos=[%.2f,%.2f,%.2f]\n",
                    tcpPose.posX, tcpPose.posY, tcpPose.posZ);
                c2pX = tcpPose.posX;
                c2pY = tcpPose.posY;
                c2pZ = tcpPose.posZ;
                c2AngleX = tcpPose.rotX;
                c2AngleY = tcpPose.rotY;
                c2AngleZ = tcpPose.rotZ;
            }
            // Update input state if received
            if (tcpPose.input.inputUpdated)
            {
                ctrl2Input = tcpPose.input;
            }
        }
    }

    if (ControllerIndex == 1)
    {
        pose.vecPosition[0] = cpX;
        pose.vecPosition[1] = cpY;
        pose.vecPosition[2] = cpZ;

        Quaternion currentRotation = anglesToQuaternion1(cAngleX, cAngleY, cAngleZ);
        currentRotation = normalizeQuaternion1(currentRotation);

        pose.qRotation.w = currentRotation.w;
        pose.qRotation.x = currentRotation.x;
        pose.qRotation.y = currentRotation.y;
        pose.qRotation.z = currentRotation.z;
    }
    else
    {
        pose.vecPosition[0] = c2pX;
        pose.vecPosition[1] = c2pY;
        pose.vecPosition[2] = c2pZ;

        Quaternion currentRotation = anglesToQuaternion1(c2AngleX, c2AngleY, c2AngleZ);
        currentRotation = normalizeQuaternion1(currentRotation);

        pose.qRotation.w = currentRotation.w;
        pose.qRotation.x = currentRotation.x;
        pose.qRotation.y = currentRotation.y;
        pose.qRotation.z = currentRotation.z;
    }

    return pose;
}

void CSampleControllerDriver::RunFrame() {
    ControllerInput* input = (ControllerIndex == 1) ? &ctrl1Input : &ctrl2Input;
    
    // Check for Shift+F10 toggle (only check once per frame, on controller 1)
    if (ControllerIndex == 1) {
        bool shiftPressed = (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
        bool f10Pressed = (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
        bool toggleKeyPressed = shiftPressed && f10Pressed;
        
        if (toggleKeyPressed && !g_toggleKeyWasPressed) {
            g_keyboardInputEnabled = !g_keyboardInputEnabled;
            DriverLog("Keyboard input %s (Shift+F10)\n", g_keyboardInputEnabled ? "ENABLED" : "DISABLED");
        }
        g_toggleKeyWasPressed = toggleKeyPressed;
    }
    
    if (ControllerIndex == 1) {
        // Use TCP input if available, otherwise fall back to keyboard (only if enabled)
        bool useKeyboard = !input->inputUpdated && g_keyboardInputEnabled;
        
        // Buttons - TCP input OR keyboard fallback
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[0], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('Q')) != 0) : input->system, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[1], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('W')) != 0) : input->menu, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[2], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('E')) != 0) : input->grip, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[7], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('I')) != 0) : input->buttonA, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[8], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('O')) != 0) : input->buttonB, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[11], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('S')) != 0) : input->triggerClick, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[13], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('F')) != 0) : input->trackpadClick, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[14], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('G')) != 0) : input->trackpadTouch, 0);

        // D-pad (keyboard only for now, respects toggle)
        bool dpadEnabled = g_keyboardInputEnabled;
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[3], dpadEnabled && (0x8000 & GetAsyncKeyState('R')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[4], dpadEnabled && (0x8000 & GetAsyncKeyState('T')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[5], dpadEnabled && (0x8000 & GetAsyncKeyState('Y')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[6], dpadEnabled && (0x8000 & GetAsyncKeyState('U')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[9], dpadEnabled && (0x8000 & GetAsyncKeyState('P')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[10], dpadEnabled && (0x8000 & GetAsyncKeyState('A')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[12], dpadEnabled && (0x8000 & GetAsyncKeyState('D')) != 0, 0);

        // Analog inputs - TCP or keyboard
        if (useKeyboard) {
            float trackpadX = 0.0f, trackpadY = 0.0f;
            if ((GetAsyncKeyState('2') & 0x8000) != 0) trackpadX = 1.0f;
            if ((GetAsyncKeyState('3') & 0x8000) != 0) trackpadY = 1.0f;
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[0], trackpadX, 0);
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[1], trackpadY, 0);
            
            float trigger = ((GetAsyncKeyState('X') & 0x8000) != 0) ? 1.0f : 0.0f;
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[2], trigger, 0);
        } else {
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[0], input->joystickX, 0);
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[1], input->joystickY, 0);
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[2], input->triggerValue, 0);
        }
    } else {
        // Controller2
        bool useKeyboard = !input->inputUpdated && g_keyboardInputEnabled;
        
        // Buttons - TCP input OR keyboard fallback
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[0], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('H')) != 0) : input->system, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[1], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('J')) != 0) : input->menu, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[2], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('K')) != 0) : input->grip, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[7], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('V')) != 0) : input->buttonA, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[8], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('B')) != 0) : input->buttonB, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[11], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('1')) != 0) : input->triggerClick, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[13], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('3')) != 0) : input->trackpadClick, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[14], 
            useKeyboard ? ((0x8000 & GetAsyncKeyState('4')) != 0) : input->trackpadTouch, 0);

        // D-pad (keyboard only, respects toggle)
        bool dpadEnabled = g_keyboardInputEnabled;
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[3], dpadEnabled && (0x8000 & GetAsyncKeyState('L')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[4], dpadEnabled && (0x8000 & GetAsyncKeyState('Z')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[5], dpadEnabled && (0x8000 & GetAsyncKeyState('X')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[6], dpadEnabled && (0x8000 & GetAsyncKeyState('C')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[9], dpadEnabled && (0x8000 & GetAsyncKeyState('N')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[10], dpadEnabled && (0x8000 & GetAsyncKeyState('M')) != 0, 0);
        vr::VRDriverInput()->UpdateBooleanComponent(HButtons[12], dpadEnabled && (0x8000 & GetAsyncKeyState('2')) != 0, 0);

        // Analog inputs
        if (useKeyboard) {
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[0], 0.0f, 0);
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[1], 0.0f, 0);
            float trigger = ((GetAsyncKeyState('4') & 0x8000) != 0) ? 1.0f : 0.0f;
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[2], trigger, 0);
        } else {
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[0], input->joystickX, 0);
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[1], input->joystickY, 0);
            vr::VRDriverInput()->UpdateScalarComponent(HAnalog[2], input->triggerValue, 0);
        }
    }

    if (m_unObjectId != vr::k_unTrackedDeviceIndexInvalid) {
        vr::VRServerDriverHost()->TrackedDevicePoseUpdated(m_unObjectId, GetPose(), sizeof(DriverPose_t));
    }
}

void CSampleControllerDriver::ProcessEvent(const vr::VREvent_t &vrEvent) {
    switch (vrEvent.eventType) {
    case vr::VREvent_Input_HapticVibration:
        if (vrEvent.data.hapticVibration.componentHandle == m_compHaptic) {
            // This is where you would send a signal to your hardware to trigger actual haptic feedback
            //DriverLog( "BUZZ!\n" );
        }
        break;
    }
}

std::string CSampleControllerDriver::GetSerialNumber() const {
    switch (ControllerIndex) {
    case 1:
        return "CTRL1Serial";
        break;
    case 2:
        return "CTRL2Serial";
        break;
    }
}
