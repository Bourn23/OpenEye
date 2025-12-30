#ifndef CPOSEDATARECEIVER_H
#define CPOSEDATARECEIVER_H

#include "ctcpclient.h"
#include <mutex>
#include <string>
#include <functional>

struct ControllerInput
{
    // Buttons (matching VR controller standard)
    bool system;          // System button
    bool menu;            // Application menu
    bool grip;            // Grip button
    bool triggerClick;    // Trigger fully pressed
    bool trackpadClick;   // Trackpad/joystick click
    bool trackpadTouch;   // Trackpad/joystick touch
    bool buttonA;         // A button
    bool buttonB;         // B button

    // Analog inputs
    float triggerValue;   // Trigger analog (0.0 - 1.0)
    float joystickX;      // Joystick/trackpad X (-1.0 to 1.0)
    float joystickY;      // Joystick/trackpad Y (-1.0 to 1.0)

    bool inputUpdated;    // Flag to indicate input was updated via TCP

    ControllerInput() : system(false), menu(false), grip(false), triggerClick(false),
                        trackpadClick(false), trackpadTouch(false), buttonA(false), buttonB(false),
                        triggerValue(0), joystickX(0), joystickY(0), inputUpdated(false) {}
};

struct PoseData
{
    double posX, posY, posZ;
    double rotX, rotY, rotZ;
    bool updated;
    
    ControllerInput input;  // Controller input state

    PoseData() : posX(0), posY(0), posZ(0), rotX(0), rotY(0), rotZ(0), updated(false) {}
};

// Callback for sending responses back to the server
using SendResponseCallback = std::function<void(const std::string&)>;

class CPoseDataReceiver
{
public:
    CPoseDataReceiver();
    ~CPoseDataReceiver();

    bool Start(const std::string& host, int port);
    void Stop();
    bool IsConnected() const;

    // Get pose data for each device (thread-safe)
    PoseData GetHeadsetPose();
    PoseData GetController1Pose();
    PoseData GetController2Pose();

    // Set callback for sending responses (for vision requests)
    void SetSendCallback(SendResponseCallback callback) { m_sendCallback = callback; }
    
    // Get TCP client for sending responses
    CTcpClient& GetTcpClient() { return m_tcpClient; }

private:
    void OnMessageReceived(const std::string& message);
    bool ParseJson(const std::string& json, std::string& device, PoseData& pose);
    bool IsVisionRequest(const std::string& json);
    void HandleVisionRequest(const std::string& json);

    CTcpClient m_tcpClient;
    SendResponseCallback m_sendCallback;

    std::mutex m_mutex;
    PoseData m_headsetPose;
    PoseData m_controller1Pose;
    PoseData m_controller2Pose;
};

// Global instance
extern CPoseDataReceiver* g_pPoseDataReceiver;

#endif // CPOSEDATARECEIVER_H
