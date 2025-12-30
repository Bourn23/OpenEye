#!/usr/bin/env python3
"""
Test script for the Vision-enabled MCP Server.
Simulates what the C++ driver would send back for vision requests.
"""

import socket
import json
import threading
import time
import base64

HOST = '127.0.0.1'
PORT = 5555

def create_test_jpeg():
    """Create a minimal valid JPEG for testing (1x1 red pixel)."""
    # Minimal JPEG: 1x1 red pixel
    jpeg_bytes = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB,
        0xD5, 0xDB, 0x20, 0xFF, 0xD9
    ])
    return base64.b64encode(jpeg_bytes).decode('utf-8')

def simulate_driver():
    """Simulate the C++ driver connecting and responding to vision requests."""
    print("Simulated driver connecting to MCP server...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")
        
        buffer = ""
        while True:
            try:
                data = sock.recv(4096)
                if not data:
                    break
                    
                buffer += data.decode('utf-8')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        msg = json.loads(line)
                        print(f"Received: {msg.get('type', 'unknown')}")
                        
                        # Handle vision requests
                        if msg.get('type') == 'vision_request':
                            action = msg.get('action', '')
                            
                            if action == 'capture_frame':
                                response = {
                                    "type": "frame",
                                    "width": 1600,
                                    "height": 800,
                                    "frameCount": 1,
                                    "frames": [create_test_jpeg()]
                                }
                                sock.sendall((json.dumps(response) + '\n').encode())
                                print("Sent frame response")
                                
                            elif action == 'capture_video':
                                duration = msg.get('duration', 3.0)
                                fps = msg.get('fps', 10)
                                frame_count = int(duration * fps)
                                
                                response = {
                                    "type": "video",
                                    "width": 1600,
                                    "height": 800,
                                    "frameCount": frame_count,
                                    "frames": [create_test_jpeg() for _ in range(frame_count)]
                                }
                                sock.sendall((json.dumps(response) + '\n').encode())
                                print(f"Sent video response ({frame_count} frames)")
                                
            except socket.timeout:
                continue
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()

if __name__ == '__main__':
    print("Vision Test - Simulated Driver")
    print("==============================")
    print("This simulates the C++ driver responding to vision requests.")
    print("Run the MCP server first, then run this script.")
    print("")
    simulate_driver()
