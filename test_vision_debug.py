#!/usr/bin/env python3
"""
Debug Test Script for Vision-enabled MCP Server.

This script tests all vision and movement functionality without needing
the actual C++ driver or SteamVR running. It simulates the driver side
and tests the MCP tools directly.

Usage:
    python test_vision_debug.py

Requirements:
    pip install mcp
"""

import socket
import json
import threading
import time
import base64
import sys
from io import BytesIO

# Test configuration
HOST = '127.0.0.1'
PORT = 5555

# ============================================================================
# SIMULATED DRIVER (runs in background thread)
# ============================================================================

class SimulatedDriver:
    """Simulates the C++ OpenVR driver for testing."""
    
    def __init__(self):
        self.sock = None
        self.running = False
        self.thread = None
        self.connected = False
        self.frames_sent = 0
        
    def create_test_image(self, width=320, height=240, color=(255, 0, 0)):
        """Create a simple colored test image as JPEG bytes."""
        try:
            from PIL import Image
            img = Image.new('RGB', (width, height), color)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except ImportError:
            # Fallback: minimal valid JPEG (1x1 pixel)
            jpeg_bytes = bytes([
                0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
                0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB,
                0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07,
                0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B,
                0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E,
                0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C,
                0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34,
                0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
                0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01,
                0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                0x09, 0x0A, 0x0B, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00,
                0x3F, 0x00, 0x7F, 0xFF, 0xD9
            ])
            return base64.b64encode(jpeg_bytes).decode('utf-8')
    
    def connect(self):
        """Connect to the MCP server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((HOST, PORT))
            self.sock.settimeout(1.0)
            self.connected = True
            print(f"  [Driver] Connected to MCP server at {HOST}:{PORT}")
            return True
        except Exception as e:
            print(f"  [Driver] Connection failed: {e}")
            return False
    
    def run(self):
        """Main driver loop - receives commands and sends responses."""
        buffer = ""
        
        while self.running and self.connected:
            try:
                data = self.sock.recv(8192)
                if not data:
                    break
                
                buffer += data.decode('utf-8')
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self.handle_message(line)
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"  [Driver] Error: {e}")
                break
        
        self.connected = False

    def handle_message(self, message):
        """Handle incoming message from MCP server."""
        try:
            msg = json.loads(message)
            msg_type = msg.get('type', '')
            
            if msg_type == 'update':
                # Pose update - just acknowledge
                data = msg.get('data', {})
                headset = data.get('headset', {})
                pos = headset.get('pos', [0, 0, 0])
                rot = headset.get('rot', [0, 0, 0])
                print(f"  [Driver] Pose update: pos={pos}, rot={rot}")
                
            elif msg_type == 'vision_request':
                action = msg.get('action', '')
                print(f"  [Driver] Vision request: {action}")
                
                if action == 'capture_frame':
                    self.send_frame_response()
                elif action == 'capture_video':
                    duration = msg.get('duration', 3.0)
                    fps = msg.get('fps', 10)
                    self.send_video_response(duration, fps)
                elif action == 'get_status':
                    self.send_status_response()
                    
        except json.JSONDecodeError as e:
            print(f"  [Driver] JSON parse error: {e}")
    
    def send_frame_response(self):
        """Send a single frame response."""
        response = {
            "type": "frame",
            "width": 1600,
            "height": 800,
            "frameCount": 1,
            "message": "",
            "frames": [self.create_test_image(1600, 800, (100, 150, 200))]
        }
        self.sock.sendall((json.dumps(response) + '\n').encode())
        self.frames_sent += 1
        print(f"  [Driver] Sent frame response (total: {self.frames_sent})")
    
    def send_video_response(self, duration, fps):
        """Send a video (multiple frames) response."""
        frame_count = int(duration * fps)
        frames = []
        
        # Create frames with varying colors to simulate video
        for i in range(frame_count):
            r = int(100 + (i / frame_count) * 100)
            g = int(50 + (i / frame_count) * 150)
            b = 200
            frames.append(self.create_test_image(1600, 800, (r, g, b)))
        
        response = {
            "type": "video",
            "width": 1600,
            "height": 800,
            "frameCount": frame_count,
            "message": "",
            "frames": frames
        }
        self.sock.sendall((json.dumps(response) + '\n').encode())
        self.frames_sent += frame_count
        print(f"  [Driver] Sent video response ({frame_count} frames)")
    
    def send_status_response(self):
        """Send status response."""
        response = {
            "type": "status",
            "width": 0,
            "height": 0,
            "frameCount": 0,
            "message": "ready"
        }
        self.sock.sendall((json.dumps(response) + '\n').encode())
        print("  [Driver] Sent status response")
    
    def start(self):
        """Start the simulated driver in a background thread."""
        self.running = True
        if self.connect():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            return True
        return False
    
    def stop(self):
        """Stop the simulated driver."""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2.0)

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_separator(name):
    print(f"\n{'='*60}")
    print(f" TEST: {name}")
    print('='*60)

def test_connection():
    """Test basic server startup and driver connection."""
    test_separator("Connection & Bridge Startup")
    
    # Import the MCP server module
    import mcp_server
    
    # Start the VR bridge
    result = mcp_server.start_vr_bridge()
    print(f"  start_vr_bridge(): {result}")
    
    time.sleep(0.5)  # Let server start
    
    # Check status before driver connects
    result = mcp_server.get_connection_status()
    print(f"  get_connection_status() [before]: {result}")
    
    # Start simulated driver
    driver = SimulatedDriver()
    if driver.start():
        time.sleep(0.5)  # Let driver connect
        
        # Check status after driver connects
        result = mcp_server.get_connection_status()
        print(f"  get_connection_status() [after]: {result}")
        
        return driver
    else:
        print("  ERROR: Failed to start simulated driver")
        return None

def test_movement(mcp_server):
    """Test movement and pose tools."""
    test_separator("Movement & Pose Control")
    
    # Test get_current_pose
    result = mcp_server.get_current_pose("headset")
    print(f"  get_current_pose('headset'): {result}")
    
    # Test teleport
    result = mcp_server.teleport("headset", 1.0, 2.0, 3.0)
    print(f"  teleport('headset', 1, 2, 3): {result}")
    
    time.sleep(0.3)
    
    # Verify pose changed
    result = mcp_server.get_current_pose("headset")
    print(f"  get_current_pose() [after teleport]: {result}")
    
    # Test rotate_device
    result = mcp_server.rotate_device("headset", 10.0, 45.0, 0.0)
    print(f"  rotate_device('headset', 10, 45, 0): {result}")
    
    time.sleep(0.3)
    
    # Test move_relative
    result = mcp_server.move_relative("headset", dx=0.5, dy=0, dz=-0.5)
    print(f"  move_relative('headset', 0.5, 0, -0.5): {result}")
    
    time.sleep(0.3)
    
    # Test look_at
    result = mcp_server.look_at(5.0, 1.5, -5.0)
    print(f"  look_at(5, 1.5, -5): {result}")
    
    time.sleep(0.3)

def test_walk_path(mcp_server):
    """Test walk_path (animated movement)."""
    test_separator("Walk Path (Animated Movement)")
    
    # Reset position first
    mcp_server.teleport("headset", 0, 1.5, 0)
    time.sleep(0.3)
    
    print("  Walking from (0, 1.5, 0) to (2, 1.5, -2)...")
    result = mcp_server.walk_path(2.0, -2.0, steps=5)
    print(f"  walk_path(2, -2, steps=5): {result}")
    
    # Verify final position
    result = mcp_server.get_current_pose("headset")
    print(f"  Final pose: {result}")

def test_vision_frame(mcp_server):
    """Test single frame capture."""
    test_separator("Vision: Single Frame Capture")
    
    print("  Calling inspect_surroundings()...")
    result = mcp_server.inspect_surroundings()
    
    try:
        data = json.loads(result)
        if data.get('type') == 'image':
            print(f"  SUCCESS: Received image")
            print(f"    - Format: {data.get('format')}")
            print(f"    - Size: {data.get('width')}x{data.get('height')}")
            print(f"    - Data length: {len(data.get('data', ''))} chars (base64)")
            
            # Try to decode and verify it's valid
            try:
                img_bytes = base64.b64decode(data.get('data', ''))
                print(f"    - Decoded size: {len(img_bytes)} bytes")
                if img_bytes[:2] == b'\xff\xd8':
                    print(f"    - Valid JPEG header: YES")
                else:
                    print(f"    - Valid JPEG header: NO (got {img_bytes[:2].hex()})")
            except Exception as e:
                print(f"    - Decode error: {e}")
        else:
            print(f"  Result: {result[:200]}...")
    except json.JSONDecodeError:
        print(f"  Result (not JSON): {result[:200]}...")

def test_vision_video(mcp_server):
    """Test video capture."""
    test_separator("Vision: Video Capture (3 seconds)")
    
    print("  Calling capture_video(duration=1.0, fps=5)...")
    start = time.time()
    result = mcp_server.capture_video(duration=1.0, fps=5)
    elapsed = time.time() - start
    print(f"  Capture took {elapsed:.2f} seconds")
    
    try:
        data = json.loads(result)
        if data.get('type') == 'video':
            print(f"  SUCCESS: Received video")
            print(f"    - Format: {data.get('format')}")
            print(f"    - Size: {data.get('width')}x{data.get('height')}")
            print(f"    - Frame count: {data.get('frameCount')}")
            print(f"    - FPS: {data.get('fps')}")
            print(f"    - Duration: {data.get('duration')}s")
            
            frames = data.get('frames', [])
            print(f"    - Actual frames received: {len(frames)}")
            
            if frames:
                # Check first frame
                try:
                    img_bytes = base64.b64decode(frames[0])
                    print(f"    - First frame size: {len(img_bytes)} bytes")
                except Exception as e:
                    print(f"    - First frame decode error: {e}")
        else:
            print(f"  Result: {result[:200]}...")
    except json.JSONDecodeError:
        print(f"  Result (not JSON): {result[:200]}...")

def test_vision_panorama(mcp_server):
    """Test 360-degree panorama scan."""
    test_separator("Vision: 360° Panorama Scan")
    
    print("  Calling look_around_and_observe()...")
    print("  (This rotates the headset and captures 4 frames)")
    start = time.time()
    result = mcp_server.look_around_and_observe()
    elapsed = time.time() - start
    print(f"  Scan took {elapsed:.2f} seconds")
    
    try:
        data = json.loads(result)
        if data.get('type') == 'panorama_scan':
            print(f"  SUCCESS: Received panorama")
            print(f"    - Format: {data.get('format')}")
            print(f"    - Description: {data.get('description')}")
            
            directions = data.get('directions', [])
            print(f"    - Directions captured: {len(directions)}")
            
            for d in directions:
                angle = d.get('angle', '?')
                data_len = len(d.get('data', ''))
                print(f"      - {angle}°: {data_len} chars")
        else:
            print(f"  Result: {result[:200]}...")
    except json.JSONDecodeError:
        print(f"  Result (not JSON): {result[:200]}...")

def test_navigation_helper(mcp_server):
    """Test the navigation helper tool."""
    test_separator("Navigation Helper")
    
    result = mcp_server.navigate_to_object("red cube")
    print(f"  navigate_to_object('red cube'):")
    print("  " + result.replace('\n', '\n  '))

def test_controller_movement(mcp_server):
    """Test controller positioning."""
    test_separator("Controller Movement")
    
    # Test controller1 (left)
    result = mcp_server.teleport("controller1", -0.5, 1.2, -0.4)
    print(f"  teleport('controller1', -0.5, 1.2, -0.4): {result}")
    
    time.sleep(0.2)
    
    # Test controller2 (right)
    result = mcp_server.teleport("controller2", 0.5, 1.2, -0.4)
    print(f"  teleport('controller2', 0.5, 1.2, -0.4): {result}")
    
    time.sleep(0.2)
    
    # Get poses
    result = mcp_server.get_current_pose("controller1")
    print(f"  get_current_pose('controller1'): {result}")
    
    result = mcp_server.get_current_pose("controller2")
    print(f"  get_current_pose('controller2'): {result}")

def test_error_handling(mcp_server):
    """Test error handling."""
    test_separator("Error Handling")
    
    # Invalid device
    result = mcp_server.teleport("invalid_device", 0, 0, 0)
    print(f"  teleport('invalid_device', ...): {result}")
    
    result = mcp_server.get_current_pose("nonexistent")
    print(f"  get_current_pose('nonexistent'): {result}")

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print(" VISION-ENABLED MCP SERVER DEBUG TEST")
    print(" Testing all movement and vision functionality")
    print("="*60)
    
    driver = None
    
    try:
        # Import the MCP server
        import mcp_server
        
        # Test 1: Connection
        driver = test_connection()
        if not driver:
            print("\nERROR: Could not establish connection. Aborting tests.")
            return False
        
        time.sleep(0.5)
        
        # Test 2: Movement
        test_movement(mcp_server)
        
        # Test 3: Walk path
        test_walk_path(mcp_server)
        
        # Test 4: Controller movement
        test_controller_movement(mcp_server)
        
        # Test 5: Vision - single frame
        test_vision_frame(mcp_server)
        
        # Test 6: Vision - video
        test_vision_video(mcp_server)
        
        # Test 7: Vision - panorama
        test_vision_panorama(mcp_server)
        
        # Test 8: Navigation helper
        test_navigation_helper(mcp_server)
        
        # Test 9: Error handling
        test_error_handling(mcp_server)
        
        # Summary
        print("\n" + "="*60)
        print(" TEST SUMMARY")
        print("="*60)
        print(f"  Driver frames sent: {driver.frames_sent}")
        print("  All tests completed!")
        print("="*60 + "\n")
        
        return True
        
    except ImportError as e:
        print(f"\nERROR: Could not import mcp_server: {e}")
        print("Make sure mcp_server.py is in the same directory.")
        return False
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if driver:
            driver.stop()
            print("\n  [Driver] Stopped")

if __name__ == '__main__':
    print("\nVision Debug Test Script")
    print("========================")
    print("This script tests the MCP server with a simulated driver.")
    print("No SteamVR or C++ driver needed!\n")
    
    # Check for PIL (optional, for better test images)
    try:
        from PIL import Image
        print("PIL/Pillow: Available (will generate colored test images)")
    except ImportError:
        print("PIL/Pillow: Not available (will use minimal JPEG fallback)")
        print("  Install with: pip install pillow")
    
    print("")
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
