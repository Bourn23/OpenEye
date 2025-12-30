#!/usr/bin/env python3
"""
Interactive Test Script for Vision-enabled MCP Server.

This provides a simple REPL to test individual MCP tools manually.
Useful for debugging specific functionality.

Usage:
    python test_interactive.py
"""

import socket
import json
import threading
import time
import base64
import sys
from io import BytesIO

HOST = '127.0.0.1'
PORT = 5555

# Global state
driver_sock = None
driver_running = False
mcp = None

def create_test_jpeg():
    """Create a test JPEG image."""
    try:
        from PIL import Image
        img = Image.new('RGB', (320, 240), (100, 150, 200))
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=80)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except ImportError:
        # Minimal JPEG fallback
        jpeg = bytes([0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
                      0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
                      0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
                      0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
                      0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
                      0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
                      0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
                      0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
                      0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
                      0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
                      0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
                      0x09,0x0A,0x0B,0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,0x00,0x3F,
                      0x00,0x7F,0xFF,0xD9])
        return base64.b64encode(jpeg).decode('utf-8')

def driver_thread():
    """Simulated driver that responds to vision requests."""
    global driver_sock, driver_running
    
    buffer = ""
    while driver_running:
        try:
            driver_sock.settimeout(0.5)
            data = driver_sock.recv(8192)
            if not data:
                break
            
            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if not line.strip():
                    continue
                    
                msg = json.loads(line)
                if msg.get('type') == 'vision_request':
                    action = msg.get('action', '')
                    print(f"\n[Driver] Vision request: {action}")
                    
                    if action == 'capture_frame':
                        resp = {"type":"frame","width":1600,"height":800,
                                "frameCount":1,"message":"","frames":[create_test_jpeg()]}
                        driver_sock.sendall((json.dumps(resp)+'\n').encode())
                        print("[Driver] Sent frame")
                    elif action == 'capture_video':
                        n = int(msg.get('duration',3) * msg.get('fps',10))
                        resp = {"type":"video","width":1600,"height":800,
                                "frameCount":n,"message":"",
                                "frames":[create_test_jpeg() for _ in range(n)]}
                        driver_sock.sendall((json.dumps(resp)+'\n').encode())
                        print(f"[Driver] Sent {n} frames")
                elif msg.get('type') == 'update':
                    pass  # Pose update, ignore
                    
        except socket.timeout:
            continue
        except Exception as e:
            if driver_running:
                print(f"[Driver] Error: {e}")
            break

def start_system():
    """Start the MCP server and simulated driver."""
    global driver_sock, driver_running, mcp
    
    print("Starting MCP server...")
    import mcp_server
    mcp = mcp_server
    
    result = mcp.start_vr_bridge()
    print(f"  {result}")
    time.sleep(0.3)
    
    print("Connecting simulated driver...")
    driver_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    driver_sock.connect((HOST, PORT))
    driver_running = True
    
    t = threading.Thread(target=driver_thread, daemon=True)
    t.start()
    time.sleep(0.3)
    
    print(f"  {mcp.get_connection_status()}")
    print("\nSystem ready!")

def print_help():
    print("""
Available commands:
  Movement:
    pose [device]           - Get current pose (default: headset)
    tp <device> <x> <y> <z> - Teleport device to position
    move <dx> <dy> <dz>     - Move headset relative
    look <x> <y> <z>        - Look at position
    walk <x> <z> [steps]    - Walk to position
    rot <pitch> <yaw> <roll>- Rotate headset
    
  Vision (saves to disk!):
    frame                   - Capture frame -> capture_<timestamp>.jpg
    video [duration] [fps]  - Capture video -> video_<timestamp>/frame_XXXX.jpg
    scan                    - 360° panorama -> panorama_<timestamp>/angle_XXX.jpg
    
  Other:
    status                  - Connection status
    help                    - Show this help
    quit                    - Exit
""")

def run_command(cmd):
    """Execute a command."""
    parts = cmd.strip().split()
    if not parts:
        return
    
    action = parts[0].lower()
    
    try:
        if action == 'help':
            print_help()
            
        elif action == 'status':
            print(mcp.get_connection_status())
            
        elif action == 'pose':
            device = parts[1] if len(parts) > 1 else 'headset'
            print(mcp.get_current_pose(device))
            
        elif action == 'tp':
            if len(parts) < 5:
                print("Usage: tp <device> <x> <y> <z>")
                return
            device, x, y, z = parts[1], float(parts[2]), float(parts[3]), float(parts[4])
            print(mcp.teleport(device, x, y, z))
            
        elif action == 'move':
            if len(parts) < 4:
                print("Usage: move <dx> <dy> <dz>")
                return
            dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
            print(mcp.move_relative('headset', dx, dy, dz))
            
        elif action == 'look':
            if len(parts) < 4:
                print("Usage: look <x> <y> <z>")
                return
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            print(mcp.look_at(x, y, z))
            
        elif action == 'walk':
            if len(parts) < 3:
                print("Usage: walk <x> <z> [steps]")
                return
            x, z = float(parts[1]), float(parts[2])
            steps = int(parts[3]) if len(parts) > 3 else 10
            print(mcp.walk_path(x, z, steps))
            
        elif action == 'rot':
            if len(parts) < 4:
                print("Usage: rot <pitch> <yaw> <roll>")
                return
            p, y, r = float(parts[1]), float(parts[2]), float(parts[3])
            print(mcp.rotate_device('headset', p, y, r))
            
        elif action == 'frame':
            print("Capturing frame...")
            result = mcp.inspect_surroundings()
            data = json.loads(result)
            if data.get('type') == 'image':
                # Save to file
                img_bytes = base64.b64decode(data['data'])
                filename = f"capture_{int(time.time())}.jpg"
                with open(filename, 'wb') as f:
                    f.write(img_bytes)
                print(f"Saved {data['width']}x{data['height']} image to: {filename}")
            else:
                print(result[:200])
                
        elif action == 'video':
            dur = float(parts[1]) if len(parts) > 1 else 3.0
            fps = int(parts[2]) if len(parts) > 2 else 10
            print(f"Capturing {dur}s video at {fps}fps...")
            result = mcp.capture_video(dur, fps)
            data = json.loads(result)
            if data.get('type') == 'video':
                # Save frames to folder
                import os
                folder = f"video_{int(time.time())}"
                os.makedirs(folder, exist_ok=True)
                
                frames = data.get('frames', [])
                for i, frame_b64 in enumerate(frames):
                    img_bytes = base64.b64decode(frame_b64)
                    with open(f"{folder}/frame_{i:04d}.jpg", 'wb') as f:
                        f.write(img_bytes)
                
                print(f"Saved {len(frames)} frames to: {folder}/")
                print(f"  Resolution: {data['width']}x{data['height']}")
                
                # Try to create MP4 if ffmpeg available
                try:
                    import subprocess
                    mp4_file = f"{folder}.mp4"
                    subprocess.run([
                        'ffmpeg', '-y', '-framerate', str(fps),
                        '-i', f'{folder}/frame_%04d.jpg',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        mp4_file
                    ], capture_output=True, check=True)
                    print(f"  Created video: {mp4_file}")
                except:
                    print(f"  (Install ffmpeg to auto-create MP4)")
            else:
                print(result[:200])
                
        elif action == 'scan':
            print("Scanning 360°...")
            result = mcp.look_around_and_observe()
            data = json.loads(result)
            if data.get('type') == 'panorama_scan':
                # Save panorama frames
                import os
                folder = f"panorama_{int(time.time())}"
                os.makedirs(folder, exist_ok=True)
                
                dirs = data.get('directions', [])
                for d in dirs:
                    angle = d['angle']
                    img_bytes = base64.b64decode(d['data'])
                    with open(f"{folder}/angle_{angle:03d}.jpg", 'wb') as f:
                        f.write(img_bytes)
                
                print(f"Saved {len(dirs)} panorama frames to: {folder}/")
                print(f"  Angles: {[d['angle'] for d in dirs]}")
            else:
                print(result[:200])
                
        elif action in ('quit', 'exit', 'q'):
            return 'quit'
            
        else:
            print(f"Unknown command: {action}. Type 'help' for commands.")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    global driver_running
    
    print("="*50)
    print(" Interactive MCP Server Test")
    print("="*50)
    
    try:
        start_system()
    except Exception as e:
        print(f"Failed to start: {e}")
        return
    
    print_help()
    
    while True:
        try:
            cmd = input("\n> ").strip()
            if run_command(cmd) == 'quit':
                break
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except EOFError:
            break
    
    driver_running = False
    print("Goodbye!")

if __name__ == '__main__':
    main()
