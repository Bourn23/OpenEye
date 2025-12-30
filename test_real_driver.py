#!/usr/bin/env python3
"""
Test Script for REAL SteamVR Driver Vision Feed.

This script tests the vision capture with the actual C++ driver and SteamVR.
It saves captured frames/videos to disk so you can verify the feed is correct.

SETUP ORDER:
1. Build the C++ driver (msbuild driver_sample.vcxproj /p:Configuration=Release /p:Platform=x64)
2. Install driver in SteamVR
3. Run this script: python test_real_driver.py
4. Start SteamVR (driver will auto-connect)
5. Use the interactive commands to capture frames

Usage:
    python test_real_driver.py
"""

import sys
import os
import time
import json
import base64

# Add current directory to path for mcp_server import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def wait_for_driver(mcp_server, timeout=60):
    """Wait for the real driver to connect."""
    print(f"\nWaiting for SteamVR driver to connect (timeout: {timeout}s)...")
    print("  -> Start SteamVR now!")
    print("")
    
    start = time.time()
    while time.time() - start < timeout:
        status = mcp_server.get_connection_status()
        if "Connected" in status:
            print(f"\n  {status}")
            return True
        
        # Show waiting indicator
        elapsed = int(time.time() - start)
        print(f"\r  Waiting... {elapsed}s", end="", flush=True)
        time.sleep(1)
    
    print(f"\n  Timeout! No driver connected after {timeout}s")
    return False

def save_frame(data, prefix="frame"):
    """Save a single frame to disk."""
    timestamp = int(time.time())
    filename = f"{prefix}_{timestamp}.jpg"
    
    img_bytes = base64.b64decode(data)
    with open(filename, 'wb') as f:
        f.write(img_bytes)
    
    return filename, len(img_bytes)

def save_video_frames(frames, fps):
    """Save video frames to a folder and optionally create MP4."""
    timestamp = int(time.time())
    folder = f"video_{timestamp}"
    os.makedirs(folder, exist_ok=True)
    
    for i, frame_b64 in enumerate(frames):
        img_bytes = base64.b64decode(frame_b64)
        with open(f"{folder}/frame_{i:04d}.jpg", 'wb') as f:
            f.write(img_bytes)
    
    # Try to create MP4
    mp4_file = None
    try:
        import subprocess
        mp4_file = f"{folder}.mp4"
        result = subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', f'{folder}/frame_%04d.jpg',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            mp4_file
        ], capture_output=True)
        if result.returncode != 0:
            mp4_file = None
    except:
        mp4_file = None
    
    return folder, mp4_file

def print_help():
    print("""
Commands (saves files to current directory):
  
  Vision Capture:
    f, frame              - Capture single frame -> frame_<timestamp>.jpg
    v, video [dur] [fps]  - Capture video (default: 3s, 10fps) -> video_<timestamp>/
    s, scan               - 360° panorama -> panorama_<timestamp>/
    
  Movement (to test if capture updates):
    look <x> <y> <z>      - Look at position
    tp <x> <y> <z>        - Teleport headset
    walk <x> <z>          - Walk to position
    rot <pitch> <yaw>     - Rotate headset
    
  Info:
    pose                  - Show current headset pose
    status                - Check driver connection
    
  Other:
    h, help               - Show this help
    q, quit               - Exit

Tips:
  - Capture a frame, move the headset, capture again to verify feed updates
  - Check the saved JPG files to see what the driver is capturing
  - If frames are black/wrong, check windowX/Y/Width/Height in default.vrsettings
""")

def cmd_frame(mcp):
    """Capture single frame."""
    print("Capturing frame from SteamVR...")
    result = mcp.inspect_surroundings()
    
    try:
        data = json.loads(result)
        if data.get('type') == 'image':
            filename, size = save_frame(data['data'])
            print(f"  SAVED: {filename}")
            print(f"  Size: {data['width']}x{data['height']}, {size} bytes")
            print(f"  -> Open this file to verify the VR feed!")
        elif data.get('type') == 'error':
            print(f"  ERROR: {data.get('message', 'Unknown error')}")
        else:
            print(f"  Unexpected response: {result[:200]}")
    except json.JSONDecodeError:
        print(f"  Failed to parse response: {result[:200]}")

def cmd_video(mcp, duration=3.0, fps=10):
    """Capture video."""
    print(f"Capturing {duration}s video at {fps}fps from SteamVR...")
    print("  (This may take a moment...)")
    
    start = time.time()
    result = mcp.capture_video(duration, fps)
    elapsed = time.time() - start
    
    try:
        data = json.loads(result)
        if data.get('type') == 'video':
            frames = data.get('frames', [])
            folder, mp4 = save_video_frames(frames, fps)
            
            print(f"  SAVED: {folder}/ ({len(frames)} frames)")
            print(f"  Size: {data['width']}x{data['height']}")
            print(f"  Capture took: {elapsed:.1f}s")
            if mp4:
                print(f"  MP4: {mp4}")
            else:
                print(f"  (Install ffmpeg to auto-create MP4)")
        elif data.get('type') == 'error':
            print(f"  ERROR: {data.get('message', 'Unknown error')}")
        else:
            print(f"  Unexpected response: {result[:200]}")
    except json.JSONDecodeError:
        print(f"  Failed to parse response: {result[:200]}")

def cmd_scan(mcp):
    """360° panorama scan."""
    print("Capturing 360° panorama from SteamVR...")
    print("  (Rotating headset and capturing 4 frames...)")
    
    result = mcp.look_around_and_observe()
    
    try:
        data = json.loads(result)
        if data.get('type') == 'panorama_scan':
            timestamp = int(time.time())
            folder = f"panorama_{timestamp}"
            os.makedirs(folder, exist_ok=True)
            
            dirs = data.get('directions', [])
            for d in dirs:
                angle = d['angle']
                img_bytes = base64.b64decode(d['data'])
                with open(f"{folder}/angle_{angle:03d}.jpg", 'wb') as f:
                    f.write(img_bytes)
            
            print(f"  SAVED: {folder}/ ({len(dirs)} angles)")
            print(f"  Angles: {[d['angle'] for d in dirs]}")
        elif data.get('type') == 'error':
            print(f"  ERROR: {data.get('message', 'Unknown error')}")
        else:
            print(f"  Unexpected response: {result[:200]}")
    except json.JSONDecodeError:
        print(f"  Failed to parse response: {result[:200]}")

def run_interactive(mcp):
    """Interactive command loop."""
    print_help()
    
    while True:
        try:
            cmd = input("\n[SteamVR] > ").strip()
            if not cmd:
                continue
            
            parts = cmd.split()
            action = parts[0].lower()
            
            # Vision commands
            if action in ('f', 'frame'):
                cmd_frame(mcp)
                
            elif action in ('v', 'video'):
                dur = float(parts[1]) if len(parts) > 1 else 3.0
                fps = int(parts[2]) if len(parts) > 2 else 10
                cmd_video(mcp, dur, fps)
                
            elif action in ('s', 'scan'):
                cmd_scan(mcp)
            
            # Movement commands
            elif action == 'look':
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    print(mcp.look_at(x, y, z))
                else:
                    print("Usage: look <x> <y> <z>")
                    
            elif action == 'tp':
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    print(mcp.teleport('headset', x, y, z))
                else:
                    print("Usage: tp <x> <y> <z>")
                    
            elif action == 'walk':
                if len(parts) >= 3:
                    x, z = float(parts[1]), float(parts[2])
                    print(mcp.walk_path(x, z))
                else:
                    print("Usage: walk <x> <z>")
                    
            elif action == 'rot':
                if len(parts) >= 3:
                    pitch, yaw = float(parts[1]), float(parts[2])
                    print(mcp.rotate_device('headset', pitch, yaw, 0))
                else:
                    print("Usage: rot <pitch> <yaw>")
            
            # Info commands
            elif action == 'pose':
                print(mcp.get_current_pose('headset'))
                
            elif action == 'status':
                print(mcp.get_connection_status())
            
            # Other
            elif action in ('h', 'help'):
                print_help()
                
            elif action in ('q', 'quit', 'exit'):
                print("Exiting...")
                break
                
            else:
                print(f"Unknown command: {action}. Type 'help' for commands.")
                
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("="*60)
    print(" SteamVR Vision Feed Test")
    print(" Tests the REAL driver capture (not simulated)")
    print("="*60)
    
    # Import and start MCP server
    try:
        import mcp_server
    except ImportError as e:
        print(f"\nERROR: Could not import mcp_server.py: {e}")
        print("Make sure mcp_server.py is in the same directory.")
        return 1
    
    print("\nStarting MCP server on port 5555...")
    result = mcp_server.start_vr_bridge()
    print(f"  {result}")
    
    # Wait for real driver
    if not wait_for_driver(mcp_server, timeout=120):
        print("\nNo driver connected. Make sure:")
        print("  1. Driver is built and installed in SteamVR")
        print("  2. default.vrsettings has tcpEnabled=true")
        print("  3. SteamVR is running")
        return 1
    
    print("\n" + "="*60)
    print(" Driver connected! Ready to capture.")
    print("="*60)
    
    # Run interactive mode
    run_interactive(mcp_server)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
