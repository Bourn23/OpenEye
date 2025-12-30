#!/usr/bin/env python3
"""
Test TCP server for OpenVR driver pose data.
Sends JSON pose updates for headset and controllers.

Usage:
    python test_server.py

Controls:
    WASD    - Move headset X/Z
    QE      - Move headset Y (up/down)
    Arrows  - Rotate headset pitch/yaw
    Space   - Reset all poses
    Esc     - Quit
"""

import socket
import json
import time
import threading
import sys

HOST = '127.0.0.1'
PORT = 5555

# Current pose state
poses = {
    'headset': {'pos': [0.0, 1.5, 0.0], 'rot': [0.0, 0.0, 0.0]},
    'controller1': {'pos': [-0.3, 1.0, -0.3], 'rot': [0.0, 0.0, 0.0]},
    'controller2': {'pos': [0.3, 1.0, -0.3], 'rot': [0.0, 0.0, 0.0]},
}

clients = []
running = True


def send_pose(client_socket, device):
    """Send a single pose update to a client."""
    try:
        msg = json.dumps({
            'device': device,
            'pos': poses[device]['pos'],
            'rot': poses[device]['rot']
        }) + '\n'
        client_socket.sendall(msg.encode('utf-8'))
        return True
    except:
        return False


def broadcast_poses():
    """Send all poses to all connected clients."""
    for client in clients[:]:
        for device in poses:
            if not send_pose(client, device):
                clients.remove(client)
                print(f"Client disconnected. Total clients: {len(clients)}")
                break


def handle_client(client_socket, addr):
    """Handle a new client connection."""
    print(f"Client connected from {addr}")
    clients.append(client_socket)
    
    # Send initial poses
    for device in poses:
        send_pose(client_socket, device)
    
    # Keep connection alive until client disconnects
    while running and client_socket in clients:
        try:
            client_socket.settimeout(1.0)
            data = client_socket.recv(1024)
            if not data:
                break
        except socket.timeout:
            continue
        except:
            break
    
    if client_socket in clients:
        clients.remove(client_socket)
    client_socket.close()
    print(f"Client {addr} disconnected. Total clients: {len(clients)}")


def server_thread():
    """Run the TCP server."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    server.settimeout(1.0)
    
    print(f"Server listening on {HOST}:{PORT}")
    
    while running:
        try:
            client_socket, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(client_socket, addr))
            thread.daemon = True
            thread.start()
        except socket.timeout:
            continue
        except:
            break
    
    server.close()


def interactive_mode():
    """Interactive mode with keyboard input (requires additional setup)."""
    global running
    
    print("\nInteractive mode:")
    print("  Type 'h' for headset controls")
    print("  Type 'c1' for controller1 controls")
    print("  Type 'c2' for controller2 controls")
    print("  Use +pos or +rot for incremental changes")
    print("  Type 'reset' to reset all poses")
    print("  Type 'quit' to exit\n")
    print("Examples:")
    print("  h rot 0 90 0      - Set headset yaw to 90 degrees")
    print("  h +rot 0 45 0     - Add 45 degrees to headset yaw")
    print("  c1 +pos 0.1 0 0   - Move controller1 right by 0.1m\n")
    
    while running:
        try:
            cmd = input("> ").strip().lower()
            
            if cmd == 'quit':
                running = False
                break
            elif cmd == 'reset':
                poses['headset'] = {'pos': [0.0, 1.5, 0.0], 'rot': [0.0, 0.0, 0.0]}
                poses['controller1'] = {'pos': [-0.3, 1.0, -0.3], 'rot': [0.0, 0.0, 0.0]}
                poses['controller2'] = {'pos': [0.3, 1.0, -0.3], 'rot': [0.0, 0.0, 0.0]}
                broadcast_poses()
                print("Poses reset")
            elif cmd.startswith('h ') or cmd.startswith('c1 ') or cmd.startswith('c2 '):
                parts = cmd.split()
                device = {'h': 'headset', 'c1': 'controller1', 'c2': 'controller2'}[parts[0]]
                
                if len(parts) >= 4 and parts[1] == 'pos':
                    poses[device]['pos'] = [float(parts[2]), float(parts[3]), float(parts[4])]
                    broadcast_poses()
                    print(f"{device} pos: {poses[device]['pos']}")
                elif len(parts) >= 4 and parts[1] == '+pos':
                    # Incremental position
                    poses[device]['pos'][0] += float(parts[2])
                    poses[device]['pos'][1] += float(parts[3])
                    poses[device]['pos'][2] += float(parts[4])
                    broadcast_poses()
                    print(f"{device} pos: {poses[device]['pos']}")
                elif len(parts) >= 4 and parts[1] == 'rot':
                    poses[device]['rot'] = [float(parts[2]), float(parts[3]), float(parts[4])]
                    broadcast_poses()
                    print(f"{device} rot: {poses[device]['rot']}")
                elif len(parts) >= 4 and parts[1] == '+rot':
                    # Incremental rotation
                    poses[device]['rot'][0] += float(parts[2])
                    poses[device]['rot'][1] += float(parts[3])
                    poses[device]['rot'][2] += float(parts[4])
                    broadcast_poses()
                    print(f"{device} rot: {poses[device]['rot']}")
                else:
                    print(f"Usage: {parts[0]} pos X Y Z  or  {parts[0]} +pos X Y Z  or  {parts[0]} rot X Y Z  or  {parts[0]} +rot X Y Z")
            elif cmd == 'status':
                print(f"Connected clients: {len(clients)}")
                for device, pose in poses.items():
                    print(f"  {device}: pos={pose['pos']}, rot={pose['rot']}")
            else:
                print("Commands: h/c1/c2 pos X Y Z | h/c1/c2 rot X Y Z | reset | status | quit")
        except EOFError:
            running = False
            break
        except Exception as e:
            print(f"Error: {e}")


def demo_mode():
    """Demo mode - automatically animate poses."""
    global running
    
    print("\nDemo mode - animating poses...")
    print("Press Ctrl+C to stop\n")
    
    t = 0
    while running:
        try:
            import math
            
            # Animate headset - gentle swaying
            poses['headset']['pos'][0] = math.sin(t * 0.5) * 0.1
            poses['headset']['rot'][1] = math.sin(t * 0.3) * 10
            
            # Animate controllers - circular motion
            poses['controller1']['pos'][0] = -0.3 + math.sin(t) * 0.1
            poses['controller1']['pos'][2] = -0.3 + math.cos(t) * 0.1
            
            poses['controller2']['pos'][0] = 0.3 + math.sin(t + 3.14) * 0.1
            poses['controller2']['pos'][2] = -0.3 + math.cos(t + 3.14) * 0.1
            
            broadcast_poses()
            
            t += 0.05
            time.sleep(0.016)  # ~60 Hz
        except KeyboardInterrupt:
            running = False
            break


if __name__ == '__main__':
    # Start server thread
    server = threading.Thread(target=server_thread)
    server.daemon = True
    server.start()
    
    time.sleep(0.5)  # Let server start
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_mode()
    else:
        interactive_mode()
    
    print("\nShutting down...")
    running = False
    time.sleep(0.5)
