import socket
import json
import base64

HOST = '127.0.0.1'
PORT = 5555

def test_capture():
    print(f"Connecting to {HOST}:{PORT}...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        
        # 1. Read initial pose update (clears the buffer)
        print("Waiting for initial handshake...")
        s.recv(4096) 
        
        # 2. Send Vision Request
        print("Sending vision request...")
        request = json.dumps({"type": "vision_request", "action": "capture_frame"}) + "\n"
        s.sendall(request.encode('utf-8'))
        
        # 3. Receive Response
        buffer = ""
        print("Receiving image data...")
        while True:
            chunk = s.recv(4096)
            if not chunk: break
            buffer += chunk.decode('utf-8', errors='ignore')
            if '\n' in buffer: break
            
        # 4. Parse and Save
        for line in buffer.split('\n'):
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if data.get('type') == 'frame':
                    img_data = base64.b64decode(data['frames'][0])
                    with open("debug_view.jpg", "wb") as f:
                        f.write(img_data)
                    print(f"SUCCESS! Saved debug_view.jpg ({data['width']}x{data['height']})")
                    print("Open this file. If it shows the game, YOU ARE READY.")
                    return
                elif data.get('type') == 'error':
                    print(f"Driver Error: {data.get('message')}")
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        s.close()

if __name__ == "__main__":
    test_capture()