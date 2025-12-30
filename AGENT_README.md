# Gemini VR Agent

AI agent that controls your VR driver through the MCP server using Google's Gemini for planning and vision.

## Setup

1. Install dependencies:
```bash
pip install -r requirements_agent.txt
```

2. Get a Gemini API key from https://makersuite.google.com/app/apikey

3. Set the environment variable:
```bash
# Windows
set GEMINI_API_KEY=your_key_here

# Linux/Mac
export GEMINI_API_KEY=your_key_here
```

## Usage

1. Start SteamVR with your driver loaded
2. Run the agent:
```bash
python gemini_vr_agent.py
```

3. Give it tasks in natural language:
```
You: Look around and describe what you see
You: Walk forward and pick up the red cube
You: Navigate to the door and open it
```

## How It Works

The agent:
1. Receives your task description
2. Uses vision tools to observe the VR environment
3. Plans a sequence of actions
4. Executes each step and verifies with vision
5. Adapts if something doesn't work as expected

## Movement Modes

The agent understands different movement methods:

- **Joystick locomotion**: For games with smooth movement (uses controller joystick)
- **Teleportation**: For games with point-and-teleport (aims controller + trigger)
- **Direct positioning**: For debugging or apps without locomotion systems

Tell the agent what game you're in and it will choose the appropriate method.

## Example Tasks

- "Look around and tell me what you see"
- "Move forward using the joystick"
- "Pick up the object in front of me"
- "Press the menu button"
- "Do a 360 scan of the environment"
