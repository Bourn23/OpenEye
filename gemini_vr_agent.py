#!/usr/bin/env python3
"""
Gemini VR Agent - An AI agent that controls VR through the MCP server.
Uses Google's Gemini API for planning, vision analysis, and task execution.

Install:
    pip install google-genai mcp pillow

Usage:
    python gemini_vr_agent.py
"""

import os
import json
import time
import base64
import traceback
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Try the new google-genai package first, fall back to deprecated one
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
    print("[DEBUG] Using new google-genai SDK")
except ImportError:
    try:
        import google.generativeai as genai_old
        USE_NEW_SDK = False
        print("[DEBUG] Using deprecated google.generativeai SDK")
    except ImportError:
        print("Please install google-genai: pip install google-genai")
        exit(1)

# ============================================================================
# Configuration
# ============================================================================

GEMINI_MODEL = "gemini-3-flash-preview"  # or "gemini-1.5-pro" for better reasoning
DEBUG = True  # Enable debug output

def debug_print(msg: str):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[DEBUG] {msg}")

# ============================================================================
# System Prompt for the VR Agent
# ============================================================================

SYSTEM_PROMPT = """You are an intelligent VR Agent that controls a virtual reality headset and two controllers through an MCP server. You can see what the VR headset sees, move around, and interact with the virtual environment.

## Your Capabilities

### Devices You Control
- **Headset**: The VR display/camera - controls what you see and your position in VR space
- **Controller1**: Left hand controller - for interactions on the left side
- **Controller2**: Right hand controller - for interactions on the right side

### Movement & Navigation
You have multiple ways to move in VR. **Choose the right method based on the game/application:**

1. **Direct Position Movement** (teleport/walk_path/move_relative):
   - Moves the actual position of devices in 3D space
   - Best for: Applications without locomotion systems, debugging, precise positioning
   - Use `teleport` for instant movement, `walk_path` for smooth transitions

2. **Joystick-Based Locomotion** (set_joystick/move_joystick_direction):
   - Uses the controller joystick to trigger in-game movement
   - Best for: Most VR games with smooth locomotion (walking, running)
   - Typically: Left joystick = movement, Right joystick = turning
   - Push joystick forward (y=1.0) to walk forward in-game

3. **Teleportation (In-Game)**:
   - Many VR games use controller pointing + trigger for teleport
   - Point controller at destination, press trigger to teleport
   - Use `rotate_device` on controller to aim, then `click_button` trigger

**IMPORTANT**: Always consider what type of movement the current VR application expects:
- Some games ONLY support joystick locomotion
- Some games ONLY support teleportation
- Some support both
- If unsure, try joystick first, then teleportation

### Controller Inputs
- **Buttons**: trigger, grip, menu, system, trackpad, a, b
- **Analog**: trigger value (0-1), joystick X/Y (-1 to 1)
- **Actions**: grab (grip+trigger), release

### Vision
- `inspect_surroundings`: Capture current view as image
- `capture_video`: Record a sequence of frames
- `look_around_and_observe`: 360° panoramic scan

## Planning & Execution Strategy

When given a task, follow this approach:

### 1. UNDERSTAND
- What is the goal?
- What game/application are we in? (affects movement method)
- What information do I need?

### 2. OBSERVE
- Use vision tools to see the current state
- Identify relevant objects, UI elements, obstacles
- Note positions and orientations

### 3. PLAN
- Break the task into discrete steps
- Choose appropriate movement method for this application
- Identify verification points

### 4. EXECUTE & VERIFY
- Execute one step at a time
- After each significant action, verify with vision
- Adjust plan if needed based on observations

### 5. CONFIRM
- Verify the task is complete
- Report results to user

## Coordinate System
- X: Left (-) / Right (+)
- Y: Down (-) / Up (+)  
- Z: Forward (-) / Backward (+)
- Rotation: Pitch (up/down), Yaw (left/right), Roll (tilt)

## Important Notes

- Controllers are tethered to headset (max 0.8m reach) - they move with you
- Always verify actions with vision when possible
- If something doesn't work, try alternative approaches
- Report what you see and your reasoning to the user
"""

# ============================================================================
# MCP Tool Definitions
# ============================================================================

MCP_TOOLS_DEFINITIONS = [
    # Connection
    {
        "name": "start_vr_bridge",
        "description": "Start the TCP server to listen for the OpenVR driver. Call this first before any other actions.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_connection_status", 
        "description": "Check if the VR driver is connected to SteamVR.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    
    # Movement
    {
        "name": "teleport",
        "description": "Instantly teleport a device to exact coordinates. Use for direct positioning.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {"type": "string", "enum": ["headset", "controller1", "controller2"]},
                "x": {"type": "number", "description": "X coordinate"},
                "y": {"type": "number", "description": "Y coordinate (height)"},
                "z": {"type": "number", "description": "Z coordinate"}
            },
            "required": ["device", "x", "y", "z"]
        }
    },
    {
        "name": "walk_path",
        "description": "Smoothly walk the headset to a destination over multiple steps. Prevents motion sickness.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Target X coordinate"},
                "z": {"type": "number", "description": "Target Z coordinate"},
                "steps": {"type": "integer", "description": "Number of steps (default 10)"}
            },
            "required": ["x", "z"]
        }
    },
    {
        "name": "move_relative",
        "description": "Move a device relative to its current position.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {"type": "string", "enum": ["headset", "controller1", "controller2"]},
                "dx": {"type": "number", "description": "Change in X"},
                "dy": {"type": "number", "description": "Change in Y"},
                "dz": {"type": "number", "description": "Change in Z"}
            },
            "required": ["device"]
        }
    },
    {
        "name": "look_at",
        "description": "Make the headset look at a specific point in 3D space.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_x": {"type": "number"},
                "target_y": {"type": "number"},
                "target_z": {"type": "number"}
            },
            "required": ["target_x", "target_y", "target_z"]
        }
    },
    {
        "name": "rotate_device",
        "description": "Set the rotation of a device in degrees.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {"type": "string", "enum": ["headset", "controller1", "controller2"]},
                "pitch": {"type": "number", "description": "Up/down rotation"},
                "yaw": {"type": "number", "description": "Left/right rotation"},
                "roll": {"type": "number", "description": "Tilt rotation"}
            },
            "required": ["device", "pitch", "yaw", "roll"]
        }
    },
    {
        "name": "get_current_pose",
        "description": "Get the current position and rotation of a device.",
        "parameters": {
            "type": "object",
            "properties": {
                "device": {"type": "string", "enum": ["headset", "controller1", "controller2"]}
            },
            "required": []
        }
    },
    
    # Controller positioning
    {
        "name": "position_controller_relative_to_headset",
        "description": "Position a controller relative to headset. Useful for natural hand positions.",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]},
                "forward": {"type": "number", "description": "Distance forward (negative=in front)"},
                "right": {"type": "number", "description": "Distance right (negative=left)"},
                "up": {"type": "number", "description": "Distance up (negative=below)"}
            },
            "required": ["controller"]
        }
    },
    {
        "name": "reset_controller_positions",
        "description": "Reset both controllers to natural resting positions near the headset.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    
    # Buttons
    {
        "name": "press_button",
        "description": "Press and hold a button on a controller.",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]},
                "button": {"type": "string", "enum": ["trigger", "grip", "menu", "system", "trackpad", "a", "b"]}
            },
            "required": ["controller", "button"]
        }
    },
    {
        "name": "release_button",
        "description": "Release a previously pressed button.",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]},
                "button": {"type": "string", "enum": ["trigger", "grip", "menu", "system", "trackpad", "a", "b"]}
            },
            "required": ["controller", "button"]
        }
    },
    {
        "name": "click_button",
        "description": "Click a button (press and release quickly).",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]},
                "button": {"type": "string", "enum": ["trigger", "grip", "menu", "system", "trackpad", "a", "b"]},
                "duration": {"type": "number", "description": "Hold duration in seconds"}
            },
            "required": ["controller", "button"]
        }
    },
    {
        "name": "set_trigger",
        "description": "Set analog trigger value (0.0 to 1.0).",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]},
                "value": {"type": "number"}
            },
            "required": ["controller", "value"]
        }
    },
    
    # Joystick
    {
        "name": "set_joystick",
        "description": "Set joystick position. X: -1(left) to 1(right), Y: -1(down) to 1(up). Use for in-game locomotion.",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]},
                "x": {"type": "number"},
                "y": {"type": "number"}
            },
            "required": ["controller", "x", "y"]
        }
    },
    {
        "name": "move_joystick_direction",
        "description": "Move joystick in a direction. Use for in-game movement (usually left controller).",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]},
                "direction": {"type": "string", "enum": ["up", "down", "left", "right", "center", "forward", "backward"]},
                "magnitude": {"type": "number"}
            },
            "required": ["controller", "direction"]
        }
    },
    {
        "name": "release_all_inputs",
        "description": "Release all buttons and center joystick.",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2", "both"]}
            },
            "required": []
        }
    },
    {
        "name": "get_controller_state",
        "description": "Get current input state of a controller.",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]}
            },
            "required": ["controller"]
        }
    },
    
    # Actions
    {
        "name": "perform_grab",
        "description": "Grab action - press grip and trigger together. Use to pick up objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]}
            },
            "required": ["controller"]
        }
    },
    {
        "name": "perform_release",
        "description": "Release a grabbed object.",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string", "enum": ["controller1", "controller2"]}
            },
            "required": ["controller"]
        }
    },
    
    # Vision
    {
        "name": "inspect_surroundings",
        "description": "Capture a single frame of what the VR headset sees. Returns base64 JPEG image.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "capture_video",
        "description": "Capture video (sequence of frames) for temporal analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "description": "Video length in seconds (max 10)"},
                "fps": {"type": "integer", "description": "Frames per second (max 30)"}
            },
            "required": []
        }
    },
    {
        "name": "look_around_and_observe",
        "description": "360° panoramic scan - captures frames at 0°, 90°, 180°, 270° angles.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
]


# ============================================================================
# Direct MCP Tool Executor (imports mcp_server.py functions directly)
# ============================================================================

class DirectMCPExecutor:
    """
    Executes MCP tools by importing and calling them directly.
    This is simpler than socket communication for local use.
    """
    
    def __init__(self):
        self.mcp_module = None
        self._load_mcp_module()
        
    def _load_mcp_module(self):
        """Import the MCP server module."""
        debug_print("Loading MCP server module...")
        try:
            import importlib.util
            import sys
            
            spec = importlib.util.spec_from_file_location("mcp_server", "mcp_server.py")
            if spec is None:
                raise ImportError("Could not find mcp_server.py")
            
            self.mcp_module = importlib.util.module_from_spec(spec)
            sys.modules["mcp_server"] = self.mcp_module
            spec.loader.exec_module(self.mcp_module)
            debug_print("MCP server module loaded successfully")
        except Exception as e:
            debug_print(f"Failed to load MCP module: {e}")
            traceback.print_exc()
            raise
        
    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> str:
        """Call an MCP tool function directly."""
        if arguments is None:
            arguments = {}
            
        debug_print(f"Calling tool: {tool_name} with args: {arguments}")
        
        # Get the function from the module
        if not hasattr(self.mcp_module, tool_name):
            error_msg = f"Error: Unknown tool '{tool_name}'"
            debug_print(error_msg)
            return error_msg
            
        func = getattr(self.mcp_module, tool_name)
        
        try:
            result = func(**arguments)
            debug_print(f"Tool result: {str(result)[:200]}...")
            return str(result)
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            debug_print(error_msg)
            traceback.print_exc()
            return error_msg


# ============================================================================
# Gemini VR Agent
# ============================================================================

@dataclass
class AgentState:
    """Tracks the agent's current state and history."""
    task: str = ""
    plan: List[str] = field(default_factory=list)
    current_step: int = 0
    observations: List[Dict] = field(default_factory=list)
    actions_taken: List[Dict] = field(default_factory=list)
    completed: bool = False


class GeminiVRAgent:
    """
    AI Agent that uses Gemini to control VR through MCP tools.
    Supports planning, execution, and vision-based verification.
    """
    
    def __init__(self, api_key: str = None):
        debug_print("Initializing GeminiVRAgent...")
        
        # Get API key
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        debug_print(f"API key found (length: {len(api_key)})")
        
        # Initialize based on SDK version
        if USE_NEW_SDK:
            self._init_new_sdk(api_key)
        else:
            self._init_old_sdk(api_key)
        
        # Initialize MCP executor
        debug_print("Initializing MCP executor...")
        self.executor = DirectMCPExecutor()
        
        # Agent state
        self.state = AgentState()
        
        # Start VR bridge automatically
        debug_print("Starting VR bridge...")
        result = self.executor.call_tool("start_vr_bridge")
        print(f"VR Bridge: {result}")
        
        debug_print("GeminiVRAgent initialized successfully")
    
    def _init_new_sdk(self, api_key: str):
        """Initialize with new google-genai SDK."""
        debug_print("Configuring new google-genai SDK...")
        
        try:
            self.client = genai.Client(api_key=api_key)
            debug_print(f"Client created: {type(self.client)}")
            
            # Build function declarations for the new SDK
            self.tools = self._build_tools_new_sdk()
            debug_print(f"Tools built: {len(self.tools)} tools")
            
            self.model_name = GEMINI_MODEL
            self.chat_history = []
            
        except Exception as e:
            debug_print(f"Error initializing new SDK: {e}")
            traceback.print_exc()
            raise
    
    def _init_old_sdk(self, api_key: str):
        """Initialize with deprecated google.generativeai SDK."""
        debug_print("Configuring deprecated google.generativeai SDK...")
        
        try:
            genai_old.configure(api_key=api_key)
            
            # Create the model with function calling
            self.model = genai_old.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=SYSTEM_PROMPT,
            )
            
            self.chat = None
            debug_print("Old SDK configured successfully")
            
        except Exception as e:
            debug_print(f"Error initializing old SDK: {e}")
            traceback.print_exc()
            raise
    
    def _build_tools_new_sdk(self) -> List:
        """Build tool declarations for new SDK."""
        tools = []
        
        for tool_def in MCP_TOOLS_DEFINITIONS:
            try:
                # Convert to the format expected by new SDK
                func_decl = types.FunctionDeclaration(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    parameters=tool_def["parameters"] if tool_def["parameters"]["properties"] else None
                )
                tools.append(func_decl)
            except Exception as e:
                debug_print(f"Error building tool {tool_def['name']}: {e}")
        
        return tools
    
    def _execute_function_call(self, function_name: str, function_args: Dict) -> str:
        """Execute a function call and return the result."""
        print(f"  → Executing: {function_name}({function_args})")
        
        # Execute the tool
        result = self.executor.call_tool(function_name, function_args)
        
        # Track the action
        self.state.actions_taken.append({
            "tool": function_name,
            "args": function_args,
            "result": result[:500] if len(result) > 500 else result
        })
        
        return result
    
    def execute_task(self, task: str, max_iterations: int = 20) -> str:
        """
        Execute a VR task with planning and verification.
        """
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")
        
        # Reset state
        self.state = AgentState(task=task)
        
        if USE_NEW_SDK:
            return self._execute_task_new_sdk(task, max_iterations)
        else:
            return self._execute_task_old_sdk(task, max_iterations)
    
    def _execute_task_new_sdk(self, task: str, max_iterations: int) -> str:
        """Execute task using new google-genai SDK."""
        debug_print("Executing task with new SDK...")
        
        prompt = f"""Execute this VR task: {task}

First, observe the current state using vision tools if needed, then plan your approach, and execute step by step. Verify each significant action.

Remember:
- For in-game movement, use joystick (controller1 for locomotion, controller2 for turning)
- For direct positioning, use teleport/walk_path
- Always verify with vision after important actions
- Report what you see and your reasoning"""

        try:
            # Create tool config
            tool_config = types.Tool(function_declarations=self.tools)
            
            # Build contents with system instruction
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=SYSTEM_PROMPT + "\n\n" + prompt)]
                )
            ]
            
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                debug_print(f"Iteration {iteration}")
                
                # Generate response
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=[tool_config],
                        temperature=0.7,
                    )
                )
                
                debug_print(f"Response received: {type(response)}")
                
                # Check for function calls
                has_function_call = False
                text_response = ""
                
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            has_function_call = True
                            fc = part.function_call
                            
                            # Execute the function
                            args = dict(fc.args) if fc.args else {}
                            result = self._execute_function_call(fc.name, args)
                            
                            # Add assistant response and function result to contents
                            contents.append(candidate.content)
                            contents.append(types.Content(
                                role="user",
                                parts=[types.Part(
                                    function_response=types.FunctionResponse(
                                        name=fc.name,
                                        response={"result": result}
                                    )
                                )]
                            ))
                            break
                        
                        if hasattr(part, 'text') and part.text:
                            text_response += part.text
                    
                    if has_function_call:
                        break
                
                if not has_function_call:
                    # No more function calls, we're done
                    self.state.completed = True
                    print(f"\n{'='*60}")
                    print(f"Task Complete - {len(self.state.actions_taken)} actions taken")
                    print(f"{'='*60}\n")
                    return text_response
            
            return "Max iterations reached"
            
        except Exception as e:
            debug_print(f"Error during task execution: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def _execute_task_old_sdk(self, task: str, max_iterations: int) -> str:
        """Execute task using deprecated SDK."""
        debug_print("Executing task with old SDK...")
        
        prompt = f"""Execute this VR task: {task}

First, observe the current state, then plan your approach, and execute step by step."""

        try:
            self.chat = self.model.start_chat()
            response = self.chat.send_message(prompt)
            
            # Simple text response for old SDK (function calling is more complex)
            text = ""
            for part in response.parts:
                if hasattr(part, 'text'):
                    text += part.text
            
            return text
            
        except Exception as e:
            debug_print(f"Error: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def chat_interactive(self):
        """Run an interactive chat session."""
        print("\n" + "="*60)
        print("Gemini VR Agent - Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  Type a task to execute it")
        print("  'status' - Check VR connection status")
        print("  'reset' - Reset controller positions")
        print("  'quit' - Exit")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                    
                if user_input.lower() == 'status':
                    result = self.executor.call_tool("get_connection_status")
                    print(f"Status: {result}")
                    continue
                    
                if user_input.lower() == 'reset':
                    result = self.executor.call_tool("reset_controller_positions")
                    print(f"Reset: {result}")
                    continue
                
                # Execute the task
                response = self.execute_task(user_input)
                print(f"\nAgent: {response}")
                
            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                if DEBUG:
                    traceback.print_exc()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    print("="*60)
    print("Gemini VR Agent")
    print("="*60)
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\nError: GEMINI_API_KEY environment variable not set.")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        print("\nSet it with:")
        print("  Windows CMD: set GEMINI_API_KEY=your_key_here")
        print("  Windows PowerShell: $env:GEMINI_API_KEY='your_key_here'")
        print("  Linux/Mac: export GEMINI_API_KEY=your_key_here")
        return
    
    try:
        debug_print("Creating agent...")
        agent = GeminiVRAgent(api_key=api_key)
        agent.chat_interactive()
    except Exception as e:
        print(f"\nFailed to initialize agent: {e}")
        if DEBUG:
            print("\nFull traceback:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
