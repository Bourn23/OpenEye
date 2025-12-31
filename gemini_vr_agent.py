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
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# Try the new google-genai package first, fall back to deprecated one
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
    print("[DEBUG] Using new google-genai SDK")
except ImportError:
    print("Please install google-genai: pip install google-genai")
    exit(1)

# ============================================================================
# Configuration
# ============================================================================

GEMINI_MODEL = "gemini-3-flash-preview"  # or "gemini-1.5-pro" for better reasoning
DEBUG = True  # Enable debug output
LOG_DIR = Path("agent_logs")  # Directory for log files
SAVE_VISION_FRAMES = True  # Save captured images to disk for inspection
SHOW_VISION_PREVIEW = True  # Show real-time vision preview with cv2.imshow

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 35  # Stay under 40 RPM limit with buffer
MIN_REQUEST_INTERVAL = 2.0  # Minimum seconds between requests (spreads out calls)
MAX_CONCURRENT_REQUESTS = 6  # Stay under 8 concurrent limit with buffer

# Response configuration
MAX_OUTPUT_TOKENS = 2048  # Limit response length to prevent rambling
API_TIMEOUT_MS = 60000  # 60 second timeout for API calls

# Try to import OpenCV for vision preview
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    if SHOW_VISION_PREVIEW:
        print("[WARNING] OpenCV not installed. Run: pip install opencv-python")

# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """
    Rate limiter for API calls with RPM tracking and automatic delays.
    Thread-safe implementation.
    """
    
    def __init__(
        self,
        max_rpm: int = MAX_REQUESTS_PER_MINUTE,
        min_interval: float = MIN_REQUEST_INTERVAL,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS
    ):
        self.max_rpm = max_rpm
        self.min_interval = min_interval
        self.max_concurrent = max_concurrent
        
        # Track request timestamps (last 60 seconds)
        self.request_times: deque = deque()
        self.last_request_time: float = 0
        self.active_requests: int = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Stats
        self.total_requests = 0
        self.total_wait_time = 0.0
    
    def _cleanup_old_requests(self):
        """Remove request timestamps older than 60 seconds."""
        now = time.time()
        while self.request_times and (now - self.request_times[0]) > 60:
            self.request_times.popleft()
    
    def get_current_rpm(self) -> int:
        """Get current requests per minute."""
        with self._lock:
            self._cleanup_old_requests()
            return len(self.request_times)
    
    def wait_if_needed(self, logger: 'AgentLogger' = None) -> float:
        """
        Wait if necessary to respect rate limits.
        Returns the time waited in seconds.
        """
        wait_time = 0.0
        
        with self._lock:
            now = time.time()
            self._cleanup_old_requests()
            
            # Check 1: Minimum interval between requests
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_interval:
                interval_wait = self.min_interval - time_since_last
                wait_time = max(wait_time, interval_wait)
            
            # Check 2: RPM limit
            current_rpm = len(self.request_times)
            if current_rpm >= self.max_rpm:
                # Wait until oldest request falls out of the 60s window
                oldest = self.request_times[0]
                rpm_wait = 60 - (now - oldest) + 0.1  # +0.1s buffer
                wait_time = max(wait_time, rpm_wait)
            
            # Check 3: Concurrent request limit
            if self.active_requests >= self.max_concurrent:
                # This shouldn't happen in single-threaded use, but safety first
                wait_time = max(wait_time, 1.0)
        
        # Actually wait (outside lock)
        if wait_time > 0:
            if logger:
                logger.info(
                    f"Rate limit: waiting {wait_time:.1f}s "
                    f"(RPM: {self.get_current_rpm()}/{self.max_rpm})"
                )
            time.sleep(wait_time)
            self.total_wait_time += wait_time
        
        return wait_time
    
    def acquire(self, logger: 'AgentLogger' = None):
        """Acquire permission to make a request. Blocks if rate limited."""
        self.wait_if_needed(logger)
        
        with self._lock:
            now = time.time()
            self.request_times.append(now)
            self.last_request_time = now
            self.active_requests += 1
            self.total_requests += 1
    
    def release(self):
        """Release after request completes."""
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "total_requests": self.total_requests,
            "current_rpm": self.get_current_rpm(),
            "max_rpm": self.max_rpm,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "active_requests": self.active_requests
        }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

# ============================================================================
# Structured Logger
# ============================================================================

class AgentLogger:
    """
    Structured logger for VR Agent debugging.
    Outputs to both console and file with clear formatting.
    """
    
    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session-specific log file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"agent_session_{self.session_id}.log"
        self.llm_log_file = self.log_dir / f"llm_conversation_{self.session_id}.log"
        self.vision_dir = self.log_dir / f"vision_{self.session_id}"
        self.media_dir = self.log_dir / f"media_{self.session_id}"
        
        # Always create vision and media directories
        self.vision_dir.mkdir(exist_ok=True)
        self.media_dir.mkdir(exist_ok=True)
        
        # Setup file logger
        self.file_logger = logging.getLogger(f"vr_agent_{self.session_id}")
        self.file_logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        self.file_logger.addHandler(file_handler)
        
        # Setup dedicated LLM conversation logger
        self.llm_logger = logging.getLogger(f"llm_conv_{self.session_id}")
        self.llm_logger.setLevel(logging.DEBUG)
        
        llm_handler = logging.FileHandler(self.llm_log_file, encoding='utf-8')
        llm_handler.setFormatter(logging.Formatter('%(message)s'))
        self.llm_logger.addHandler(llm_handler)
        
        # Counters
        self.action_count = 0
        self.vision_frame_count = 0
        self.api_call_count = 0
        self.llm_message_count = 0
        
        self._log_header()
    
    def _log_header(self):
        """Log session header."""
        header = f"""
{'='*70}
VR AGENT SESSION: {self.session_id}
Started: {datetime.now().isoformat()}
Log File: {self.log_file}
LLM Conversation Log: {self.llm_log_file}
Vision Frames: {self.vision_dir}
Media Files: {self.media_dir}
{'='*70}
"""
        print(header)
        self.file_logger.info(header)
        
        # LLM log header
        llm_header = f"""{'='*80}
LLM CONVERSATION LOG - Session: {self.session_id}
Started: {datetime.now().isoformat()}
{'='*80}

This log contains all messages sent to and received from the LLM.
Media files are saved to: {self.media_dir}
{'='*80}
"""
        self.llm_logger.info(llm_header)
    
    def _format_and_log(self, level: str, category: str, message: str, data: Dict = None):
        """Format and output log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Build log line
        prefix = f"[{timestamp}] [{level}] [{category}]"
        log_line = f"{prefix} {message}"
        
        # Console output with colors (ANSI)
        colors = {
            "INFO": "\033[94m",      # Blue
            "ACTION": "\033[92m",    # Green
            "VISION": "\033[95m",    # Magenta
            "THINK": "\033[93m",     # Yellow
            "ERROR": "\033[91m",     # Red
            "API": "\033[96m",       # Cyan
        }
        reset = "\033[0m"
        color = colors.get(level, "")
        
        print(f"{color}{log_line}{reset}")
        
        # Add data if present
        if data:
            data_str = json.dumps(data, indent=2, default=str)
            # Truncate large data for console
            if len(data_str) > 500:
                print(f"  └─ Data: {data_str[700:]}... (truncated)")
            else:
                for line in data_str.split('\n'):
                    print(f"  │ {line}")
        
        # File output (full data)
        self.file_logger.info(log_line)
        if data:
            self.file_logger.info(f"  DATA: {json.dumps(data, default=str)}")
    
    def info(self, message: str, data: Dict = None):
        """General info log."""
        self._format_and_log("INFO", "SYSTEM", message, data)
    
    def action(self, tool_name: str, args: Dict, result: str):
        """Log a tool/action execution."""
        self.action_count += 1
        
        # Detect vision-related actions
        is_vision = tool_name in ["inspect_surroundings", "capture_video", "look_around_and_observe"]
        
        self._format_and_log(
            "ACTION", 
            f"TOOL #{self.action_count}",
            f"{tool_name}",
            {"arguments": args}
        )
        
        # Handle vision results specially - check for image data patterns
        if is_vision and ('"data"' in result or '"type": "image"' in result or '"type": "panorama_scan"' in result):
            self._handle_vision_result(tool_name, result)
        else:
            # Truncate long results for display
            display_result = result[:300] + "..." if len(result) > 300 else result
            self._format_and_log("ACTION", "RESULT", display_result)
    
    def _show_image_preview(self, img_data: bytes, title: str):
        """Show image preview using OpenCV if available and enabled."""
        if not SHOW_VISION_PREVIEW or not CV2_AVAILABLE:
            return
        
        try:
            # Validate JPEG data before decoding
            if len(img_data) < 100:
                self._format_and_log("VISION", "PREVIEW", f"Image data too small ({len(img_data)} bytes), skipping preview")
                return
            
            # Check JPEG header (FFD8) and footer (FFD9)
            if img_data[:2] != b'\xff\xd8':
                self._format_and_log("VISION", "PREVIEW", "Invalid JPEG header, skipping preview")
                return
            
            if img_data[-2:] != b'\xff\xd9':
                self._format_and_log("VISION", "PREVIEW", f"JPEG data appears truncated (missing FFD9 footer, size={len(img_data)}), skipping preview")
                return
            
            # Decode JPEG bytes to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Resize if too large (max 800px width for preview)
                h, w = img.shape[:2]
                if w > 800:
                    scale = 800 / w
                    img = cv2.resize(img, (800, int(h * scale)))
                
                cv2.imshow(title, img)
                cv2.waitKey(500)  # Show for 500ms minimum
            else:
                self._format_and_log("VISION", "PREVIEW", "cv2.imdecode returned None - JPEG may be corrupt")
        except Exception as e:
            self._format_and_log("VISION", "PREVIEW", f"Failed to show preview: {e}")
    
    def _handle_vision_result(self, tool_name: str, result: str):
        """Handle and optionally save vision data from various vision tools."""
        
        try:
            # Try to parse as JSON first
            result_data = None
            try:
                result_data = json.loads(result)
            except json.JSONDecodeError:
                pass
            
            if not isinstance(result_data, dict):
                self._format_and_log("VISION", "CAPTURE", f"Vision data received ({len(result)} chars) - could not parse JSON")
                return
            
            result_type = result_data.get('type', '')
            
            # Handle panorama scan (look_around_and_observe) - multiple directions
            if result_type == 'panorama_scan' and 'directions' in result_data:
                directions = result_data['directions']
                preview_images = []
                
                for direction in directions:
                    angle = direction.get('angle', 0)
                    b64_data = direction.get('data')
                    if b64_data:
                        self._save_and_show_frame(b64_data, f"{tool_name}_angle{angle}", f"VR Vision - {angle} deg")
                        
                        # Collect for combined preview
                        if SHOW_VISION_PREVIEW and CV2_AVAILABLE:
                            try:
                                img_data = base64.b64decode(b64_data)
                                nparr = np.frombuffer(img_data, np.uint8)
                                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if img is not None:
                                    preview_images.append((angle, img))
                            except:
                                pass
                
                # Create combined panorama preview (2x2 grid)
                self._show_panorama_grid(preview_images)
                return
            
            # Handle video (capture_video) - multiple frames
            if result_type == 'video' and 'frames' in result_data:
                frames = result_data['frames']
                fps = result_data.get('fps', 10)
                
                for i, b64_data in enumerate(frames):
                    self._save_and_show_frame(b64_data, f"{tool_name}_frame{i:03d}", f"VR Video - Frame {i}")
                    
                    # Brief delay between frames for video preview effect
                    if SHOW_VISION_PREVIEW and CV2_AVAILABLE and i < len(frames) - 1:
                        cv2.waitKey(int(1000 / fps))
                return
            
            # Handle single image (inspect_surroundings) - direct data field
            if result_type == 'image' and 'data' in result_data:
                b64_data = result_data['data']
                self._save_and_show_frame(b64_data, tool_name, f"VR Vision - {tool_name}")
                return
            
            # Fallback: try to find any 'data' field with base64
            if 'data' in result_data:
                self._save_and_show_frame(result_data['data'], tool_name, f"VR Vision - {tool_name}")
                return
            
            self._format_and_log("VISION", "CAPTURE", f"Vision data received but format not recognized: type={result_type}")
            
        except Exception as e:
            self._format_and_log("VISION", "ERROR", f"Failed to process vision data: {e}")
    
    def _save_and_show_frame(self, b64_data: str, name_suffix: str, window_title: str):
        """Decode, save, and optionally display a single base64 frame."""
        try:
            img_data = base64.b64decode(b64_data)
            self.vision_frame_count += 1
            timestamp = datetime.now().strftime("%H%M%S")
            frame_path = self.vision_dir / f"frame_{self.vision_frame_count:04d}_{timestamp}_{name_suffix}.jpg"
            frame_path.write_bytes(img_data)
            
            self._format_and_log(
                "VISION",
                f"FRAME #{self.vision_frame_count}",
                f"Saved to: {frame_path}",
                {"size_bytes": len(img_data)}
            )
            
            # Show preview
            self._show_image_preview(img_data, window_title)
            
        except Exception as e:
            self._format_and_log("VISION", "ERROR", f"Failed to save frame {name_suffix}: {e}")
    
    def _show_panorama_grid(self, preview_images: list):
        """Create and show a 2x2 grid of panorama images."""
        if not SHOW_VISION_PREVIEW or not CV2_AVAILABLE or len(preview_images) < 4:
            return
        
        try:
            # Sort by angle and resize to same size
            preview_images.sort(key=lambda x: x[0])
            target_h, target_w = 240, 320
            resized = [cv2.resize(img, (target_w, target_h)) for _, img in preview_images[:4]]
            
            # Create 2x2 grid
            top_row = np.hstack([resized[0], resized[1]])
            bottom_row = np.hstack([resized[2], resized[3]])
            combined = np.vstack([top_row, bottom_row])
            
            # Add angle labels
            angles = [0, 90, 180, 270]
            positions = [(10, 30), (target_w + 10, 30), (10, target_h + 30), (target_w + 10, target_h + 30)]
            for angle, pos in zip(angles, positions):
                cv2.putText(combined, f"{angle} deg", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("VR Vision - 360 Panorama", combined)
            cv2.waitKey(2000)
        except Exception as e:
            self._format_and_log("VISION", "PREVIEW", f"Failed to create panorama grid: {e}")
    
    def save_media(self, data: bytes, media_type: str, source: str) -> Path:
        """Save any media data (images, etc.) and return the path."""
        timestamp = datetime.now().strftime("%H%M%S_%f")
        
        # Determine extension based on type
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "jpeg": ".jpg",
            "png": ".png",
        }
        ext = ext_map.get(media_type.lower(), ".bin")
        
        filename = f"{source}_{timestamp}{ext}"
        filepath = self.media_dir / filename
        filepath.write_bytes(data)
        
        self._format_and_log("INFO", "MEDIA", f"Saved: {filepath} ({len(data)} bytes)")
        return filepath
    
    def log_llm_request(self, contents: List, iteration: int):
        """Log the full request being sent to the LLM."""
        self.llm_message_count += 1
        
        separator = f"\n{'='*80}\n"
        header = f"{separator}[REQUEST #{self.llm_message_count}] Iteration {iteration} - {datetime.now().isoformat()}{separator}"
        self.llm_logger.info(header)
        
        for i, content in enumerate(contents):
            role = getattr(content, 'role', 'unknown')
            self.llm_logger.info(f"\n--- Message {i+1} (role: {role}) ---")
            
            parts = getattr(content, 'parts', [])
            for j, part in enumerate(parts):
                # Handle different part types
                if hasattr(part, 'text') and part.text:
                    text = part.text
                    # Truncate very long text but note the full length
                    if len(text) > 2000:
                        self.llm_logger.info(f"[Part {j+1} - TEXT ({len(text)} chars, truncated)]")
                        self.llm_logger.info(text[:2000] + "\n... [TRUNCATED]")
                    else:
                        self.llm_logger.info(f"[Part {j+1} - TEXT]")
                        self.llm_logger.info(text)
                
                elif hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    self.llm_logger.info(f"[Part {j+1} - FUNCTION_CALL]")
                    self.llm_logger.info(f"  Function: {fc.name}")
                    self.llm_logger.info(f"  Args: {dict(fc.args) if fc.args else {}}")
                
                elif hasattr(part, 'function_response') and part.function_response:
                    fr = part.function_response
                    self.llm_logger.info(f"[Part {j+1} - FUNCTION_RESPONSE]")
                    self.llm_logger.info(f"  Function: {fr.name}")
                    response_str = str(fr.response)
                    # Check for base64 data and save it
                    if len(response_str) > 1000 and 'base64' in response_str.lower():
                        self.llm_logger.info(f"  Response: [Contains base64 data, {len(response_str)} chars - see media folder]")
                        self._save_base64_from_response(response_str, fr.name)
                    elif len(response_str) > 2000:
                        self.llm_logger.info(f"  Response ({len(response_str)} chars, truncated): {response_str[:2000]}...")
                    else:
                        self.llm_logger.info(f"  Response: {response_str}")
                
                elif hasattr(part, 'inline_data') and part.inline_data:
                    # Handle inline media
                    inline = part.inline_data
                    mime = getattr(inline, 'mime_type', 'unknown')
                    data = getattr(inline, 'data', b'')
                    self.llm_logger.info(f"[Part {j+1} - INLINE_DATA]")
                    self.llm_logger.info(f"  MIME: {mime}, Size: {len(data)} bytes")
                    # Save the media
                    if data:
                        self.save_media(data if isinstance(data, bytes) else base64.b64decode(data), mime, f"inline_{i}_{j}")
                
                else:
                    self.llm_logger.info(f"[Part {j+1} - OTHER: {type(part).__name__}]")
    
    def _save_base64_from_response(self, response_str: str, source: str):
        """Extract and save base64 data from a response string."""
        try:
            # Try to parse as JSON first
            try:
                result_data = json.loads(response_str)
                if isinstance(result_data, dict):
                    # Check for nested 'result' key
                    if 'result' in result_data:
                        inner = result_data['result']
                        if isinstance(inner, str):
                            try:
                                inner = json.loads(inner)
                            except:
                                pass
                        if isinstance(inner, dict) and 'data' in inner:
                            b64_data = inner['data']
                            img_data = base64.b64decode(b64_data)
                            self.save_media(img_data, "jpeg", f"response_{source}")
                            return
                    # Check for direct 'data' key
                    if 'data' in result_data:
                        b64_data = result_data['data']
                        img_data = base64.b64decode(b64_data)
                        self.save_media(img_data, "jpeg", f"response_{source}")
                        return
            except json.JSONDecodeError:
                pass
            
            # Fall back to regex
            import re
            b64_match = re.search(r'"data"\s*:\s*"([A-Za-z0-9+/=]+)"', response_str)
            if b64_match:
                b64_data = b64_match.group(1)
                img_data = base64.b64decode(b64_data)
                self.save_media(img_data, "jpeg", f"response_{source}")
        except Exception as e:
            self.llm_logger.info(f"  [Failed to extract/save base64: {e}]")
    
    def log_llm_response(self, response, iteration: int):
        """Log the full response received from the LLM."""
        separator = f"\n{'-'*80}\n"
        header = f"{separator}[RESPONSE] Iteration {iteration} - {datetime.now().isoformat()}{separator}"
        self.llm_logger.info(header)
        
        for i, candidate in enumerate(response.candidates):
            self.llm_logger.info(f"\n--- Candidate {i+1} ---")
            
            content = candidate.content
            role = getattr(content, 'role', 'unknown')
            self.llm_logger.info(f"Role: {role}")
            
            for j, part in enumerate(content.parts):
                if hasattr(part, 'text') and part.text:
                    text = part.text
                    self.llm_logger.info(f"[Part {j+1} - TEXT ({len(text)} chars)]")
                    self.llm_logger.info(text)
                
                elif hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    self.llm_logger.info(f"[Part {j+1} - FUNCTION_CALL]")
                    self.llm_logger.info(f"  Function: {fc.name}")
                    self.llm_logger.info(f"  Args: {json.dumps(dict(fc.args) if fc.args else {}, indent=2)}")
                
                else:
                    self.llm_logger.info(f"[Part {j+1} - {type(part).__name__}]")
        
        # Log finish reason if available
        if hasattr(response.candidates[0], 'finish_reason'):
            self.llm_logger.info(f"\nFinish Reason: {response.candidates[0].finish_reason}")
    
    def thinking(self, thought: str):
        """Log agent's reasoning/thinking."""
        # Clean up and format thought
        thought = thought.strip()
        if thought:
            self._format_and_log("THINK", "REASONING", thought[:500] + ("..." if len(thought) > 500 else ""))
    
    def api_call(self, direction: str, details: str = ""):
        """Log API calls to Gemini."""
        # Only count actual requests, not responses
        if direction == "REQUEST":
            self.api_call_count += 1
        self._format_and_log(
            "API",
            f"GEMINI #{self.api_call_count}",
            f"{direction} {details}"
        )
    
    def error(self, message: str, exception: Exception = None):
        """Log errors."""
        self._format_and_log("ERROR", "ERROR", message)
        if exception:
            self.file_logger.error(traceback.format_exc())
    
    def task_start(self, task: str):
        """Log task start."""
        separator = "=" * 70
        msg = f"\n{separator}\nNEW TASK: {task}\n{separator}"
        print(f"\033[1m{msg}\033[0m")  # Bold
        self.file_logger.info(msg)
    
    def task_complete(self, actions_taken: int, summary: str = ""):
        """Log task completion."""
        separator = "=" * 70
        msg = f"""
{separator}
TASK COMPLETE
  Actions: {actions_taken}
  API Calls: {self.api_call_count}
  Vision Frames: {self.vision_frame_count}
{separator}
"""
        print(f"\033[1;92m{msg}\033[0m")  # Bold green
        self.file_logger.info(msg)
        if summary:
            self.file_logger.info(f"Summary: {summary}")
    
    def state_update(self, device: str, position: Dict = None, rotation: Dict = None):
        """Log device state changes."""
        data = {}
        if position:
            data["position"] = position
        if rotation:
            data["rotation"] = rotation
        self._format_and_log("INFO", f"STATE:{device.upper()}", "Pose updated", data)


# Global logger instance
_logger: Optional[AgentLogger] = None

def get_logger() -> AgentLogger:
    """Get or create the global logger."""
    global _logger
    if _logger is None:
        _logger = AgentLogger()
    return _logger

def debug_print(msg: str):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        get_logger().info(msg)

# ============================================================================
# System Prompt for the VR Agent
# ============================================================================

SYSTEM_PROMPT = """You are an intelligent VR Agent that controls a virtual reality headset and two controllers through an MCP server.

## CRITICAL: Be Efficient with API Calls
- **DO NOT** use vision tools for simple movement commands (move, turn, look)
- **ONLY** use vision when explicitly asked to "look", "see", "observe", or when you need to find something
- For simple commands like "move back", "turn left", "go forward" - just execute the movement directly in ONE call
- Avoid verification steps unless the task requires visual confirmation

## Devices You Control
- **Headset**: VR display/camera - your position in VR space
- **Controller1**: Left hand controller
- **Controller2**: Right hand controller

## Movement Methods

### Direct Position Movement (DEFAULT for simple commands)
- `move_relative`: Move relative to current position - USE THIS for "move back/forward/left/right"
- `teleport`: Instant move to exact coordinates
- `walk_path`: Smooth walking to destination

### Joystick Locomotion (for VR games with locomotion systems)
- `move_joystick_direction`: Push joystick in a direction
- Only use if you know the app uses joystick locomotion

## Simple Command Mappings
- "move back" → `move_relative(device="headset", dz=1)` (positive Z = backward)
- "move forward" → `move_relative(device="headset", dz=-1)` (negative Z = forward)
- "move left" → `move_relative(device="headset", dx=-1)`
- "move right" → `move_relative(device="headset", dx=1)`
- "turn left" → `rotate_device(device="headset", yaw=-45, pitch=0, roll=0)`
- "turn right" → `rotate_device(device="headset", yaw=45, pitch=0, roll=0)`
- "look up" → `rotate_device(device="headset", pitch=-20, yaw=0, roll=0)`
- "look down" → `rotate_device(device="headset", pitch=20, yaw=0, roll=0)`

## Coordinate System
- X: Left (-) / Right (+)
- Y: Down (-) / Up (+)  
- Z: Forward (-) / Backward (+)

## When to Use Vision
- User asks "what do you see?" or "look around"
- User asks to find or locate something
- User asks to interact with a specific object you need to identify
- **NOT** for simple movement commands

## Controller Inputs
- Buttons: trigger, grip, menu, system, trackpad, a, b
- Analog: trigger value (0-1), joystick X/Y (-1 to 1)
"""

# ============================================================================
# MCP Tool Definitions
# ============================================================================

MCP_TOOLS_DEFINITIONS = [
    # # Connection
    # {
    #     "name": "start_vr_bridge",
    #     "description": "Start the TCP server to listen for the OpenVR driver. Call this first before any other actions.",
    #     "parameters": {"type": "object", "properties": {}, "required": []}
    # },
    # {
    #     "name": "get_connection_status", 
    #     "description": "Check if the VR driver is connected to SteamVR.",
    #     "parameters": {"type": "object", "properties": {}, "required": []}
    # },
    
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
        # if USE_NEW_SDK:
        self._init_new_sdk(api_key)
        # else:
        #     self._init_old_sdk(api_key)
        
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
            # Use v1alpha API version for media_resolution support
            # Add timeout to prevent hanging on slow responses
            self.client = genai.Client(
                api_key=api_key,
                http_options={
                    'api_version': 'v1alpha',
                    'timeout': API_TIMEOUT_MS,
                }
            )
            debug_print(f"Client created: {type(self.client)} with v1alpha API and {API_TIMEOUT_MS/1000}s timeout")
            
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
    
    def _execute_function_call(self, function_name: str, function_args: Dict) -> tuple:
        """
        Execute a function call and return the result.
        Returns: (result_text, image_parts) where image_parts is a list of image Part objects for Gemini
        """
        logger = get_logger()
        
        # Execute the tool
        result = self.executor.call_tool(function_name, function_args)
        
        # Log the action with full details
        logger.action(function_name, function_args, result)
        
        # Track the action
        self.state.actions_taken.append({
            "tool": function_name,
            "args": function_args,
            "result": result[:500] if len(result) > 500 else result
        })
        
        # Extract images from vision tool results for Gemini
        image_parts = self._extract_images_from_result(function_name, result)
        
        return result, image_parts
    
    def _extract_images_from_result(self, function_name: str, result: str) -> list:
        """
        Extract base64 images from vision tool results and convert to Gemini image parts.
        Returns a list of (description, types.Part) tuples using the new Gemini 3 API format.
        Uses inline_data with types.Blob and media_resolution for optimal quality.
        """
        image_parts = []
        
        # Only process vision-related tools
        vision_tools = ["inspect_surroundings", "capture_video", "look_around_and_observe"]
        if function_name not in vision_tools:
            return image_parts
        
        try:
            result_data = json.loads(result)
            if not isinstance(result_data, dict):
                return image_parts
            
            result_type = result_data.get('type', '')
            
            # Determine media resolution based on content type
            # For still images: use high resolution for best quality
            # For video frames: use medium (treated as 70 tokens per frame)
            is_video = result_type == 'video'
            resolution_level = "media_resolution_medium" if is_video else "media_resolution_high"
            
            # Handle panorama scan (look_around_and_observe) - multiple directions
            if result_type == 'panorama_scan' and 'directions' in result_data:
                for direction in result_data['directions']:
                    angle = direction.get('angle', 0)
                    b64_data = direction.get('data')
                    if b64_data:
                        try:
                            img_bytes = base64.b64decode(b64_data)
                            # Use new Gemini 3 API format with inline_data and media_resolution
                            image_part = types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=img_bytes,
                                ),
                                media_resolution={"level": resolution_level}
                            )
                            image_parts.append((f"View at {angle}°", image_part))
                        except Exception as e:
                            debug_print(f"Failed to decode image at angle {angle}: {e}")
            
            # Handle video (capture_video) - multiple frames
            elif result_type == 'video' and 'frames' in result_data:
                frames = result_data['frames']
                # Only send a subset of frames to avoid overwhelming the model
                max_frames = min(5, len(frames))
                step = max(1, len(frames) // max_frames)
                for i in range(0, len(frames), step):
                    if len(image_parts) >= max_frames:
                        break
                    b64_data = frames[i]
                    try:
                        img_bytes = base64.b64decode(b64_data)
                        # Use medium resolution for video frames (70 tokens per frame)
                        image_part = types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg",
                                data=img_bytes,
                            ),
                            media_resolution={"level": "media_resolution_medium"}
                        )
                        image_parts.append((f"Frame {i+1}", image_part))
                    except Exception as e:
                        debug_print(f"Failed to decode video frame {i}: {e}")
            
            # Handle single image (inspect_surroundings)
            elif result_type == 'image' and 'data' in result_data:
                b64_data = result_data['data']
                try:
                    img_bytes = base64.b64decode(b64_data)
                    # Use high resolution for single images (1120 tokens)
                    image_part = types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=img_bytes,
                        ),
                        media_resolution={"level": "media_resolution_high"}
                    )
                    image_parts.append(("Current view", image_part))
                except Exception as e:
                    debug_print(f"Failed to decode image: {e}")
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            debug_print(f"Error extracting images: {e}")
        
        return image_parts
    
    def execute_task(self, task: str, max_iterations: int = 20) -> str:
        """
        Execute a VR task with planning and verification.
        """
        logger = get_logger()
        logger.task_start(task)
        
        # Reset state
        self.state = AgentState(task=task)
        
        if USE_NEW_SDK:
            return self._execute_task_new_sdk(task, max_iterations)
        else:
            return self._execute_task_old_sdk(task, max_iterations)
    
    def _execute_task_new_sdk(self, task: str, max_iterations: int) -> str:
        """Execute task using new google-genai SDK."""
        logger = get_logger()
        logger.info("Starting task execution with new SDK")
        
        prompt = f"""Execute this VR task: {task}

Be efficient - for simple movement commands, just execute the action directly without checking status or using vision.
Only use vision tools if the task requires seeing something."""

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
                logger.info(f"Iteration {iteration}/{max_iterations}")
                
                # Rate limiting - wait if needed before making API call
                rate_limiter = get_rate_limiter()
                rate_limiter.acquire(logger)
                
                try:
                    # Log the full request to LLM conversation log
                    logger.log_llm_request(contents, iteration)
                    
                    # Generate response
                    # Note: Gemini 3 recommends temperature=1.0 (default) for optimal reasoning
                    logger.api_call("REQUEST", f"Sending to {self.model_name} (RPM: {rate_limiter.get_current_rpm()}/{rate_limiter.max_rpm})")
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            tools=[tool_config],
                            temperature=1.0,  # Gemini 3 optimal - don't change!
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                        )
                    )
                    logger.api_call("RESPONSE", "Received from Gemini")
                    
                    # Log the full response to LLM conversation log
                    logger.log_llm_response(response, iteration)
                finally:
                    rate_limiter.release()
                
                # Check for function calls
                has_function_call = False
                text_response = ""
                
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            has_function_call = True
                            fc = part.function_call
                            
                            # Execute the function and get any images
                            args = dict(fc.args) if fc.args else {}
                            result, image_parts = self._execute_function_call(fc.name, args)
                            
                            # Add assistant response to contents
                            contents.append(candidate.content)
                            
                            # Build the function response parts
                            # For vision tools, include a simplified text result + actual images
                            response_parts = []
                            
                            if image_parts:
                                # For vision results, send a brief text description + actual images
                                # This allows Gemini to actually SEE the images
                                num_images = len(image_parts)
                                image_descriptions = [desc for desc, _ in image_parts]
                                
                                response_parts.append(types.Part(
                                    function_response=types.FunctionResponse(
                                        name=fc.name,
                                        response={
                                            "status": "success",
                                            "message": f"Captured {num_images} image(s): {', '.join(image_descriptions)}. The images are provided below for your analysis."
                                        }
                                    )
                                ))
                                
                                # Add each image as a separate part with a label
                                for desc, img_part in image_parts:
                                    response_parts.append(types.Part(text=f"\n[Image: {desc}]"))
                                    response_parts.append(img_part)
                            else:
                                # Non-vision tool, just return the text result
                                response_parts.append(types.Part(
                                    function_response=types.FunctionResponse(
                                        name=fc.name,
                                        response={"result": result}
                                    )
                                ))
                            
                            contents.append(types.Content(
                                role="user",
                                parts=response_parts
                            ))
                            break
                        
                        if hasattr(part, 'text') and part.text:
                            text_response += part.text
                            # Log the agent's thinking/reasoning
                            logger.thinking(part.text)
                    
                    if has_function_call:
                        break
                
                if not has_function_call:
                    # No more function calls, we're done
                    self.state.completed = True
                    rate_stats = get_rate_limiter().get_stats()
                    logger.info(f"Rate limiter stats: {rate_stats}")
                    logger.task_complete(len(self.state.actions_taken), text_response[:200])
                    return text_response
            
            logger.info(f"Max iterations ({max_iterations}) reached")
            rate_stats = get_rate_limiter().get_stats()
            logger.info(f"Rate limiter stats: {rate_stats}")
            return "Max iterations reached"
            
        except Exception as e:
            logger.error(f"Error during task execution: {e}", e)
            return f"Error: {str(e)}"
    
    def _execute_task_old_sdk(self, task: str, max_iterations: int) -> str:
        """Execute task using deprecated SDK."""
        logger = get_logger()
        logger.info("Executing task with old SDK (limited functionality)")
        
        prompt = f"""Execute this VR task: {task}

First, observe the current state, then plan your approach, and execute step by step."""

        try:
            self.chat = self.model.start_chat()
            
            # Rate limiting
            rate_limiter = get_rate_limiter()
            rate_limiter.acquire(logger)
            
            try:
                logger.api_call("REQUEST", "Sending to Gemini (old SDK)")
                response = self.chat.send_message(prompt)
                logger.api_call("RESPONSE", "Received from Gemini")
            finally:
                rate_limiter.release()
            
            # Simple text response for old SDK (function calling is more complex)
            text = ""
            for part in response.parts:
                if hasattr(part, 'text'):
                    text += part.text
                    logger.thinking(part.text)
            
            logger.task_complete(0, text[:200])
            return text
            
        except Exception as e:
            logger.error(f"Error: {e}", e)
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
    
    # Check for API key (loaded from .env or system environment)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\nError: GEMINI_API_KEY not found.")
        print("\nOption 1 - Create a .env file in this directory:")
        print("  GEMINI_API_KEY=your_key_here")
        print("\nOption 2 - Set environment variable:")
        print("  Windows CMD: set GEMINI_API_KEY=your_key_here")
        print("  Windows PowerShell: $env:GEMINI_API_KEY='your_key_here'")
        print("  Linux/Mac: export GEMINI_API_KEY=your_key_here")
        print("\nGet your API key from: https://aistudio.google.com/app/apikey")
        return
    
    print(f"API key loaded (length: {len(api_key)})")
    
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
