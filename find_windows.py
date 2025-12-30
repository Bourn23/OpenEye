"""List all visible windows to find the SteamVR mirror window title"""
import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32

# Callback type
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

def get_window_title(hwnd):
    length = user32.GetWindowTextLengthW(hwnd)
    if length > 0:
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    return ""

def get_window_class(hwnd):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value

def is_window_visible(hwnd):
    return user32.IsWindowVisible(hwnd)

def get_window_rect(hwnd):
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return (rect.right - rect.left, rect.bottom - rect.top)

windows = []

def enum_callback(hwnd, lparam):
    if is_window_visible(hwnd):
        title = get_window_title(hwnd)
        if title:  # Only windows with titles
            width, height = get_window_rect(hwnd)
            if width > 50 and height > 50:  # Skip tiny windows
                class_name = get_window_class(hwnd)
                windows.append({
                    'hwnd': hwnd,
                    'title': title,
                    'class': class_name,
                    'size': f"{width}x{height}"
                })
    return True

# Enumerate all windows
user32.EnumWindows(EnumWindowsProc(enum_callback), 0)

# Print results, filtering for potentially relevant windows
print("=" * 80)
print("All visible windows with titles:")
print("=" * 80)

# Look for VR-related windows
vr_keywords = ['steam', 'vr', 'mirror', 'headset', 'openvr', 'compositor', 'overlay', 'null']
vr_windows = []
other_windows = []

for w in windows:
    title_lower = w['title'].lower()
    class_lower = w['class'].lower()
    if any(kw in title_lower or kw in class_lower for kw in vr_keywords):
        vr_windows.append(w)
    else:
        other_windows.append(w)

print("\n*** VR-RELATED WINDOWS ***")
for w in vr_windows:
    print(f"  Title: '{w['title']}'")
    print(f"  Class: '{w['class']}'")
    print(f"  Size:  {w['size']}")
    print(f"  HWND:  {w['hwnd']}")
    print()

print("\n*** OTHER WINDOWS (first 20) ***")
for w in other_windows[:20]:
    print(f"  '{w['title']}' ({w['class']}) - {w['size']}")

print(f"\nTotal windows found: {len(windows)}")
print(f"VR-related: {len(vr_windows)}")
