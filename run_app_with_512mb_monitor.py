import os
import sys
import time
import threading
import ctypes
from ctypes import wintypes

# Windows Memory Query setup
psapi = ctypes.WinDLL('psapi')
kernel32 = ctypes.WinDLL('kernel32')

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]

psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

LIMIT_MB = 512.0
stop_monitor = False

def get_ram_mb():
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    h_proc = kernel32.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(counters), counters.cb):
        return counters.WorkingSetSize / (1024 * 1024), counters.PeakWorkingSetSize / (1024 * 1024)
    return 0.0, 0.0

def memory_watcher():
    last_printed_peak = 0.0
    while not stop_monitor:
        cur, peak = get_ram_mb()
        if peak > last_printed_peak + 5.0 or cur > last_printed_peak:
            last_printed_peak = max(peak, cur)
            pct = (last_printed_peak / LIMIT_MB) * 100.0
            status = "OK" if last_printed_peak < LIMIT_MB else "EXCEEDED 512 MB CAP!"
            bar_len = 20
            filled = min(bar_len, int(bar_len * (last_printed_peak / LIMIT_MB)))
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\n[RAM MONITOR] Current: {cur:5.1f} MB | Peak: {peak:5.1f} MB / {LIMIT_MB:.0f} MB [{bar}] {pct:4.1f}% -> {status}")
        time.sleep(0.5)

if __name__ == "__main__":
    print("=" * 65)
    print("  HARDWOOD APP - LIVE 512 MB MEMORY WATCHER")
    print(f"  Enforcing Render 512 MB Cap Simulation")
    print(f"  Open in your browser: http://localhost:5002")
    print("=" * 65)

    # Start memory watcher thread
    watcher_thread = threading.Thread(target=memory_watcher, daemon=True)
    watcher_thread.start()

    # Import and run app
    from app import app
    try:
        app.run(port=5002, debug=False)
    except KeyboardInterrupt:
        stop_monitor = True
        print("\nStopping server...")
