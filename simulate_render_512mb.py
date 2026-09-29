import os
import sys
import time
import ctypes
from ctypes import wintypes

# Windows Process Memory Query setup
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

def get_ram_mb():
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    h_proc = kernel32.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(counters), counters.cb):
        return counters.WorkingSetSize / (1024 * 1024), counters.PeakWorkingSetSize / (1024 * 1024)
    return 0.0, 0.0

def print_stage(stage_name, ram_mb, peak_mb, limit_mb=512.0):
    pct = (peak_mb / limit_mb) * 100.0
    bar_len = 30
    filled = int(bar_len * (peak_mb / limit_mb))
    bar = "#" * filled + "-" * (bar_len - filled)
    status = "PASS" if peak_mb < limit_mb else "OOM CRASH"
    print(f"\n[{stage_name}]")
    print(f"  Current RAM : {ram_mb:6.2f} MB")
    print(f"  Peak RAM    : {peak_mb:6.2f} MB / {limit_mb:.0f} MB [{bar}] {pct:5.1f}%")
    print(f"  Status      : {status}")

def run_simulation():
    LIMIT_MB = 512.0
    print("=" * 60)
    print(f"  RENDER 512 MB MEMORY & PERFORMANCE SIMULATION")
    print("=" * 60)

    cur, peak = get_ram_mb()
    print_stage("1. Python Process Startup", cur, peak, LIMIT_MB)

    # Stage 2: App Import
    t0 = time.time()
    import app
    t_boot = time.time() - t0
    cur, peak = get_ram_mb()
    print_stage(f"2. Flask App & Libraries Boot ({t_boot:.2f}s)", cur, peak, LIMIT_MB)

    client = app.app.test_client()

    # Stage 3: Team EPM Analysis
    t0 = time.time()
    team_html_path = "scripts/TeamEPMContent/533-2053.htm"
    if os.path.exists(team_html_path):
        with open(team_html_path, "r", encoding="utf-8") as f:
            team_html = f.read()
        from scripts.TeamEPMContent.getTeamPlayerStatsFlask import get_team_player_stats
        get_team_player_stats(team_html)
    t_team = time.time() - t0
    cur, peak = get_ram_mb()
    print_stage(f"3. Team Performance EPM Analysis ({t_team:.2f}s)", cur, peak, LIMIT_MB)

    # Stage 4: Player Prediction (Mock Player)
    t0 = time.time()
    from scripts.flaskGetPredictedStats import givePlayerStats
    sample_html = """
    <html>
    <body>
      <h1>John Doe</h1>
      <table>
        <tr><td>#12 John Doe</td></tr>
        <tr><td>Age: 20</td></tr>
        <tr><td>Outside Shot: 65</td></tr>
        <tr><td>Height: 6' 6"</td></tr>
        <tr><td>Inside Shot: 55</td></tr>
        <tr><td>Wingspan: 6' 10"</td></tr>
        <tr><td>Vertical: 34"</td></tr>
        <tr><td>Weight: 210 lbs.</td></tr>
        <tr><td>Finishing: 60 Passing: 50 Shooting Range: 65</td></tr>
        <tr><td>Ball Handling: 55 Driving: 58 Rebounding: 50</td></tr>
        <tr><td>Strength: 60 Interior Defense: 50 Speed: 70</td></tr>
        <tr><td>Perimeter Defense: 65 Stamina: 75 Basketball IQ: 60</td></tr>
      </table>
    </body>
    </html>
    """
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        try:
            givePlayerStats(sample_html, pos, from_file=True)
        except Exception as e:
            print(f"  [Warning: givePlayerStats({pos}) encountered error: {e}]")
            break
    t_player = time.time() - t0
    cur, peak = get_ram_mb()
    print_stage(f"4. Player Stat Prediction ({t_player:.2f}s)", cur, peak, LIMIT_MB)

    # Summary
    print("\n" + "=" * 60)
    print("  SIMULATION RESULTS")
    print("=" * 60)
    headroom = LIMIT_MB - peak
    headroom_pct = (headroom / LIMIT_MB) * 100.0
    print(f"  Maximum Peak RAM   : {peak:.2f} MB")
    print(f"  Render Memory Limit: {LIMIT_MB:.2f} MB")
    print(f"  Free Headroom      : {headroom:.2f} MB ({headroom_pct:.1f}% safety margin)")
    if peak < LIMIT_MB:
        print(f"  FINAL VERDICT      : PASS - READY FOR RENDER (Under 512 MB)")
    else:
        print(f"  FINAL VERDICT      : FAIL - WILL CRASH ON RENDER (Exceeds 512 MB)")
    print("=" * 60)

if __name__ == "__main__":
    run_simulation()
