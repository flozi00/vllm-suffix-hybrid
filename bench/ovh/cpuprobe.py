"""cpuprobe.py <url> <secs>: sample vLLM /metrics process_cpu_seconds_total -> cores used."""
import re, ssl, sys, time, urllib.request, socket, subprocess
_g = socket.getaddrinfo
def g(h, *a, **k):
    try: return _g(h, *a, **k)
    except socket.gaierror:
        ip = subprocess.run(["host", h], capture_output=True, text=True).stdout.split()[-1]; return _g(ip, *a, **k)
socket.getaddrinfo = g
ctx = ssl._create_unverified_context()
def cpu():
    t = urllib.request.urlopen(sys.argv[1] + "/metrics", context=ctx, timeout=20).read().decode()
    m = re.search(r"^process_cpu_seconds_total(?:\{[^}]*\})? ([0-9.e+]+)", t, re.M)
    return float(m.group(1)) if m else None
prev, tp = cpu(), time.time()
end = time.time() + float(sys.argv[2])
while time.time() < end:
    time.sleep(10)
    c, t = cpu(), time.time()
    print(f"api-server cores: {(c - prev) / (t - tp):.2f}", flush=True)
    prev, tp = c, t
