"""Probe: Modal GPU driver version, CUDA, EGL/OSMesa availability. App t3-tut01-probe."""
import modal

app = modal.App("t3-tut01-probe")
image = modal.Image.debian_slim(python_version="3.12").apt_install("libegl1", "libosmesa6", "libgl1")


@app.function(gpu="T4", image=image, timeout=300)
def probe():
    import glob, subprocess
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    libs = sorted(set(glob.glob("/usr/lib/x86_64-linux-gnu/libEGL_nvidia*") + glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-egl*")
                      + glob.glob("/usr/share/glvnd/egl_vendor.d/*")))
    import os
    return out[:1200], libs, os.cpu_count()


@app.local_entrypoint()
def main():
    print(probe.remote())
