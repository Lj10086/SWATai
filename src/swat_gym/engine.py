"""Find and run the platform-native SWAT+ rev 62.0.0 engine in model/TxtInOut."""
from __future__ import annotations

import os
import platform
import subprocess
import threading
from pathlib import Path

_MAGIC = {
    "macho": (
        b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce",
        b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
    ),
    "elf": (b"\x7fELF",),
    "pe": (b"MZ",),
}
_ALL_MAGIC = tuple(m for group in _MAGIC.values() for m in group)


class EngineError(RuntimeError):
    """The SWAT+ engine could not be found, or exited without producing output."""


def _family(path: Path) -> str | None:
    try:
        head = path.open("rb").read(4)
    except OSError:
        return None
    for name, magics in _MAGIC.items():
        if head.startswith(magics):
            return name
    return None


def _is_backup(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".bak") or ".bak." in n or n.endswith(".old")


def _wanted_family() -> str:
    system = platform.system()
    if system == "Linux":
        return "elf"
    if system == "Darwin":
        return "macho"
    if system == "Windows":
        return "pe"
    return "elf"


def _is_executable_binary(path: Path) -> bool:
    if not path.is_file() or _is_backup(path):
        return False
    if os.name != "nt" and not os.access(path, os.X_OK):
        return False
    return _family(path) is not None


def find_engine(txtinout: Path) -> Path:
    """Platform-native engine in ``txtinout``; ignores *.bak and wrong-OS binaries."""
    wanted = _wanted_family()
    candidates = sorted(
        p for p in txtinout.iterdir()
        if _is_executable_binary(p) and _family(p) == wanted
    )
    if not candidates:
        raise EngineError(
            f"No native SWAT+ executable ({wanted}) found in {txtinout}. "
            f"Place a rev 62.0.0 build for this OS there "
            f"(Mac: Mach-O, Linux: ELF). Backups named *.bak are ignored."
        )
    if len(candidates) > 1:
        # Prefer a name that advertises rev 62.
        preferred = [p for p in candidates if "62" in p.name]
        if len(preferred) == 1:
            return preferred[0]
        names = ", ".join(p.name for p in candidates)
        raise EngineError(f"Multiple native executables in {txtinout}: {names}. Keep one.")
    return candidates[0]


#: The engine's console output (one progress line per simulated day) goes here, not a pipe:
#: a file is ~10 % faster per run. Not an input, so FastRunner deletes it before each run.
ENGINE_LOG = "swat_gym_engine.log"


def run_engine(exe: Path, run_dir: Path, timeout: float = 600.0) -> str:
    """Run ``exe`` with ``run_dir`` as the working directory; return its console output."""
    log = Path(run_dir) / ENGINE_LOG
    with open(log, "wb") as f:
        proc = subprocess.Popen([str(exe)], cwd=run_dir, stdout=f, stderr=subprocess.STDOUT)
        # A blocking wait plus a kill timer: wait(timeout=...) polls with sleeps of up to
        # 50 ms, which cost ~30 ms per ~40 ms run.
        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(timeout, kill)
        timer.start()
        try:
            proc.wait()
        except BaseException:  # e.g. Ctrl-C: don't leave the engine running
            proc.kill()
            proc.wait()
            raise
        finally:
            timer.cancel()
    if timed_out.is_set():
        raise EngineError(f"SWAT+ timed out after {timeout}s in {run_dir}")
    out = log.read_text(errors="replace")
    if proc.returncode != 0:
        raise EngineError(
            f"SWAT+ exited {proc.returncode} in {run_dir}\n"
            f"--- output tail ---\n{out[-3000:]}"
        )
    return out
