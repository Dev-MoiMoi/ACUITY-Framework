import os
import sys
import time
import random

from setuptools import setup
from setuptools.command.build_py import build_py

BANNER = r"""
     _   ___  ___ ___ ____ ___  _____
    /_\ / _ \/ __|_  |_   _/ _ \_   _|
   / _ \ (_) | (__ / /  | || (_) || |
  /_/ \_\___/ \___/___| |_| \___/ |_|
  _   ___  _____ ___ ___  ___
 /_\ / __||_   _|_ _| _ \/ _ \
/ _ \\__ \  | |  | ||  _/ (_) |
/_/ \_\___/ |_| |___|_|  \___/
"""

FUN_MESSAGES = [
    "Brewing the finest pandesal...",
    "Warming up the neurons...",
    "Asking Mang Juan for directions...",
    "Calibrating the Haversine compass...",
    "Folding TF-IDF vectors...",
    "Coaxing the CRF model awake...",
    "Sweeping the sari-sari store for NLTK...",
    "Whetting the cosine blades...",
    "Teleporting between barangays...",
    "Sharpening the fuzzy matcher...",
    "Tying a knot on the registry ribbon...",
    "Near done, kuya / ate...",
]

BAR_WIDTH = 32

ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_BOLD = "\033[1m"


def _use_color():
    return sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""


def _color(text, code):
    return f"{code}{text}{ANSI_RESET}" if _use_color() else text


def _is_fun_enabled():
    return os.environ.get("ACUITY_NO_FUN", "") == ""


def _animate(duration=1.5):
    if not sys.stdout.isatty():
        _static_bar()
        return
    deadline = time.monotonic() + duration
    message = random.choice(FUN_MESSAGES)
    ticks = 0
    while time.monotonic() < deadline:
        ticks += 1
        filled = (ticks % (BAR_WIDTH - 1)) + 1
        bar = "=" * (filled - 1) + ">" + " " * (BAR_WIDTH - filled)
        status = message if ticks % 4 == 0 else " "
        sys.stdout.write(
            f"\r  [{bar}] {status[:34]:<34}"
        )
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write("\r" + " " * (BAR_WIDTH + 44) + "\r")
    sys.stdout.flush()


def _static_bar():
    message = random.choice(FUN_MESSAGES)
    bar = "=" * (BAR_WIDTH - 1) + ">"
    print(f"  [{bar}] {message}")


class AcuityBuild(build_py):
    def run(self):
        if _is_fun_enabled():
            print(_color(ANSI_BOLD + BANNER, ANSI_CYAN))
            print(_color("  ACUITY installer v1.0.0 — extracting local visibility...", ANSI_YELLOW))
            print()
            _animate()
            print(_color("  Installing ACUITY Framework...", ANSI_YELLOW))
            print()
        super().run()
        if _is_fun_enabled():
            print()
            print(_color("  " + "=" * 44, ANSI_GREEN))
            print(_color("  ACUITY installed successfully! Shine that visibility!", ANSI_BOLD + ANSI_GREEN))
            print(_color("  " + "=" * 44, ANSI_GREEN))


setup(cmdclass={"build_py": AcuityBuild})
