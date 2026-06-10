"""Terminal output helpers — color, sections, spinner. No business logic.

_NO_COLOR / _VERBOSE are module-level mutable state; cli.main() sets them via
`from . import output; output._NO_COLOR = ...`. Color functions read this
module's globals, so the change is seen everywhere."""
import os
import sys
import threading
import time


# --------------------------------------------------------------------------- #
# Demo-friendly output helpers — colored, sectioned, subprocess-output-suppressed.
# --------------------------------------------------------------------------- #
_VERBOSE = False    # set from --verbose; suppresses noise from milvus-backup CLI
_NO_COLOR = not (sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb")

def _c(code, s):
    return s if _NO_COLOR else f"\033[{code}m{s}\033[0m"

def _bold(s):  return _c("1",    s)
def _dim(s):   return _c("2",    s)
def _green(s): return _c("1;32", s)
def _red(s):   return _c("1;31", s)
def _yel(s):   return _c("1;33", s)
def _cyan(s):  return _c("1;36", s)
def _grey(s):  return _c("90",   s)

_WIDTH = 60  # alignment column for "……… ✓ (1.2s)"
_step_t = None  # set by step(), read by done()

# spinner state — animates "the script is still working" between step() and done()
_spinner_thread = None
_spinner_stop = None

def _spinner_loop(stop_ev, start_t):
    frames = "|/-\\"
    last = ""
    i = 0
    while not stop_ev.is_set():
        elapsed = time.time() - start_t
        s = f"{frames[i % 4]} ({elapsed:.0f}s)"
        # erase previous, write new — all on the same line, no newline
        sys.stdout.write("\b" * len(last) + " " * len(last) + "\b" * len(last))
        sys.stdout.write(s)
        sys.stdout.flush()
        last = s
        time.sleep(0.1)
        i += 1
    # erase the spinner so done()/fail() can write cleanly
    sys.stdout.write("\b" * len(last) + " " * len(last) + "\b" * len(last))
    sys.stdout.flush()

def _spinner_start():
    """Animate a |/- spinner with elapsed seconds, until _spinner_stop() is called.
    Disabled when output isn't a TTY (e.g. piping to a file) or color is off, so
    captured/logged output stays clean."""
    global _spinner_thread, _spinner_stop
    if _NO_COLOR or not sys.stdout.isatty() or _VERBOSE:
        return
    _spinner_stop = threading.Event()
    _spinner_thread = threading.Thread(
        target=_spinner_loop, args=(_spinner_stop, time.time()), daemon=True)
    _spinner_thread.start()

def _spinner_end():
    global _spinner_thread, _spinner_stop
    if _spinner_thread:
        _spinner_stop.set()
        _spinner_thread.join(timeout=0.5)
        _spinner_thread = None
        _spinner_stop = None

def header(title, subtitle=None):
    _spinner_end()  # kill any spinner left dangling by an error path
    bar = "━" * 4
    print(f"\n{_bold(_cyan(bar + ' ' + title + ' '))}{_dim('━' * max(1, _WIDTH - len(title) - 7))}", flush=True)
    if subtitle:
        for line in subtitle.split("\n"):
            print(f"  {_dim(line)}", flush=True)
        print()

def step(n_of_n, label):
    """Begin a step. Print left-aligned label, no newline; done()/fail() finish it.
    Auto-starts a spinner that animates until done()/fail() is called."""
    global _step_t
    _step_t = time.time()
    pad = label + " " + _dim("…" * max(3, _WIDTH - len(n_of_n) - len(label) - 6))
    print(f"  [{n_of_n}] {pad} ", end="", flush=True)
    _spinner_start()

def done(extra=""):
    _spinner_end()
    dt = time.time() - (_step_t or time.time())
    tail = f" {_dim(f'({dt:.1f}s)')}"
    extra_s = f" {_dim(extra)}" if extra else ""
    print(f"{_green('✓')}{tail}{extra_s}", flush=True)

def fail(msg=""):
    _spinner_end()
    dt = time.time() - (_step_t or time.time())
    print(f"{_red('✗')} {_dim(f'({dt:.1f}s)')} {_red(msg)}", flush=True)

def info(msg):
    _spinner_end()
    print(f"  {_dim('·')} {msg}", flush=True)

def warn(msg):
    _spinner_end()
    print(f"  {_yel('⚠')} {msg}", flush=True)

def kv(k, v, color=None):
    _spinner_end()
    val = v if color is None else color(v)
    print(f"  {_dim(k + ':')} {val}", flush=True)

# Backward-compat: keep the older `log()` so I don't have to touch every call.
# In demo mode it prints to the same demo-friendly format.
def log(msg):
    _spinner_end()
    print(f"  {_dim('·')} {msg}", flush=True)


