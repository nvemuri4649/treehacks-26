"""
Backend resolver — picks the right GPU server URL from config or CLI args.
===========================================================================
Supports:
    --backend gx10       Look up named backend from backends.json
    --backend runpod     Look up RunPod backend (substitutes pod_id into URL)
    --server http://...  Direct URL override (ignores backends.json)

Environment variable overrides:
    DIFFGUARD_SERVER     Fallback URL if neither --backend nor --server given
    RUNPOD_POD_ID        RunPod pod ID (overrides backends.json pod_id)
    RUNPOD_API_KEY       RunPod API key
"""

import json
import os
import sys

# Path to backends.json (at repo root)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "backends.json")


def load_backends_config() -> dict:
    """Load backends.json, return empty structure if missing."""
    path = os.path.abspath(_CONFIG_PATH)
    if not os.path.isfile(path):
        return {"backends": {}, "default": "local"}
    with open(path) as f:
        return json.load(f)


def resolve_backend(backend_name: str | None, server_url: str | None) -> str:
    """
    Resolve a server URL from the given arguments.

    Priority:
        1. --server <url>         (explicit URL, always wins)
        2. --backend <name>       (look up in backends.json)
        3. DIFFGUARD_SERVER env   (fallback)
        4. default from config    (last resort)

    Returns the fully-resolved HTTP URL string.
    """
    # 1. Explicit URL
    if server_url:
        return server_url.rstrip("/")

    # 2. Named backend
    config = load_backends_config()
    name = backend_name or os.environ.get("DIFFGUARD_BACKEND") or config.get("default", "local")
    backends = config.get("backends", {})

    if name not in backends:
        # Maybe it's a raw URL passed as --backend
        if name.startswith("http://") or name.startswith("https://"):
            return name.rstrip("/")
        available = ", ".join(backends.keys()) or "(none configured)"
        print(f"ERROR: Unknown backend '{name}'. Available: {available}")
        print(f"  Edit backends.json to add it, or use --server <url> directly.")
        sys.exit(1)

    entry = backends[name]
    url = entry.get("url", "")

    # Handle RunPod URL template
    if entry.get("type") == "runpod":
        pod_id = os.environ.get("RUNPOD_POD_ID") or entry.get("pod_id", "")
        if not pod_id:
            print(f"ERROR: RunPod backend '{name}' requires a pod_id.")
            print(f"  Set RUNPOD_POD_ID env var, or edit pod_id in backends.json.")
            sys.exit(1)
        url = url.replace("{POD_ID}", pod_id)

    if not url:
        print(f"ERROR: Backend '{name}' has no URL configured.")
        sys.exit(1)

    return url.rstrip("/")


def add_backend_args(parser):
    """
    Add --backend and --server arguments to an argparse parser.
    These are mutually supportive (--server overrides --backend).
    """
    config = load_backends_config()
    available = list(config.get("backends", {}).keys())
    default_name = config.get("default", "local")

    backend_help = f"Named backend from backends.json (available: {', '.join(available) or 'none'}; default: {default_name})"
    parser.add_argument(
        "--backend", "-b",
        default=None,
        help=backend_help,
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Direct server URL (overrides --backend)",
    )


def get_server_url(args) -> str:
    """
    Convenience: resolve backend from parsed argparse args.
    Expects args to have .backend and .server attributes (from add_backend_args).
    """
    url = resolve_backend(
        getattr(args, "backend", None),
        getattr(args, "server", None),
    )
    return url


def print_backend_info(url: str, name: str | None = None):
    """Print which backend is being used."""
    label = f"'{name}' " if name else ""
    print(f"  Backend: {label}{url}")
