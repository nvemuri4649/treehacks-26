#!/usr/bin/env python3
"""
Setup Checker for Agent Loop
==============================
Verifies that all dependencies and configurations are ready for running
the adversarial agent loop.

Usage:
    python client/check_setup.py
    python client/check_setup.py --backend gx10
"""

import argparse
import os
import sys


def check_color(status: bool) -> str:
    """Return colored checkmark or X."""
    return "\033[92m✓\033[0m" if status else "\033[91m✗\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Check agent loop setup")
    parser.add_argument("--backend", default=None, help="Backend to check (gx10, runpod, local)")
    args = parser.parse_args()

    print("=" * 70)
    print("AGENT LOOP SETUP CHECKER")
    print("=" * 70)
    print()

    all_good = True

    # 1. Check Python packages
    print("[1] Checking Python dependencies...")

    packages = {
        "requests": "requests",
        "PIL": "pillow",
        "anthropic": "anthropic",
    }

    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  {check_color(True)} {package} installed")
        except ImportError:
            print(f"  {check_color(False)} {package} NOT installed")
            print(f"      Install with: pip install {package}")
            all_good = False

    print()

    # 2. Check Anthropic API key
    print("[2] Checking Anthropic API key...")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        masked = f"{api_key[:10]}...{api_key[-4:]}"
        print(f"  {check_color(True)} ANTHROPIC_API_KEY set ({masked})")
    else:
        print(f"  {check_color(False)} ANTHROPIC_API_KEY not set")
        print(f"      Get your key from: https://console.anthropic.com/")
        print(f"      Then run: export ANTHROPIC_API_KEY='sk-ant-api03-...'")
        all_good = False

    print()

    # 3. Check backends.json
    print("[3] Checking backend configuration...")
    backends_path = os.path.join(os.path.dirname(__file__), "..", "backends.json")
    if os.path.exists(backends_path):
        print(f"  {check_color(True)} backends.json found")

        try:
            import json
            with open(backends_path) as f:
                config = json.load(f)
            backends = config.get("backends", {})
            default = config.get("default", "")
            print(f"      Available backends: {', '.join(backends.keys())}")
            print(f"      Default: {default}")
        except Exception as e:
            print(f"  {check_color(False)} Error reading backends.json: {e}")
            all_good = False
    else:
        print(f"  {check_color(False)} backends.json not found")
        all_good = False

    print()

    # 4. Check GPU backend connectivity (if specified)
    if args.backend:
        print(f"[4] Checking GPU backend connectivity ({args.backend})...")

        try:
            import requests
            from backends import resolve_backend

            server_url = resolve_backend(args.backend, None)
            print(f"      Server URL: {server_url}")

            try:
                resp = requests.get(f"{server_url}/health", timeout=5)
                if resp.status_code == 200:
                    info = resp.json()
                    print(f"  {check_color(True)} Server is reachable")
                    print(f"      GPU: {info.get('gpu', '?')}")
                    print(f"      Memory: {info.get('gpu_memory', '?')}")
                    print(f"      Model loaded: {info.get('model_loaded', '?')}")
                else:
                    print(f"  {check_color(False)} Server returned {resp.status_code}")
                    all_good = False
            except requests.exceptions.ConnectionError:
                print(f"  {check_color(False)} Cannot connect to server")
                print(f"      Make sure the DiffusionGuard server is running")
                all_good = False
            except requests.exceptions.Timeout:
                print(f"  {check_color(False)} Connection timed out")
                all_good = False

        except Exception as e:
            print(f"  {check_color(False)} Error: {e}")
            all_good = False

        print()

    # 5. Test Claude API (if key is set)
    if api_key:
        print("[5] Testing Claude API connection...")
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            # Simple test message
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}],
            )

            if message.content:
                print(f"  {check_color(True)} Claude API is working")
                print(f"      Model: claude-opus-4-6")
            else:
                print(f"  {check_color(False)} Unexpected response from Claude API")
                all_good = False

        except Exception as e:
            print(f"  {check_color(False)} Claude API error: {e}")
            print(f"      Check your API key is valid")
            all_good = False

        print()

    # Summary
    print("=" * 70)
    if all_good:
        print("\033[92m✓ ALL CHECKS PASSED\033[0m")
        print()
        print("You're ready to run the agent loop!")
        print()
        print("Quick start:")
        print("  python client/agent_loop.py \\")
        print("      --image face.png \\")
        print("      --mask mask.png \\")
        print("      --prompt 'a person in jail' \\")
        print("      --threshold 6")
        if args.backend:
            print(f"      --backend {args.backend}")
    else:
        print("\033[91m✗ SOME CHECKS FAILED\033[0m")
        print()
        print("Fix the issues above before running the agent loop.")
        print("See AGENT_LOOP.md for full setup instructions.")

    print("=" * 70)

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
