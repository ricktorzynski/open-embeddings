#!/usr/bin/env python3
"""Setup script for quick start demo."""

import subprocess
import sys
import time
import requests


def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0


def wait_for_server(url="http://localhost:8765/health", timeout=30):
    """Wait for server to become healthy."""
    print("Waiting for server to start...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.RequestException:
            pass

        time.sleep(2)

    return False


def main():
    """Run the setup and demo."""
    print("=== Open Embeddings Setup ===")

    # Install dependencies
    if not run_command("pip install -e ."):
        print("Failed to install dependencies")
        sys.exit(1)

    # Start server in background
    print("\nStarting server...")
    server_process = subprocess.Popen([
        sys.executable, "-m", "open_embeddings.main"
    ])

    try:
        # Wait for server to be ready
        if not wait_for_server():
            print("Server failed to start within timeout")
            return

        # Run example
        print("\nRunning client example...")
        run_command("python examples/client_example.py")

        print("\n=== Setup Complete! ===")
        print("Open Embeddings is running at http://localhost:8765")
        print("Visit http://localhost:8765/docs for API documentation")
        print("Press Ctrl+C to stop the server")

        # Keep server running
        server_process.wait()

    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server_process.terminate()
        server_process.wait()


if __name__ == "__main__":
    main()