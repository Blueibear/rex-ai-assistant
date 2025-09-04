#!/usr/bin/env python
"""
rex_launcher.py
---------------

This script is a launcher for advanced users who want to run Rex with a
local LLM inference server (e.g. KoboldCpp).  It prompts the user to
select a user profile from the ``Memory`` directory, displays some
information about the user, and then launches the inference server.

This script is provided as a starting point and may need to be
customized to fit your local setup.

Configuration:
    - KOBOLDCPP_PATH: Path to your KoboldCpp executable.
    - MODEL_PATH: Path to the GGUF model file you want to use.
"""

import os
import json
import subprocess
from datetime import datetime

# --- User Configuration ---
# TODO: Set these paths to match your local setup.
KOBOLDCPP_PATH = "path/to/your/koboldcpp.exe"
MODEL_PATH = "path/to/your/model.gguf"
# --------------------------


def main():
    """Main function to select user and launch the LLM server."""
    # The "Memory" directory is expected to be in the same directory as this script.
    script_dir = os.path.dirname(__file__)
    memory_dir = os.path.join(script_dir, "Memory")

    if not os.path.isdir(memory_dir):
        print(f"Error: Could not find the 'Memory' directory at '{memory_dir}'")
        return

    # Choose user
    try:
        users = [d for d in os.listdir(memory_dir) if os.path.isdir(os.path.join(memory_dir, d))]
        if not users:
            print("No user profiles found in the 'Memory' directory.")
            return
    except FileNotFoundError:
        print(f"Error: Could not list users in '{memory_dir}'.")
        return

    print("Choose your user:")
    for i, user in enumerate(users):
        print(f"{i + 1}. {user.capitalize()}")

    try:
        choice = input(f"Enter a number (1-{len(users)}): ")
        selected_user = users[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Exiting.")
        return

    user_dir = os.path.join(memory_dir, selected_user)

    # Load memory
    core_path = os.path.join(user_dir, "core.json")
    history_path = os.path.join(user_dir, "history.log")

    try:
        with open(core_path, "r") as f:
            core_memory = json.load(f)
    except FileNotFoundError:
        print(f"Error: 'core.json' not found for user '{selected_user}'.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode 'core.json' for user '{selected_user}'.")
        return

    # Display who you're logged in as
    name = core_memory.get('name', 'Unknown')
    role = core_memory.get('role', 'Unknown')
    topics = core_memory.get('preferences', {}).get('topics', [])
    print(f"\n✅ Launching session as {name}...\n")
    print(f"Role: {role}")
    print(f"Preferred topics: {', '.join(topics)}")

    # Log session start to history
    with open(history_path, "a") as log:
        log.write(f"\n[{datetime.now().strftime('%Y-%m-%d %I:%M %p')}] Session started for {name}\n")

    # Check if the KoboldCpp path is configured
    if KOBOLDCPP_PATH == "path/to/your/koboldcpp.exe" or not os.path.exists(KOBOLDCPP_PATH):
        print("\n---")
        print("⚠️  Action Required: KoboldCpp path is not configured or is invalid.")
        print(f"Please edit '{__file__}' and set the KOBOLDCPP_PATH variable.")
        print("---\n")
        return

    # Launch KoboldCpp
    print(f"\n🚀 Launching KoboldCpp with model '{os.path.basename(MODEL_PATH)}'...")
    try:
        subprocess.run([
            KOBOLDCPP_PATH,
            "--model", MODEL_PATH,
            "--gpulayers", "99", # Example flag, adjust as needed
            "--threads", "8",
            "--smartcontext",
        ], check=True)
    except FileNotFoundError:
        print(f"Error: Could not find the KoboldCpp executable at '{KOBOLDCPP_PATH}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error launching KoboldCpp: {e}")


if __name__ == "__main__":
    main()
