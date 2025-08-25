#!/usr/bin/env python3
"""
Monitor running workflows in real-time
"""
import sys  # noqa : 
import time
from datetime import datetime

import requests

BASE_URL = "http://localhost:8000"


def monitor_workflows():
    """Monitor all workflows continuously"""
    print("📊 Flowza Workflow Monitor")
    print("Press Ctrl+C to stop monitoring")

    try:
        while True:
            # Clear screen
            print("\033[2J\033[H")
            print(
                f"Flowza Workflow Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"  # noqa :
            )
            print("=" * 60)

            # Get all workflows
            response = requests.get(f"{BASE_URL}/api/workflows/")
            if response.status_code == 200:
                workflows = response.json()

                if not workflows:
                    print("No workflows found")
                else:
                    for workflow in workflows:
                        status_icon = {
                            "pending": "⏳",
                            "running": "🔄",
                            "completed": "✅",
                            "failed": "❌",
                            "draft": "📝",
                        }.get(workflow["status"], "❓")

                        print(
                            f"{status_icon} Workflow {workflow['id']}: {workflow['name']}"  # noqa :
                        )
                        print(f"   Status: {workflow['status']}")
                        print(f"   Created: {workflow['created_at']}")
                        print(f"   Nodes: {len(workflow.get('nodes', []))}")
                        print()

            else:
                print("Failed to fetch workflows")

            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\nMonitoring stopped")


if __name__ == "__main__":
    monitor_workflows()
