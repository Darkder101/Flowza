#!/usr/bin/env python3
"""
Test script for basic workflow execution in Flowza (fixed for /api prefix)
"""
import time

import requests

BASE_URL = "http://localhost:8000/api"  # <-- updated base URL


def test_csv_loader():
    print("\n1️⃣ Testing CSV Loader...")

    workflow = {
        "name": "CSV Loader Test",
        "nodes": [
            {
                "id": "csv_loader_node",
                "type": "csv_loader",
                "position": {"x": 100, "y": 100},
                "data": {"label": "CSV Loader"},
                "parameters": {"file_path": "datasets/test_data.csv"},
            }
        ],
        "connections": [],
    }

    # Create workflow
    response = requests.post(f"{BASE_URL}/workflows/", json=workflow)
    if response.status_code != 200:
        print("❌ Failed to create workflow")
        print(response.text)
        return

    workflow_id = response.json().get("id") or response.json().get("workflow_id")

    # Execute workflow
    exec_resp = requests.post(f"{BASE_URL}/workflows/{workflow_id}/execute")
    if exec_resp.status_code != 200:
        print("❌ Failed to execute workflow")
        print(exec_resp.text)
        return

    time.sleep(2)
    progress_resp = requests.get(f"{BASE_URL}/workflows/{workflow_id}/progress")
    print("✅ Workflow executed:", progress_resp.status_code == 200)


def test_drop_nulls_pipeline():
    print("\n2️⃣ Testing Drop Nulls Pipeline...")

    workflow = {
        "name": "Drop Nulls Pipeline",
        "nodes": [
            {
                "id": "csv_loader_node",
                "type": "csv_loader",
                "position": {"x": 100, "y": 100},
                "data": {"label": "CSV Loader"},
                "parameters": {"file_path": "datasets/test_data.csv"},
            },
            {
                "id": "drop_nulls_node",
                "type": "drop_nulls",
                "position": {"x": 300, "y": 100},
                "data": {"label": "Drop Nulls"},
                "parameters": {"how": "any"},
            },
        ],
        "connections": [{"source": "csv_loader_node", "target": "drop_nulls_node"}],
    }

    # Create workflow
    response = requests.post(f"{BASE_URL}/workflows/", json=workflow)
    if response.status_code != 200:
        print("❌ Failed to create workflow")
        print(response.text)
        return

    workflow_id = response.json().get("id") or response.json().get("workflow_id")

    # Execute workflow
    exec_resp = requests.post(f"{BASE_URL}/workflows/{workflow_id}/execute")
    if exec_resp.status_code != 200:
        print("❌ Failed to execute workflow")
        print(exec_resp.text)
        return

    time.sleep(2)
    progress_resp = requests.get(f"{BASE_URL}/workflows/{workflow_id}/progress")
    print("✅ Pipeline executed:", progress_resp.status_code == 200)


if __name__ == "__main__":
    test_csv_loader()
    test_drop_nulls_pipeline()
    test_drop_nulls_pipeline()
