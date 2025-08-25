#!/usr/bin/env python3
"""
Test script for Flowza workflow execution
"""

import time
import requests

BASE_URL = "http://localhost:8000"


def create_and_run_workflow(workflow_def: dict):
    """Helper: create workflow, execute, and track progress"""
    # Step 1: Create workflow
    create_resp = requests.post(f"{BASE_URL}/api/workflows/", json=workflow_def)  # noqa : 
    if create_resp.status_code != 200:
        print("❌ Failed to create workflow")
        print(create_resp.text)
        return None

    workflow_id = create_resp.json().get("id")
    print(f"✅ Workflow created: {workflow_id}")

    # Step 2: Execute workflow
    exec_resp = requests.post(f"{BASE_URL}/api/workflows/{workflow_id}/execute")  # noqa : 
    if exec_resp.status_code != 200:
        print("❌ Failed to execute workflow")
        print(exec_resp.text)
        return None
    print(f"🚀 Workflow {workflow_id} execution started")

    # Step 3: Poll progress
    for i in range(10):  # wait up to 10s
        time.sleep(1)
        prog_resp = requests.get(f"{BASE_URL}/api/workflows/{workflow_id}/progress")  # noqa : 
        if prog_resp.status_code == 200:
            progress = prog_resp.json()
            print(f"📊 Progress: {progress}")
            if progress.get("status") in ["completed", "failed"]:
                break
        else:
            print("❌ Failed to fetch progress")
            break

    return workflow_id


def test_workflow_execution():
    """Run two sample workflows"""

    # Test 1: Simple CSV Load
    print("\n1️⃣ Testing CSV Loader...")
    workflow_csv = {
        "name": "Test CSV Load",
        "description": "Simple CSV loading test",
        "nodes": [
            {
                "id": "load_csv",
                "type": "csv_loader",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Load CSV"},
                "parameters": {"file_path": "datasets/test_data.csv"},
            }
        ],
        "connections": [],
    }
    create_and_run_workflow(workflow_csv)

    # Test 2: Drop Nulls pipeline
    print("\n2️⃣ Testing Drop Nulls Pipeline...")
    workflow_pipeline = {
        "name": "Drop Nulls Pipeline",
        "description": "CSV → Drop Nulls",
        "nodes": [
            {
                "id": "load_csv",
                "type": "csv_loader",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Load CSV"},
                "parameters": {"file_path": "datasets/test_data.csv"},
            },
            {
                "id": "drop_nulls",
                "type": "preprocess",
                "position": {"x": 300, "y": 100},
                "data": {"label": "Drop Nulls"},
                "parameters": {"drop_nulls": True, "encode_categorical": False, "normalize": False},
            },
        ],
        "connections": [{"source": "load_csv", "target": "drop_nulls"}],
    }
    create_and_run_workflow(workflow_pipeline)


if __name__ == "__main__":
    test_workflow_execution()
