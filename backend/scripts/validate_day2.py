#!/usr/bin/env python3
"""
Validation script for Day 2 completion (fixed for Flowza API)
"""
import sys
import time
import requests

BASE_URL = "http://localhost:8000"


def validate_day2_completion():
    """Validate that all Day 2 features are working"""
    print("🔍 Validating Flowza Day 2 Implementation...")

    tests = []

    # Test 1: API Health Check
    try:
        response = requests.get(f"{BASE_URL}/health")
        tests.append(("API Health Check", response.status_code == 200))
    except Exception:
        tests.append(("API Health Check", False))

    # Test 2: Available Nodes
    try:
        response = requests.get(f"{BASE_URL}/api/tasks/available")
        data = response.json()
        has_nodes = len(data.get("preprocessing_nodes", [])) > 0
        tests.append((
            "Available ML Nodes",
            response.status_code == 200 and has_nodes,
        ))
    except Exception:
        tests.append(("Available ML Nodes", False))

    # Test 3: Workflow Execution
    try:
        workflow = {
            "name": "Validation Test",
            "nodes": [
                {
                    "id": "test_node",
                    "type": "csv_loader",
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Test"},
                    "parameters": {
                        "file_path": "datasets/test_data.csv",
                    },
                }
            ],
            "connections": [],
        }

        # Step 1: Create workflow
        response = requests.post(f"{BASE_URL}/api/workflows/", json=workflow)
        workflow_created = response.status_code == 200

        if workflow_created:
            workflow_id = response.json().get("id") or response.json().get("workflow_id")  # noqa : 

            # Step 2: Execute workflow
            exec_response = requests.post(
                f"{BASE_URL}/api/workflows/{workflow_id}/execute"
            )
            execution_started = exec_response.status_code == 200

            # Step 3: Check progress
            time.sleep(2)
            status_response = requests.get(
                f"{BASE_URL}/api/workflows/{workflow_id}/progress"
            )
            status_check = status_response.status_code == 200

            tests.append((
                "Workflow Execution",
                workflow_created and execution_started and status_check,
            ))
        else:
            tests.append(("Workflow Execution", False))
    except Exception:
        tests.append(("Workflow Execution", False))

    # Test 4: Pipeline Execution
    try:
        pipeline = {
            "name": "Pipeline Test",
            "nodes": [
                {
                    "id": "load",
                    "type": "csv_loader",
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Load"},
                    "parameters": {
                        "file_path": "datasets/test_data.csv",
                    },
                },
                {
                    "id": "clean",
                    "type": "drop_nulls",
                    "position": {"x": 300, "y": 100},
                    "data": {"label": "Clean"},
                    "parameters": {"how": "any"},
                },
            ],
            "connections": [{"source": "load", "target": "clean"}],
        }

        # Create workflow
        response = requests.post(f"{BASE_URL}/api/workflows/", json=pipeline)
        pipeline_created = response.status_code == 200

        if pipeline_created:
            workflow_id = response.json().get("id") or response.json().get("workflow_id")  # noqa : 
            exec_response = requests.post(
                f"{BASE_URL}/api/workflows/{workflow_id}/execute"
            )
            execution_started = exec_response.status_code == 200
            tests.append(("Pipeline Execution", pipeline_created and execution_started))  # noqa : 
        else:
            tests.append(("Pipeline Execution", False))
    except Exception:
        tests.append(("Pipeline Execution", False))

    # Print results
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)

    passed = 0
    total = len(tests)

    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print("=" * 50)
    print(f"Score: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Day 2 implementation is complete and working!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = validate_day2_completion()
    sys.exit(0 if success else 1)
