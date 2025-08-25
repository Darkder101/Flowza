#!/usr/bin/env python3
"""
List all registered FastAPI routes
"""

import sys
import os

from fastapi.routing import APIRoute

# Go one level up (backend/) so "app" becomes importable
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from app.main import app  # import your FastAPI app  # noqa : 


def list_routes():
    print("📌 Registered FastAPI Routes:")
    for route in app.routes:
        if isinstance(route, APIRoute):
            methods = ",".join(route.methods)
            print(f"{methods:10s} {route.path}")


if __name__ == "__main__":
    list_routes()
