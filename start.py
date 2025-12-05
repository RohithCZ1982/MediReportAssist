#!/usr/bin/env python3
"""
Startup script for Patient Discharge Instructions Assistant
"""

import uvicorn
import sys
from pathlib import Path

# Ensure static directory exists
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Ensure uploads directory exists
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

if __name__ == "__main__":
    print("=" * 60)
    print("Patient Discharge Instructions Assistant")
    print("=" * 60)
    print("\nStarting server...")
    print("Access the application at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)


