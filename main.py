import uvicorn
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.api import app

if __name__ == "__main__":
    print("=" * 70)
    print(" Starting Real-Time Sentiment Analysis System Server")
    print(" Department of Computer Science & Engineering | Aditya University")
    print("=" * 70)
    print(" Dashboard UI: http://127.0.0.1:8000/")
    print(" API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
