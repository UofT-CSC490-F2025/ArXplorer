import uvicorn
import sys
import os

if __name__ == "__main__":
    # Add the project root to the Python path to ensure 'src' can be found
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Project root added to path: {project_root}")
    print("Starting Uvicorn server for 'src.main:app'...")
    
    # Run the Uvicorn server programmatically
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, workers=1)
