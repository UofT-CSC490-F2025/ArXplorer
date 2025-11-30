from fastapi import FastAPI

app = FastAPI(
    title="ArXplorer API",
    description="API for ArXplorer, a research paper discovery tool.",
    version="0.1.0",
)

@app.get("/")
def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to ArXplorer API"}

# Add other endpoints here as needed for the application
