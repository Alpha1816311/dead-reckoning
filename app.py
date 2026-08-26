from fastapi import FastAPI

app = FastAPI(title="Intelligent Dead Reckoning")

@app.get("/")
def home():
    return {
        "project": "Intelligent Dead Reckoning",
        "status": "ONLINE",
        "message": "IDR system is running"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}