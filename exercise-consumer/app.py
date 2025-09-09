from fastapi import FastAPI, Request, HTTPException
from google.cloud import storage
import os
import json
from datetime import datetime

app = FastAPI()

# Get bucket and folder from environment variables
BUCKET_NAME = os.getenv("BUCKET")
FOLDER = os.getenv("FOLDER", "exercise/")

@app.get("/")
def root():
    return {"service": "exercise-consumer", "version": "1.0.0"}

@app.post("/write")
async def write(request: Request):
    try:
        # Parse JSON body
        body = await request.json()
        name = body.get("name", "unknown")
        demo = body.get("demo", False)

        # Init GCS client
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # Unique filename
        filename = f"{FOLDER}{name}_{datetime.utcnow().isoformat()}.json"
        blob = bucket.blob(filename)

        # Upload to GCS
        blob.upload_from_string(
            json.dumps(
                {"name": name, "demo": demo, "timestamp": datetime.utcnow().isoformat()}
            ),
            content_type="application/json",
        )

        return {"status": "ok", "file": filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
