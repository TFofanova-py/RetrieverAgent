import pymupdf
from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
import tempfile
import os

app = FastAPI()


async def extract_text(file_name: str) -> str:
    doc = pymupdf.open(file_name)

    result = ""

    for i, page in enumerate(doc):
        # Get a TextPage object for the current page
        text_page = page.get_text()
        result += text_page + "\n"

    doc.close()
    return result


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_path = tmp.name
            print(temp_path)
            shutil.copyfileobj(file.file, tmp)

        result = await extract_text(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"extracted_text": result}


@app.get("/")
def root():
    return {"message": "Extract PDF API is running"}

# uvicorn extract:app --port 8004 --reload