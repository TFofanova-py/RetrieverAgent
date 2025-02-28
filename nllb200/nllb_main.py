from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Define the model name (replace with your desired model)
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize FastAPI app
app = FastAPI()

class RequestModel(BaseModel):
    text: str
    lang: str = "deu_Latn"


@app.post("/translate")
async def translate(req: RequestModel):
    try:
        input_ids = tokenizer(req.text, return_tensors="pt").input_ids.to(device)
        output_ids = model.generate(input_ids, forced_bos_token_id=tokenizer.convert_tokens_to_ids(req.lang), max_length=700)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"translated_text": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Hugging Face Translation API is running"}

