FROM python:3.12

WORKDIR /pdf_extract

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8004

CMD ["uvicorn", "extract:app", "--host", "0.0.0.0", "--port", "8004"]
