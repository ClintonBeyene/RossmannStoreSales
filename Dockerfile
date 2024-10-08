FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]