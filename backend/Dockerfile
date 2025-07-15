FROM python:3.12

WORKDIR /app

COPY requirements.txt ./

RUN pip install --retries=3 --timeout=30 -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "fastapi", "run", "main.py", "--port", "8000" ]
