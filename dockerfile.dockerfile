FROM docker.io/python:3.7
WORKDIR /api
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["gunicorn",""--bind","0.0.0.0:5000",main:api"]