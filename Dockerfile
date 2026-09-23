# syntax=docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt

COPY app.py .
COPY models ./models
COPY results ./results
COPY DataSet/attack_sample.csv ./DataSet/attack_sample.csv

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
