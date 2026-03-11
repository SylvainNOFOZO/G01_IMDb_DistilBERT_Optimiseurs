FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir dash dash-bootstrap-components plotly pandas numpy

COPY dashboard.py .
COPY *.csv *.json outputs/results/

EXPOSE 7860

CMD ["python", "dashboard.py"]
