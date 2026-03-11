FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir dash dash-bootstrap-components plotly pandas numpy

COPY dashboard.py .
COPY outputs/ outputs/

EXPOSE 7860

CMD ["python", "dashboard.py"]
