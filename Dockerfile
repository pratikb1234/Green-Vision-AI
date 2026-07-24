FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY greenplan/ greenplan/
COPY config/ config/
RUN mkdir -p models outputs

ENV PYTHONUNBUFFERED=1

# Full offline demo by default. For a real DeepSeek run:
#   docker run -e OPENROUTER_API_KEY=sk-or-... greengrid \
#     python -m greenplan run --config config/city.yaml --recommend
CMD ["python", "-m", "greenplan", "run", "--config", "config/city.yaml", "--mock", "--recommend"]
