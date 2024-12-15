# Basis-Image mit Python
FROM python:3.9-slim

# Setze das Arbeitsverzeichnis innerhalb des Containers
WORKDIR /app

# Kopiere alle Dateien und Ordner in den Container
COPY . /app

# Installiere Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Stelle sicher, dass Hugging Face API-Token gesetzt wird
RUN mkdir -p /root/.huggingface && cp "HF_token.txt" /root/.huggingface/token

# Exponiere den Port 8501 für Streamlit
EXPOSE 8501

# Starte die Streamlit-App
CMD ["streamlit", "run", "app.py"]