# Hugging Face Space image for the Wikiwise Streamlit app.
FROM python:3.13-slim

# Spaces run the container as uid 1000; give it a home so model downloads are writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py generate.py retrieve.py load_vectordb.py ./

EXPOSE 8501
# fileWatcherType=none: no live reload needed in prod, and it avoids the torchvision traceback spam.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]
