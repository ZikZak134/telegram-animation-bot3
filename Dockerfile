FROM python:3.10
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY . .
RUN pip install -r requirements.txt
RUN mkdir -p hf_cache && chmod -R 777 hf_cache
CMD ["python", "app.py"]