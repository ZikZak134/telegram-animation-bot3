FROM python:3.10-slim  # Slim-версия для ускорения

# Устанавливаем системные зависимости для компиляции (если нужно для torch)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip и wheel для быстрой установки
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Создаём пользователя
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Копируем requirements и устанавливаем зависимости по шагам (для кэша Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # CPU-версия torch, быстрее
RUN pip install --no-cache-dir transformers accelerate
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальное
COPY . .

# Создаём кэш для модели
RUN mkdir -p hf_cache && chmod -R 777 hf_cache

CMD ["python", "app.py"]
