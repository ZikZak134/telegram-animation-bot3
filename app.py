import os
import logging
from flask import Flask, request, abort
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
import threading
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from PIL import Image
import torch

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ['BOT_TOKEN']
bot = Bot(token=BOT_TOKEN)
dispatcher = Dispatcher(bot, None, workers=1)

os.environ['HF_HOME'] = 'hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.environ['HF_HUB_DISABLE_XET_DOWNLOAD'] = 'true'

pipe = None

def load_model():
    global pipe
    if pipe is None:
        try:
            model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=os.environ['HF_HOME'])
            pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, cache_dir=os.environ['HF_HOME'])
            pipe.to("cpu")
            logger.info("Модель загружена!")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            pipe = None

def generate_video(image_path, prompt, chat_id):
    load_model()
    if pipe is None:
        bot.send_message(chat_id, "Ошибка модели.")
        return
    try:
        image = Image.open(image_path).convert("RGB").resize((832, 480))
        output = pipe(
            prompt=prompt or "Реалистичное движение",
            negative_prompt="размыто, низкое качество, статично",
            height=480,
            width=832,
            num_frames=10,
            guidance_scale=5.0
        ).frames[0]
        video_path = "output.mp4"
        export_to_video(output, video_path, fps=15)
        bot.send_video(chat_id=chat_id, video=open(video_path, "rb"))
        os.remove(video_path)
        logger.info("Видео отправлено!")
    except Exception as e:
        bot.send_message(chat_id, f"Ошибка: {str(e)}")
        logger.error(f"Ошибка генерации: {e}")

def start(update, context):
    update.message.reply_text("Отправьте фото для видео!")
    logger.info("Получена команда /start")

def handle_photo(update, context):
    update.message.reply_text("Обрабатываю...")
    photo = update.message.photo[-1].get_file()
    photo.download("input.jpg")
    prompt = update.message.caption or None
    threading.Thread(target=generate_video, args=("input.jpg", prompt, update.message.chat_id)).start()
    logger.info("Получено фото, генерация запущена")

dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))

@app.route('/webhook', methods=['POST'])
def webhook():
    logger.info("Получен POST запрос на /webhook")
    if request.method == 'POST':
        update = Update.de_json(request.get_json(force=True), bot)
        logger.debug(f"Обработка обновления: {update}")
        dispatcher.process_update(update)
        logger.info("Webhook обработан")
        return "ok", 200
    logger.warning("Получен некорректный метод запроса")
    abort(405)  # Явно возвращаем 405 для GET

@app.route('/', methods=['GET'])
def home():
    logger.info("Получен GET запрос на /")
    return "Bot is running", 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Запуск на порту {port}")
    app.run(host='0.0.0.0', port=port)
