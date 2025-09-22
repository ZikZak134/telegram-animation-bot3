import os
from flask import Flask, request, abort
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
import threading
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from PIL import Image
import torch

app = Flask(__name__)

BOT_TOKEN = "6261311126:AAHC90o9g51JYLLw47TFgfwcL02r5nqNhBw"  # Замени на свой токен от @BotFather
bot = Bot(token=BOT_TOKEN)
dispatcher = Dispatcher(bot, None, workers=1)

os.environ['HF_HOME'] = 'hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.environ['HF_HUB_DISABLE_XET_DOWNLOAD'] = 'true'

pipe = None  # Ленивая загрузка

def load_model():
    global pipe
    if pipe is None:
        try:
            model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # Легкая версия для теста
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=os.environ['HF_HOME'])
            pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, cache_dir=os.environ['HF_HOME'])
            pipe.to("cpu")
            print("Модель загружена!")
        except Exception as e:
            print(f"Ошибка: {e}")
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
    except Exception as e:
        bot.send_message(chat_id, f"Ошибка: {str(e)}")

def start(update, context):
    update.message.reply_text("Отправьте фото для видео!")

def handle_photo(update, context):
    update.message.reply_text("Обрабатываю...")
    photo = update.message.photo[-1].get_file()
    photo.download("input.jpg")
    prompt = update.message.caption or None
    threading.Thread(target=generate_video, args=("input.jpg", prompt, update.message.chat_id)).start()

dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), bot)
        dispatcher.process_update(update)
        return "ok", 200
    abort(400)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
