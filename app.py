import os
from flask import Flask, request, abort
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
import threading

app = Flask(__name__)

BOT_TOKEN = os.environ['BOT_TOKEN']
bot = Bot(token=BOT_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0)

os.environ['HF_HOME'] = 'hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']
os.environ['HF_HUB_DISABLE_XET_DOWNLOAD'] = 'true'

try:
    from diffusers import AutoencoderKLWan, WanPipeline
    import torch

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, cache_dir=os.environ['HF_HOME'])
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, cache_dir=os.environ['HF_HOME'])
    pipe.to("cpu")
except ImportError as e:
    print(f"Ошибка загрузки модели: {e}. Убедитесь, что все зависимости установлены.")
    pipe = None

def generate_video(image_path, prompt, chat_id):
    if pipe is None:
        bot.send_message(chat_id, "Ошибка: Модель не загружена. Попробуйте позже.")
        return
    output = pipe(
        prompt=prompt,
        negative_prompt="Яркие тона, переэкспонировано, статично, размытые детали, субтитры, стиль, картины, изображения, статичная картинка, серый фон, низкое качество, артефакты JPEG, уродливо, незавершенно, лишние пальцы, плохо нарисованные руки, плохо нарисованные лица, деформировано, искажено, лишние конечности, сросшиеся пальцы, неподвижная картинка, грязный фон, три ноги, много людей на заднем плане, ходьба назад",
        height=480,
        width=832,
        num_frames=10,  # Уменьшено для скорости
        guidance_scale=5.0
    ).frames[0]
    export_to_video(output, "output.mp4", fps=15)
    bot.send_video(chat_id=chat_id, video=open("output.mp4", "rb"))

def start(update, context):
    update.message.reply_text("Отправьте фото для анимации!")

def handle_photo(update, context):
    update.message.reply_text("Обрабатываю...")
    photo = update.message.photo[-1].get_file()
    photo.download("input.jpg")
    prompt = update.message.caption or "Реалистичное движение"
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