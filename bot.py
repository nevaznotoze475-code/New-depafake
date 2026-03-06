import telebot
from telebot import types
import cv2
import numpy as np
import os
import time
import uuid
from moviepy.editor import VideoFileClip
import ffmpeg
import threading

# Импорты для дипфейков
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# Импорты для улучшения качества
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ========== НАСТРОЙКИ ==========
TOKEN = "8366196461:AAHD95QhRtLPd9sPdJFP5X9wZun_FVEx4Ww"  # замените на свой
YOUR_ID = 5858391454  # замените на свой ID

# Инициализация бота
bot = telebot.TeleBot(TOKEN)

# Состояния пользователя
user_states = {}
user_data = {}

# ========== ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ==========
print("Загрузка моделей... Это займет минуту")

# Модель для обнаружения лиц
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Модель для замены лиц
swapper = get_model('inswapper_128.onnx')

# Модель для улучшения качества (опционально)
try:
    model = RRDBNet(num_in_ch=3, num_out_ch=3)
    upsampler = RealESRGANer(
        scale=2,
        model_path=None,  # Автоматически скачает модель
        model=model
    )
    print("Модель улучшения качества загружена")
except:
    upsampler = None
    print("Режим улучшения недоступен (установите realesrgan)")

print("✅ Бот готов к работе!")

# ========== ФУНКЦИИ ДЛЯ ДИПФЕЙКОВ ==========

def swap_faces_in_image(source_img, target_img, enhance=False):
    """
    Замена лица с одного фото на другое
    """
    try:
        # Находим лица на обоих фото
        source_faces = app.get(source_img)
        target_faces = app.get(target_img)
        
        if len(source_faces) == 0:
            return None, "На исходном фото не найдено лиц"
        if len(target_faces) == 0:
            return None, "На целевом фото не найдено лиц"
        
        # Берем первые лица
        source_face = source_faces[0]
        
        # Копируем целевое изображение
        result = target_img.copy()
        
        # Заменяем все найденные лица (или только первое)
        for face in target_faces:
            result = swapper.get(result, face, source_face, paste_back=True)
        
        # Улучшение качества если нужно
        if enhance and upsampler:
            result, _ = upsampler.enhance(result, outscale=1.5)
        
        return result, "✅ Готово!"
        
    except Exception as e:
        return None, f"❌ Ошибка: {str(e)}"

def process_video_frames(video_path, source_img_path, output_path, enhance=False):
    """
    Обработка видео покадрово
    """
    try:
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Загружаем исходное лицо один раз
        source_img = cv2.imread(source_img_path)
        source_faces = app.get(source_img)
        if len(source_faces) == 0:
            return False, "На исходном фото не найдено лиц"
        source_face = source_faces[0]
        
        # Кодек для сохранения
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Находим лица в кадре
            faces = app.get(frame)
            
            # Заменяем лица
            result_frame = frame.copy()
            for face in faces:
                result_frame = swapper.get(result_frame, face, source_face, paste_back=True)
            
            # Улучшение если нужно
            if enhance and upsampler and processed % 10 == 0:  # Каждый 10-й кадр для скорости
                result_frame, _ = upsampler.enhance(result_frame, outscale=1)
            
            out.write(result_frame)
            processed += 1
            
            # Прогресс каждые 30 кадров
            if processed % 30 == 0:
                progress = (processed / total_frames) * 100
                print(f"Прогресс: {progress:.1f}%")
        
        cap.release()
        out.release()
        return True, f"✅ Видео готово! Обработано кадров: {processed}"
        
    except Exception as e:
        return False, f"❌ Ошибка: {str(e)}"

# ========== КОМАНДЫ БОТА ==========

@bot.message_handler(commands=['start'])
def start(message):
    if message.from_user.id != YOUR_ID:
        bot.reply_to(message, "❌ Этот бот только для личного использования")
        return
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    btn1 = types.KeyboardButton("📷 Замена лица (фото)")
    btn2 = types.KeyboardButton("🎥 Замена лица (видео)")
    btn3 = types.KeyboardButton("⚡ Улучшить качество")
    btn4 = types.KeyboardButton("❓ Помощь")
    markup.add(btn1, btn2, btn3, btn4)
    
    bot.send_message(
        message.chat.id,
        "🔮 *Универсальный дипфейк бот*\n\n"
        "Выбери режим работы:\n"
        "• 📷 Фото → замена на другом фото\n"
        "• 🎥 Видео → замена лица в видео\n"
        "• ⚡ Улучшение качества (Real-ESRGAN)\n\n"
        "_Все данные хранятся локально_",
        parse_mode="Markdown",
        reply_markup=markup
    )

@bot.message_handler(func=lambda m: m.text == "📷 Замена лица (фото)")
def photo_mode(message):
    if message.from_user.id != YOUR_ID:
        return
    
    user_states[message.chat.id] = "waiting_source_photo"
    msg = bot.send_message(
        message.chat.id,
        "📸 Отправь **исходное фото** (чьё лицо будем вставлять)",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(msg, get_source_photo)

@bot.message_handler(func=lambda m: m.text == "🎥 Замена лица (видео)")
def video_mode(message):
    if message.from_user.id != YOUR_ID:
        return
    
    user_states[message.chat.id] = "waiting_source_video"
    msg = bot.send_message(
        message.chat.id,
        "🎬 Отправь **исходное фото** (чьё лицо будем вставлять в видео)",
        parse_mode="Markdown"
    )
    bot.register_next_step_handler(msg, get_source_photo_for_video)

def get_source_photo(message):
    if message.photo:
        # Скачиваем исходное фото
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded = bot.download_file(file_info.file_path)
        
        # Сохраняем с уникальным именем
        source_path = f"source_{uuid.uuid4()}.jpg"
        with open(source_path, 'wb') as f:
            f.write(downloaded)
        
        # Сохраняем в данные пользователя
        user_data[message.chat.id] = {"source_path": source_path}
        
        user_states[message.chat.id] = "waiting_target_photo"
        msg = bot.send_message(
            message.chat.id,
            "🖼 Теперь отправь **целевое фото** (куда вставим лицо)",
            parse_mode="Markdown"
        )
        bot.register_next_step_handler(msg, get_target_photo)

def get_target_photo(message):
    if message.photo:
        # Скачиваем целевое фото
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded = bot.download_file(file_info.file_path)
        
        target_path = f"target_{uuid.uuid4()}.jpg"
        with open(target_path, 'wb') as f:
            f.write(downloaded)
        
        # Получаем исходное фото
        source_path = user_data[message.chat.id]["source_path"]
        
        # Спрашиваем про улучшение
        markup = types.InlineKeyboardMarkup()
        markup.add(
            types.InlineKeyboardButton("✅ Да", callback_data="enhance_yes"),
            types.InlineKeyboardButton("❌ Нет", callback_data="enhance_no")
        )
        
        # Сохраняем пути
        user_data[message.chat.id]["target_path"] = target_path
        
        bot.send_message(
            message.chat.id,
            "✨ Улучшить качество результата? (замедлит обработку)",
            reply_markup=markup
        )

def get_source_photo_for_video(message):
    if message.photo:
        # Скачиваем исходное фото для видео
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded = bot.download_file(file_info.file_path)
        
        source_path = f"source_{uuid.uuid4()}.jpg"
        with open(source_path, 'wb') as f:
            f.write(downloaded)
        
        user_data[message.chat.id] = {"source_path": source_path}
        
        user_states[message.chat.id] = "waiting_video"
        msg = bot.send_message(
            message.chat.id,
            "🎥 Теперь отправь **видео** для обработки",
            parse_mode="Markdown"
        )
        bot.register_next_step_handler(msg, get_video)

def get_video(message):
    if message.video:
        # Скачиваем видео
        file_id = message.video.file_id
        file_info = bot.get_file(file_id)
        downloaded = bot.download_file(file_info.file_path)
        
        video_path = f"video_{uuid.uuid4()}.mp4"
        with open(video_path, 'wb') as f:
            f.write(downloaded)
        
        # Сохраняем путь к видео
        user_data[message.chat.id]["video_path"] = video_path
        
        # Спрашиваем про улучшение
        markup = types.InlineKeyboardMarkup()
        markup.add(
            types.InlineKeyboardButton("✅ Да", callback_data="video_enhance_yes"),
            types.InlineKeyboardButton("❌ Нет", callback_data="video_enhance_no")
        )
        
        bot.send_message(
            message.chat.id,
            "🎬 Улучшать качество видео? (очень медленно!)",
            reply_markup=markup
        )

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    if call.data == "enhance_yes":
        bot.edit_message_text(
            "⏳ Обрабатываю фото с улучшением...",
            call.message.chat.id,
            call.message.message_id
        )
        
        # Запускаем обработку в отдельном потоке
        thread = threading.Thread(
            target=process_photo,
            args=(call.message.chat.id, True)
        )
        thread.start()
        
    elif call.data == "enhance_no":
        bot.edit_message_text(
            "⏳ Обрабатываю фото...",
            call.message.chat.id,
            call.message.message_id
        )
        
        thread = threading.Thread(
            target=process_photo,
            args=(call.message.chat.id, False)
        )
        thread.start()
        
    elif call.data == "video_enhance_yes":
        bot.edit_message_text(
            "⏳ Обрабатываю видео с улучшением... Это может занять много времени",
            call.message.chat.id,
            call.message.message_id
        )
        
        thread = threading.Thread(
            target=process_video,
            args=(call.message.chat.id, True)
        )
        thread.start()
        
    elif call.data == "video_enhance_no":
        bot.edit_message_text(
            "⏳ Обрабатываю видео... Это займет несколько минут",
            call.message.chat.id,
            call.message.message_id
        )
        
        thread = threading.Thread(
            target=process_video,
            args=(call.message.chat.id, False)
        )
        thread.start()

def process_photo(chat_id, enhance):
    """Обработка фото в отдельном потоке"""
    try:
        source_path = user_data[chat_id]["source_path"]
        target_path = user_data[chat_id]["target_path"]
        
        # Загружаем изображения
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)
        
        # Заменяем лица
        result, message = swap_faces_in_image(source_img, target_img, enhance)
        
        if result is not None:
            # Сохраняем результат
            result_path = f"result_{uuid.uuid4()}.jpg"
            cv2.imwrite(result_path, result)
            
            # Отправляем результат
            with open(result_path, 'rb') as photo:
                bot.send_photo(chat_id, photo, caption=message)
            
            # Очищаем временные файлы
            os.remove(source_path)
            os.remove(target_path)
            os.remove(result_path)
        else:
            bot.send_message(chat_id, message)
            
    except Exception as e:
        bot.send_message(chat_id, f"❌ Ошибка: {str(e)}")
    finally:
        if chat_id in user_data:
            del user_data[chat_id]

def process_video(chat_id, enhance):
    """Обработка видео в отдельном потоке"""
    try:
        source_path = user_data[chat_id]["source_path"]
        video_path = user_data[chat_id]["video_path"]
        output_path = f"output_{uuid.uuid4()}.mp4"
        
        # Обрабатываем видео
        success, message = process_video_frames(
            video_path, 
            source_path, 
            output_path,
            enhance
        )
        
        if success:
            # Отправляем результат
            with open(output_path, 'rb') as video:
                bot.send_video(chat_id, video, caption=message)
            
            # Очищаем временные файлы
            os.remove(source_path)
            os.remove(video_path)
            os.remove(output_path)
        else:
            bot.send_message(chat_id, message)
            
    except Exception as e:
        bot.send_message(chat_id, f"❌ Ошибка: {str(e)}")
    finally:
        if chat_id in user_data:
            del user_data[chat_id]

@bot.message_handler(func=lambda m: m.text == "⚡ Улучшить качество")
def enhance_mode(message):
    if message.from_user.id != YOUR_ID:
        return
    
    bot.send_message(
        message.chat.id,
        "⚡ Отправь фото для улучшения качества (Real-ESRGAN)"
    )
    user_states[message.chat.id] = "waiting_enhance"

@bot.message_handler(func=lambda m: m.text == "❓ Помощь")
def help_command(message):
    if message.from_user.id != YOUR_ID:
        return
    
    help_text = """
*📖 Инструкция:*

1️⃣ *Замена лица на фото:*
   - Выбери режим фото
   - Отправь исходное лицо
   - Отправь целевое фото

2️⃣ *Замена лица в видео:*
   - Выбери режим видео
   - Отправь исходное лицо
   - Отправь видео

3️⃣ *Улучшение качества:*
   - Отправь любое фото
   - Получишь апскейл 2x

*⚠️ Важно:*
• Видео обрабатывается долго
• Для качества нужно мощное GPU
• Все файлы удаляются после отправки
    """
    bot.send_message(message.chat.id, help_text, parse_mode="Markdown")

@bot.message_handler(func=lambda m: m.text and m.text not in [
    "📷 Замена лица (фото)", "🎥 Замена лица (видео)", 
    "⚡ Улучшить качество", "❓ Помощь"
])
def handle_enhance(message):
    if message.chat.id in user_states and user_states[message.chat.id] == "waiting_enhance":
        if message.photo:
            bot.send_message(message.chat.id, "⏳ Улучшаю качество...")
            
            # Скачиваем фото
            file_id = message.photo[-1].file_id
            file_info = bot.get_file(file_id)
            downloaded = bot.download_file(file_info.file_path)
            
            input_path = f"enhance_input_{uuid.uuid4()}.jpg"
            output_path = f"enhance_output_{uuid.uuid4()}.jpg"
            
            with open(input_path, 'wb') as f:
                f.write(downloaded)
            
            # Улучшаем
            img = cv2.imread(input_path)
            if upsampler:
                output, _ = upsampler.enhance(img, outscale=2)
                cv2.imwrite(output_path, output)
                
                with open(output_path, 'rb') as photo:
                    bot.send_photo(message.chat.id, photo, caption="✅ Качество улучшено")
                
                os.remove(input_path)
                os.remove(output_path)
            else:
                bot.send_message(message.chat.id, "❌ Модуль улучшения не установлен")
            
            del user_states[message.chat.id]

# Запуск бота
if __name__ == "__main__":
    print("🚀 Бот запущен...")
    bot.infinity_polling()
