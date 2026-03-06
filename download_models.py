import os
import requests
import sys

def download_file(url, filename):
    """Скачивание файла с прогрессом"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Прогресс
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rПрогресс: {percent:.1f}%")
                        sys.stdout.flush()
        print(f"\n✅ {filename} загружен")
        return True
    except Exception as e:
        print(f"\n❌ Ошибка загрузки {filename}: {e}")
        return False

def main():
    # Создаём папку для моделей
    os.makedirs("models", exist_ok=True)
    os.chdir("models")
    
    print("📥 Скачивание моделей...")
    
    # Модель для замены лиц
    print("\n1. inswapper_128.onnx (200 MB)")
    if not os.path.exists("inswapper_128.onnx"):
        download_file(
            "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
            "inswapper_128.onnx"
        )
    else:
        print("✅ уже существует")
    
    # Модель для улучшения
    print("\n2. RealESRGAN_x4plus.pth (60 MB)")
    if not os.path.exists("RealESRGAN_x4plus.pth"):
        download_file(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN_x4plus.pth"
        )
    else:
        print("✅ уже существует")
    
    print("\n🎉 Все модели готовы!")

if __name__ == "__main__":
    main()
