import ollama
import os

# --- Konfigurasi ---
# Nama model sudah diperbarui sesuai link yang Anda berikan.
MODEL_NAME = 'qwen2.5vl:7b' # <-- SUDAH DIPERBARUI

# Path ke gambar yang ingin dianalisis.
# Ganti dengan path gambar di komputer Anda!
# Contoh Windows: 'C:/Users/raida/Pictures/kucing.jpg'
# Contoh Linux: '/home/raida/Pictures/kucing.jpg'
IMAGE_PATH = 'C:/Nugas/TA/Tugas-Akhir/image/image_orang.jpg' 

# Pertanyaan Anda tentang gambar tersebut
PROMPT = 'Berikan saran baju yang cocok untuk orang yang ada pada gambar ini.'

def run_qwen_vision_local(model, prompt, image_path):
    """
    Fungsi untuk menjalankan model Qwen2.5-VL 7B secara lokal dengan Ollama.
    """
    print(f"🚀 Memproses gambar: {os.path.basename(image_path)}")
    print(f"🤔 Pertanyaan: {prompt}")
    print(f"📦 Menggunakan model: {model}")
    print("-" * 30)

    # Memeriksa apakah file gambar ada
    if not os.path.exists(image_path):
        print(f"❌ Error: File gambar tidak ditemukan di '{image_path}'")
        return

    try:
        # Mengirim request ke Ollama API yang berjalan secara lokal
        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path] # Menyertakan path gambar
                }
            ]
        )

        # Menampilkan jawaban dari model
        print("🤖 Jawaban Model:")
        print(response['message']['content'])

    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")
        print(f"\nPastikan Ollama sudah berjalan dan model '{model}' sudah Anda pull.")
        print(f"Jalankan 'ollama pull {model}' di terminal jika belum.")

if __name__ == '__main__':
    # Pastikan Anda sudah mengubah IMAGE_PATH di atas
    if IMAGE_PATH == 'C:/path/to/your/image.jpg':
        print("⚠️  Peringatan: Harap ubah variabel IMAGE_PATH di dalam skrip terlebih dahulu!")
    else:
        run_qwen_vision_local(MODEL_NAME, PROMPT, IMAGE_PATH)