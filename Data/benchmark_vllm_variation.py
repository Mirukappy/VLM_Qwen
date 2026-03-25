import requests
import base64
import time
import csv
import os
import re
import io
from PIL import Image
from datetime import datetime

# --- KONFIGURASI BENCHMARK vLLM ---
# UBAH INI SESUAI DENGAN MODEL YANG SEDANG RUNNING DI TERMINAL
CURRENT_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct" 

TEMPERATURES = [0.1, 0.4, 0.7, 1.0] # 4 Titik Temperatur
ITERATIONS = 3 # 3 percobaan per suhu (Total 12 per gambar)
OCCASION = "wawancara kerja di kantor korporat"

# --- DAFTAR GAMBAR YANG AKAN DIUJI ---
DAFTAR_GAMBAR = [
    "Data/Shirt_0.jpg", "Data/Shirt_1.jpg", "Data/Shirt_2.jpg",
    "Data/TShirt_0.jpg", "Data/TShirt_1.jpg", "Data/TShirt_2.jpg",
    "Data/Jacket_0.jpg", "Data/Jacket_1.jpg", "Data/Jacket_2.jpg"
]

PROMPT = f"""Kamu adalah Fashion Stylist profesional. Lihat foto orang ini. 
Dia akan pergi ke acara: '{OCCASION}'.

TUGASMU:
1. DESKRIPSI: Sebutkan jenis pakaian dan warnanya dengan detail.
2. ANALISIS: Apakah pakaian tersebut cocok untuk '{OCCASION}'?
3. SARAN: Berikan 1 saran singkat untuk memperbaiki penampilannya.

Jawab singkat dan ikuti format angka 1, 2, 3 di atas.
PENTING: LANGSUNG JAWAB SAJA. JANGAN MENGGUNAKAN TAG <think>.
"""

API_URL = "http://localhost:8000/v1/chat/completions"

def encode_image(image_path):
    """Mengecilkan gambar otomatis agar token tidak overlimit di vLLM"""
    with Image.open(image_path) as img:
        # Resize ke 896x896 (Aman untuk max-model-len 4096)
        img.thumbnail((896, 896), Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def run_benchmark():
    print(f"=== MEMULAI BENCHMARK vLLM ({CURRENT_MODEL_NAME}) ===")
    print(f"Total gambar yang akan diuji: {len(DAFTAR_GAMBAR)}\n")

    # LOOP 1: Mengambil setiap gambar dari daftar
    for img_path in DAFTAR_GAMBAR:
        if not os.path.exists(img_path):
            print(f"❌ ERROR: File '{img_path}' tidak ditemukan! Lewati...")
            continue

        # Membuat nama file CSV secara dinamis untuk vLLM
        nama_file_asli = os.path.basename(img_path)
        nama_tanpa_ekstensi = os.path.splitext(nama_file_asli)[0]
        csv_file = f"benchmark_vllm_{nama_tanpa_ekstensi}.csv"

        # Siapkan CSV untuk gambar ini
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Backend", "Model", "Temp", "Iterasi", "Waktu (s)", "Respons", "Status"])

        print("="*60)
        print(f"📷 MENGUJI GAMBAR : {img_path}")
        print(f"📄 Target CSV     : {csv_file}")
        print("="*60)

        base64_image = encode_image(img_path)

        # LOOP 2: Menguji setiap temperatur
        for temp in TEMPERATURES:
            print(f"  > Suhu: {temp}")
            
            # LOOP 3: Mengulang iterasi
            for i in range(1, ITERATIONS + 1):
                start_time = time.time()
                status = "Success"
                response_text = ""

                payload = {
                    "model": CURRENT_MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    "temperature": temp,
                    "max_tokens": 512 # Token cukup besar untuk format 1,2,3
                }

                try:
                    # Timeout dinaikkan agar server tidak diputus sepihak oleh Python jika sedang sibuk
                    response = requests.post(API_URL, json=payload, timeout=90)
                    
                    if response.status_code == 200:
                        data = response.json()
                        raw_text = data['choices'][0]['message']['content']
                        
                        # Antisipasi tambahan: Bersihkan tag <think> jika Qwen 3 membandel
                        clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
                        response_text = clean_text.replace('\n', ' ').strip()
                        
                        if len(response_text) < 5:
                            response_text = "[WARNING] Output kosong/terpotong."
                            status = "Truncated"
                            
                    else:
                        status = f"HTTP Error {response.status_code}: {response.text}"
                except Exception as e:
                    status = f"Error: {str(e)}"

                duration = time.time() - start_time
                
                print(f"    [Iter {i}/{ITERATIONS}] Waktu: {duration:.2f}s | Status: {status}")

                # Simpan ke CSV spesifik untuk gambar tersebut
                with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "vLLM", CURRENT_MODEL_NAME, temp, i, f"{duration:.2f}", response_text, status
                    ])

    print(f"\n✅ SEMUA BENCHMARK vLLM UNTUK MODEL {CURRENT_MODEL_NAME} SELESAI!")

if __name__ == "__main__":
    run_benchmark()