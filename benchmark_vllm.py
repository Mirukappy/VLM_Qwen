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
# Nama model harus SAMA PERSIS dengan yang dijalankan di command terminal vLLM
CURRENT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

TEMPERATURES = [0.1, 0.4, 0.7, 1.0] # 4 Titik Temperatur
ITERATIONS = 20 # 20 iterasi per suhu (Total 80 percobaan)
TEST_IMAGE = "test_image.jpg"
OCCASION = "wawancara kerja di kantor korporat"

PROMPT = f"""Kamu adalah Fashion Stylist profesional. Lihat foto orang ini. 
Dia akan pergi ke acara: '{OCCASION}'.

TUGASMU:
1. DESKRIPSI: Sebutkan jenis pakaian dan warnanya dengan detail.
2. ANALISIS: Apakah pakaian tersebut cocok untuk '{OCCASION}'?
3. SARAN: Berikan 1 saran singkat untuk memperbaiki penampilannya.

Jawab singkat dan ikuti format angka 1, 2, 3 di atas.
"""

API_URL = "http://localhost:8000/v1/chat/completions"
CSV_FILE = "benchmark_vllm_part2.csv" # Nama file dibedakan agar rapi

def encode_image(image_path):
    """Karena max_model_len sekarang 4096, kita bisa pakai gambar lebih besar (896px)"""
    with Image.open(image_path) as img:
        # Resize ke 896x896 (Sangat tajam, tapi tokennya masih aman di bawah 4096)
        img.thumbnail((896, 896), Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def run_benchmark():
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ ERROR: File gambar '{TEST_IMAGE}' tidak ditemukan!")
        return

    base64_image = encode_image(TEST_IMAGE)
    
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Backend", "Model", "Temp", "Iterasi", "Waktu (s)", "Respons", "Status"])

    print(f"=== MEMULAI BENCHMARK vLLM ({CURRENT_MODEL_NAME}) ===")
    print(f"Total percobaan: {len(TEMPERATURES) * ITERATIONS} request.\n")

    for temp in TEMPERATURES:
        print(f"\n>>> Menguji Suhu: {temp} <<<")
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
                "max_tokens": 512 # Kita naikkan kembali ke 512 karena sisa memori sekarang sangat lega
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=90)
                if response.status_code == 200:
                    data = response.json()
                    raw_text = data['choices'][0]['message']['content']
                    
                    # Qwen 2.5 biasanya tidak pakai <think>, tapi kita pasang regex ini untuk jaga-jaga
                    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
                    response_text = clean_text.replace('\n', ' ').strip()
                        
                else:
                    status = f"HTTP Error {response.status_code}: {response.text}"
            except Exception as e:
                status = f"Error: {str(e)}"

            duration = time.time() - start_time
            
            print(f"    [Iter {i}/{ITERATIONS}] Waktu: {duration:.2f}s | Status: {status}")

            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "vLLM", CURRENT_MODEL_NAME, temp, i, f"{duration:.2f}", response_text, status
                ])

    print("\n✅ BENCHMARK vLLM SELESAI!")

if __name__ == "__main__":
    run_benchmark()