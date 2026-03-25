import ollama
import time
import csv
import os
import re
from datetime import datetime

# --- KONFIGURASI BENCHMARK ---
MODELS = ['qwen2.5vl:3b', 'qwen2.5vl:7b', 'qwen3-vl:2b', 'qwen3-vl:4b']
TEMPERATURES = [0.1, 0.4, 0.7, 1.0] # 4 Titik Temperatur
ITERATIONS = 3 # 3 percobaan per suhu (Total 12 per model)
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

def run_benchmark():
    print("=== MEMULAI BENCHMARK OLLAMA (MULTI-IMAGE & MULTI-CSV) ===")
    print(f"Total gambar yang akan diuji: {len(DAFTAR_GAMBAR)}\n")

    # LOOP 1: Mengambil setiap gambar dari daftar
    for img_path in DAFTAR_GAMBAR:
        if not os.path.exists(img_path):
            print(f"❌ ERROR: File gambar '{img_path}' tidak ditemukan! Lewati...")
            continue

        # Membuat nama file CSV secara dinamis
        # Contoh: dari "Data/Shirt_0.jpg" -> "Shirt_0" -> "benchmark_ollama_Shirt_0.csv"
        nama_file_asli = os.path.basename(img_path)
        nama_tanpa_ekstensi = os.path.splitext(nama_file_asli)[0]
        csv_file = f"benchmark_ollama_{nama_tanpa_ekstensi}.csv"

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

        # LOOP 2: Menguji setiap model
        for model in MODELS:
            print(f"\n>>> Model: {model} <<<")
            try:
                # Pancingan agar model dimuat ke VRAM
                ollama.chat(model=model, messages=[{'role': 'user', 'content': 'hi'}])
            except:
                print(f"Lewati {model} - Tidak ditemukan / Gagal dimuat.")
                continue

            # LOOP 3: Menguji setiap temperatur
            for temp in TEMPERATURES:
                print(f"  > Suhu: {temp}")
                
                # LOOP 4: Mengulang iterasi
                for i in range(1, ITERATIONS + 1):
                    start_time = time.time()
                    status = "Success"
                    response_text = ""
                    
                    try:
                        res = ollama.chat(
                            model=model,
                            options={'temperature': temp, 'num_predict': 512, 'num_ctx': 4096},
                            messages=[{'role': 'user', 'content': PROMPT, 'images': [img_path]}]
                        )
                        raw_text = res['message']['content']
                        
                        # Antisipasi tambahan: Bersihkan tag <think> jika Qwen 3 membandel
                        clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
                        response_text = clean_text.replace('\n', ' ').strip()
                        
                    except Exception as e:
                        status = f"Error: {str(e)}"
                        print(f"    [Iter {i}] ❌ GAGAL - {status}")

                    duration = time.time() - start_time
                    
                    print(f"    [Iter {i}/{ITERATIONS}] Waktu: {duration:.2f}s | Status: {status}")

                    # Simpan ke CSV spesifik untuk gambar tersebut
                    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Ollama", model, temp, i, f"{duration:.2f}", response_text, status
                        ])

    print("\n✅ SEMUA BENCHMARK OLLAMA SELESAI!")

if __name__ == "__main__":
    run_benchmark()