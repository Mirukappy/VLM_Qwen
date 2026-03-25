import ollama
import time
import csv
import os
from datetime import datetime

# --- KONFIGURASI BENCHMARK ---
MODELS = ['qwen2.5vl:3b', 'qwen2.5vl:7b', 'qwen3vl:2b', 'qwen3vl:4b']
TEMPERATURES = [0.1, 0.4, 0.7, 1.0] # 4 Titik Temperatur
ITERATIONS = 3 # 3 percobaan per suhu (Total 12 per model)
TEST_IMAGE = "Data/Shirt_0.jpg" 
OCCASION = "wawancara kerja di kantor korporat"

PROMPT = f"""Kamu adalah Fashion Stylist profesional. Lihat foto orang ini. 
Dia akan pergi ke acara: '{OCCASION}'.

TUGASMU:
1. DESKRIPSI: Sebutkan jenis pakaian dan warnanya dengan detail.
2. ANALISIS: Apakah pakaian tersebut cocok untuk '{OCCASION}'?
3. SARAN: Berikan 1 saran singkat untuk memperbaiki penampilannya.

Jawab singkat dan ikuti format angka 1, 2, 3 di atas.
"""

CSV_FILE = "benchmark_ollama_shirt_0.csv"

def run_benchmark():
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ ERROR: File gambar '{TEST_IMAGE}' tidak ditemukan!")
        return

    # Siapkan CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Backend", "Model", "Temp", "Iterasi", "Waktu (s)", "Respons", "Status"])

    print("=== MEMULAI BENCHMARK OLLAMA ===")
    print(f"Total percobaan per model: {len(TEMPERATURES) * ITERATIONS} request.\n")

    for model in MODELS:
        print(f"\n>>> Menguji Model: {model} <<<")
        try:
            ollama.chat(model=model, messages=[{'role': 'user', 'content': 'hi'}])
        except:
            print(f"Lewati {model} - Tidak ditemukan / Gagal dimuat.")
            continue

        for temp in TEMPERATURES:
            print(f"  > Suhu: {temp}")
            for i in range(1, ITERATIONS + 1):
                start_time = time.time()
                status = "Success"
                response_text = ""
                
                try:
                    res = ollama.chat(
                        model=model,
                        options={'temperature': temp, 'num_predict': 512, 'num_ctx': 4096},
                        messages=[{'role': 'user', 'content': PROMPT, 'images': [TEST_IMAGE]}]
                    )
                    response_text = res['message']['content'].replace('\n', ' ').strip()
                except Exception as e:
                    status = f"Error: {str(e)}"
                    print(f"    [Iter {i}] ❌ GAGAL - {status}")

                duration = time.time() - start_time
                
                print(f"    [Iter {i}/{ITERATIONS}] Waktu: {duration:.2f}s | Status: {status}")

                # Simpan ke CSV
                with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Ollama", model, temp, i, f"{duration:.2f}", response_text, status
                    ])

    print("\n✅ BENCHMARK OLLAMA SELESAI!")

if __name__ == "__main__":
    run_benchmark()