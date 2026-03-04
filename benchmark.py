import ollama
import pandas as pd
import time
import os
from difflib import SequenceMatcher
from tqdm import tqdm

# ==========================================
# 1. KONFIGURASI EKSPERIMEN (Sesuai Bab 3)
# ==========================================
MODEL_NAME = "fashion-advisor"  # Pastikan model ini sudah di-pull di Ollama
TEST_IMAGE_PATH = "sample_test.jpg" # Ganti dengan foto baju yang akan dites
OUTPUT_FILE = "Hasil_Benchmark_Bab4.xlsx"

# Rentang Temperature (0.1 sd 1.0)
TEMPERATURE_LIST = [round(x * 0.1, 1) for x in range(1, 11)]

# Jumlah pengulangan per temperature (Sesuai Bab 3: 10x)
NUM_ITERATIONS = 10 

# System Prompt (In-Context Learning / Chain-of-Thought)
SYSTEM_PROMPT = """
Kamu adalah Personal Fashion Stylist profesional bernama 'Trion'.
Tugasmu adalah menganalisis penampilan pengguna dari foto dan memberikan saran.

Aturan Output:
1. Jawab dalam Bahasa Indonesia yang luwes.
2. Ikuti format ketat berikut:
   [KARAKTER] <Analisis fisik pengguna: warna kulit, bentuk wajah, rambut>
   [OUTFIT] <Analisis pakaian yang dipakai sekarang: warna, jenis, gaya>
   [SARAN] <Rekomendasi mix-and-match atau perbaikan gaya>
"""

# ==========================================
# 2. MODUL PENILAIAN OTOMATIS (HEURISTIC)
# ==========================================
def calculate_consistency(text1, text2):
    """Mengukur kemiripan teks menggunakan Sequence Matcher Ratio (0.0 - 1.0)"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_quality_score(text):
    """
    Rule-Based Heuristic Scoring (0-100)
    Sesuai Bab 3.4: Kelengkapan Atribut, Format, dan Panjang.
    """
    score = 0
    text_lower = text.lower()

    # A. Kepatuhan Format (Bobot 30)
    if "[karakter]" in text_lower: score += 15
    if "[outfit]" in text_lower: score += 15

    # B. Kelengkapan Atribut Visual (Bobot 40)
    keywords = ["kulit", "wajah", "rambut", "warna", "cocok", "gaya"]
    found_keywords = sum(1 for word in keywords if word in text_lower)
    # Maksimal 40 poin proporsional
    score += min(40, found_keywords * 7) 

    # C. Kedalaman Informasi / Panjang Teks (Bobot 30)
    length = len(text)
    if 200 <= length <= 1000: # Ideal
        score += 30
    elif length > 1000: # Terlalu panjang
        score += 15
    elif 50 < length < 200: # Terlalu pendek
        score += 10
    
    return score

# ==========================================
# 3. ENGINE UTAMA BENCHMARK
# ==========================================
def run_benchmark():
    print(f"🚀 Memulai Benchmark Skripsi...")
    print(f"Model: {MODEL_NAME}")
    print(f"Total Eksekusi: {len(TEMPERATURE_LIST) * NUM_ITERATIONS} kali running\n")

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: File gambar '{TEST_IMAGE_PATH}' tidak ditemukan!")
        print("Silakan taruh 1 foto sampel (jpg/png) di folder yang sama.")
        return

    results = []
    
    # Progress Bar Total
    pbar = tqdm(total=len(TEMPERATURE_LIST) * NUM_ITERATIONS, desc="Testing", unit="run")

    for temp in TEMPERATURE_LIST:
        baseline_response = "" # Jawaban pertama dijadikan patokan konsistensi
        
        for i in range(1, NUM_ITERATIONS + 1):
            try:
                # --- 1. Catat Waktu Mulai ---
                start_time = time.time()
                
                # --- 2. Inferensi ke Ollama (Prompt Engineering) ---
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': "Analisis gaya saya.", 'images': [TEST_IMAGE_PATH]}
                    ],
                    options={
                        'temperature': temp,
                        'num_predict': 500, # Batasi output token agar tidak terlalu lama
                    }
                )
                
                # --- 3. Hitung Durasi ---
                end_time = time.time()
                duration = round(end_time - start_time, 2)
                
                output_text = response['message']['content']

                # --- 4. Hitung Metrik ---
                # Skor Kualitas (Heuristik)
                quality_score = calculate_quality_score(output_text)

                # Skor Konsistensi (Bandingkan dengan iterasi ke-1)
                if i == 1:
                    baseline_response = output_text
                    consistency_score = 100.0 # Iterasi pertama selalu 100% konsisten dgn dirinya
                else:
                    ratio = calculate_consistency(baseline_response, output_text)
                    consistency_score = round(ratio * 100, 2)

                # Simpan Data
                results.append({
                    "Temperature": temp,
                    "Iterasi": i,
                    "Durasi (detik)": duration,
                    "Konsistensi (%)": consistency_score,
                    "Skor Kualitas": quality_score,
                    "Output Model": output_text[:100] + "..." # Simpan potongan teks
                })

            except Exception as e:
                print(f"\nError pada Temp {temp} Iterasi {i}: {e}")
            
            pbar.update(1)
            
    pbar.close()

    # ==========================================
    # 4. SIMPAN HASIL KE EXCEL
    # ==========================================
    print("\n📊 Menyimpan data...")
    df = pd.DataFrame(results)

    # Buat Pivot Table (Rata-rata per Temperature) - Ini untuk Tabel Bab 4 Anda
    df_summary = df.groupby("Temperature")[["Konsistensi (%)", "Skor Kualitas", "Durasi (detik)"]].mean().reset_index()

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        df_summary.to_excel(writer, sheet_name="Tabel Bab 4", index=False)
        df.to_excel(writer, sheet_name="Data Mentah", index=False)

    print(f"✅ Selesai! File tersimpan: {OUTPUT_FILE}")
    print("----------------------------------------------------")
    print("REVIEW HASIL (Rata-rata):")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()