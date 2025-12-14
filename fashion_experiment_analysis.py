import cv2
import ollama
import pandas as pd
import time
import os
from tqdm import tqdm
from difflib import SequenceMatcher

# --- KONFIGURASI EKSPERIMEN ---
MODEL_NAME = 'fashion-advisor' 
OUTPUT_FILE = 'data_skripsi_bab4_robust.xlsx'
TEMP_IMAGE = 'experiment_capture.jpg'

# Berapa kali tes ulang untuk satu angka temperatur?
# Disarankan 5 atau 10 kali agar data valid.
NUM_ITERATIONS = 10 

# Prompt
PROMPT_TEXT = """
Kamu adalah Trion. Konteks: Pengguna ingin pergi ke 'Pesta Pernikahan'.
Lakukan analisis visual dan jawab dengan format:
[KARAKTER] (Analisis fisik)
[OUTFIT] (Analisis baju)
[SARAN] (Rekomendasi)
Gunakan Bahasa Indonesia.
"""

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return False
    print("📸 Tekan SPASI untuk ambil foto sampel...")
    while True:
        ret, frame = cap.read()
        cv2.imshow('Ambil Foto', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(TEMP_IMAGE, frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return True

def calculate_similarity(text1, text2):
    if not text1: return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_auto_score(text):
    score = 0
    lower_text = text.lower()
    # Kriteria Penilaian
    if "kulit" in lower_text or "tone" in lower_text: score += 10
    if "wajah" in lower_text or "muka" in lower_text: score += 10
    if "tubuh" in lower_text or "badan" in lower_text: score += 10
    if "warna" in lower_text: score += 10
    
    length = len(lower_text)
    if 200 < length < 1000: score += 30 # Panjang ideal
    elif length >= 1000: score += 10
    else: score += 5

    if "[KARAKTER]" in text and "[OUTFIT]" in text: score += 30
    return score

def run_experiment():
    if not capture_image(): return

    results = []
    temperatures = [round(x * 0.1, 1) for x in range(1, 11)] # 0.1 s/d 1.0

    print(f"\n🧪 Memulai Eksperimen ({len(temperatures)} Suhu x {NUM_ITERATIONS} Pengulangan)...")
    
    # Progress bar total
    total_runs = len(temperatures) * NUM_ITERATIONS
    pbar = tqdm(total=total_runs, desc="Processing")

    for temp in temperatures:
        
        # Variabel untuk menampung jawaban PERTAMA di suhu ini
        # Gunanya untuk membandingkan konsistensi: Jawaban ke-2,3,4,5 mirip gak sama Jawaban ke-1?
        baseline_response = "" 

        for i in range(1, NUM_ITERATIONS + 1):
            try:
                start_time = time.time()
                
                response = ollama.chat(
                    model=MODEL_NAME,
                    options={'temperature': temp, 'num_predict': 300},
                    messages=[{'role': 'user', 'content': PROMPT_TEXT, 'images': [TEMP_IMAGE]}]
                )
                current_response = response['message']['content']
                duration = round(time.time() - start_time, 2)
                
                # --- HITUNG KONSISTENSI ---
                # Jika ini iterasi pertama, dia jadi patokan (baseline)
                if i == 1:
                    baseline_response = current_response
                    similarity_score = 1.0 # Mirip 100% dengan dirinya sendiri
                else:
                    # Bandingkan jawaban ini dengan jawaban pertama tadi
                    similarity_score = calculate_similarity(baseline_response, current_response)
                
                # Hitung Kualitas
                quality_score = calculate_auto_score(current_response)

                results.append({
                    'Temperature': temp,
                    'Iterasi Ke': i,
                    'Durasi (s)': duration,
                    'Konsistensi (%)': round(similarity_score * 100, 2),
                    'Skor Kualitas': quality_score,
                    'Response': current_response
                })
                
            except Exception as e:
                print(f"Error {temp}-{i}: {e}")
            
            pbar.update(1)

    pbar.close()
    
    # Simpan Excel
    df = pd.DataFrame(results)
    
    # --- FITUR BONUS: MENGHITUNG RATA-RATA OTOMATIS ---
    print("\n📊 Menghitung Rata-Rata...")
    # Group by Temperature dan cari rata-rata skor
    summary = df.groupby('Temperature')[['Konsistensi (%)', 'Skor Kualitas', 'Durasi (s)']].mean()
    
    # Simpan ke sheet berbeda dalam satu file Excel
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        df.to_excel(writer, sheet_name='Data Mentah', index=False)
        summary.to_excel(writer, sheet_name='Rata-Rata (Analisa Bab 4)')
        
    print(f"✅ Selesai! Cek file '{OUTPUT_FILE}'.")
    print("Gunakan Sheet 'Rata-Rata' untuk membuat grafik Skripsi.")

    if os.path.exists(TEMP_IMAGE):
        os.remove(TEMP_IMAGE)

if __name__ == "__main__":
    run_experiment()