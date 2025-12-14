import cv2
import ollama
import pandas as pd
import time
import os
from tqdm import tqdm # Progress bar

# --- KONFIGURASI ---
MODEL_NAME = 'fashion-advisor' 
TEMP_IMAGE = 'experiment_source.jpg'
CONTEXT = "Pesta Pernikahan" # Konteks tetap untuk pengujian

# --- DEFINISI 3 PROMPT BERBEDA ---

# PROMPT 1: KHUSUS FISIK
PROMPT_FISIK = """
Analisis karakteristik fisik orang di gambar ini secara objektif.
Fokus HANYA pada:
1. Warna kulit (Skin tone).
2. Bentuk wajah.
3. Tipe tubuh.
Jawab singkat dan padat dalam Bahasa Indonesia.
"""

# PROMPT 2: KHUSUS OUTFIT
PROMPT_BAJU = """
Deskripsikan HANYA pakaian yang dikenakan orang ini.
Abaikan wajah dan tubuh.
Sebutkan: Warna, Jenis Pakaian, dan Pola/Motif jika ada.
Jawab singkat dalam Bahasa Indonesia.
"""

# PROMPT 3: ADVISOR (LOGIKA GABUNGAN)
# Di sini kita minta AI melihat gambar + konteks untuk memberi saran
PROMPT_ADVISOR = f"""
Kamu adalah Trion, Fashion Advisor.
Konteks Acara: {CONTEXT}.

Berdasarkan gambar, analisis apakah baju yang dipakai COCOK untuk {CONTEXT}?
Jika tidak, berikan saran perbaikan.
Jika ya, berikan pujian.
Jawab dalam Bahasa Indonesia yang santai.
"""

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Webcam tidak terdeteksi")
        return False
    
    print(f"\n📸 KAMERA AKTIF.")
    print(f"   Tekan [SPASI] untuk mengambil foto sampel.")
    print(f"   Foto ini akan dipakai untuk SEMUA 300 pengujian.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Tampilkan panduan
        cv2.imshow('Ambil Foto Sumber Eksperimen', frame)
        
        if cv2.waitKey(1) & 0xFF == ord(' '): # Tekan Spasi
            cv2.imwrite(TEMP_IMAGE, frame)
            print("✅ Foto sumber tersimpan!")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return True

def run_module_test(module_name, prompt_text, filename):
    """
    Fungsi generik untuk menjalankan tes 1 modul
    """
    print(f"\n🚀 MEMULAI MODUL: {module_name}")
    print(f"📂 Output nanti disimpan di: {filename}")
    
    results = []
    
    # Range Temperature: 0.1, 0.2 ... 1.0
    temperatures = [round(x * 0.1, 1) for x in range(1, 11)]
    
    # Iterasi per temperature: 10 kali
    ITERATIONS = 10
    
    total_runs = len(temperatures) * ITERATIONS
    
    # Progress Bar
    pbar = tqdm(total=total_runs, desc=f"Testing {module_name}")
    
    for temp in temperatures:
        for i in range(1, ITERATIONS + 1):
            start_time = time.time()
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    options={
                        'temperature': temp,
                        'num_predict': 200, # Batas panjang jawaban
                    },
                    messages=[{
                        'role': 'user', 
                        'content': prompt_text, 
                        'images': [TEMP_IMAGE]
                    }]
                )
                
                answer = response['message']['content']
                duration = round(time.time() - start_time, 2)
                
                results.append({
                    'Modul': module_name,
                    'Temperature': temp,
                    'Iterasi Ke': i,
                    'Durasi (s)': duration,
                    'Respon AI': answer
                })
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({'Temperature': temp, 'Respon AI': "ERROR"})
            
            pbar.update(1)
            
    pbar.close()
    
    # Simpan ke Excel khusus modul ini
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"✅ Data {module_name} tersimpan di {filename}")

def main():
    # 1. Ambil Foto Sekali
    if not capture_image(): return

    # 2. Jalankan Tes Modul 1: FISIK
    run_module_test(
        module_name="Analisis Fisik", 
        prompt_text=PROMPT_FISIK, 
        filename="data_modul_fisik.xlsx"
    )

    # 3. Jalankan Tes Modul 2: BAJU
    run_module_test(
        module_name="Analisis Outfit", 
        prompt_text=PROMPT_BAJU, 
        filename="data_modul_baju.xlsx"
    )

    # 4. Jalankan Tes Modul 3: SARAN (ADVISOR)
    run_module_test(
        module_name="Rekomendasi Advisor", 
        prompt_text=PROMPT_ADVISOR, 
        filename="data_modul_saran.xlsx"
    )

    print("\n🎉🎉 SEMUA EKSPERIMEN SELESAI! 🎉🎉")
    print("Silakan cek 3 file Excel yang baru dibuat di folder ini.")
    
    # Bersihkan gambar
    if os.path.exists(TEMP_IMAGE):
        os.remove(TEMP_IMAGE)

if __name__ == "__main__":
    main()