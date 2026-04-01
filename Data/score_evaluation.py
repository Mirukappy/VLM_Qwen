import pandas as pd
import ollama
import re
import os
import glob
import time

# --- 1. KONFIGURASI JURI & KUNCI JAWABAN ---
MODEL_JURI = "qwen2.5:7b" 
KONTEKS_ACARA = "wawancara kerja di kantor korporat"

# ⚠️ PENTING: Ganti deskripsi ini sesuai dengan baju asli yang ada di foto Anda!
KUNCI_JAWABAN = {
    # Kategori Kaos Oblong Ungu
    "Shirt_0": "Pria memakai kaos oblong warna ungu polos (tidak memakai kacamata maupun topi).",
    "Shirt_1": "Pria memakai kaos oblong warna ungu dan memakai kacamata.",
    "Shirt_2": "Pria memakai kaos oblong warna ungu dan memakai topi warna putih.",
    
    # Kategori Kaos Kerah Abu-abu
    "TShirt_0": "Pria memakai kaos kerah (polo shirt) warna abu-abu (tidak memakai kacamata maupun topi).",
    "TShirt_1": "Pria memakai kaos kerah (polo shirt) warna abu-abu dan memakai kacamata.",
    "TShirt_2": "Pria memakai kaos kerah (polo shirt) warna abu-abu dan memakai topi warna putih.",
    
    # Kategori Jaket Hitam
    "Jacket_0": "Pria memakai jaket warna hitam (tidak memakai kacamata maupun topi).",
    "Jacket_1": "Pria memakai jaket warna hitam dan memakai kacamata.",
    "Jacket_2": "Pria memakai jaket warna hitam dan memakai topi warna putih."
}

def ekstrak_kategori_dari_nama_file(nama_file):
    nama_bersih = os.path.splitext(os.path.basename(nama_file))[0]
    for key in KUNCI_JAWABAN.keys():
        if key in nama_bersih:
            return key
    return None

def ambil_skor_juri(teks_jawaban, fakta_visual):
    # Cek apakah teks jawaban dari CSV kosong/error
    if pd.isna(teks_jawaban) or len(str(teks_jawaban)) < 5 or "Error" in str(teks_jawaban):
        print(f"   ⚠️ SKIP: Jawaban di CSV kosong atau berisi error -> {teks_jawaban}")
        return 1
    
    prompt = f"""Tugas: Berikan skor 1-5 untuk jawaban AI Fashion Assistant.
Fakta Gambar Sebenarnya: {fakta_visual}
Konteks: {KONTEKS_ACARA}

Jawaban AI yang dinilai: "{teks_jawaban}"

Kriteria:
5: Visual 100% akurat (jenis & warna sama dengan fakta), format 1-2-3 lengkap, saran sangat profesional.
4: Visual akurat, format lengkap, saran standar.
3: Ada kesalahan kecil (misal: warna sedikit meleset), atau format tidak lengkap.
2: Halusinasi parah (menyebut pakaian yang sama sekali berbeda), saran tidak masuk akal.
1: Jawaban rusak, kosong, atau ngawur.

Balas HANYA dengan SATU ANGKA (1, 2, 3, 4, atau 5)."""

    try:
        response = ollama.chat(
            model=MODEL_JURI, 
            options={'temperature': 0.0, 'num_predict': 5},
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        raw_output = response['message']['content']
        skor = re.findall(r'[1-5]', raw_output)
        
        if skor:
            return int(skor[0])
        else:
            print(f"   ⚠️ JURI NGAWUR: Juri tidak mengeluarkan angka 1-5. Output Juri: '{raw_output}'")
            return 1
            
    except Exception as e:
        # INI YANG PALING PENTING: Menampilkan error koneksi ke Ollama
        print(f"   ❌ ERROR OLLAMA JURI: {str(e)}")
        return 1

def proses_master_evaluasi():
    print("=== MEMULAI MASTER EVALUATOR ===")
    
    semua_file_csv = glob.glob("Data/benchmark_*.csv")
    
    if not semua_file_csv:
        print("❌ Tidak ada file CSV benchmark ditemukan di folder ini.")
        return

    print(f"Ditemukan {len(semua_file_csv)} file untuk dievaluasi.\n")
    list_df_hasil = []

    for file_csv in semua_file_csv:
        backend = "vLLM" if "vllm" in file_csv.lower() else "Ollama"
        kategori_gambar = ekstrak_kategori_dari_nama_file(file_csv)
        
        if not kategori_gambar:
            print(f"⚠️ Skip: Tidak dapat menentukan kategori gambar untuk file {file_csv}")
            continue
            
        fakta_visual_aktif = KUNCI_JAWABAN[kategori_gambar]
        
        print(f"📄 Menilai: {file_csv}")
        
        df = pd.read_csv(file_csv)
        skor_kualitas = []
        
        start_eval_waktu = time.time()
        
        for i, row in df.iterrows():
            skor = ambil_skor_juri(row['Respons'], fakta_visual_aktif)
            skor_kualitas.append(skor)
        
        durasi_eval = time.time() - start_eval_waktu
        print(f"   ✅ Selesai ({len(df)} baris) dalam {durasi_eval:.2f} detik.")
        
        df['Skor_Kualitas'] = skor_kualitas
        df['Kategori_Baju'] = kategori_gambar 
        list_df_hasil.append(df)

    if not list_df_hasil:
        return

    print("\nMenggabungkan semua data...")
    df_total = pd.concat(list_df_hasil, ignore_index=True)
    
    file_database_akhir = "DATABASE_EVALUASI_FINAL_3.csv"
    df_total.to_csv(file_database_akhir, index=False)

    print("\n" + "="*60)
    print("HASIL ANALISIS RATA-RATA (FORMAT CSV SIAP COPY)")
    print("="*60)

    # Grouping 1: Rata-rata Skor per Model
    analisis_model = df_total.groupby(['Backend', 'Model']).agg({
        'Skor_Kualitas': ['mean', 'std'],
        'Waktu (s)': 'mean',
        'Respons': 'count'
    }).round(2)
    
    analisis_model.columns = ['Rata_Skor_MOS', 'Std_Deviasi', 'Rata_Waktu_s', 'Total_Tes']
    
    # Mencetak format CSV murni ke terminal
    print("\n--- DATA PERFORMA MODEL (COPY DI BAWAH INI) ---")
    print(analisis_model.to_csv())
    
    # Grouping 2: Rata-rata Skor per Kategori Baju
    analisis_kategori = df_total.groupby('Kategori_Baju')['Skor_Kualitas'].mean().round(2).reset_index()
    analisis_kategori.columns = ['Kategori_Pakaian', 'Rata_Skor_Akurasi']
    
    # Mencetak format CSV murni ke terminal
    print("\n--- DATA TINGKAT KESULITAN KATEGORI (COPY DI BAWAH INI) ---")
    print(analisis_kategori.to_csv(index=False))

    print("\n✅ Database lengkap juga tersimpan di file:", file_database_akhir)

if __name__ == "__main__":
    proses_master_evaluasi()