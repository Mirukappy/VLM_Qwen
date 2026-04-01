import pandas as pd
import os

# --- KONFIGURASI FILE ---
FILE_DATABASE = "DATABASE_EVALUASI_FINAL_3.csv"

def buat_ringkasan_skripsi():
    print("=== MEMULAI ANALISIS HIRARKI UNTUK BAB 4 ===")
    
    # Pastikan file database ada
    if not os.path.exists(FILE_DATABASE):
        print(f"❌ ERROR: File '{FILE_DATABASE}' tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan evaluasi_master.py terlebih dahulu.")
        return

    # Membaca database
    df = pd.read_csv(FILE_DATABASE)
    
    # Pastikan data yang dihitung hanya yang statusnya "Success" 
    # (opsional, tapi bagus agar error tidak merusak rata-rata waktu)
    # df = df[df['Status'].str.contains("Success", na=False)] 

    # =====================================================================
    # TABEL 1: KOMPARASI LEVEL BACKEND (vLLM vs Ollama)
    # =====================================================================
    print("\n" + "="*60)
    print("TABEL 1: PERBANDINGAN PERFORMA BACKEND (OLLAMA vs vLLM)")
    print("="*60)
    
    tabel_backend = df.groupby('Backend').agg({
        'Skor_Kualitas': 'mean',
        'Waktu (s)': 'mean',
        'Respons': 'count'
    }).reset_index().round(2)
    
    tabel_backend.columns = ['Backend', 'Rata_Skor_MOS', 'Rata_Waktu_Detik', 'Jumlah_Data']
    print(tabel_backend.to_csv(index=False))
    tabel_backend.to_csv("Ringkasan_1_Backend.csv", index=False)


    # =====================================================================
    # TABEL 2: KOMPARASI LEVEL MODEL (Spesifik per Backend)
    # =====================================================================
    print("\n" + "="*60)
    print("TABEL 2: PERBANDINGAN PERFORMA MODEL PER BACKEND")
    print("="*60)
    
    tabel_model = df.groupby(['Backend', 'Model']).agg({
        'Skor_Kualitas': 'mean',
        'Waktu (s)': 'mean',
        'Respons': 'count'
    }).reset_index().round(2)
    
    tabel_model.columns = ['Backend', 'Nama_Model', 'Rata_Skor_MOS', 'Rata_Waktu_Detik', 'Jumlah_Data']
    print(tabel_model.to_csv(index=False))
    tabel_model.to_csv("Ringkasan_2_Model.csv", index=False)


    # =====================================================================
    # TABEL 3: KOMPARASI LEVEL TEMPERATUR (Paling Detail)
    # =====================================================================
    print("\n" + "="*60)
    print("TABEL 3: PENGARUH TEMPERATUR TERHADAP MODEL & BACKEND")
    print("="*60)
    
    tabel_temp = df.groupby(['Backend', 'Model', 'Temp']).agg({
        'Skor_Kualitas': 'mean',
        'Waktu (s)': 'mean',
        'Respons': 'count'
    }).reset_index().round(2)
    
    tabel_temp.columns = ['Backend', 'Nama_Model', 'Temperatur', 'Rata_Skor_MOS', 'Rata_Waktu_Detik', 'Jumlah_Data']
    print(tabel_temp.to_csv(index=False))
    tabel_temp.to_csv("Ringkasan_3_Temperatur.csv", index=False)

    print("\n✅ ANALISIS SELESAI!")
    print("💡 Tips: Anda bisa memblok teks yang dipisahkan koma di atas, copy, lalu paste langsung ke Excel.")
    print("📁 Program juga telah menyimpan 3 file CSV baru (Ringkasan_1, 2, dan 3) di folder ini.")

if __name__ == "__main__":
    buat_ringkasan_skripsi()