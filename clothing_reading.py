import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
import ollama
import os
import tempfile
import threading
from PIL import Image, ImageTk

# --- KONFIGURASI ---
# Qwen2.5-VL sangat jago membaca teks (OCR)
MODEL_NAME = 'qwen2.5vl:7b' 

class ClothingReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Clothing & Text Reader")
        self.root.geometry("800x600")

        self.cap = cv2.VideoCapture(0)
        self.webcam_running = True

        # Setup GUI
        self.setup_ui()

        # Mulai Webcam
        self.update_webcam()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # Frame Utama
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Judul / Instruksi
        info_label = ttk.Label(
            main_frame, 
            text="Arahkan kamera ke baju. AI akan mendeskripsikan pakaian & membaca tulisannya.",
            font=("Helvetica", 12),
            wraplength=700,
            justify="center"
        )
        info_label.pack(pady=(0, 10))

        # Tombol Analisis
        # Kita buat tombolnya besar agar mudah ditekan
        self.analyze_button = ttk.Button(
            main_frame, 
            text="📸 Cek Pakaian & Baca Teks", 
            command=self.capture_and_analyze
        )
        self.analyze_button.pack(pady=5, ipadx=10, ipady=5)

        # Area Video Webcam
        self.webcam_label = ttk.Label(main_frame)
        self.webcam_label.pack(pady=10, fill="both", expand=True)

    def update_webcam(self):
        if not self.webcam_running:
            return
        ret, frame = self.cap.read()
        if ret:
            # Mirror effect (supaya tulisan di preview terbaca normal bagi user, 
            # TAPI nanti saat dikirim ke AI kita harus kirim gambar asli agar teks tidak terbalik)
            frame_display = cv2.flip(frame, 1)
            
            cv_image = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
        self.root.after(15, self.update_webcam)

    def capture_and_analyze(self):
        print("Mengambil gambar...")
        
        # 1. Ambil gambar RAW dari webcam (JANGAN di-flip/mirror)
        # Qwen butuh gambar asli agar teks terbaca dengan benar (kiri ke kanan)
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Gagal mengambil gambar webcam.")
            return

        # 2. Simpan ke file sementara
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            self.temp_filename = tmp.name
            cv2.imwrite(self.temp_filename, frame)

        # 3. Matikan tombol & mulai thread analisis
        self.analyze_button.config(text="Sedang Membaca...", state="disabled")
        threading.Thread(target=self.run_analysis, daemon=True).start()

    def run_analysis(self):
        # Prompt khusus untuk Deskripsi + OCR
        prompt_teks = """
        Analisis gambar ini secara detail. Fokus pada pakaian yang dikenakan orang tersebut.
        
        Lakukan 2 hal berikut:
        1. DESKRIPSI: Jelaskan warna, jenis, dan gaya pakaian yang dipakai.
        2. MEMBACA TEKS (PENTING): Jika ada tulisan, logo, atau teks apa pun pada baju/pakaian, tolong TULISKAN ULANG teks tersebut persis seperti yang tertulis. Jika teksnya dalam bahasa asing, coba terjemahkan.
        
        Jika tidak ada teks pada baju, katakan "Tidak ada teks yang terdeteksi".
        """

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': prompt_teks,
                    'images': [self.temp_filename]
                }]
            )
            result_text = response['message']['content']
            self.root.after(0, self.show_result, result_text)

        except Exception as e:
            err = f"Gagal: {e}"
            print(err)
            self.root.after(0, lambda: messagebox.showerror("Error", err))
        finally:
            # Bersihkan file & aktifkan tombol lagi
            if os.path.exists(self.temp_filename):
                os.remove(self.temp_filename)
            self.root.after(0, lambda: self.analyze_button.config(text="📸 Cek Pakaian & Baca Teks", state="normal"))

    def show_result(self, text):
        # Jendela Pop-up Hasil
        window = tk.Toplevel(self.root)
        window.title("Hasil Analisis Pakaian")
        window.geometry("500x400")

        lbl = ttk.Label(window, text="🔍 Hasil Deteksi:", font=("Helvetica", 12, "bold"))
        lbl.pack(pady=10)

        text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Helvetica", 11))
        text_area.pack(padx=10, pady=10, fill="both", expand=True)
        text_area.insert(tk.INSERT, text)
        text_area.config(state='disabled')

    def on_close(self):
        self.webcam_running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    # Cek koneksi lokal ke Ollama
    try:
        ollama.list()
    except:
        messagebox.showerror("Error", "Ollama belum berjalan! Jalankan aplikasi Ollama dulu.")
        exit()

    root = tk.Tk()
    app = ClothingReaderApp(root)
    root.mainloop()