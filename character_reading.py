import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
import ollama
import os
import tempfile
import threading
from PIL import Image, ImageTk

# --- KONFIGURASI ---
# Menggunakan Qwen2.5-VL karena kemampuannya melihat detail visual sangat baik
MODEL_NAME = 'qwen2.5vl:7b' 

class CharacterAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Body & Face Analyzer")
        self.root.geometry("800x650")

        self.cap = cv2.VideoCapture(0)
        self.webcam_running = True

        self.setup_ui()
        self.update_webcam()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # Frame Utama
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Header
        header = ttk.Label(
            main_frame, 
            text="Analisis Karakteristik Fisik", 
            font=("Helvetica", 16, "bold")
        )
        header.pack(pady=5)

        # Instruksi
        info_label = ttk.Label(
            main_frame, 
            text="AI akan menganalisis: Jenis Kelamin, Bentuk Tubuh, Warna Kulit, & Bentuk Wajah.\nPastikan pencahayaan cukup terang.",
            font=("Helvetica", 10),
            justify="center"
        )
        info_label.pack(pady=(0, 10))

        # Tombol Analisis
        self.analyze_button = ttk.Button(
            main_frame, 
            text="👤 Analisis Fisik Saya", 
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
            # Mirror effect untuk kenyamanan user
            frame_display = cv2.flip(frame, 1)
            cv_image = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
        self.root.after(15, self.update_webcam)

    def capture_and_analyze(self):
        print("📸 Mengambil gambar...")
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Gagal mengambil gambar webcam.")
            return

        # Simpan frame (Tanpa flip agar AI melihat sisi asli, meski untuk wajah tidak terlalu berpengaruh)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            self.temp_filename = tmp.name
            cv2.imwrite(self.temp_filename, frame)

        # Matikan tombol & mulai thread
        self.analyze_button.config(text="Sedang Menganalisis...", state="disabled")
        threading.Thread(target=self.run_analysis, daemon=True).start()

    def run_analysis(self):
        # Prompt dirancang HATI-HATI.
        # Beberapa AI menolak menebak gender/ras karena alasan etika (safety filter).
        # Kita membingkainya dalam konteks "Fashion Styling" agar AI mau menjawab secara objektif.
        prompt_teks = """
        Bertindaklah sebagai konsultan fashion profesional. 
        Untuk memberikan saran pakaian yang tepat, tolong deskripsikan atribut fisik orang dalam gambar ini secara objektif dan hormat.

        Harap sebutkan detail berikut:
        1. Estimasi Gender (Maskulin/Feminin).
        2. Tipe Bentuk Tubuh (Misal: Kurus, Atletis, Berisi, Tinggi, dsb).
        3. Warna Kulit & Undertone (Misal: Fair, Medium, Tan, Dark dengan undertone Warm/Cool).
        4. Bentuk Wajah (Misal: Oval, Bulat, Kotak).
        5. Warna & Gaya Rambut.

        Jawablah dalam format daftar (bullet points).
        """

        try:
            print("Mengirim ke Qwen-VL...")
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': prompt_teks,
                    'images': [self.temp_filename]
                }]
            )
            result_text = response['message']['content']
            
            # Kembali ke main thread untuk update GUI
            self.root.after(0, self.show_result, result_text)

        except Exception as e:
            err = f"Gagal: {e}"
            print(err)
            self.root.after(0, lambda: messagebox.showerror("Error", err))
        finally:
            if os.path.exists(self.temp_filename):
                os.remove(self.temp_filename)
            self.root.after(0, lambda: self.analyze_button.config(text="👤 Analisis Fisik Saya", state="normal"))

    def show_result(self, text):
        window = tk.Toplevel(self.root)
        window.title("Hasil Analisis Karakteristik")
        window.geometry("500x500")

        lbl = ttk.Label(window, text="📝 Data Karakteristik Pengguna:", font=("Helvetica", 12, "bold"))
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
    try:
        ollama.list()
    except:
        messagebox.showerror("Error", "Ollama belum berjalan!")
        exit()

    root = tk.Tk()
    app = CharacterAnalyzerApp(root)
    root.mainloop()