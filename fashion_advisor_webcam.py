import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
import ollama
import os
import threading
import time
from gtts import gTTS
import pygame
from PIL import Image, ImageTk

# --- KONFIGURASI ---
# Gunakan model kustom Anda 'fashion-advisor' atau 'qwen2.5vl:7b'
MODEL_NAME = 'fashion-advisor' 
AUDIO_FILENAME = "trion_voice.mp3"
TEMP_IMAGE_PATH = "temp_capture.jpg"

class FashionAdvisorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trion: AI Personal Fashion Stylist")
        self.root.geometry("950x800")
        
        # Inisialisasi Audio Mixer
        pygame.mixer.init()

        # Variabel Status
        self.webcam_running = True

        # Setup Webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Webcam tidak terdeteksi!")
            self.root.destroy()
            return
        
        # Setup UI
        self.setup_ui()
        
        # Mulai Loop Webcam
        self.update_webcam()
        
        # Event saat aplikasi ditutup
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # Styling
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        style.configure("TLabel", font=("Segoe UI", 12))

        # --- 1. Panel Atas (Input) ---
        input_frame = ttk.LabelFrame(self.root, text="Konteks Acara", padding=15)
        input_frame.pack(fill="x", padx=15, pady=10)

        # Input Text
        ttk.Label(input_frame, text="Mau pergi ke mana?").pack(side=tk.LEFT, padx=(0, 10))
        self.occasion_entry = ttk.Entry(input_frame, width=40, font=("Segoe UI", 11))
        self.occasion_entry.pack(side=tk.LEFT, padx=5)
        self.occasion_entry.insert(0, "Kerja di Kafe") # Contoh default

        # Tombol Analisis
        self.analyze_button = ttk.Button(
            input_frame, 
            text="✨ Cek Penampilanku", 
            command=self.capture_and_analyze
        )
        self.analyze_button.pack(side=tk.LEFT, padx=15)

        # Tombol Stop Suara
        self.stop_button = ttk.Button(
            input_frame, 
            text="⛔ Stop Suara", 
            command=self.stop_audio,
            state="disabled"
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Label Status
        self.status_label = ttk.Label(input_frame, text="Siap", foreground="gray")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # --- 2. Panel Tengah (Webcam) ---
        video_frame = ttk.Frame(self.root, padding=5, relief="groove", borderwidth=2)
        video_frame.pack(fill="both", expand=True, padx=15, pady=5)
        
        self.webcam_label = ttk.Label(video_frame)
        self.webcam_label.pack(fill="both", expand=True)

        # --- 3. Panel Bawah (Output Teks) ---
        output_frame = ttk.LabelFrame(self.root, text="Saran Trion", padding=10)
        output_frame.pack(fill="both", expand=True, padx=15, pady=15)

        self.output_text = scrolledtext.ScrolledText(
            output_frame, 
            height=8, 
            font=("Segoe UI", 12), 
            wrap=tk.WORD,
            bg="#f0f0f0",
            bd=0
        )
        self.output_text.pack(fill="both", expand=True)
        self.output_text.insert(tk.END, "Halo! Aku Trion. Tulis acaramu di atas lalu klik tombol untuk mulai analisis.")
        self.output_text.config(state='disabled')

    def update_webcam(self):
        """Mengupdate tampilan webcam secara real-time"""
        if not self.webcam_running: return
        
        ret, frame = self.cap.read()
        if ret:
            # Flip horizontal (efek cermin) untuk preview
            frame_flip = cv2.flip(frame, 1)
            
            # Konversi warna BGR ke RGB
            cv_image = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            
            # Resize agar pas di window (opsional, menjaga aspek rasio)
            # pil_image.thumbnail((800, 600)) 
            
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
        
        self.root.after(15, self.update_webcam)

    def capture_and_analyze(self):
        occasion = self.occasion_entry.get()
        if not occasion.strip():
            messagebox.showwarning("Input Kosong", "Tulis dulu acaranya, biar sarannya pas!")
            return

        # 1. Ambil Frame Webcam Asli (Tanpa Flip)
        ret, frame = self.cap.read()
        if not ret: return

        # Simpan gambar sementara
        cv2.imwrite(TEMP_IMAGE_PATH, frame)

        # 2. Update UI
        self.analyze_button.config(state="disabled")
        self.status_label.config(text="Sedang Menganalisis... 🧠", foreground="blue")
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Sedang melihat penampilanmu...")
        self.output_text.config(state='disabled')

        # 3. Jalankan Logic AI di Thread Terpisah
        threading.Thread(target=self.run_ai_pipeline, args=(occasion,), daemon=True).start()

    def run_ai_pipeline(self, occasion):
        try:
            # --- TAHAP 1: MEMBANGUN PROMPT CERDAS ---
            prompt = f"""
            Kamu adalah Trion, Fashion Stylist pribadi yang cerdas, gaul, dan sangat teliti.
            Tugasmu: Berikan saran fashion untuk acara '{occasion}'.

            Lakukan langkah berpikir ini:
            1. LIHAT ORANGNYA: Analisis warna kulit (undertone), bentuk wajah, dan tipe tubuh pengguna di gambar.
            2. LIHAT BAJUNYA: Identifikasi apa yang sedang dia pakai sekarang.
            3. NILAI KECOCOKAN: Apakah baju itu cocok dengan fisik dia DAN cocok dengan acara '{occasion}'?

            OUTPUT (Penting):
            - Jika sudah oke: Puji dia dengan menyebutkan detail kenapa itu cocok (misal: "Warna ini bikin kulitmu cerah").
            - Jika kurang oke: Berikan saran perbaikan yang sopan tapi konkret (misal: "Ganti ke warna X biar lebih fresh").
            - Gunakan BAHASA INDONESIA yang santai, akrab (pakai 'kamu'/'aku'), dan natural.
            - Jawab dalam bentuk paragraf pendek (cerita). JANGAN pakai poin-poin/bullet list.
            - Maksimal 4 kalimat.
            """

            print(">> Mengirim ke Ollama...")
            
            # --- TAHAP 2: INFERENCE (OLLAMA) ---
            response = ollama.chat(
                model=MODEL_NAME,
                # Parameter untuk hasil optimal (Kreatif tapi Fokus)
                options={
                    'temperature': 0.6, 
                    'num_predict': 250, 
                    'top_p': 0.9,
                },
                messages=[{'role': 'user', 'content': prompt, 'images': [TEMP_IMAGE_PATH]}]
            )
            
            raw_text = response['message']['content']
            
            # Bersihkan teks agar enak dibaca TTS
            clean_text = raw_text.replace("*", "").replace("#", "").replace("- ", "").replace("\n", " ")

            # Update GUI Teks
            self.root.after(0, self.update_text_result, clean_text)

            # --- TAHAP 3: TEXT-TO-SPEECH (GOOGLE) ---
            print(">> Mengunduh suara Google...")
            self.root.after(0, lambda: self.status_label.config(text="Mengunduh Suara... 🔊", foreground="orange"))
            
            tts = gTTS(text=clean_text, lang='id')
            
            # Hapus file lama jika ada (trik agar tidak error permission)
            if os.path.exists(AUDIO_FILENAME):
                try:
                    os.remove(AUDIO_FILENAME)
                except: pass 

            tts.save(AUDIO_FILENAME)

            # --- TAHAP 4: PLAY AUDIO ---
            self.root.after(0, self.play_audio)

        except Exception as e:
            print(f"Error Pipeline: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Terjadi kesalahan:\n{str(e)}"))
            self.root.after(0, self.reset_ui)
        finally:
            # Hapus gambar temp
            if os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)

    def update_text_result(self, text):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state='disabled')

    def play_audio(self):
        """Memutar audio dengan Pygame"""
        try:
            self.status_label.config(text="Sedang Bicara... 🗣️", foreground="green")
            self.stop_button.config(state="normal")
            
            pygame.mixer.music.load(AUDIO_FILENAME)
            pygame.mixer.music.play()
            
            # Thread untuk memantau kapan audio selesai
            threading.Thread(target=self.wait_audio_finish, daemon=True).start()
            
        except Exception as e:
            print(f"Gagal play audio: {e}")
            self.reset_ui()

    def wait_audio_finish(self):
        """Looping menunggu audio selesai"""
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        self.root.after(0, self.reset_ui)

    def stop_audio(self):
        """Mematikan suara paksa"""
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        self.reset_ui()

    def reset_ui(self):
        """Kembalikan tombol ke keadaan semula"""
        self.analyze_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Siap", foreground="gray")

    def on_close(self):
        """Bersih-bersih saat keluar"""
        print("Menutup aplikasi...")
        self.webcam_running = False
        self.cap.release()
        
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            
            # Tunggu sebentar sebelum hapus file audio agar tidak error 'file in use'
            time.sleep(0.5) 
            if os.path.exists(AUDIO_FILENAME):
                os.remove(AUDIO_FILENAME)
            if os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)
        except: pass
        
        self.root.destroy()

if __name__ == "__main__":
    # Cek Koneksi Ollama
    try:
        ollama.list()
        print("✅ Koneksi Ollama berhasil.")
    except:
        messagebox.showerror("Fatal Error", "Ollama belum berjalan!\nJalankan 'ollama serve' di terminal.")
        exit()

    root = tk.Tk()
    app = FashionAdvisorApp(root)
    root.mainloop()