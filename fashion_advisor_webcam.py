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
MODEL_NAME = 'qwen2.5vl:7b' 
AUDIO_FILENAME = "trion_voice.mp3"
TEMP_IMAGE_PATH = "temp_capture.jpg"

# 🛠️ PERBAIKAN 1: SYSTEM PROMPT (ATURAN MUTLAK)
# Dengan ditaruh sebagai 'system', AI tidak akan berani melanggar format ini.
SYSTEM_PROMPT = """Kamu adalah Fashion Stylist AI profesional. 
Aturan Mutlak: Kamu WAJIB menghasilkan HANYA 3 poin berikut secara berurutan. Dilarang melewatkan poin 2 dan 3!

1. DESKRIPSI: (Sebutkan baju, warna, dan celana/aksesoris jika ada).
2. ANALISIS: (Jelaskan apakah cocok atau tidak untuk acara user).
3. SARAN: (Berikan 1 saran perbaikan warna/pakaian yang konkret).

Jawab dengan singkat, padat, dan gunakan Bahasa Indonesia yang natural."""

class FashionAdvisorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trion: AI Personal Fashion Stylist (Fast Mode)")
        self.root.geometry("950x800")
        
        pygame.mixer.init()
        self.webcam_running = True

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Webcam tidak terdeteksi!")
            self.root.destroy()
            return
        
        self.setup_ui()
        self.update_webcam()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        threading.Thread(target=self.warmup_model, daemon=True).start()

    def warmup_model(self):
        """Pemanasan agar model langsung masuk VRAM GPU dan tidak lambat saat klik pertama"""
        print(f">> Memanaskan Model {MODEL_NAME} ke GPU...")
        try:
            # keep_alive='1h' memastikan model standby di GPU selama 1 jam, bikin inferensi selanjutnya secepat kilat!
            ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': 'hi'}], keep_alive='1h')
            print(">> Model Siap di VRAM GPU! 🚀")
            self.root.after(0, lambda: self.status_label.config(text="Sistem Siap (Fast Mode)", foreground="green"))
        except:
            print(">> Gagal warmup. Pastikan Ollama menyala.")

    def setup_ui(self):
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        style.configure("TLabel", font=("Segoe UI", 12))

        input_frame = ttk.LabelFrame(self.root, text="Konteks Acara", padding=15)
        input_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(input_frame, text="Mau pergi ke mana?").pack(side=tk.LEFT, padx=(0, 10))
        self.occasion_entry = ttk.Entry(input_frame, width=40, font=("Segoe UI", 11))
        self.occasion_entry.pack(side=tk.LEFT, padx=5)
        self.occasion_entry.insert(0, "Wawancara kerja di kantor korporat") 

        self.analyze_button = ttk.Button(input_frame, text="✨ Cek Penampilanku", command=self.capture_and_analyze)
        self.analyze_button.pack(side=tk.LEFT, padx=15)

        self.stop_button = ttk.Button(input_frame, text="⛔ Stop Suara", command=self.stop_audio, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(input_frame, text="Memuat Model...", foreground="orange")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        video_frame = ttk.Frame(self.root, padding=5, relief="groove", borderwidth=2)
        video_frame.pack(fill="both", expand=True, padx=15, pady=5)
        
        self.webcam_label = ttk.Label(video_frame)
        self.webcam_label.pack(fill="both", expand=True)

        output_frame = ttk.LabelFrame(self.root, text="Analisis Trion (Deskripsi, Analisis, Saran)", padding=10)
        output_frame.pack(fill="both", expand=True, padx=15, pady=15)

        self.output_text = scrolledtext.ScrolledText(output_frame, height=10, font=("Segoe UI", 12), wrap=tk.WORD, bg="#f8f9fa", bd=0)
        self.output_text.pack(fill="both", expand=True)
        self.output_text.insert(tk.END, "Halo! Aku Trion. Tulis acaramu di atas lalu klik 'Cek Penampilanku' untuk mulai analisis.")
        self.output_text.config(state='disabled')

    def update_webcam(self):
        if not self.webcam_running: return
        ret, frame = self.cap.read()
        if ret:
            frame_flip = cv2.flip(frame, 1)
            cv_image = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
        self.root.after(15, self.update_webcam)

    def capture_and_analyze(self):
        occasion = self.occasion_entry.get()
        if not occasion:
            messagebox.showwarning("Peringatan", "Isi dulu mau pergi ke acara apa!")
            return

        ret, frame = self.cap.read()
        if not ret: return

        # Resolusi sangat optimal untuk VRAM & Kecepatan
        frame_resized = cv2.resize(frame, (640, 480))
        cv2.imwrite(TEMP_IMAGE_PATH, frame_resized)

        self.analyze_button.config(state="disabled")
        self.status_label.config(text="Menganalisis Gayamu... 🧠", foreground="blue")
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Mengekstrak visual dan menyusun analisis (Mohon tunggu beberapa detik)...")
        self.output_text.config(state='disabled')

        threading.Thread(target=self.run_ai_pipeline, args=(occasion,), daemon=True).start()

    def run_ai_pipeline(self, occasion):
        start_time = time.time()
        
        # 🛠️ PERBAIKAN 2: MEMISAHKAN ROLE SYSTEM DAN USER
        pesan_user = f"Acara yang akan aku datangi: '{occasion}'. Tolong nilai pakaianku di foto ini sesuai 3 aturan mutlakmu!"

        try:
            print(f">> Mengirim request ke Ollama...")
            
            response = ollama.chat(
                model=MODEL_NAME,
                keep_alive='1h', # 🛠️ PERBAIKAN 3: Kunci model di VRAM agar tes berikutnya langsung jalan!
                options={
                    'temperature': 0.3, # Suhu diturunkan agar AI lebih fokus mematuhi format 3 poin
                    'num_ctx': 4096,       
                    'num_predict': 512,    
                },
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': pesan_user, 'images': [TEMP_IMAGE_PATH]}
                ]
            )

            final_answer = response['message']['content'].strip()
            
            durasi = time.time() - start_time
            print(f">> Output Final ({durasi:.2f} detik):\n{final_answer}")

            self.root.after(0, self.update_text_result, final_answer)
            
            self.root.after(0, lambda: self.status_label.config(text="Membuat Suara... 🔊", foreground="orange"))
            tts = gTTS(text=final_answer, lang='id')
            if os.path.exists(AUDIO_FILENAME):
                try: os.remove(AUDIO_FILENAME)
                except: pass 
            tts.save(AUDIO_FILENAME)
            
            self.root.after(0, self.play_audio)

        except Exception as e:
            print(f"Error Pipeline: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Gagal menghubungi AI:\n{str(e)}"))
            self.root.after(0, self.reset_ui)
        finally:
            if os.path.exists(TEMP_IMAGE_PATH):
                os.remove(TEMP_IMAGE_PATH)

    def update_text_result(self, text):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state='disabled')

    def play_audio(self):
        try:
            self.status_label.config(text="Sedang Bicara... 🗣️", foreground="green")
            self.stop_button.config(state="normal")
            pygame.mixer.music.load(AUDIO_FILENAME)
            pygame.mixer.music.play()
            threading.Thread(target=self.wait_audio_finish, daemon=True).start()
        except Exception as e:
            print(f"Gagal play audio: {e}")
            self.reset_ui()

    def wait_audio_finish(self):
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        self.root.after(0, self.reset_ui)

    def stop_audio(self):
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        self.reset_ui()

    def reset_ui(self):
        self.analyze_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Sistem Siap", foreground="green")

    def on_close(self):
        print("Menutup aplikasi...")
        self.webcam_running = False
        self.cap.release()
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            time.sleep(0.5) 
            if os.path.exists(AUDIO_FILENAME): os.remove(AUDIO_FILENAME)
            if os.path.exists(TEMP_IMAGE_PATH): os.remove(TEMP_IMAGE_PATH)
        except: pass
        self.root.destroy()

if __name__ == "__main__":
    try:
        ollama.list()
        print("✅ Koneksi Ollama berhasil.")
    except:
        messagebox.showerror("Fatal Error", "Ollama belum berjalan!\nJalankan 'ollama serve' di terminal.")
        exit()

    root = tk.Tk()
    app = FashionAdvisorApp(root)
    root.mainloop()