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
import re

# --- KONFIGURASI ---
MODEL_NAME = 'qwen2.5vl:3b' 
AUDIO_FILENAME = "trion_voice.mp3"
TEMP_IMAGE_PATH = "temp_capture.jpg"

class FashionAdvisorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trion: AI Personal Fashion Stylist (Unleashed Mode)")
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

        # Panggil fungsi pemanasan di background
        threading.Thread(target=self.warmup_model, daemon=True).start()

    def warmup_model(self):
        """Memuat model diam-diam saat aplikasi baru dibuka"""
        print(">> Melakukan Warmup Model...")
        try:
            # Kirim prompt kosong/dummy agar model masuk VRAM
            ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': 'hi'}])
            print(">> Model Siap & Hangat! 🚀")
            self.status_label.config(text="Sistem Siap (Unleashed Mode)")
        except:
            pass

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
        self.occasion_entry.insert(0, "Kerja di Kafe") 

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
        output_frame = ttk.LabelFrame(self.root, text="Analisis Trion (Full Detail)", padding=10)
        output_frame.pack(fill="both", expand=True, padx=15, pady=15)

        self.output_text = scrolledtext.ScrolledText(
            output_frame, 
            height=10, 
            font=("Segoe UI", 11), 
            wrap=tk.WORD,
            bg="#f0f0f0",
            bd=0
        )
        self.output_text.pack(fill="both", expand=True)
        self.output_text.insert(tk.END, "Halo! Aku Trion. Tulis acaramu di atas lalu klik tombol untuk mulai analisis mendalam.")
        self.output_text.config(state='disabled')

    def update_webcam(self):
        """Mengupdate tampilan webcam secara real-time"""
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

        # === MODE: UNLEASHED (896px) ===
        # Resolusi tinggi agar dia bisa melihat detail jahitan, jam tangan, logo, dll.
        # Menggunakan Lanczos4 agar gambar sangat tajam saat di-resize.
        target_size = (896, 896) 
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(TEMP_IMAGE_PATH, frame_resized)

        # Update UI
        self.analyze_button.config(state="disabled")
        self.status_label.config(text="Menganalisis Detail Visual... 🧠", foreground="blue")
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Sedang melakukan observasi mendalam (High-Res Analysis)...")
        self.output_text.config(state='disabled')

        # Jalankan Logic AI di Thread Terpisah
        threading.Thread(target=self.run_ai_pipeline, args=(occasion,), daemon=True).start()

    def run_ai_pipeline(self, occasion):
        import time
        start_time = time.time()
        
        # --- PROMPT --
        # Kita sederhanakan sedikit agar model tidak pusing
        prompt = f"""
        Kamu adalah Fashion Stylist.
        Lihat gambar orang ini yang mau ke acara: '{occasion}'.
        
        TUGAS:
        1. Identifikasi: Gender, Warna Kulit, dan Baju yang dipakai.
        2. Berikan saran: Apakah cocok untuk '{occasion}'? Jika tidak, apa yang harus diganti?
        
        FORMAT JAWABAN (WAJIB):
        "Halo! Aku lihat kamu pakai [Baju]. Ini [Cocok/Kurang] untuk {occasion}. 
        Saranku, coba [Saran Konkret] agar lebih pas dengan [Warna Kulit/Bentuk Tubuh] kamu."
        
        PENTING: Jawab dalam Bahasa Indonesia yang santai.
        """

        try:
            print(f">> [FORCE MODE] Mengirim gambar ke GPU...")
            
            response = ollama.chat(
                model=MODEL_NAME,
                keep_alive=-1,
                options={
                    'temperature': 0.5,    # Sedikit naik biar kreatif
                    'num_ctx': 4096,       
                    'num_predict': 300,    # Limit sedang
                    'num_thread': 6,       
                },
                messages=[{'role': 'user', 'content': prompt, 'images': [TEMP_IMAGE_PATH]}]
            )

            raw_text = response['message']['content']
            print(f"\n[RAW DEBUG]: {raw_text}\n") # WAJIB LIHAT INI DI TERMINAL

            # =========================================================
            # 🛠️ LOGIKA BRUTE FORCE (JANGAN PAKAI REGEX RUMIT)
            # =========================================================
            
            # 1. Buang tag <think> dan isinya (biar analisis panjang gak dibaca)
            #    Kita pakai split untuk membuang bagian depan (think)
            import re
            
            # Cek apakah ada tag think?
            if '<think>' in raw_text:
                # Hapus semua yang ada di dalam <think>...</think>
                clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
            else:
                # Kalau gak ada tag, ya ambil semua
                clean_text = raw_text

            # 2. Bersihkan sisa-sisa kotoran
            clean_text = clean_text.replace("Assistant:", "").replace("User:", "")
            clean_text = clean_text.replace("*", "").replace("#", "").replace('"', '')
            clean_text = " ".join(clean_text.split()) # Hapus spasi ganda/enter
            
            # 3. KUNCI PENYELAMATAN:
            # Jika setelah dibersihkan ternyata kosong (artinya semua teks ada di dalam think),
            # KITA AMBIL PAKSA DARI DALAM THINK.
            if len(clean_text) < 5:
                print(">> PERINGATAN: Jawaban ada di dalam <think>. Mengambil paksa...")
                raw_clean = raw_text.replace("<think>", "").replace("</think>", "")
                # Ambil 200 karakter terakhir
                clean_text = raw_clean[-200:]

            # 4. Potong biar gak kepanjangan buat TTS
            # Ambil maksimal 2 kalimat pertama yang ditemukan
            sentences = clean_text.split('.')
            if len(sentences) >= 2:
                final_answer = sentences[0] + ". " + sentences[1] + "."
            else:
                final_answer = clean_text

            print(f">> Output Final: {final_answer}")

            # Update UI & TTS
            self.root.after(0, self.update_text_result, final_answer)
            
            # TTS
            self.root.after(0, lambda: self.status_label.config(text="Mengunduh Suara... 🔊", foreground="orange"))
            tts = gTTS(text=final_answer, lang='id')
            if os.path.exists(AUDIO_FILENAME):
                try: os.remove(AUDIO_FILENAME)
                except: pass 
            tts.save(AUDIO_FILENAME)
            self.root.after(0, self.play_audio)

        except Exception as e:
            print(f"Error Pipeline: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
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
        self.status_label.config(text="Siap", foreground="gray")

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