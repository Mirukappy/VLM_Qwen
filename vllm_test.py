import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
import os
import threading
import time
from gtts import gTTS
import pygame
from PIL import Image, ImageTk
import base64
from openai import OpenAI

# --- KONFIGURASI vLLM ---
# Pastikan Anda menjalankan vLLM di terminal sebelum membuka aplikasi ini
# URL default vLLM biasanya http://localhost:8000/v1
VLLM_API_URL = "http://localhost:8000/v1" 
VLLM_API_KEY = "EMPTY" # vLLM lokal tidak butuh key

# Nama model harus SAMA PERSIS dengan yang Anda load di terminal vLLM
# Contoh: "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
CURRENT_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

AUDIO_FILENAME = "trion_voice.mp3"
TEMP_IMAGE_PATH = "temp_capture.jpg"

class FashionAdvisorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Trion vLLM Client - Connected to {CURRENT_MODEL_NAME}")
        self.root.geometry("950x850")
        
        # Inisialisasi Audio
        pygame.mixer.init()

        # Setup OpenAI Client (Bridge ke vLLM)
        self.client = OpenAI(
            base_url=VLLM_API_URL,
            api_key=VLLM_API_KEY,
        )

        self.webcam_running = True
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Webcam tidak terdeteksi!")
            self.root.destroy()
            return
        
        self.setup_ui()
        self.update_webcam()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Cek Koneksi ke vLLM
        threading.Thread(target=self.check_vllm_connection, daemon=True).start()

    def check_vllm_connection(self):
        """Cek apakah server vLLM sudah nyala"""
        try:
            print(f">> Menghubungi vLLM di {VLLM_API_URL}...")
            self.client.models.list()
            print(">> ✅ Terhubung ke vLLM Server!")
            self.status_label.config(text="vLLM Server: ONLINE ✅", foreground="green")
        except Exception as e:
            print(f">> ❌ Gagal konek ke vLLM: {e}")
            self.status_label.config(text="vLLM Server: OFFLINE ❌ (Cek Terminal)", foreground="red")

    def setup_ui(self):
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 11), padding=6)

        # Panel Input
        input_frame = ttk.LabelFrame(self.root, text="Panel Kontrol", padding=15)
        input_frame.pack(fill="x", padx=15, pady=10)

        ttk.Label(input_frame, text="Acara:").pack(side=tk.LEFT, padx=(0, 5))
        self.occasion_entry = ttk.Entry(input_frame, width=30, font=("Segoe UI", 11))
        self.occasion_entry.pack(side=tk.LEFT, padx=5)
        self.occasion_entry.insert(0, "Hangout Santai")

        self.analyze_button = ttk.Button(input_frame, text="✨ Analisis (vLLM)", command=self.capture_and_analyze)
        self.analyze_button.pack(side=tk.LEFT, padx=15)
        
        self.stop_button = ttk.Button(input_frame, text="⛔ Stop Audio", command=self.stop_audio, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(input_frame, text="Mencari Server...", foreground="orange")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Webcam
        video_frame = ttk.Frame(self.root, padding=5, relief="groove")
        video_frame.pack(fill="both", expand=True, padx=15, pady=5)
        self.webcam_label = ttk.Label(video_frame)
        self.webcam_label.pack(fill="both", expand=True)

        # Output Text
        output_frame = ttk.LabelFrame(self.root, text="Saran AI", padding=10)
        output_frame.pack(fill="both", expand=True, padx=15, pady=15)
        self.output_text = scrolledtext.ScrolledText(output_frame, height=8, font=("Segoe UI", 12), bg="#f0f0f0")
        self.output_text.pack(fill="both", expand=True)
        self.output_text.config(state='disabled')

    def update_webcam(self):
        if self.webcam_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.webcam_label.imgtk = imgtk
                self.webcam_label.configure(image=imgtk)
            self.root.after(15, self.update_webcam)

    def encode_image_base64(self, image_path):
        """Ubah gambar jadi Base64 untuk dikirim ke vLLM"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def capture_and_analyze(self):
        occasion = self.occasion_entry.get()
        ret, frame = self.cap.read()
        if not ret: return

        # Resize untuk vLLM (Qwen VL suka resolusi kelipatan 14/28)
        # Kita pakai 896x896 agar tajam
        target_size = (896, 896) 
        frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(TEMP_IMAGE_PATH, frame_resized)

        self.analyze_button.config(state="disabled")
        self.status_label.config(text="Mengirim ke vLLM... 🚀", foreground="blue")
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Sedang berpikir di server vLLM...")
        self.output_text.config(state='disabled')

        threading.Thread(target=self.run_vllm_pipeline, args=(occasion,), daemon=True).start()

    def run_vllm_pipeline(self, occasion):
        try:
            # 1. Encode Gambar
            base64_image = self.encode_image_base64(TEMP_IMAGE_PATH)

            # 2. Siapkan Prompt
            prompt_text = f"""
            Kamu Stylist. User mau ke acara: '{occasion}'.
            
            TUGAS:
            1. Analisis Baju & Warna Kulit.
            2. Beri 1 saran singkat.
            
            FORMAT:
            "Baju [Warna] kamu [Cocok/Kurang]. Coba [Saran Konkret]."
            """

            print(f">> Mengirim Request ke Model: {CURRENT_MODEL_NAME}")
            
            # 3. Request ke vLLM (OpenAI Compatible)
            # vLLM memproses gambar melalui format content list
            response = self.client.chat.completions.create(
                model=CURRENT_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=300, # Sama dengan num_predict
                temperature=0.4,
            )

            # 4. Ambil Hasil
            final_answer = response.choices[0].message.content
            
            # Bersihkan Thinking Process (jika model Qwen 3 bocor)
            import re
            final_answer = re.sub(r'<think>.*?</think>', '', final_answer, flags=re.DOTALL).strip()
            
            if not final_answer:
                final_answer = "Maaf, aku tidak melihat jelas. Coba foto lagi."

            print(f">> Output vLLM: {final_answer}")
            self.root.after(0, self.update_text_result, final_answer)

            # 5. TTS
            self.root.after(0, lambda: self.status_label.config(text="Suara... 🔊", foreground="orange"))
            tts = gTTS(text=final_answer, lang='id')
            if os.path.exists(AUDIO_FILENAME): os.remove(AUDIO_FILENAME)
            tts.save(AUDIO_FILENAME)
            self.root.after(0, self.play_audio)

        except Exception as e:
            print(f"Error vLLM: {e}")
            self.root.after(0, lambda: messagebox.showerror("vLLM Error", f"Pastikan Server vLLM jalan!\nError: {e}"))
            self.root.after(0, self.reset_ui)
        finally:
             if os.path.exists(TEMP_IMAGE_PATH): os.remove(TEMP_IMAGE_PATH)

    def update_text_result(self, text):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state='disabled')

    def play_audio(self):
        try:
            self.status_label.config(text="Bicara... 🗣️", foreground="green")
            self.stop_button.config(state="normal")
            pygame.mixer.music.load(AUDIO_FILENAME)
            pygame.mixer.music.play()
            threading.Thread(target=self.wait_audio_finish, daemon=True).start()
        except: self.reset_ui()

    def wait_audio_finish(self):
        while pygame.mixer.music.get_busy(): time.sleep(0.1)
        self.root.after(0, self.reset_ui)

    def stop_audio(self):
        if pygame.mixer.get_init(): pygame.mixer.music.stop()
        self.reset_ui()

    def reset_ui(self):
        self.analyze_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Siap", foreground="gray")

    def on_close(self):
        self.webcam_running = False
        self.cap.release()
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except: pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FashionAdvisorApp(root)
    root.mainloop()