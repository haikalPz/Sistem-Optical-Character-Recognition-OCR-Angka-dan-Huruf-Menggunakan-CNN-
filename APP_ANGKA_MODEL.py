import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# ===============================
# LOAD MODEL
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_ANGKA_cnn.h5")

if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Error", "Model tidak ditemukan!")
    exit()

model = load_model(MODEL_PATH)

# ===============================
# FUNGSI PREPROCESS GAMBAR
# ===============================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")  # grayscale
    img = img.resize((28, 28))
    img = np.array(img)
    
    # Binarisasi ringan (opsional tapi bagus)
    img = 255 - img  

    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# ===============================
# FUNGSI PILIH GAMBAR
# ===============================
def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not file_path:
        return

    try:
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        result = np.argmax(prediction)

        # tampilkan gambar
        show_img = Image.open(file_path).resize((200, 200))
        photo = ImageTk.PhotoImage(show_img)
        panel.config(image=photo)
        panel.image = photo

        result_label.config(text=f"Hasil Prediksi: {result}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ===============================
# GUI TKINTER
# ===============================
root = tk.Tk()
root.title("Aplikasi Deteksi Angka CNN")
root.geometry("350x450")

title = tk.Label(root, text="DETEKSI ANGKA CNN", font=("Arial", 16, "bold"))
title.pack(pady=10)

panel = tk.Label(root)
panel.pack(pady=10)

btn = tk.Button(
    root, text="Pilih Gambar",
    command=open_image,
    font=("Arial", 12),
    bg="blue", fg="white"
)
btn.pack(pady=10)

result_label = tk.Label(root, text="Hasil Prediksi: -", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()
