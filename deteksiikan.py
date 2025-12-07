import cv2
from ultralytics import YOLO
import sys

# Load model YOLOv8 dengan file best.pt
try:
    model = YOLO('bestikan.pt')
    print("✓ Model berhasil dimuat")
except FileNotFoundError:
    print("✗ Error: File 'best.pt' tidak ditemukan")
    print("Pastikan file 'best.pt' berada di direktori yang sama dengan script ini")
    sys.exit(1)

# Buka webcam
cap = cv2.VideoCapture(0)

# Cek apakah webcam berhasil dibuka
if not cap.isOpened():
    print("✗ Error: Tidak dapat membuka webcam")
    sys.exit(1)

print("✓ Webcam berhasil dibuka")
print("Tekan 'q' untuk keluar dari program")

# Loop untuk membaca frame dari webcam
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("✗ Error: Tidak dapat membaca frame dari webcam")
        break
    
    # Lakukan deteksi objek
    results = model(frame)
    
    # Visualisasi hasil deteksi
    annotated_frame = results[0].plot()
    
    # Tampilkan hasil deteksi
    cv2.imshow('YOLOv8 Deteksi Objek', annotated_frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Tampilkan informasi deteksi
    detections = results[0].boxes
    if len(detections) > 0:
        print(f"Terdeteksi {len(detections)} objek")

# Tutup webcam dan jendela
cap.release()
cv2.destroyAllWindows()
print("✓ Program selesai")