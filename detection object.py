from ultralytics import YOLO
import cv2

# Load model YOLOv8 nano (pretrained COCO dataset)
model = YOLO('yolov8n.pt')

# Buka kamera (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak bisa membuka kamera.")
    exit()

print("Kamera berhasil dibuka. Tekan 'q' atau 'ESC' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Deteksi objek dengan confidence threshold 25%
    results = model.predict(frame, conf=0.25)

    # Proses hasil deteksi
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat kotak
            confidence = float(box.conf[0])
            cls = int(box.cls[0])  # Kelas deteksi
            label = model.names[cls]  # Nama label objek

            # Gambar kotak dan label deteksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow('YOLOv8 Real-time Detection', frame)

    # Tekan 'q' atau ESC untuk keluar
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 adalah kode ESC
        print("Program dihentikan oleh user.")
        break

cap.release()
cv2.destroyAllWindows()
