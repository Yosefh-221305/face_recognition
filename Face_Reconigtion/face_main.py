import cv2
import face_recognition
import numpy as np
from PIL import Image
import os

# 1. FUNGSI UNTUK MEMBERSIHKAN FORMAT GAMBAR
def muat_gambar_standard(file_path):
    """
    Memastikan gambar dalam format RGB 8-bit agar tidak menyebabkan 
    RuntimeError: Unsupported image type.
    """
    try:
        # Membuka gambar dengan Pillow
        with Image.open(file_path) as img:
            # Konversi ke RGB (menghilangkan channel Alpha/Transparansi)
            img_rgb = img.convert('RGB')
            # Konversi ke array numpy dengan tipe data uint8 (8-bit)
            return np.array(img_rgb, dtype=np.uint8)
    except Exception as e:
        print(f"Gagal memuat {file_path}: {e}")
        return None

# 2. PERSIAPAN DATA WAJAH
known_face_encodings = []
known_faces_name = []

# Daftar foto dan nama (Pastikan file ini ada di folder yang sama dengan script)
data_wajah = {
    "Yosefh": r"C:\Users\Alo-Novita\Desktop\All Project\Face_Reconigtion\yosefh.jpg",
    "Emberlly": r"C:\Users\Alo-Novita\Desktop\All Project\Face_Reconigtion\edellyn.jpg"
}

print("Sedang memproses database wajah...")

for nama, file_foto in data_wajah.items():
    img = muat_gambar_standard(file_foto)
    if img is not None:
        # Mencari encoding wajah
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_faces_name.append(nama)
            print(f"Berhasil memuat wajah: {nama}")
        else:
            print(f"Peringatan: Wajah tidak ditemukan di foto {file_foto}")
    else:
        print(f"Peringatan: File {file_foto} tidak ditemukan di folder script.")

# 3. AKSES WEBCAM
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Tidak bisa membuka webcam.")
    exit()

print("\nWebcam Aktif. Tekan 'q' untuk keluar.")

while True:
    # Ambil frame dari webcam
    ret, frame = video_capture.read()
    if not ret:
        print("Gagal mengambil gambar dari webcam.")
        break

    # OpenCV membaca BGR, face_recognition butuh RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi lokasi dan encoding wajah di webcam
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Bandingkan dengan database
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Gunakan wajah dengan jarak (distance) terkecil agar lebih akurat
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_faces_name[best_match_index]

        # GAMBAR HASIL PADA FRAME
        # Kotak Wajah
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Label Nama
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # Tampilkan jendela video
    cv2.imshow('Sistem Pengenal Wajah', frame)

    # Berhenti jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
video_capture.release()
cv2.destroyAllWindows()