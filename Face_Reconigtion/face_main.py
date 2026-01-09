import cv2
import face_recognition
import numpy as np
from PIL import Image  # Use Pillow for robust image loading

def load_and_fix_image(file_path):
    """
    Uses PIL to open the image, strips ICC profiles/metadata, 
    and forces it into a standard 8-bit RGB format.
    """
    try:
        # 1. Open with PIL (Better at handling strange formats/metadata than OpenCV)
        with Image.open(file_path) as pil_img:
            # 2. Convert to RGB (removes Alpha channel/CMYK/Grayscale)
            pil_img = pil_img.convert('RGB')
            # 3. Convert to Numpy array (uint8)
            img_array = np.array(pil_img)
            return img_array
    except Exception as e:
        print(f"❌ Critical Error loading {file_path}: {e}")
        return None

# --- DATABASE PREPARATION ---
known_face_encodings = []
known_faces_name = []

data_wajah = {
    "Yosefh": r"C:\Users\Alo-Novita\Desktop\All Project\Face_Reconigtion\yosefh.jpg",
    "Emberlly": r"C:\Users\Alo-Novita\Desktop\All Project\Face_Reconigtion\edellyn.jpg"
}

print("Registering faces...")

for nama, path in data_wajah.items():
    image = load_and_fix_image(path)
    
    if image is not None:
        # Check if the image is actually valid before encoding
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_faces_name.append(nama)
            print(f"✅ Registered: {nama}")
        else:
            print(f"⚠️ No face found in {nama}. Try a clearer photo.")

# --- WEBCAM PROCESS ---
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret: break

    # Convert webcam BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_faces_name[best_match_index]

        # Draw UI
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Face System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

video_capture.release()
cv2.destroyAllWindows()