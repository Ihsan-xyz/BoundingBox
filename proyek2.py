import cv2
import numpy as np
from collections import OrderedDict
import math

LEBAR_FRAME = 480
TINGGI_FRAME = 360
PROSES_SETIAP_N_FRAME = 16
JEDA_LOOP_MS = 30
CONFIDENCE_THRESHOLD = 0.4  

AREA_THRESHOLD = 8000
MOVEMENT_THRESHOLD = 10
WARNA_TEKS = (255, 255, 255)

WARNA_TENGAH = (0, 255, 0)
WARNA_KIRI   = (255, 0, 0)
WARNA_KANAN  = (0, 255, 255)
WARNA_JAUH   = (0, 0, 255)
WARNA_ATAS   = (255, 255, 255)
WARNA_BAWAH  = (0, 0, 0)

try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Tidak dapat memuat file cascade: {face_cascade_path}")
except Exception as e:
    print(f"Gagal memuat model Haar Cascade: {e}")
    exit()

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, LEBAR_FRAME)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, TINGGI_FRAME)

ret, frame_test = camera.read()
if not ret:
    print("Error: Tidak dapat membaca frame.")
    exit()
FRAME_WIDTH = frame_test.shape[1]
CENTER_THRESHOLD = FRAME_WIDTH * 0.1

tracked_faces = OrderedDict()
next_face_id = 0

def process_frame(frame):
    global tracked_faces, next_face_id

    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(40, 40)
    )

    current_boxes = []
    for (x, y, w, h) in detected_faces:
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        current_boxes.append((x1, y1, x2, y2))

    if len(tracked_faces) == 0:
        for (x1, y1, x2, y2) in current_boxes:
            cX, cY = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
            tracked_faces[next_face_id] = (cX, cY)
            next_face_id += 1
    else:
        face_ids = list(tracked_faces.keys())
        previous_centroids = np.array(list(tracked_faces.values()))

        current_centroids = []
        for (x1, y1, x2, y2) in current_boxes:
            cX, cY = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
            current_centroids.append((cX, cY))

        if len(current_centroids) == 0:
            tracked_faces.clear()
        else:
            current_centroids = np.array(current_centroids)
            D = np.zeros((len(previous_centroids), len(current_centroids)))
            for i in range(len(previous_centroids)):
                for j in range(len(current_centroids)):
                    D[i, j] = math.dist(previous_centroids[i], current_centroids[j])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            new_tracked_faces = OrderedDict()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                face_id = face_ids[row]
                new_tracked_faces[face_id] = tuple(current_centroids[col])
                used_rows.add(row)
                used_cols.add(col)

            unused_cols = set(range(0, len(current_centroids))).difference(used_cols)
            for col in unused_cols:
                new_tracked_faces[next_face_id] = tuple(current_centroids[col])
                next_face_id += 1

            tracked_faces = new_tracked_faces

    for (face_id, (cX, cY)) in tracked_faces.items():
        box_ditemukan = None
        min_dist = float('inf')
        for (x1, y1, x2, y2) in current_boxes:
            box_cX, box_cY = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
            dist = math.dist((cX, cY), (box_cX, box_cY))
            if dist < min_dist:
                min_dist = dist
                box_ditemukan = (x1, y1, x2 - x1, y2 - y1)

        if box_ditemukan:
            (x, y, w, h) = box_ditemukan
            area = w * h

            perubahan_x, perubahan_y = 0, 0
            list_gerakan = []
            if 'previous_centroids' in locals() and face_id in face_ids:
                idx_lama = face_ids.index(face_id)
                if idx_lama < len(previous_centroids):
                    x_sebelumnya, y_sebelumnya = previous_centroids[idx_lama]
                    perubahan_x = cX - x_sebelumnya
                    perubahan_y = cY - y_sebelumnya
                    if perubahan_x > MOVEMENT_THRESHOLD: list_gerakan.append("Kanan")
                    elif perubahan_x < -MOVEMENT_THRESHOLD: list_gerakan.append("Kiri")
                    if perubahan_y > MOVEMENT_THRESHOLD: list_gerakan.append("Bawah")
                    elif perubahan_y < -MOVEMENT_THRESHOLD: list_gerakan.append("Atas")

            if area < AREA_THRESHOLD:
                warna_kotak = WARNA_JAUH
            else:
                if perubahan_y < -MOVEMENT_THRESHOLD:
                    warna_kotak = WARNA_ATAS
                elif perubahan_y > MOVEMENT_THRESHOLD:
                    warna_kotak = WARNA_BAWAH
                else:
                    if cX > FRAME_WIDTH / 2 - CENTER_THRESHOLD and cX < FRAME_WIDTH / 2 + CENTER_THRESHOLD:
                        warna_kotak = WARNA_TENGAH
                    elif cX <= FRAME_WIDTH / 2 - CENTER_THRESHOLD:
                        warna_kotak = WARNA_KIRI
                    else:
                        warna_kotak = WARNA_KANAN

            teks_gerakan = " & ".join(list_gerakan)
            if teks_gerakan:
                teks_gerakan = "Bergerak ke " + teks_gerakan

            cv2.rectangle(frame, (x, y), (x + w, y + h), warna_kotak, 2)

            if teks_gerakan and warna_kotak != WARNA_JAUH:
                warna_teks_gerakan = WARNA_BAWAH if warna_kotak == WARNA_ATAS else WARNA_ATAS if warna_kotak == WARNA_BAWAH else warna_kotak
                cv2.putText(frame, teks_gerakan, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_teks_gerakan, 2)

    frame = cv2.resize(frame, (LEBAR_FRAME, TINGGI_FRAME))
    cv2.putText(frame, f"Jumlah Wajah: {len(tracked_faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WARNA_TEKS, 2)
    return frame

def main():
    frame_counter = 0
    processed_frame = None

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % PROSES_SETIAP_N_FRAME == 0 or processed_frame is None:
            processed_frame = process_frame(frame.copy())

        cv2.imshow("Deteksi Gerakan Wajah", processed_frame)

        if cv2.waitKey(JEDA_LOOP_MS) & 0xFF == ord('c'):
            break

    print("Menutup program...")
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
