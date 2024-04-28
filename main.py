import cv2
import dlib
from scipy.spatial import distance as dist


    # EAR hesaplanması
    ear = (A + B) / (2.0 * C)
    return ear

# Kamera akışının başlatılması
cap = cv2.VideoCapture(0)

while True:
    # Kameradan görüntü alınması
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntü üzerinde yüz tespiti
    faces = face_detector(frame)

    for face in faces:
        # Yüz üzerindeki landmark'ların tespiti
        landmarks = face_predictor(frame, face)

        # Göz landmark'larının belirlenmesi
        left_eye = []
        right_eye = []

        for n in range(36, 42):
            left_eye.append((landmarks.part(n).x, landmarks.part(n).y))
        for n in range(42, 48):
            right_eye.append((landmarks.part(n).x, landmarks.part(n).y))

        # Her iki göz için EAR hesaplanması
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Gözlerin açık olup olmadığının kontrol edilmesi
        if left_ear < 0.2 and right_ear < 0.2:
            cv2.putText(frame, "Gozler Kapali", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Gozler Acik", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sonuçların ekranda gösterilmesi
    cv2.imshow("Frame", frame)

    # 'q' tuşuna basıldığında döngüden çıkılması
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera akışının ve pencerelerin kapatılması
cap.release()
cv2.destroyAllWindows()
