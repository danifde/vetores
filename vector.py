import cv2
import face_recognition
import numpy as np
import os
import time

# Inicializar el capturador de video
video_capture = cv2.VideoCapture(0)

# Crear un directorio para guardar los vectores de embedding
embeddings_dir = 'embeddings'
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

# Crear un directorio para guardar las imágenes de rostros
faces_dir = 'faces'
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)

current_face = None  # Variable para rastrear la cara actual

while True:
    # Capturar un cuadro de video
    ret, frame = video_capture.read()

    # Detectar caras en el cuadro
    face_locations = face_recognition.face_locations(frame)

    # Si se detecta al menos una cara
    if face_locations:
        if current_face is None:
            # Si esta es la primera cara detectada, solicitar el nombre de la persona
            name = input("Por favor, ingresa el nombre de la persona: ")
            current_face = name

            # Crear una carpeta para la persona si no existe
            if not os.path.exists(os.path.join(faces_dir, name)):
                os.makedirs(os.path.join(faces_dir, name))

            # Crear una cadarpeta para los embeddings de la persona si no existe
            if not os.path.exists(os.path.join(embeddings_dir, name)):
                os.makedirs(os.path.join(embeddings_dir, name))

    # Mostrar el cuadro de video en tiempo real con OpenCV
    cv2.imshow("Video", frame)
    cv2.waitKey(1)

    # Esperar hasta que el usuario esté listo para tomar una foto
    input("Presiona Enter para tomar una foto cuando estés listo...")

    # Capturar una foto inmediatamente después de que el usuario presione Enter
    ret, frame = video_capture.read()

    if current_face:
        # Recopilar la imagen del rostro
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Recortar y guardar la imagen del rostro en la carpeta de la persona
            face_image = frame[top:bottom, left:right]
            face_filename = os.path.join(faces_dir, current_face, f'{current_face}_{i}_{int(time.time())}.jpg')
            cv2.imwrite(face_filename, face_image)

            # Extraer embeddings de las caras detectadas
            face_encodings = face_recognition.face_encodings(frame, [(top, right, bottom, left)])

            # Por ejemplo, aquí se imprime el primer embedding
            if len(face_encodings) > 0:
                print(f"Vector de embedding de la cara {i} de {current_face}:", face_encodings[0])

                # Guardar el embedding en un archivo (nombre basado en el nombre de la persona y la hora actual)
                timestamp = int(time.time())
                embedding_filename = os.path.join(embeddings_dir, current_face, f'{current_face}_{i}_{timestamp}.npy')
                np.save(embedding_filename, face_encodings[0])

    # Esperar 3 segundos antes de tomar otra foto
    time.sleep(3)

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()
