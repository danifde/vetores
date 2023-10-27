import cv2
import face_recognition
import numpy as np
import os

# Directorio donde se almacenan los vectores de embedding
embeddings_dir = 'embeddings'

# Directorio donde se almacenan las imágenes de referencia (personas conocidas)
known_faces_dir = 'faces'

# Cargar los vectores de embedding de todas las personas conocidas
known_face_encodings = {}
for person_dir in os.listdir(embeddings_dir):
    person_name = person_dir
    person_embeddings = []

    for embedding_file in os.listdir(os.path.join(embeddings_dir, person_dir)):
        embedding_path = os.path.join(embeddings_dir, person_dir, embedding_file)
        embedding = np.load(embedding_path)
        person_embeddings.append(embedding)

    known_face_encodings[person_name] = person_embeddings

# Inicializar el capturador de video
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar un cuadro de video
    ret, frame = video_capture.read()

    # Detectar caras en el cuadro
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for i, face_encoding in enumerate(face_encodings):
        # Comparar el rostro con todas las personas conocidas
        for person_name, known_embeddings in known_face_encodings.items():
            results = face_recognition.compare_faces(known_embeddings, face_encoding)
            if any(results):
                # Dibujar un recuadro y mostrar el nombre de la persona reconocida
                top, right, bottom, left = face_locations[i]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, person_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            
                
    # Mostrar el cuadro de video en tiempo real con OpenCV
    cv2.imshow("Video", frame)

    # Romper el bucle cuando se presione la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()
