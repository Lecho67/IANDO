import torch
import cv2
import time

# Verifica si tienes acceso a la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Carga del modelo YOLO con detección y segmentación
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Configuración del umbral de confianza para mejorar la precisión
conf_threshold = 0.4  # Puedes ajustar este valor

# Configura la cámara
cap = cv2.VideoCapture(0)  # Usa la cámara predeterminada
if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir el cuadro de la cámara")
        break

    # Preprocesamiento: convierte la imagen en el formato correcto
    frame_resized = cv2.resize(frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Realiza la detección y segmentación con YOLO
    results = model(frame_rgb)  # Llama al modelo sin conf ni iou

    # Filtro por umbral de confianza
    detections = results.pandas().xyxy[0]  # coordenadas, etiquetas y confianza
    detections = detections[detections['confidence'] > conf_threshold]  # Aplicar umbral de confianza

    # Dibuja los cuadros en la imagen original
    for index, row in detections.iterrows():
        # Coordenadas del bounding box
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Etiqueta con nombre del objeto y confianza
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Muestra la imagen en tiempo real
    cv2.imshow('YOLO Object Detection', frame)

    # Sal del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Pausa para estabilizar la tasa de cuadros
    time.sleep(0.01)

# Libera los recursos de la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()