import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado con capacidad de segmentación
model = YOLO("yolov8n-seg.pt")  # Modelo YOLOv8 con capacidad de segmentación

# Inicializar la cámara (0 suele ser la cámara integrada)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

# Loop para capturar frame a frame de la cámara
while True:
    # Capturar cada frame
    ret, frame = cap.read()

    # Verificar que el frame se haya capturado correctamente
    if not ret:
        print("No se puede recibir el frame (se ha alcanzado el final del video o hay un error)")
        break

    # Realizar la segmentación con YOLO (confidencia mínima de 0.3)
    # Personalización: Solo detectar personas (0) y coches (2)
    results = model(frame, task="segment", conf=0.3)

    # Mostrar el frame con las máscaras de segmentación y las cajas
    annotated_frame = results[0].plot()  # .plot() dibuja las máscaras y las cajas en el frame

    # Mostrar el frame anotado en una ventana
    cv2.imshow('Segmentación en tiempo real con YOLOv8', annotated_frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
