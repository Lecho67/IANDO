import cv2
from ultralytics import YOLO
import torch
import torchvision
import time

# Verificar la instalación de PyTorch y CUDA
print(f"Versión de PyTorch: {torch.__version__}")
print(f"Versión de Torchvision: {torchvision.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

# Verificar si la GPU está disponible
if torch.cuda.is_available():
    print("La GPU está disponible y será utilizada.")
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda'  # Usar la GPU
else:
    print("No se ha detectado una GPU, se utilizará la CPU.")
    device = 'cpu'  # Usar la CPU

# Cargar el modelo YOLOv8 con capacidad de segmentación y moverlo al dispositivo adecuado
model = YOLO("yolov8n-seg.pt").to(device)

# Inicializar la cámara (0 suele ser la cámara integrada)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

# Loop para capturar frame a frame de la cámara
while True:
    start_time = time.time()  # Para calcular los FPS

    # Capturar cada frame
    ret, frame = cap.read()

    # Verificar que el frame se haya capturado correctamente
    if not ret:
        print("No se puede recibir el frame (se ha alcanzado el final del video o hay un error)")
        break

    # Convertir el frame de BGR (formato de OpenCV) a RGB (formato de YOLO)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar la segmentación con YOLO (confidencia mínima de 0.3)
    results = model(frame_rgb, task="segment", conf=0.3)

    # Mostrar el frame con las máscaras de segmentación y las cajas
    annotated_frame = results[0].plot()  # .plot() dibuja las máscaras y las cajas en el frame

    # Calcular los FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame anotado en una ventana
    cv2.imshow('Segmentación en tiempo real con YOLOv8', annotated_frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
