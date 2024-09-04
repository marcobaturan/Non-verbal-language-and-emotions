import cv2
import numpy as np
import scipy.fftpack
import time

# Inicializar el detector de caras de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Acceder a la cámara
cap = cv2.VideoCapture(0)

# Variables para almacenar valores de color y tiempo
heartbeat_values = [0] * 150  # Ventana de valores para FFT
heartbeat_times = [0] * 150
start_time = time.time()

def extract_pulse(cap, x, y, w, h):
    """
    Extraer el ritmo cardíaco usando procesamiento de imágenes y FFT.
    """
    global heartbeat_values, heartbeat_times
    
    # Capturar el frame y procesar la imagen
    ret, frame = cap.read()
    if not ret:
        return 0, frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop_img = img[y:y + h, x:x + w]
    
    # Actualizar los datos del pulso
    mean_val = np.mean(crop_img)
    heartbeat_values = heartbeat_values[1:] + [mean_val]
    heartbeat_times = heartbeat_times[1:] + [time.time()]
    
    # Aplicar un filtro de paso bajo simple
    filtered_values = np.convolve(heartbeat_values, np.ones(3)/3, mode='same')
    
    # Calcular la FFT para encontrar la frecuencia dominante
    fft_values = np.abs(scipy.fftpack.fft(filtered_values - np.mean(filtered_values)))
    freqs = scipy.fftpack.fftfreq(len(fft_values), d=(heartbeat_times[-1] - heartbeat_times[0]) / len(fft_values))
    
    # Encontrar la frecuencia dominante dentro del rango típico de pulsaciones (0.8 - 3 Hz, equivalente a 48 - 180 BPM)
    freq_in_bpm = np.abs(freqs * 60)
    valid_idxs = np.where((freq_in_bpm > 48) & (freq_in_bpm < 180))
    
    if len(valid_idxs[0]) == 0:
        return 0, frame
    
    dominant_freq_idx = valid_idxs[0][np.argmax(fft_values[valid_idxs])]
    bpm = freq_in_bpm[dominant_freq_idx]
    
    return int(bpm), frame

def get_color_intensity(bpm):
    """
    Obtener la intensidad del color en función del BPM.
    """
    if bpm <= 30:
        return 0  # Rojo muy bajo
    elif bpm <= 60:
        return int(255 * 0.25)  # Rojo bajo
    elif bpm <= 90:
        return int(255 * 0.5)  # Rojo medio
    elif bpm <= 130:
        return int(255 * 0.75)  # Rojo alto
    elif bpm <= 180:
        return 255  # Rojo máximo
    else:
        return 0  # Rojo muy alto, fuera de rango

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir la imagen a escala de grises para la detección de la cara
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar caras en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    bpm = 0  # Inicializar el BPM a cero por defecto

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extraer el ritmo cardíaco
            bpm, frame = extract_pulse(cap, x, y, w, h)
            
            # Crear una máscara para resaltar la cara según el pulso
            mask = np.zeros_like(frame[y:y+h, x:x+w])
            color_intensity = get_color_intensity(bpm)
            mask[:, :] = (0, 0, color_intensity)  # Rojo según la intensidad del BPM
            
            # Aplicar la máscara a la imagen original
            highlighted = cv2.addWeighted(frame[y:y+h, x:x+w], 1.0, mask, 0.5, 0)
            frame[y:y+h, x:x+w] = highlighted

    # Mostrar el BPM en la esquina superior izquierda
    cv2.putText(frame, f'HB/s: {bpm}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Mostrar la imagen con la cara resaltada
    cv2.imshow('Heartbeat Detection', frame)
    print(f'Heart rate: {bpm}')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
