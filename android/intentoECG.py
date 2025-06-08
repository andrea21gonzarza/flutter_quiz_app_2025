import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

print('Simulando señal ECG...')

# 1. Crear una forma de onda parecida a una señal ECG con wavelet Daubechies
pqrst = signal.wavelets.daub(10)  # Onda similar a pqrst
samples_rest = 10                 # Descanso entre latidos
descanso = np.zeros(samples_rest)

# 2. Unir latido + descanso
latido_completo = np.concatenate([pqrst, descanso])

# 3. Configurar parámetros de simulación
bpm = 75                         # Latidos por minuto simulados
duracion_segundos = 10          # Duración total de la señal
beats_total = int((bpm / 60) * duracion_segundos)

# 4. Crear señal ECG completa repitiendo el patrón
ecg = np.tile(latido_completo, beats_total)

# 5. Agregar ruido gaussiano
ruido = np.random.normal(0, 0.01, len(ecg))
ecg_con_ruido = ecg + ruido

# 6. Simular muestreo (como un ADC real)
frecuencia_muestreo = 50  # Hz
total_muestras = int(frecuencia_muestreo * duracion_segundos)
ecg_muestreado = signal.resample(ecg_con_ruido, total_muestras)

# 7. Convertir amplitudes a valores de un ADC (resolución de 10 bits)
adc_bits = 1024
ecg_final = np.interp(ecg_muestreado, (ecg_muestreado.min(), ecg_muestreado.max()), (0, adc_bits))

# 8. Graficar la señal final
plt.plot(ecg_final)
plt.title(f"ECG simulado a {bpm} bpm, {frecuencia_muestreo}Hz")
plt.xlabel("Muestra")
plt.ylabel("Valor ADC")
plt.grid(True)
plt.show()

# 9. Guardar en archivo CSV
np.savetxt("ecg_values.csv", ecg_final, delimiter="\\n", fmt="%.2f")
print("Archivo 'ecg_values.csv' guardado.")
