"""
Created on Sun Jul 11 08:07:13 2021

@author: Alfonso Castillo Orozco
"""

# Universidad de Costa Rica
# Escuela de Ingeniería Eléctrica
# IE0405 Modelos Probabilísticos de Señales y Sistemas
# PROYECTO 4

# Bibliotecas necesarias
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fftpack import fft

############## 4.1. - Modulación 16-QAM

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)

def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)

# MODULACIÓN

def modulador16QAM(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital BPSK.
    '''

    # Se toman los bits de 4 en 4: 16-QAM
    bits_4_en_4 = []
    for i in range(0,len(bits),4):
        bits_4_en_4.append(format(int(str(bits[i])+str(bits[i+1])+str(bits[i+2])+str(bits[i+3])),'04d'))

    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    portadora_I = np.cos(2*np.pi*fc*t_periodo)
    portadora_Q = np.sin(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # (opcional) señal de bits

    # 4. Asignar las formas de onda según los bits (BPSK)
    for i, bit in enumerate(bits_4_en_4):
        if int(bit[0])==0 and int(bit[1])==0 and int(bit[2])==0 and int(bit[3])== 0:
            senal_Tx[i*mpp : (i+1)*mpp] = 0.25*portadora_I + 0.25*portadora_Q
        elif int(bit[0])==0 and int(bit[1])==0 and int(bit[2])==0 and int(bit[3])== 1:
            senal_Tx[i*mpp : (i+1)*mpp] = 0.75*portadora_I + 0.25*portadora_Q
        elif int(bit[0])==0 and int(bit[1])==0 and int(bit[2])==1 and int(bit[3])== 0:
            senal_Tx[i*mpp : (i+1)*mpp] = 0.25*portadora_I + 0.75*portadora_Q
        elif int(bit[0])==0 and int(bit[1])==0 and int(bit[2])==1 and int(bit[3])== 1:
            senal_Tx[i*mpp : (i+1)*mpp] = 0.75*portadora_I + 0.75*portadora_Q
        elif int(bit[0])==0 and int(bit[1])==1 and int(bit[2])==0 and int(bit[3])== 0:
            senal_Tx[i*mpp : (i+1)*mpp] = 0.25*portadora_I + -0.25*portadora_Q
        elif int(bit[0])==0 and int(bit[1])==1 and int(bit[2])==0 and int(bit[3])== 1:
            senal_Tx[i*mpp : (i+1)*mpp] = 0.25*portadora_I + -0.75*portadora_Q
        elif int(bit[0])==0 and int(bit[1])==1 and int(bit[2])==1 and int(bit[3])== 0:
            senal_Tx[i*mpp : (i+1)*mpp] = 0.75*portadora_I + -0.25*portadora_Q
        elif int(bit[0])==0 and int(bit[1])==1 and int(bit[2])==1 and int(bit[3])== 1:
            senal_Tx[i*mpp : (i+1)*mpp] = 0.75*portadora_I + -0.75*portadora_Q
        elif int(bit[0])==1 and int(bit[1])==0 and int(bit[2])==0 and int(bit[3])== 0:
            senal_Tx[i*mpp : (i+1)*mpp] = -0.25*portadora_I + 0.25*portadora_Q
        elif int(bit[0])==1 and int(bit[1])==0 and int(bit[2])==0 and int(bit[3])== 1:
            senal_Tx[i*mpp : (i+1)*mpp] = -0.25*portadora_I + 0.75*portadora_Q
        elif int(bit[0])==1 and int(bit[1])==0 and int(bit[2])==1 and int(bit[3])== 0:
            senal_Tx[i*mpp : (i+1)*mpp] = -0.75*portadora_I + 0.25*portadora_Q
        elif int(bit[0])==1 and int(bit[1])==0 and int(bit[2])==1 and int(bit[3])== 1:
            senal_Tx[i*mpp : (i+1)*mpp] = -0.75*portadora_I + 0.75*portadora_Q
        elif int(bit[0])==1 and int(bit[1])==1 and int(bit[2])==0 and int(bit[3])== 0:
            senal_Tx[i*mpp : (i+1)*mpp] = -0.25*portadora_I + -0.25*portadora_Q
        elif int(bit[0])==1 and int(bit[1])==1 and int(bit[2])==0 and int(bit[3])== 1:
            senal_Tx[i*mpp : (i+1)*mpp] = -0.75*portadora_I + -0.25*portadora_Q
        elif int(bit[0])==1 and int(bit[1])==1 and int(bit[2])==1 and int(bit[3])== 0:
            senal_Tx[i*mpp : (i+1)*mpp] = -0.25*portadora_I + -0.75*portadora_Q
        else: 
            senal_Tx[i*mpp : (i+1)*mpp] = -0.75*portadora_I + -0.75*portadora_Q
    
    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    return senal_Tx, P_senal_Tx, portadora_I, portadora_Q, moduladora  


# Ruido
def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx 

# DEMODULACIÓN

def demodulador_16QAM(senal_Rx, portadora_I, portadora_Q, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Demodulación
    for i, j, k in zip(range(N), range(0,N,4), range(1,N,4)):
        # Producto interno de dos funciones
        producto_I = senal_Rx[i*mpp : (i+1)*mpp] * portadora_I
        Ep_I = np.sum(producto_I)
        
        producto_Q = senal_Rx[i*mpp : (i+1)*mpp] * portadora_Q
        Ep_Q = np.sum(producto_Q)
        
        senal_demodulada[i*mpp : (i+1)*mpp] = producto_I + producto_Q

        # Criterio de decisión por detección de energía
        if Ep_I < 0:
            bits_Rx[j] = 1
        else:
            bits_Rx[j] = 0
            
        if Ep_Q < 0:
            bits_Rx[k] = 1
        else:
            bits_Rx[k] = 0 

    return bits_Rx.astype(int), senal_demodulada


# Reconstrucción de la Imagen
def bits_a_rgb(bits_Rx, dimensiones):
    '''Un bloque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


# ****************Simulación************************ 
# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = -2   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3 Modulación
senal_Tx, Pm, portadora_I, portadora_Q, moduladora = modulador16QAM(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador_16QAM(senal_Rx, portadora_I, portadora_Q, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)

# ****************Señales************************
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='r', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()





############## 4.2. - Estacionaridad y ergodicidad

# Tiempos en segundos (s)
t_final   = 0.1
periodo     = 500 
t_muestras = np.linspace(0, t_final, periodo)
ensayos = 4
xt = np.empty((ensayos, len(t_muestras)))
# Muestras
Aj = [-1,1]
V_cos =  np.cos(2*(np.pi)*fc*t_muestras)
V_sen =  np.sin(2*(np.pi)*fc*t_muestras)
for i in Aj:
    x = i*V_cos + i*V_sen 
    y = -i*V_cos + i*V_sen 
    xt[i , :] = x
    xt[i+1 , :] = y
    plt.plot(t_muestras, x)
    plt.plot(t_muestras, y)
# Promedio de los ensayos
P = [np.mean(xt[:,i]) for i in range(len(t_muestras))]
plt.plot(t_muestras, P, '.-', color = 'r', label = 'Promedio de los Ensayos')
# Promedio teórico
E = np.mean(senal_Tx)*t_muestras
plt.plot(t_muestras, E, '--', color='g', label = 'Promedio Teórico')
# Gráfica
plt.title('Proceso aleatorio')
plt.xlabel('$tiempo (s)$')
plt.ylabel('$x(t)$')
plt.legend()
plt.show()





############## 4.3. - Densidad espectral de potencia

senal_f = fft(senal_Tx) # Transformada de Fourier
Nm = len(senal_Tx) # Muestras de la señal
Ns = Nm // mpp # Número de símbolos (198 x 89 x 8 x 3)
Tc = 1 / fc # Tiempo del símbolo = periodo de la onda portadora
Tm = Tc / mpp # Tiempo entre muestras (período de muestreo)
T = Ns * Tc # Tiempo de la simulación
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2) # Espacio de frecuencias

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.title('Densidad espectral de potencia')
plt.xlim(0, 20000)
plt.grid()
plt.show()