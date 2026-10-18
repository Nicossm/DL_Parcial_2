# Clasificador de personajes de Los Simpsons (CNN)

Parcial 2 de Deep Learning, DUOC UC (2do semestre 2026).

Red convolucional hecha en Keras para reconocer a 18 personajes de Los Simpsons a partir de imágenes de 64x64. El objetivo era pasar el 85% de accuracy en test.

## Resultados

| Modelo | Validación | Test |
|---|---|---|
| Red densa (Fully Connected) | 26,9% | 21,8% |
| CNN sencilla | 75,9% | 93,8% |
| CNN profunda con BatchNorm y Dropout | 92,3% | 96,6% |
| CNN profunda + data augmentation | 95,6% | 98,4% |

El modelo final es el último: 4 bloques convolucionales, BatchNorm, Dropout y data augmentation, entrenado con Adam. El detalle de cada prueba, las métricas por personaje y los errores están en el notebook.

## Archivos

- `CNN_SIMPSON_FINAL.ipynb`: notebook con todo el trabajo, ya ejecutado.
- `modelos/mejor_modelo_simpsons.keras`: el modelo final entrenado.
- `requirements.txt`: librerías usadas.

## Cómo correrlo

Lo entrené en Google Colab con GPU T4 (todo el notebook tarda unos 20 minutos).

1. Subir `simpsons_train.tar.gz` y `simpsons_test.tar.gz` a Google Drive, en `MyDrive/DeepLearning/P2/`. Si se usa otra carpeta hay que cambiar `RUTA_DRIVE` en la sección 2.
2. Abrir el notebook en Colab y elegir un entorno con GPU.
3. Ejecutar todas las celdas.

El dataset no está en el repo porque pesa unos 500 MB.

Para usar el modelo ya entrenado sin volver a entrenar:

```python
from tensorflow import keras
modelo = keras.models.load_model("modelos/mejor_modelo_simpsons.keras")
```

Las imágenes tienen que ir en RGB, de 64x64 y divididas por 255, igual que en el notebook.
