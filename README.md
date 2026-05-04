# Reconocimiento de órdenes cortas en audio para su integración en un videojuego  

### Autor  
Tomás Rando

### Descripción  
Se resolvió el reconocimiento de órdenes cortas en audios en inglés haciendo uso de modelos que utilizan **redes convolucionales** y **vision transformers**. Inicialmente, se elabora el espectrograma asociado a cada audio y este se utiliza para que el modelo prediga la órden. Posteriormente, se desarrolló un videojuego simple tipo ice-sliding que se controla utilizando la voz.  

Los modelos se cargan en una aplicación desarrollada en Python que implementa una API con dos endpoints que reciben un audio y devuelven la predicción de los modelos. Por otro lado, el videojuego graba audios de 1.5 segundos cuando se le indica y realiza peticiones HTTP a la API con dicho audio. Al recibir la respuesta del modelo, el juego la interpreta y realiza la acción correspondiente. Toda esta comunicación se produce muy rápidamente ya que ambos componentes se ejecutan en el mismo entorno local.  

Por último, se produjo un informe asociado que muestra el proceso realizado y los resultados obtenidos.  
[Informe](https://github.com/TomasRandoM/Proyecto_CNNyViT/InformeFinal.pdf)  

### Herramientas utilizadas  
- Python 3.12.12  
- Godot 4.6.1  
- TensorFlow 2.19.0  
- TensorFlow-Datasets 4.9.9
- Flask 3.1.2
- Google Speech Commands Dataset v0.0.3
- Keras 