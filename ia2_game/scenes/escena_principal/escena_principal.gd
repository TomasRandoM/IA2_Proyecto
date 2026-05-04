extends Node2D

@export var niveles: Array[PackedScene]
var option : String = "CNN"
var _nivel_actual: int = 1
var _nivel_instanciado: Node
func _ready() -> void:
	_crear_nivel(_nivel_actual)

func _crear_nivel(numero_nivel: int):
	_nivel_instanciado = niveles[numero_nivel - 1].instantiate()
	add_child(_nivel_instanciado)
	if (option == "CNN"):
		_nivel_instanciado.get_node("Player").url = "http://127.0.0.1:6000/cnn/predict"
	else:
		_nivel_instanciado.get_node("Player").url = "http://127.0.0.1:6000/vit/predict"
		

func _eliminar_nivel():
	_nivel_instanciado.queue_free()
	
func _reiniciar_nivel():
	_eliminar_nivel()
	_crear_nivel.call_deferred(_nivel_actual)
	
