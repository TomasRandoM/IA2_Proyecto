extends Button

@export var boton_jugar: Button
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pressed.connect(_cargar)


func _cargar():
	boton_jugar.jugar(2)
