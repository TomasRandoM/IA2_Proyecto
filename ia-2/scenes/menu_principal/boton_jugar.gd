extends Button

@export var escena_principal: PackedScene

func _ready() -> void:
	pressed.connect(jugar.bind(1))

func jugar(value):
	var escena = escena_principal.instantiate()
	if (value == 1):
		escena.option = "CNN";
	else:
		escena.option = "VIT";
		
	get_tree().change_scene_to_node(escena)
