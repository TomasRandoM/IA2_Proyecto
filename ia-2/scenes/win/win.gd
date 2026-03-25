extends Area2D

func _ready():
	pass

func _on_body_entered(body: Node2D) -> void:
	if body.is_in_group("players"):
		call_deferred("_volver_al_menu")
		
func _volver_al_menu():
	get_tree().change_scene_to_file("res://scenes/menu_principal/menu_principal.tscn")
