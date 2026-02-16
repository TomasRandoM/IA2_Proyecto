extends Area2D

func _ready():
	pass

func _on_body_entered(body: Node2D) -> void:
	if body.is_in_group("players"):
		get_tree().quit()
