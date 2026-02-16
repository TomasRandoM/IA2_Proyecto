extends CharacterBody2D

@export var animation: Node

var _velocidad: float = 100.0
var lastDir = "up"
var doneUp = true
var doneLeft = true
var doneDown = true
var doneRight = true
var path = "user://auxAudio.wav"
var effect
var recording: AudioStreamWAV
var http

func _ready():
	add_to_group("players")
	http = HTTPRequest.new()
	add_child(http) 
	http.request_completed.connect(_on_http_request_completed)
	var idx = AudioServer.get_bus_index("Record")
	effect = AudioServer.get_bus_effect(idx, 0)
	
func _physics_process(_delta):
	
	if (velocity.x == 0 && velocity.y == 0): 
		if (lastDir == "up"):
			animation.play("idleUp")
			if (doneUp == false):
				doneUp = true
				global_position.y += 2
		elif (lastDir == "down"):
			animation.play("idleDown")
			if (doneDown == false):
				doneDown = true
				global_position.y -= 2
		elif (lastDir == "right"):
			animation.play("idleRight")
			animation.flip_h = false
			if (doneRight == false):
				doneRight = true
				global_position.x -= 2
		else:
			animation.play("idleLeft")
			animation.flip_h = true
			if (doneLeft == false):
				doneLeft = true
				global_position.x += 2
		if Input.is_action_just_pressed("up"):
			lastDir = "up"
			doneUp = false
			velocity.y = -_velocidad
			animation.play("runUp")
		if Input.is_action_just_pressed("right"):
			lastDir = "right"
			velocity.x = _velocidad
			doneRight = false
			animation.play("runRight")
			animation.flip_h = false
		if Input.is_action_just_pressed("down"):
			lastDir = "down"
			velocity.y = +_velocidad
			doneDown = false
			animation.play("runDown")
		if Input.is_action_just_pressed("left"):
			lastDir = "left"
			velocity.x = -_velocidad
			doneLeft = false
			animation.play("runRight")
			animation.flip_h = true
	else:	
		if velocity.length() < 70.0:
			velocity = Vector2.ZERO
	move_and_slide()

func _input(event):
	if (event.is_action_pressed("record")):
		startRecording()
		
func startRecording():
	effect.set_recording_active(true)
	 
	await get_tree().create_timer(1.5).timeout 
	stopRecording()
		  

func stopRecording():
	effect.set_recording_active(false)
	recording = effect.get_recording()
	recording.save_to_wav(path)
	var file = FileAccess.open(path, FileAccess.READ)
	var bytes
	if file:
		var size = file.get_length()
		bytes = file.get_buffer(size)
		sendApi(bytes)

	
func sendApi(file_bytes):
	var url = "http://127.0.0.1:6000/cnn/predict"
	var boundary = "----GodotFormBoundary123456"
	
	var body := "--%s\r\n" % boundary
	body += "Content-Disposition: form-data; name=\"file\"; filename=\"auxAudio.wav\"\r\n"
	body += "Content-Type: audio/wav\r\n\r\n"
	var header_bytes = body.to_utf8_buffer()
	
	var footer_bytes = "\r\n--%s--\r\n" % boundary
	footer_bytes = footer_bytes.to_utf8_buffer()
	
	var final_bytes = header_bytes + file_bytes + footer_bytes
	
	var headers = [
		"Content-Type: multipart/form-data; boundary=%s" % boundary
	]
	var err = http.request_raw(url, headers, HTTPClient.METHOD_POST, final_bytes)
	
func _on_http_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
	var prediction
	if result != HTTPRequest.RESULT_SUCCESS:
		return

	if response_code != 200:
		return

	var respuesta = body.get_string_from_utf8()

	var json = JSON.new()
	var parse = json.parse(respuesta)
	if parse == OK:
		var data = json.data

		if data is Dictionary:
			if data.has("msg"):
				prediction = data["msg"]
				if (velocity.x == 0 && velocity.y == 0):
					if prediction == "up":
						lastDir = "up"
						doneUp = false
						velocity.y = -_velocidad
						animation.play("runUp")

					if prediction == "right":
						lastDir = "right"
						velocity.x = _velocidad
						doneRight = false
						animation.play("runRight")
						animation.flip_h = false

					if prediction == "down":
						lastDir = "down"
						velocity.y = +_velocidad
						doneDown = false
						animation.play("runDown")

					if prediction == "left":
						lastDir = "left"
						velocity.x = -_velocidad
						doneLeft = false
						animation.play("runRight")
						animation.flip_h = true
	return
