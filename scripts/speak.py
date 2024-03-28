from naoqi import ALProxy

tts = ALProxy("ALTextToSpeech", "10.61.144.12", 9559)

tts.say("hello world")
