from naoqi import ALProxy

tts = ALProxy("ALTextToSpeech", "10.61.144.12", 9559)
motion = ALProxy("ALMotion", "10.61.144.12", 9559)


threadMove = motion.post.moveTo(0.5, 0.0, 0.0)
tts.say("I can walk! Let us explore the area.")

motion.wait(threadMove, 0)
tts.say("I have reached the destination.")