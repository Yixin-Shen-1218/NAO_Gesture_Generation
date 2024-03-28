from naoqi import ALProxy

motion = ALProxy("ALMotion", "10.61.144.12", 9559)

motion.moveTo(1.0, 0.0, 0.0)