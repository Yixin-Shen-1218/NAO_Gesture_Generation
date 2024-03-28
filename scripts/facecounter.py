from naoqi import ALBroker, ALModule, ALProxy
import time, sys


class FaceCounterModule(ALModule):
    def __init__(self, name):
        ALModule.__init__(self, name)
        self.name = name
        self.memoryProxy = ALProxy("ALMemory")
        self.memoryProxy.subscribeToEnvent("FaceDetected", self.name, "OnFaceDetected")

    def onFaceDetected(self, key, value, message):
        """ returns the count """
        print "Face detected! value = " + str(value)

    def getCount(self):
        pass


if __name__ == '__main__':
    pip = "10.61.144.12"
    ppport = 9559

    myBroker = ALBroker("myBroker",
        "0.0.0.0",
        0,
        pip,
        ppport)

    global FaceCounter
    FaceCounter = FaceCounterModule("FaceCounter")

    