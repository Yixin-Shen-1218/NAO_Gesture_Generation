from naoqi import ALProxy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# Get the Posture
proxy = ALProxy("ALRobotPosture", "10.61.144.12", 9559)

print("Current position: ", proxy.getPostureFamily())


def myDisplayImageFunction(imageNAO):
    lena = mpimg.imread(imageNAO)
    lena.shape(512, 512, 3)

    plt.imshow(lena)
    plt.axis('off')
    plt.show()

# Get the Image
videoProxy = ALProxy("ALVideoDevice", "10.61.144.12", 9559)

subscriber = videoProxy.subscribeCamera("demo", 0, 3, 13, 1)

imageNAO = videoProxy.getImageRemote(subscriber)

# print imageNAO

# myDisplayImageFunction(imageNAO)