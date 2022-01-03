import numpy as np
import cv2

cap = cv2.VideoCapture("../media/input/Cars_On_Highway.mp4")
_, OGFrame = cap.read()
height, width, channels = OGFrame.shape
Path_Vector = '../media/output/Vector_Field.mp4'
Path_Motion = '../media/output/Vehicle_Motion.mp4'
Path_Sub = '../media/output/Background_Subtraction.mp4'
FPS = 25
Scale = 0.5
WindowSize = (int(width * Scale), int(height * Scale))
OGFrame = cv2.resize(OGFrame, WindowSize)
prev = cv2.cvtColor(OGFrame, cv2.COLOR_BGR2GRAY)
HSV_Vehicles = np.zeros_like(OGFrame)
HSV_Vehicles[..., 1] = 255
outVector = cv2.VideoWriter(Path_Vector, cv2.VideoWriter_fourcc(*'mp4v'), FPS, WindowSize)
outMotion = cv2.VideoWriter(Path_Motion, cv2.VideoWriter_fourcc(*'mp4v'), FPS, WindowSize)
outSub = cv2.VideoWriter(Path_Sub, cv2.VideoWriter_fourcc(*'mp4v'), FPS, WindowSize)


def VectorField(VectorFrame, prev, next):
    VectorFrame_Final = VectorFrame.copy()
    OpticalFlow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    OpticalFlowCopy = OpticalFlow.copy()
    Magnitude, Angle = cv2.cartToPolar(OpticalFlow[:, :, 0], OpticalFlow[:, :, 1])
    print(VectorFrame.shape)
    for i in range(VectorFrame.shape[0]):
        for j in range(VectorFrame.shape[1]):
            if i % 25 == 0 and j % 25 == 0:
                ArrowHead = (j, i)
                ArrowTail = (i + (2 * Magnitude[i, j] * np.sin(Angle[i, j])), j + (2 * Magnitude[i, j] * np.cos(Angle[i, j])))
                VectorFrame_Final = cv2.arrowedLine(VectorFrame_Final, ArrowHead, (int(ArrowTail[1]), int(ArrowTail[0])), (0, 255, 0), 2)
    return VectorFrame_Final, OpticalFlow, OpticalFlowCopy, Angle, Magnitude


def VehicleMotion(Angle, Magnitude):
    HSV_Vehicles[..., 0] = Angle * (180 / (np.pi / 2))
    HSV_Vehicles[..., 2] = cv2.normalize(Magnitude, None, 0, 255, cv2.NORM_MINMAX)
    BGR_Vehicles = cv2.cvtColor(HSV_Vehicles, cv2.COLOR_HSV2BGR)
    return BGR_Vehicles

def BackgroundSubtraction(OpticalFlowCopy, VectorFrame):
    SubtractedFrame = np.zeros_like(VectorFrame)
    x = OpticalFlowCopy[:, :, 0]
    x = 2.5 * x
    arr = np.where(x > 1)
    for i in range(len(arr[0])):
        SubtractedFrame[arr[0][i], arr[1][i], :] = VectorFrame[arr[0][i], arr[1][i], :]
    return SubtractedFrame

while True:
    _, VectorFrame = cap.read()
    if VectorFrame is None:
        break
    VectorFrame = cv2.resize(VectorFrame, WindowSize)
    next = cv2.cvtColor(VectorFrame, cv2.COLOR_BGR2GRAY)
    VectorFrame_Final, OpticalFlow, OpticalFlowCopy, Angle, Magnitude = VectorField(VectorFrame, prev, next)
    SubtractedFrame = BackgroundSubtraction(OpticalFlowCopy, VectorFrame)
    BGR_Vehicles = VehicleMotion(Angle, Magnitude)
    cv2.imshow('Vehicle Motion', BGR_Vehicles)
    cv2.imshow('Vector Field', VectorFrame_Final)
    cv2.imshow('Background Subtraction', SubtractedFrame)
    outVector.write(VectorFrame_Final)
    outMotion.write(BGR_Vehicles)
    outSub.write(SubtractedFrame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    prev = next

outVector.release()
outMotion.release()
outSub.release()
cv2.destroyAllWindows()
