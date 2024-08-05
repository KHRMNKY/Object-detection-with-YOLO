from roboflow import Roboflow
rf = Roboflow(api_key="Kp85FfPGfaxRDV4AWTG8")
project = rf.workspace("thesis-3c51t").project("cow-counting")
version = project.version(12)
dataset = version.download("yolov8n")
