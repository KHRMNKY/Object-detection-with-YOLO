import multiprocessing
from ultralytics import YOLO

def image_prediction(img_path):
    model = YOLO(".\\runs\\detect\\train3\\weights\\best.pt")
    results = model(img_path)
    print(results[0].show())


if __name__ == '__main__':
    multiprocessing.freeze_support()
    image_prediction(".\\48.jpg")
