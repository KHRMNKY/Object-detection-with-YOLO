import multiprocessing
from ultralytics import YOLO
import cv2 as cv


def video_detect():
    model = YOLO(".\\runs\\detect\\train3\\weights\\best.pt")
    capture = cv.VideoCapture(".\\test_data_video_and_imgs\\cow_video.mp4")

    frame_width = int(capture.get(3)) 
    frame_height = int(capture.get(4)) 
    size = (frame_width, frame_height)
    result = cv.VideoWriter('output.mp4',  
                            cv.VideoWriter_fourcc(*'mp4v'), 
                            10, size) 

    while True:
        ret, frame = capture.read()
        if not ret:
            break 
        preds = model(frame)
        frame = preds[0].plot()
        result.write(frame)
        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord('q'):
            break
    capture.release() 
    result.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    multiprocessing.freeze_support()
    video_detect()
