from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
import time
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model
from time import sleep

path1=r'D:\Projects\age-gender-estimation-master\age-gender-estimation-master\advertisements\child.jpg'
path2=r'D:\Projects\age-gender-estimation-master\age-gender-estimation-master\advertisements\teen.jpg'
path3=r'D:\Projects\age-gender-estimation-master\age-gender-estimation-master\advertisements\dove.png'
path4=r'D:\Projects\age-gender-estimation-master\age-gender-estimation-master\advertisements\old.jpg'
path5=r'D:\Projects\age-gender-estimation-master\age-gender-estimation-master\advertisements\lil.jpg'
path6=r'D:\Projects\age-gender-estimation-master\age-gender-estimation-master\advertisements\young.jpg'
path7=r'D:\Projects\age-gender-estimation-master\age-gender-estimation-master\advertisements\aunty.jpg'
path8=r'D:\Projects\age-gender-estimation-master\age-gender-estimation-master\advertisements\grandmom.jpg'
path9=r'D:\Projects\age-gender-estimation-master\age-gender-estimation-master\advertisements\pizza.jpg'
child=cv2.imread(path1)
teen=cv2.imread(path2)
adult=cv2.imread(path3)
old=cv2.imread(path4)
lil=cv2.imread(path5)
young=cv2.imread(path6)
aunty=cv2.imread(path7)
grandmom=cv2.imread(path8)
pizza=cv2.imread(path9)
window_name="maleadv"
windows_name="femaleadv"
windows1_name="neutraladv"

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    start_time = time.time()
    capture_duration = 15
    image_duration=2 

    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # while (True and ( int(time.time() - start_time) < capture_duration )):
        #     if not cap.isOpened():
        #         print("camera is closed")
        #         sleep(5)
        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))


def main():
    start_time = time.time()
    capture_duration = 15
    image_duration=7 
    args = get_args()
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir

    if not weight_file:
        weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()

    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            ######################################## MY CODE start #######################################################33

            age_sum = 0
            male = 0
            female = 0
            count=0
            young = 0
            adult = 0
            old = 0
            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")
                draw_label(img, (d.left(), d.top()), label)
                # age_sum = age_sum + predicted_ages[i]
                if(predicted_ages[i] < 25):
                    young = young + 1
                elif(predicted_ages[i] >=25 and predicted_ages[i]<= 45):
                    adult = adult + 1
                else:
                    old = old + 1

                count = count +1
                if (predicted_genders[i][0] < 0.5):
                    male=male+1
                    
            if(((male/count) >= 0.35) and ((male/count)<=0.65)):
                    cv2.destroyAllWindows()
                    tart_time = time.time()
                    while( int(time.time() - tart_time) < image_duration ):
                         cv2.imshow(window_name,pizza)
                        
                        #  cv2.destroyWindow(window_name)
                         if cv2.waitKey(5) == 27:  # ESC key press
                             break
            elif ((male/count)>0.65):
                 
                if(young > (count/2)):
                     cv2.destroyAllWindows()
                     tart_time = time.time()
                     while( int(time.time() - tart_time) < image_duration ):
                            cv2.imshow(window_name,child)
                        
                        #  cv2.destroyWindow(window_name)
                            if cv2.waitKey(5) == 27:  # ESC key press
                                break
                elif(adult > (count/2)):
                    #  cv2.imshow(window_name,teen)
                    #  sleep(10)
                    cv2.destroyAllWindows()
                    tart_time = time.time()
                    while( int(time.time() - tart_time) < image_duration ):
                         cv2.imshow(window_name,teen)
                        
                        #  cv2.destroyWindow(window_name)
                         if cv2.waitKey(5) == 27:  # ESC key press
                             break
                elif(old > (count/2)):
                    cv2.destroyAllWindows()
                    tart_time = time.time()
                    while( int(time.time() - tart_time) < image_duration ):
                         cv2.imshow(window_name,old)
                        
                        #  cv2.destroyWindow(window_name)
                         if cv2.waitKey(5) == 27:  # ESC key press
                             break
                else:
                    cv2.imshow(window_name,pizza)
                        #  sleep(10)
                    tart_time = time.time()
                    while( int(time.time() - tart_time) < image_duration ):
                        
                        if cv2.waitKey(5) == 27:  # ESC key press
                            break

    #for women        
            else:
                if(young > (count/2)):
                    cv2.destroyAllWindows()
                    tart_time = time.time()
                    while( int(time.time() - tart_time) < image_duration ):
                         cv2.imshow(windows_name,lil)
                        
                        #  cv2.destroyWindow(window_name)
                         if cv2.waitKey(5) == 27:  # ESC key press
                             break
                elif(adult > (count/2)):
                    cv2.destroyAllWindows()
                    tart_time = time.time()
                    while( int(time.time() - tart_time) < image_duration ):
                         cv2.imshow(windows_name,lil)
                        
                        #  cv2.destroyWindow(window_name)
                         if cv2.waitKey(5) == 27:  # ESC key press
                             break
                elif(old > (count/2)):
                    cv2.destroyAllWindows()
                    tart_time = time.time()
                    while( int(time.time() - tart_time) < image_duration ):
                         cv2.imshow(windows_name,grandmom)
                        
                        #  cv2.destroyWindow(window_name)
                         if cv2.waitKey(5) == 27:  # ESC key press
                             break
                else:
                    cv2.destroyAllWindows()
                    tart_time = time.time()
                    while( int(time.time() - tart_time) < image_duration ):
                         cv2.imshow(windows1_name,pizza)
                        
                        #  cv2.destroyWindow(window_name)
                         if cv2.waitKey(5) == 27:  # ESC key press
                             break

            # avg_age = age_sum/count
#            if(young > (count/2)):
#               print("And belongs to young category")
#           elif(adult > (count/2)):
#                    print("And belongs to adult category")
 #           elif(old > (count/2)):
#                print("And belongs to old category")
#            else:
#                print("And the audience are a mix of all age groups")
############################################# My Code end ########################################################333
            # draw results
            # for i, d in enumerate(detected):
            #     label = "{}, {}".format(int(predicted_ages[i]),
            #                             "M" if predicted_genders[i][0] < 0.5 else "F")
            #     draw_label(img, (d.left(), d.top()), label)
                # sleep(5)
                # print(label)

        cv2.imshow("result", img)
        # sleep(3)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == '__main__':
    main()
