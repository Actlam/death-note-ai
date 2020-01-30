#
#       顔検出 + 性別・年齢予測
# 
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from wide_resnet import WideResNet
from pathlib import Path
import align.detect_face
import tensorflow as tf
from skimage.transform import rescale
from keras.utils.data_utils import get_file
import os.path as os
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.3, thickness=1):
    """
    性別・年齢を表記する関数
    """
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (0,255,255), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)


class AgeGenderPrediction:
    """
    性別・年齢予測を行うためのクラス
    """
    def __init__(self, img_size):
        if os.isdir("model") == False:
            pre_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
            modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
            weight_file = get_file("weights.28-3.73.hdf5", pre_model, cache_subdir="model",
                                   file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
        else:
            weight_file = "model/weights.28-3.73.hdf5"

        model = WideResNet(img_size, depth=16, k=8)()
        model.load_weights(weight_file)

        self.model = model


    # 予測
    def __call__(self,faces):
        results = self.model.predict(faces)
        Genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        Ages = results[1].dot(ages).flatten()
        return Ages, Genders


def image_select(Ages):
    """
    エフェクトを選択する関数
    """
    if Ages > 71:
        blend_img = cv2.imread('./blend_images/input08.png')
        alpha = 0.1
    elif Ages > 61:
        blend_img = cv2.imread('./blend_images/input07.png')
        alpha = 0.2
    elif Ages > 51:
        blend_img = cv2.imread('./blend_images/input06.png')
        alpha = 0.25
    elif Ages > 41:
        blend_img = cv2.imread('./blend_images/input05.png')
        alpha = 0.35
    elif Ages > 31:
        blend_img = cv2.imread('./blend_images/input04.png')
        alpha = 0.75
    elif Ages > 21:
        blend_img = cv2.imread('./blend_images/input03.png')
        alpha = 0.8
    elif Ages > 11:
        blend_img = cv2.imread('./blend_images/input02.png')
        alpha = 0.85
    else:
        blend_img = cv2.imread('./blend_images/input01.png')
        alpha = 0.9
    return blend_img, alpha


if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    # ウィンドウ名
    ORG_WINDOW_NAME = "DEATH NOTE AI"
    CROPPED_WINDOW_NAME = "cropped faces"

    # カメラの有無
    camera_sw = True

    if camera_sw:
        DEVICE_ID = 0  # 使用するカメラ
        cap = cv2.VideoCapture(DEVICE_ID)  # カメラ映像取得

        # 初期フレームの読込
        end_flag, c_frame = cap.read()
        height, width, channels = c_frame.shape
    else:
        end_flag = True

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(CROPPED_WINDOW_NAME)

    agp_sw = True  # agp_modelを使用するかどうか

    if agp_sw:
        # 年齢と性別推定器の設定
        img_size = 64
        agp_model = AgeGenderPrediction(img_size)

    # 顔面検出器のモデルセット
    cascade_file = "haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)
    color = (0, 255, 225)
    pen_w = 3

    start = time.time()  # fps計測
    while end_flag == True:

        # カメラが存在しないとき（手動でFalseにする必要あり）
        if not camera_sw:
            c_frame = cv2.imread("./blend_images/obama.jpg")  #入力画像

        dst = c_frame  # 顔認識の外ではエフェクトをかけない
        img_gray = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)  # 二値画像化
        bb = cascade.detectMultiScale(img_gray, minSize=(100, 100))  # 画像内のbounding boxを取得する
        gra = 50  # 範囲のあそび

        # bounding boxの候補数が0以上である場合、年齢推定
        if len(bb) > 0:

            if agp_sw:
                # 候補領域の切り抜き
                c_frame_size = c_frame.shape
                y_min = np.max([0,bb[0,1]-gra])
                y_max = np.min([c_frame_size[0],bb[0,1]+bb[0,3]+gra])
                x_min = np.max([0,bb[0,0]-gra])
                x_max = np.min([c_frame_size[1],bb[0,0]+bb[0,2]+gra])
                _cropped = c_frame[y_min:y_max,x_min:x_max]

                try: # try, excpetはresize出来ないバグ対策
                    faces = cv2.resize(_cropped, (img_size, img_size))
                    faces = np.array([faces])
                    Ages, Genders = agp_model(faces)  # 年齢と性別推定
                    print(Ages)

                    cv2.imshow(CROPPED_WINDOW_NAME, _cropped)

                    # 年齢と性別の値を描画
                    for face in range(len(faces)):
                        if Genders[face][0] < 0.5:
                            gender = 'Male'
                            gender_score = 81
                        else:
                            gender = 'Female'
                            gender_score = 87
                            
                        blend_img, alpha = image_select(Ages)
                        blend_img = cv2.resize(blend_img, c_frame.shape[1::-1])
                        dst = cv2.addWeighted(c_frame, alpha, blend_img, 1-alpha, 0)

                        # フレームの表示
                        for (x, y, w, h) in bb:
                            # 候補領域の視覚化
                            fontpath = 'death_note_font_by_karlibell22.ttf'  # 同一フォルダにフォントが必要
                            font = ImageFont.truetype(fontpath, 32)
                            img_pil = Image.fromarray(dst)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((x, y-40), 'HUMAN', font = font, fill = (0,0,255))
                            draw.text((x, y), 'Life ' + str(gender_score - int(Ages[face])), font = font, fill = (0,0,255))
                            dst = np.array(img_pil)  # 描画したイメージをnumpyの配列に戻して出力
                except:
                    print('Human lost')

        cv2.imshow(ORG_WINDOW_NAME, dst)  # フレーム表示

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        if camera_sw:
            end_flag, c_frame = cap.read()

        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        print("FPS:",fps)
        start = end