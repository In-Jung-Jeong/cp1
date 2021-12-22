import flask
from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from werkzeug.utils import secure_filename

def recall(y_target, y_pred):
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) 

    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    count_true_positive_false_negative = K.sum(y_target_yn)

    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    return recall


def precision(y_target, y_pred):
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    y_target_yn = K.round(K.clip(y_target, 0, 1)) 

    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    count_true_positive_false_positive = K.sum(y_pred_yn)

    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    
    return _f1score


smooth = 1e-4

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)

app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
def index():
    return render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        file = request.files['image']

        npimg = np.fromfile(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)


        plt.imshow(np.squeeze(img))
        plt.savefig('static/uploads/' + file.filename + '.jpg')

        img = cv2.resize(img, (256, 256))
        img = img / 255
        img = img[np.newaxis, :, :, :]
        pred = model.predict(img)

        plt.imshow(np.squeeze(pred))
        plt.savefig('static/uploads/predict_' + file.filename + '.jpg')

        return render_template('view.html', origin_data='static/uploads/' + file.filename + '.jpg', 
        pred_data='static/uploads/'+'predict_' + file.filename + '.jpg')


if __name__ == "__main__":
    custom = {
    'dice_coef' : dice_coef,
    'dice_coef_loss' : dice_coef_loss,
    'iou' : iou,
    'precision': precision,
    'recall' : recall,
}
    model = tf.keras.models.load_model('unet_brain_mri_seg.h5', custom_objects=custom)
    app.run(debug=True)
