import os
from PIL import Image
import numpy as np
import datetime
from flask import Flask, render_template, request, redirect

from keras.models import load_model
from keras.preprocessing import image

# Membuat direktori Upload untuk menyimpan data input dari user
uploads_dir = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

app = Flask(__name__)
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

# Kamus untuk memetakan nomor kelas ke label yang lebih deskriptif
dic_model2 = {
    0: 'Blight. Blight pada tanaman jagung merupakan penyakit yang disebabkan oleh jamur Exserohilum turcicum, yang juga dikenal sebagai penyakit hawar daun. Penyakit ini dapat menyebabkan kerusakan signifikan pada daun tanaman jagung, mengurangi kemampuan fotosintesis dan, pada akhirnya, mengurangi hasil panen.',
    1: 'Common Rust. Common Rust pada tanaman jagung, yang disebabkan oleh jamur Puccinia sorghi, adalah penyakit yang dapat menginfeksi daun dan mengurangi hasil panen.',
    2: 'Gray Leaf Spot. Gray Leaf Spot pada tanaman jagung adalah penyakit yang disebabkan oleh jamur Cercospora zeae-maydis.',
    3: 'Healthy atau Tanaman Jagung dalam keadaan sehat.'
}

# Load models
model2 = load_model('Penyakitjagung.h5')

def predict_image(model, img_path, dic):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    class_probabilities = model.predict(img_array)
    predicted_class = np.argmax(class_probabilities)
    
    # Menggunakan dic untuk mendapatkan label yang sesuai dengan kelas yang diprediksi
    predicted_label = dic.get(predicted_class, "Unknown Class")
    # Mengambil nilai probabilitas kelas yang diprediksi
    confidence = class_probabilities[0][predicted_class] * 100
    
    return predicted_label, confidence

# Route untuk halaman utama
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

# Route untuk halaman about
@app.route("/about", methods=['GET'])
def about():
    return render_template("about.html")

# Route untuk halaman education_agriculture
@app.route("/education_agriculture", methods=['GET'])
def education_agriculture():
    return render_template("education_agriculture.html")

# Route untuk halaman team
@app.route("/team", methods=['GET'])
def team():
    return render_template("team.html")

# Route untuk detection_temperature
@app.route("/detection_temperature", methods=['GET', 'POST'])
def detection_temperature():
    return render_template("monitoring.html")

# Route untuk halaman detection_webcam
@app.route("/detection_webcam", methods=['GET'])
def detection_webcam():
    return render_template("detection_webcam.html")

# Route untuk detection_image
@app.route("/detection_image", methods=['GET', 'POST'])
def detection_image():
    return render_template("detection_image.html")

@app.route("/submit", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_name = request.form['model_name']
        img = request.files['my_image']
        img_path = "static/images/detect/" + img.filename
        img.save(img_path)

        if model_name == 'model2':
            prediction = predict_image(model2, img_path, dic_model2)
            result = f"Prediksi Penyakit Jagung: {prediction}"
        else:
            result = "Model tidak dikenali."
           
        return render_template("detection_image.html", img_path=img_path, result=result)

    return render_template("detection_image.html")

if __name__ == '__main__':
    app.run(debug=True)
