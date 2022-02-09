import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__, template_folder='template')

model = joblib.load('mushroom_classification.pkl')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        cap_shape = (request.form['cap_shape'])
        cap_surface = (request.form['cap_surface'])
        cap_color = (request.form['cap_color'])
        bruises = (request.form['bruises'])
        odor = (request.form['odor'])
        gill_attachment = (request.form['gill_attachment'])
        gill_spacing = (request.form['gill_spacing'])
        gill_size = (request.form['gill_size'])
        gill_color = (request.form['gill_color'])
        stalk_shape = (request.form['stalk_shape'])
        stalk_root = (request.form['stalk_root'])
        stalk_surface_above_ring = (request.form['stalk_surface_above_ring'])
        stalk_surface_below_ring = (request.form['stalk_surface_below_ring'])
        stalk_color_above_ring = (request.form['stalk_color_above_ring'])
        stalk_color_below_ring = (request.form['stalk_color_below_ring'])
        veil_type = (request.form['veil_type'])
        veil_color = (request.form['veil_color'])
        ring_number = (request.form['ring_number'])
        ring_type = (request.form['ring_type'])
        spore_print_color = (request.form['spore_print_color'])
        population = (request.form['population'])
        habitat = (request.form['habitat'])
        arr = np.array([[cap_shape,cap_surface,cap_color,bruises,odor,gill_attachment,gill_spacing,gill_size,
                         gill_color,stalk_shape,stalk_root,stalk_surface_above_ring,stalk_surface_below_ring,
                         stalk_color_above_ring,stalk_color_below_ring,veil_type,veil_color,ring_number,
                         ring_type,spore_print_color,population,habitat]])
        prediction = model.predict(arr)
    return render_template('after.html', data=prediction)


if __name__ == '__main__':
    app.run()
