import os

from flask import Flask, render_template, app, send_file, request, jsonify, flash, redirect, url_for
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import sys
sys.path.insert(1, './AImodel')
from efficientdet_test_videos import excuteModel

# configuration
DEBUG = True
filename = "upload.mp4"
# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})
cors = CORS(app, resources={r"/upload-file": {"origins": "*"}})
app.config['CORS_HEADER'] = 'Content-Type'

# sanity check route
UPLOAD_FOLDER = './../server/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def ping_pong():
    return jsonify('pong!')


@app.route('/upload-file', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def postdata():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], "upload.mp4")
        file.save(full_filename)
        return 'Upload Done'


@app.route('/detect', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def detector():
    path = excuteModel("upload")
    return "./../assets/upload.mp4"


if __name__ == '__main__':
    app.run()
