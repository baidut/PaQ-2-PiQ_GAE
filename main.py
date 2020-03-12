# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python37_app]
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS

from paq2piq_standalone import *
import sys
# from io import BytesIO

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)
app.model = InferenceModel(RoIPoolModel(), Path(__file__).parent/'models'/'RoIPoolModel.pth')
CORS(app)

def get_results(img_bytes):
    image = Image.open(img_bytes)
    output = app.model.predict_from_pil_image(image)
    # save traffic? (not so important)
    # Object of type 'float32' is not JSON serializable
    for key, val in output.items():
        if not isinstance(val, str):
            output[key] = np.array(val).astype(int).tolist()
    return make_response(jsonify(output), 200)

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'

@app.route('/view')
def index():
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/filepond', methods=['POST'])
def analyze():
    # note that filepond post is different
    for key in request.files:
        file = request.files[key]
        #print(data['filepond'].file)
        return get_results(file.stream)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python37_app]
