from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
"""
from fastai import *
from fastai.vision import *
from lib.fastiqa.all import *
"""
from paq2piq_standalone import *
import sys

model_file_url = 'https://github.com/baidut/PaQ-2-PiQ/releases/download/v1.0/RoIPoolModel-fit.10.bs.120.pth'
model_file_name = 'RoIPoolModel'
# classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware,
        allow_origins=["*"],
        allow_headers=["*"], # ['X-Requested-With', 'Content-Type']
        allow_methods=["*"],
        expose_headers=["X-Status"],
        allow_credentials=True,
        )
app.mount('/static', StaticFiles(directory='static'))

# async def download_file(url, dest):
#     if dest.exists(): return
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             data = await response.read()
#             with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    file_path = path/'models'/f'{model_file_name}.pth'
    # await download_file(model_file_url, file_path)
    # run on cpu
    return InferenceModel(RoIPoolModel(), file_path)

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())


def get_results(img_bytes):
    image = Image.open(BytesIO(img_bytes))
    output = model.predict_from_pil_image(image)
    # save traffic? (not so important)
    # Object of type 'float32' is not JSON serializable
    for key, val in output.items():
        if not isinstance(val, str):
            output[key] = np.array(val).astype(int).tolist()
    return JSONResponse(output)


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    return get_results(img_bytes)



@app.route('/filepond', methods=['POST'])
async def analyze(request):
    # note that filepond post is different
    data = await request.form()
    # print(data) # FormData([('filepond', '{}'), ('filepond', <starlette.datastructures.UploadFile object at 0x7f22b454fdd8>)])
    print(data['filepond'].file)
    img_bytes = (data['filepond'].file.read())
    return get_results(img_bytes)


if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)
