from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from PIL import Image as PIL_Image
"""
from fastai import *
from fastai.vision import *
"""
from lib.fastiqa.all import *

model_file_url = 'https://github.com/baidut/PaQ-2-PiQ/releases/download/v1.0/RoIPoolModel-fit.10.bs.120.pth'
model_file_name = 'model'
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
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    """
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    """
    data = Im2MOS(TestImages(path=path))
    data.device = torch.device('cpu') # run on CPU
    model = RoIPoolModel()
    learn = RoIPoolLearner(data, model, path=path)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))

    qmap = learn.predict_quality_map(img, [32, 32])
    score = f'{qmap.global_score:.2f}'
    mat = qmap.mat.astype(int).tolist()
    data = {'local': mat,
            'result': score,
            'message': 'Created', 'code': 'SUCCESS',
            'success': True, 'status': 'OK',
            'ContentType':'application/json'
           }
    return JSONResponse(data)

@app.route('/filepond', methods=['POST'])
async def analyze(request):
    # note that filepond post is different
    data = await request.form()
    # print(data) # FormData([('filepond', '{}'), ('filepond', <starlette.datastructures.UploadFile object at 0x7f22b454fdd8>)])
    print(data['filepond'].file)
    img_bytes = (data['filepond'].file.read()) # TypeError: object bytes can't be used in 'await' expression
    img = open_image(BytesIO(img_bytes))
    # img = PIL_Image.open(file.stream)
    # t = pil2tensor(img.convert("RGB"), np.float32).div_(255)
    # img = Image(t)

    qmap = learn.predict_quality_map(img, [32, 32])
    score = f'{qmap.global_score:.2f}'
    mat = qmap.mat.astype(int).tolist()
    data = {'local': mat,
            'result': score,
            'message': 'Created', 'code': 'SUCCESS',
            'success': True, 'status': 'OK',
            'ContentType':'application/json'
           }
    return JSONResponse(data, status_code=200, headers={'Access-Control-Allow-Origin': '*'})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)
