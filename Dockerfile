FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app app/

RUN python app/server.py

EXPOSE 8080

CMD ["python", "app/server.py", "serve"]