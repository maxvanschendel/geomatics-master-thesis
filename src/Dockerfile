FROM python:3.8
COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install -r requirements.txt
COPY . /opt/app

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6 -y

CMD [ "python", "./main.py"]