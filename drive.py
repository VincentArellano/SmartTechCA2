#import flask
#install flask using pip install flask
#command to isnatll socketio
# py -m pip install python-socketio==4.6.1
#py -m pip install tensorflow 
#install eventlet
#py -m pip install eventlet==0.30.2

#py -m pip install pillow
#install open cv
#py -m pip install opencv-contrib-python
#py -m pip install numpy
#install engineio
#py -m pip install python-engineio==3.13.2

import socketio

import eventlet
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from imgaug import augmenters as iaa #pip install imgaug

#initialize our server with the socketio library
sio = socketio.Server()

#create a flask application
app = Flask(__name__) #'__main__'

#speed limit for the car
speed_limit = 20

@sio.on('connect') # decorator
#connect is the event name
def connect(sid, environ):
    print('Connected')
    send_control(0,1)

#add method preprocess to preprocess the image
def image_preprocess(image):
    #crop the image
    image = image[60:135,:,:]
    #convert to YUV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    #apply gaussian blur
    image = cv2.GaussianBlur(image, (3,3), 0)
    #resize the image
    image = cv2.resize(image, (200,66))
    #normalize the image
    image = image/255
    return image

#function to preprocess the image before we send it to the model
@sio.on('telemetry')
#will take the session id and the data
def telemetry(sid, data):  
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image) 
    #preprocess the image
    image = image_preprocess(image) 
    image = np.array([image])
    #predict the steering angle from the model..passing the image to the model
    steering_angle = float(model.predict(image))
    speed = float(data['speed'])
    #relation between speed and steering angle
    print(f'{steering_angle} {speed}')
    # sendign the controls throttle = 1.0 - speed/speed_limit abd steering angle if i steer left the steering angle will be negative
    throttle = min(1.0 - abs(steering_angle), 1-speed/speed_limit ) # speed/speed_limit
    send_control(steering_angle, throttle)

#function to send the control commands to the simulator
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    #app.run(port=3000)
    model = load_model('pass bridge1.h5')
    #wrap the flask application with the socketio server
    app = socketio.Middleware(sio, app)
    #deploy the application on the eventlet web server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
