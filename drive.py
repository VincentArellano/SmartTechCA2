# import socketio
# import eventlet
# from flask import Flask
# from keras.models import load_model
# import base64
# from io import BytesIO
# from PIL import Image
# import numpy as np
# import cv2

# sio = socketio.Server()

# app = Flask(__name__)

# @sio.on('connect')
# def connect(sid, environ):
#     print('Connected')
#     send_control(0, 0)

# speed_limit = 10

# def img_preprocess(img):
#     img = img[60:140, :, :]
#     img = cv2.resize(img, (200, 66))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#     img = img / 255.0 - 0.5
#     return img

# @sio.on('telemetry')
# def telemetry(sid, data):
#     image = Image.open(BytesIO(base64.b64decode(data['image'])))
#     image = np.asarray(image)
#     image = img_preprocess(image)
#     image = np.array([image])
#     steering_angle = float(model.predict(image))
#     speed = float(data['speed'])
#     throttle = 1.0 - abs(steering_angle)
#     send_control(steering_angle, throttle)

# def send_control(steering_angle, throttle):
#     sio.emit('steer', data={
#         'steering_angle': steering_angle.__str__(),
#         'throttle': throttle.__str__()
#     })

# if __name__ == '__main__':
#     model = load_model('alpha_model.h5')
#     model.summary()
#     app = socketio.Middleware(sio, app)
#     eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

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
speed_limit = 10

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

#function to augment the image for the model
def zoom(image):
    zoom = iaa.Affine(scale=(1,1.3)) # zoom from 100% to 130% what affine means is that it will zoom the image and then it will shift the image
    image = zoom.augment_image(image)
    return image

image =image_paths[random.randint(0,100)] #randomly select an image from the dataset
original_image = mpig.imread(image) #read the image
zoomed_image = zoom(original_image) #zoom the image
fig, axs = plt.subplots(1,2, figsize=(15,10))#plot the image
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image')

#function to pan the image
def pan(image):
    pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
    image = pan.augment_image(image)
    return image

#function to pan the image
image = image_paths[random.randint(0,100)] #randomly select an image from the dataset
panned_image = mpig.imread(image) #read the image
zoomed_image = zoom(original_image) #zoom the image
fig, axs = plt.subplots(1,2, figsize=(15,10))#plot the image
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(panned_image)
axs[1].set_title('Zoomed Image')

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2,1.2))
    image = brightness.augment_image(image)
    return image

#function to flip the image
image = image_paths[random.randint(0,100)] #randomly select an image from the dataset
original_image = mpig.imread(image) #read the image
brightened_image = img_random_brightness(original_image) #zoom the image
fig, axs = plt.subplots(1,2, figsize=(15,10))#plot the image
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(brightened_image)
axs[1].set_title('brightened Image')

#function to flip the image on y axis
def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)#1 flips the image on y azis
    steering_angle = -steering_angle
    return image, steering_angle

#function to flip the image
random_index = random.randint(0,1000)
image = image_paths[random_index] #randomly select an image from the dataset
steering_angle = steerings[random_index]
original_image = mpig.imread(image) #read the image
flipped_image,flipped_steering_angle = img_random_flip(original_image,steering_angle) #zoom the image
fig, axs = plt.subplots(1,2, figsize=(15,10))#plot the image
axs[0].imshow(original_image)
axs[0].set_title('Original Image -' +'Steering angle'+ str(steering_angle))
axs[1].imshow(flipped_image)
axs[1].set_title('Original Image -' +'Steering angle'+ str(flipped_steering_angle))


def random_augment(image,sterring_angle):
    image = mpig.imread(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image = img_random_flip(image)
    return image, steering_angle
    
ncols = 2
nrows =10
fig, axs =plt.subplots(nrows,ncols, figsize=(15,50))
fig.tight_layout()
for i in range(10):
    rand_num = random.randint(0,len(image_paths)-1)
    random_image = image_paths[rand_num]
    random_steering = steerings[rand_num]
    originaimage = mping.imread(random_image)
    augumented_image, steering_angle = random_augment(random_image,random_steering)
    axs[i][0],imshow(originaimage)
    axs[i][0].set_title("orignal")
    axs[i][0],imshow(originaimage)
    axs[i][0].set_title("orignal")

#function to preprocess the image with batch generator for the model
def batch_generator(image_paths, steering_angles, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0,len(image_paths)-1)
            if istraining: #if training is true then augment the image
                im, steering = random_augment(image_paths[random_index], steering_angles[random_index])
            else:
                im = mpig.imread(image_paths[random_index])
                steering = steering_angles[random_index]
            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))

#we have to see the changes between xtrain and xtrain_gen but not x_valid and x_valid_gen
x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1)) #1 is the batch size and 1 is the training the 1s means that we are augmenting the image
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0)) #0 is the batch size and 0 is the training the 0s means that we are not augmenting the image

fig, axs = plt.subplots(1,2, figsize=(15,10))
fig.tight_layout()
axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training Image')
axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation Image')

def img_preprocess_no_imread(img):
  #convert the image to yuv color space
    img = mpig.imread(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    #crop the image
    img = img[60:135,:,:]
    #resize the image
    img = cv2.resize(img,(200,66))
    #normalize the image
    img = img/255
    return img

#function to send the control commands to the simulator
def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    #app.run(port=3000)
    model = load_model('alpha.h5')
    #wrap the flask application with the socketio server
    app = socketio.Middleware(sio, app)
    #deploy the application on the eventlet web server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
