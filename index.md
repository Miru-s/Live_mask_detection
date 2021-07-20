# Live Mask detection

      from tensorflow.keras.models import Sequential
      from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
      cnn = Sequential([Conv2D(filters=100, kernel_size=(3,3),
                    activation='relu', padding = 'same'),

                    MaxPooling2D(pool_size=(2,2)),
                    Conv2D(filters=100, kernel_size=(3,3),
                    activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Flatten(),
                    Dropout(0.5),
                    Dense(50),
                    Dense(35),
                    Dense(2)])
                    
           cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
           import cv2
           import numpy as np
           labels_dict={0:'No mask',1:'Mask'}
            color_dict={0:(0,0,255),1:(0,255,0)}
            imgsize = 4 #set image resize
            camera = cv2.VideoCapture(0) 
            classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            while True:
                (rval, im) = camera.read()
                im=cv2.flip(im,1,1) #mirrow the image
                imgs = cv2.resize(im, (im.shape[1] // imgsize, im.shape[0] //
                 imgsize))
    face_rec = classifier.detectMultiScale(imgs)
    for i in face_rec: # Overlay rectangle on face
        (x, y, l, w) = [v * imgsize for v in i]
        face_img = im[y:y+w, x:x+l]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=cnn.predict(reshaped)
        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(im,(x,y),(x+l,y+w),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+l,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-
        10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow('LIVE',im)
    key = cv2.waitKey(10)
    
    if key == 27:
        break
    webcam.release()
    cv2.destroyAllWindows()


 
Keras is a deep learning API written in Python. It is a powerful and easy-to-use free open source Python library for developing and evaluating deep learning models. It wraps the efficient numerical computation libraries Theano and TensorFlow and allows you to define and train neural network models in just a few lines of code.
 
Sequential model is a stack of layers where each layer is exactly one input tensor or an output tensor.
 
Flatten- To flatten everything into one dimensional picture.
 
Dense creates the output layer.
 
CNN Compile is used to train the data.
 
Adam optimizer- Adaptive Moment Estimation is an algorithm for optimization technique for gradient descent.
 
Binary cross entropy compares each predicted probability to actual class output which can be either 1 or 0, then calculates the score that penalises the probabilities based on the distance from the expected value. In our case, it is about wearing or not wearing a mask.
 
OpenCV is a cross-platform library using which we can develop real-time computer vision applications. It mainly focuses on image processing, video capture and analysis including features like face detection and object detection.
 
 
 
 
 
Creating a CNN (Convolutional Neural Network)
 
Conv2D or 2D Convolution Layer accepts image data.
Relu or Rectified Linear Activation function is a default activation when developing multi-layer CNN.
Padding- We have kept padding=same in our program which means the dimensions of the image won’t change after convolution.
MaxPooling2D is the first pooling layer to obtain sharp & smooth features.
 
