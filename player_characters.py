import cv2
import tensorflow as tf
import numpy as np

image_size = 128
num_channels = 3

file_path = "data/simpsons.mp4"
window_name = "Simpshons Classification"

sess = tf.Session()
saver = tf.train.import_meta_graph('./model/simpsons_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model/'))
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")

y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 42))

map_characters = {0: 'Abraham Grampa',
                  1: 'Agnes Skinner',
                  2: 'Apu Nahasapeemapetilon',
                  3: 'Barney Gumble',
                  4: 'Bart Simpson',
                  5: 'Carl Carlson',
                  6: 'Charles Montgomery Burns',
                  7: 'Chief Wiggum',
                  8: 'Cletus Spuckler',
                  9: 'Comic Book Guy',
                  10: 'Disco Stu',
                  11: 'Edna Krabappel',
                  12: 'Fat Tony',
                  13: 'Gil',
                  14: 'Groundskeeper Willie',
                  15: 'Homer Simpson',
                  16: 'Kent Brockman',
                  17: 'Krusty the Clown',
                  18: 'Lenny Leonard',
                  19: 'Lionel Hutz',
                  20: 'Lisa Simpson',
                  21: 'Maggie Simpson',
                  22: 'Marge Simpson',
                  23: 'Martin Prince',
                  24: 'Mayor Guimby',
                  25: 'Milhouse van Houten',
                  26: 'Miss Hoover',
                  27: 'Moe Szyslak',
                  28: 'Ned Flanders',
                  29: 'Nelson Muntz',
                  30: 'Otto Mann',
                  31: 'Patty Bouvier',
                  32: 'Principal Skinner',
                  33: 'Professor John Frink',
                  34: 'Rainier Wolfcastle',
                  35: 'Ralph Wiggum',
                  36: 'Selma Bouvier',
                  37: 'Sideshow Bob',
                  38: 'Sideshow Mel',
                  39: 'Snake Jailbird',
                  40: 'Troy Mcclure',
                  41: 'Waylon Smithers'
                  }

def get_square(image,square_size):

    height,width,cz=image.shape
    if(height>width):
      differ=height
    else:
      differ=width
    differ+=4

    mask = np.zeros((differ,differ,3), dtype="uint8")
    x_pos=int((differ-width)/2)
    y_pos=int((differ-height)/2)
    mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
    mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)

    return mask

def predict(frame):
    images = []
    image = get_square(frame,image_size)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)

    x_batch = images.reshape(1, image_size, image_size, num_channels)

    feed_dict_testing = {x: x_batch, y_true: y_test_images}

    result = sess.run(y_pred, feed_dict=feed_dict_testing)

    return result


def draw(frame, model,a,c,aa,cc):
    max = round(np.amax(model) * 100, 2)
    ind = np.argmax(model)
    text = map_characters[ind] + " (" + str(max) + "%)"
    cv2.rectangle(frame, (a, c), (aa, cc), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    if c>frame.shape[0]/2:
        cv2.putText(frame, text, (a-10, c-20), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, text, (a - 10, cc + 20), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


cap = cv2.VideoCapture(file_path)
play = True

face_cascade = cv2.CascadeClassifier('model/haarcascade_profileface.xml')

while cap.isOpened():

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        play = not play

    if play:
        ret, frame = cap.read()
        frame = cv2.resize(frame,None,fx=0.5,fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        for (a, c, w, h) in faces:
            model = predict(frame[a-20:a+w+20,c-20:c+h+20])
            frame = draw(frame,model,a,c,a+w,c+h)
        cv2.imshow(window_name, frame)

cap.release()
cv2.destroyAllWindows()
