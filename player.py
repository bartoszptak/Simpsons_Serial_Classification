import cv2
import tensorflow as tf
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

image_size=128
num_channels=3

file_path = "data/simpsons.mp4"
window_name = "Simpshons Classification"

sess = tf.Session()
saver = tf.train.import_meta_graph('./model/simpsons_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model/'))
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")
x= graph.get_tensor_by_name("x:0")

y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 42))

map_characters = {0: 'Abraham Grampa Simpson',
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


def predict(frame):
    images = []
    image = cv2.resize(frame, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)

    x_batch = images.reshape(1, image_size, image_size, num_channels)

    feed_dict_testing = {x: x_batch, y_true: y_test_images}

    result = sess.run(y_pred, feed_dict=feed_dict_testing)

    return result

def draw(frame, model):
    max = round(np.amax(model) * 100, 2)
    ind = np.argmax(model)
    text = map_characters[ind] + " (" + str(max) + "%)"
    cv2.rectangle(frame, (0, 0), (len(text) * 16, 40), (255, 255, 255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (20, 30), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

cap = cv2.VideoCapture(file_path)
play = True

while cap.isOpened():

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        play = not play

    if play:
        ret, frame = cap.read()
        a = predict(frame)
        frame = draw(frame, a)
        cv2.imshow(window_name, frame)

cap.release()
cv2.destroyAllWindows()