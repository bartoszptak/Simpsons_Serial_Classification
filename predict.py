import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

charakters = {'0': 'Abraham Grampa Simpson',
       '1': 'Agnes Skinner',
       '2': 'Apu Nahasapeemapetilon',
       '3': 'Barney Gumble',
       '4': 'Bart Simpson',
       '5': 'Carl Carlson',
       '6': 'Charles Montgomery Burns',
       '7': 'Chief Wiggum',
       '8': 'Cletus Spuckler',
       '9': 'Comic Book Guy',
       '10': 'Disco Stu',
       '11': 'Edna Krabappel',
       '12': 'Fat Tony',
       '13': 'Gil',
       '14': 'Groundskeeper Willie',
       '15': 'Homer Simpson',
       '16': 'Kent Brockman',
       '17': 'Krusty the Clown',
       '18': 'Lenny Leonard',
       '19': 'Lionel Hutz',
       '20': 'Lisa Simpson',
       '21': 'Maggie Simpson',
       '22': 'Marge Simpson',
       '23': 'Martin Prince',
       '24': 'Mayor Guimby',
       '25': 'Milhouse van Houten',
       '26': 'Miss Hoover',
       '27': 'Moe Szyslak',
       '28': 'Ned Flanders',
       '29': 'Nelson Muntz',
       '30': 'Otto Mann',
       '31': 'Patty Bouvier',
       '32': 'Principal Skinner',
       '33': 'Professor John Frink',
       '34': 'Rainier Wolfcastle',
       '35': 'Ralph Wiggum',
       '36': 'Selma Bouvier',
       '37': 'Sideshow Bob',
       '38': 'Sideshow Mel',
       '39': 'Snake Jailbird',
       '40': 'Troy Mcclure',
       '41': 'Waylon Smithers'
       }

# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path
image_size=128
num_channels=3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./model/simpsons_model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./model/'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 42)) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
max = round(np.amax(result) * 100, 2)
ind = np.argmax(result)
font = cv2.FONT_HERSHEY_SIMPLEX
text = "\n###\n" + charakters[str(ind)] + " (" + str(max) + "%)\n###\n"
print(text)