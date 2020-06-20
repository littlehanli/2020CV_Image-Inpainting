#python test_model.py --image examples/places356/Places365_test_00000001.jpg --mask examples/places356_mask/mask_00000001.png --output examples/places356/output_00000001.png --checkpoint_dir model_logs/release_places2_256


import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

'''
parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
'''
def Output(img_in,mask_in,img_out):
    print("import from test:",img_in,mask_in,img_out)

#if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    
    #args, unknown = parser.parse_known_args()

    model = InpaintCAModel()
    #image = cv2.imread(args.image)
    image = cv2.imread("examples/places356/"+img_in)
    #mask = cv2.imread(args.mask)
    mask = cv2.imread("places356_mask/"+mask_in)
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            
            #var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            var_value = tf.contrib.framework.load_variable("model_logs/release_places2_256", from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        #cv2.imwrite(args.output, result[0][:, :, ::-1])
        #cv2.imshow("result", result[0][:, :, ::-1])
        cv2.imwrite("examples/places356/"+img_out, result[0][:, :, ::-1])
        show1 = cv2.imread("examples/places356/"+img_in)
        show2 = cv2.imread("examples/places356/"+img_out)
        show = np.hstack([show1,show2])
        cv2.imshow("result",show)
