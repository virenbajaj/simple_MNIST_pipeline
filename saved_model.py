import sys
# sys.path.append('../')

import tensorflow as tf
import tensorflow.keras as tfk 
import keras
import os
import json




def main(_):

    with tf.Session(graph=tf.Graph()) as sess:

        
        export_path = "servable"
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        img_height = 28 
        img_width = 28
        input_shape = (1,img_height,img_width,1)
        input_tensor = tf.keras.Input(shape=input_shape)
        image_height_tensor = tf.keras.Input(shape=(img_height,))
        image_width_tensor = tf.keras.Input(shape=(None,img_width))
        model_version = 1

    #     print(model)
    #     predictions = model.predict(input_tensor)
    #     print(predictions)
        model = tf.keras.models.load_model(f'adadelta/batch_size_32/epochs_1/cp.ckpt')
        predict_tensor = model.output
        print(predict_tensor)
        saver = tf.train.Saver()
        trained_checkpoint_prefix = "adadelta/batch_size_32/epochs_1/cp.ckpt"
        saver.restore(sess, trained_checkpoint_prefix)

        tensor_info_input = tf.saved_model.utils.build_tensor_info(input_tensor)
        tensor_info_height = tf.saved_model.utils.build_tensor_info(image_height_tensor)
        tensor_info_width = tf.saved_model.utils.build_tensor_info(image_width_tensor)
        output_tensor =tf.keras.Input(shape=(None,10))
        tensor_info_output = tf.saved_model.utils.build_tensor_info(output_tensor)
        prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_input, 'height': tensor_info_height, 'width': tensor_info_width},
                    outputs={'classification': tensor_info_output},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature,
            })

        # export the model
        builder.save(as_text=True)
        print('Done exporting!')

if __name__ == '__main__':
    tf.app.run()
