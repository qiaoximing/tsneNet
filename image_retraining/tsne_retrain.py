import os.path
import numpy as np
import tensorflow as tf
import retrain

IMAGE_SIZE = 299
NUM_CHANNELS = 3
NUM_LABELS = 3
BATCH_SIZE = 100

def cache_bottlenecks(sess, cache_file, image_dir, jpeg_data_tensor,
                      bottleneck_tensor):
    image_lists = retrain.create_image_lists(
        image_dir, testing_percentage=10, validation_percentage=10)
    cache_lists = {'training':[], 'testing':[], 'validation':[]}
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                image_path = retrain.get_image_path(
                    image_lists, label_name, index, image_dir, category)
                image_data = retrain.gfile.FastGFile(image_path, 'rb').read()
                bottleneck_values = retrain.run_bottleneck_on_image(
                    sess, image_data, jpeg_data_tensor, bottleneck_tensor)
                cache_lists[category].append([image_data, bottleneck_values])
    xxx.save(cache_file, cache_lists)


def student_model():
    jpeg_data = tf.placeholder(tf.string, [None], name='jpeg_input')
    labels = tf.placeholder(tf.float32, [None, NUM_LABELS], name='labels_input')



def main():
    cache_file = './cache.file'
    if not os.path.exists(cache_file):
        retrain.maybe_download_and_extract()
        graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tesnor = (
            retrain.create_inception_graph())
        cache_bottlenecks(sess, cache_file, './images', jpeg_data_tensor,
                          bottleneck_tensor)

    sess = tf.Session()

    # calculate bottleneck values of images and cache them on disk

    # add student network
    jpeg_data, labels = student_model()

    # start training
    sess.run(tf.initialize_all_variables())
    for i in range(1000):

if __name__ == '__main__':
    main()
