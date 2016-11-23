import os.path
import Image
import numpy as np
import tensorflow as tf
import retrain

IMAGE_SIZE = 299
NUM_CHANNELS = 3
NUM_LABELS = 3

def one_hot(i, n):
    return [int(j == i) for j in range(n)] 

def cache_bottlenecks(sess, image_dir, jpeg_data_tensor, bottleneck_tensor):
    image_lists = retrain.create_image_lists(
        image_dir, testing_percentage=10, validation_percentage=10)
    cache_lists = {'training':[], 'testing':[], 'validation':[]}
    for category in ['training', 'testing', 'validation']:
        cache_lists[category] = {'images':[], 'bottlenecks':[], 'labels':[]}
    label_num = len(image_lists.items())
    for label_index, (label_name, label_lists) in enumerate(image_lists.items()):
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                image_path = retrain.get_image_path(
                    image_lists, label_name, index, image_dir, category)
                #image_data = retrain.gfile.FastGFile(image_path, 'rb').read()
                image_data_np = np.array(Image.open(image_path).resize((56,56)))
                if len(image_data_np.shape) != 3:
                    continue
                #bottleneck_values = retrain.run_bottleneck_on_image(
                #    sess, image_data, jpeg_data_tensor, bottleneck_tensor)
                cache_lists[category]['images'].append(image_data_np)
                #cache_lists[category]['bottlenecks'].append(bottleneck_values)
                cache_lists[category]['labels'].append(one_hot(label_index, label_num))
    p = np.random.permutation(len(cache_lists['training']['images']))
    np.save('train.images.npy', np.array(cache_lists['training']['images'])[p])
    #np.save('train.bottlenecks.npy', np.array(cache_lists['training']['bottlenecks'])[p])
    np.save('train.labels.npy', np.array(cache_lists['training']['labels'])[p])
    np.save('validation.images.npy', cache_lists['validation']['images'])
    #np.save('validation.bottlenecks.npy', cache_lists['validation']['bottlenecks'])
    np.save('validation.labels.npy', cache_lists['validation']['labels'])
    np.save('test.images.npy', cache_lists['testing']['images'])
    #np.save('test.bottlenecks.npy', cache_lists['testing']['bottlenecks'])
    np.save('test.labels.npy', cache_lists['testing']['labels'])

def main():
    sess = tf.Session()

    retrain.maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tesnor = (
        retrain.create_inception_graph())
    cache_bottlenecks(sess, './image_dir', jpeg_data_tensor,
                      bottleneck_tensor)

if __name__ == '__main__':
    main()
