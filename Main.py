from os import listdir
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
from mrcnn.utils import Dataset, extract_bboxes, compute_ap
from mrcnn.visualize import display_instances
from xml.etree import ElementTree
from numpy import zeros, mean, expand_dims
from numpy import asarray
from matplotlib import pyplot
from matplotlib.patches import Rectangle

class SproutsDataset(Dataset):
    """
    class that defines and loads the kangaroo dataset
    """

    def load_dataset(self, dataset_dir, is_train=True, split=True):
        """ Load the dataset definitions """
        self.add_class("dataset", 1, "sprout")
        # define data locations
        if split:
            images_dir = dataset_dir + '/images/'
        else:
            images_dir = dataset_dir + '/new_images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in listdir(images_dir):
            image_id = filename[:-4]
            if split:
                # skip all images after 85 if we are building the train set
                if is_train and int(image_id) >= 85:
                    continue
                # skip all images before 85 if we are building the test set
                if not is_train and int(image_id) < 85:
                    continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):
        """ Function to extract bounding boxes from an annotation file """
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def load_mask(self, image_id):
        """ Loads the mask for an image """
        # get the details of image
        info = self.image_info[image_id]
        # get the box file location
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        # create an array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create the masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('sprout'))
        return masks, asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        """ load an image reference """
        info = self.image_info[image_id]
        return info['path']

class SproutsConfig(Config):
    """
    Defines a configuration for the model
    """
    NAME = "sprouts_cfg"
    # number of classes ( background + sprout)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch (=number of images in the training dataset)
    STEPS_PER_EPOCH = 107

class PredictionConfig(Config):
    """ Define the prediction configuration """
    # name of the configuration
    NAME = "sprouts_cfg"
    # number of classes (background + sprout)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def evaluate_model(dataset, model, cfg):
    """ Calculate the mAP for a model on a given dataset """
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g.: center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results from first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # sote
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP

def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    """ Plot a number of photos with ground truth and predictions """
    # load image and mask
    for i in range(n_images):
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g.: center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        pyplot.subplot(n_images, 2, i*2+1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        pyplot.subplot(n_images, 2, i*2+2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2-x1, y2-y1
            # create the shape
            rect = Rectangle((x1,y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
        # show the figure
        pyplot.show()

def plot_prediction(dataset, model, cfg, n_images=5):
    """ Plot a single image at a time with predictions"""
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Prediction')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
        pyplot.show()

# train set
train_set = SproutsDataset()
train_set.load_dataset('sprout', is_train=True, split=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# test/validation set
test_set = SproutsDataset()
test_set.load_dataset('sprout', is_train=False, split=True)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# new test set
new_set = SproutsDataset()
new_set.load_dataset('sprout', is_train=False, split=False)
new_set.prepare()
print('New set: %d' % len(new_set.image_ids))

'''
# the following is only to test the class works fine
# define image id
image_id = 2
# load the image
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)
'''

''' 
#the following is to train the model 
# prepare the config
config = SproutsConfig()
config.display()
# display the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
'''

# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('mask_rcnn_sprouts_cfg_0005.h5', by_name=True)
'''
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
#evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
'''
# plot predictions for train dataset
#plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
#plot_actual_vs_predicted(test_set, model, cfg)
plot_prediction(new_set, model, cfg)