import os
import cv2 as cv
import numpy as np
from tqdm import tqdm


class Normalize:
    """
    Normalize an image.
    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].
    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def normalize(self, im, mean, std):
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        return im
    
    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.
        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = self.normalize(im, mean, std)

        if label is None:
            return (im, )
        else:
            return (im, label)


class MiniSupervisely : 
    def __init__(self, root) : 
        self.root = root
        self.val_path = os.path.join(root, 'val.txt')
        self.image_set = self.load_data(self.val_path)
        self.num_classes = 2
        self.miou = -1
        self.class_miou = -1
        self.acc = -1
        self.class_acc = -1


    @property
    def name(self):
        return self.__class__.__name__
    

    def load_data(self, val_path) : 
        """
        Load validation image set from val.txt file
        Args :
            val_path (str) : path to val.txt file
        Returns :
            image_set (list) : list of image path of input and expected image
        """

        image_set = []
        with open(val_path, 'r') as f : 
            for line in f.readlines() : 
                image_set.append(line.strip().split())

        return image_set
    
    
    def eval(self, model) : 
        """
        Evaluate model on validation set
        Args :
            model (object) : PP_HumanSeg model object
        """

        intersect_area_all = []
        pred_area_all = []
        label_area_all = []

        pbar = tqdm(self.image_set)

        pbar.set_description(
            "Evaluating {} with {} val set".format(model.name, self.name))
        
        normalize = Normalize()

        for input_image, expected_image in pbar : 
            
            input_image = cv.imread(os.path.join(self.root, input_image))
            input_image = cv.resize(input_image, (192, 192))
            input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)

            expected_image = cv.imread(os.path.join(self.root, expected_image),cv.IMREAD_GRAYSCALE)
            expected_image = cv.resize(expected_image, (192, 192))[np.newaxis, :, :]
            
            input_image, expected_image = normalize(input_image, expected_image)

            output_image = model.infer(input_image)   

            intersect_area, pred_area, label_area = self.calculate_area(
                output_image,
                expected_image,
                self.num_classes)
            intersect_area_all = intersect_area_all + intersect_area
            pred_area_all = pred_area_all + pred_area
            label_area_all = label_area_all + label_area


            
        self.class_iou, self.miou = self.mean_iou(intersect_area_all, pred_area_all,
                                           label_area_all)
        self.class_acc, self.acc = self.accuracy(intersect_area_all, pred_area_all)
    

    def get_results(self) :
        """
        Get evaluation results
        Returns :
            miou (float) : mean iou
            class_miou (list) : iou on all classes
            acc (float) : mean accuracy
            class_acc (list) : accuracy on all classes
        """
        return self.miou, self.class_miou, self.acc, self.class_acc
    

    def print_result(self) : 
        """
        Print evaluation results
        """
        print("Mean IoU : ", self.miou)
        print("Mean Accuracy : ", self.acc)

    
    def one_hot(self, arr, max_size) :
        return np.eye(max_size)[arr]


    def calculate_area(self,pred, label, num_classes, ignore_index=255):
        """
        Calculate intersect, prediction and label area
        Args:
            pred (Tensor): The prediction by model.
            label (Tensor): The ground truth of image.
            num_classes (int): The unique number of target classes.
            ignore_index (int): Specifies a target value that is ignored. Default: 255.
        Returns:
            Tensor: The intersection area of prediction and the ground on all class.
            Tensor: The prediction area on all class.
            Tensor: The ground truth area on all class
        """
        
        # Delete ignore_index
        mask = label != ignore_index
        pred = pred + 1
        label = label + 1
        pred = pred * mask
        label = label * mask


        pred = self.one_hot(pred, num_classes + 1)
        label = self.one_hot(label, num_classes + 1)

        pred = pred[:, :, :, 1:]
        label = label[:, :, :, 1:]

        pred_area = []
        label_area = []
        intersect_area = []

        #iterate over all classes and calculate their respective areas
        for i in range(num_classes):
            pred_i = pred[:, :, :, i]
            label_i = label[:, :, :, i]
            pred_area_i = np.sum(pred_i)
            label_area_i = np.sum(label_i)
            intersect_area_i = np.sum(pred_i * label_i)
            pred_area.append(pred_area_i)
            label_area.append(label_area_i)
            intersect_area.append(intersect_area_i)
        
        return intersect_area, pred_area, label_area
    

    def mean_iou(self,intersect_area, pred_area, label_area):
        """
        Calculate iou.
        Args:
            intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
            pred_area (Tensor): The prediction area on all classes.
            label_area (Tensor): The ground truth area on all classes.
        Returns:
            np.ndarray: iou on all classes.
            float: mean iou of all classes.
        """
        intersect_area = np.array(intersect_area)
        pred_area = np.array(pred_area)
        label_area = np.array(label_area)

        union = pred_area + label_area - intersect_area

        class_iou = []
        for i in range(len(intersect_area)):
            if union[i] == 0:
                iou = 0
            else:
                iou = intersect_area[i] / union[i]
            class_iou.append(iou)

        miou = np.mean(class_iou)

        return np.array(class_iou), miou
    

    def accuracy(self,intersect_area, pred_area):
        """
        Calculate accuracy
        Args:
            intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
            pred_area (Tensor): The prediction area on all classes.
        Returns:
            np.ndarray: accuracy on all classes.
            float: mean accuracy.
        """

        intersect_area = np.array(intersect_area)
        pred_area = np.array(pred_area)

        class_acc = []
        for i in range(len(intersect_area)):
            if pred_area[i] == 0:
                acc = 0
            else:
                acc = intersect_area[i] / pred_area[i]
            class_acc.append(acc)

        macc = np.sum(intersect_area) / np.sum(pred_area)

        return np.array(class_acc), macc
