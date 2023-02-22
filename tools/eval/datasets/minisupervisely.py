import os
import cv2 as cv
import numpy as np
from tqdm import tqdm


class MiniSupervisely : 

    '''
        Refer to https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/paddleseg/core/val.py
        for official evaluation implementation.
    '''

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

        intersect_area_all = np.zeros([1], dtype=np.int64)
        pred_area_all = np.zeros([1], dtype=np.int64)
        label_area_all = np.zeros([1], dtype=np.int64)

        pbar = tqdm(self.image_set)

        pbar.set_description(
            "Evaluating {} with {} val set".format(model.name, self.name))

        for input_image, expected_image in pbar : 
            
            input_image = cv.imread(os.path.join(self.root, input_image)).astype('float32')
            
            expected_image = cv.imread(os.path.join(self.root, expected_image), cv.IMREAD_GRAYSCALE)[np.newaxis, :, :]           

            output_image = model.infer(input_image) 
            
            intersect_area, pred_area, label_area = self.calculate_area(
                output_image.astype('uint32'),
                expected_image.astype('uint32'),
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
        print("Class IoU : ", self.class_iou)
        print("Class Accuracy : ", self.class_acc)


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
        
       
        if len(pred.shape) == 4:
            pred = np.squeeze(pred, axis=1)
        if len(label.shape) == 4:
            label = np.squeeze(label, axis=1)
        if not pred.shape == label.shape:
            raise ValueError('Shape of `pred` and `label should be equal, '
                            'but there are {} and {}.'.format(pred.shape,
                                                            label.shape))

        mask = label != ignore_index
        pred_area = []
        label_area = []
        intersect_area = []

        #iterate over all classes and calculate their respective areas
        for i in range(num_classes):
            pred_i = np.logical_and(pred == i, mask)
            label_i = label == i
            intersect_i = np.logical_and(pred_i, label_i)
            pred_area.append(np.sum(pred_i.astype('int32')))
            label_area.append(np.sum(label_i.astype('int32')))
            intersect_area.append(np.sum(intersect_i.astype('int32')))

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
