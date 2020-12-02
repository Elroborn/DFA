import math
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import roc_auc_score

class PADMeter(object):
    """Presentation Attack Detection Meter"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.label = np.ones(0)
        self.output = np.ones(0)
        self.threshold = None 
        self.grid_density = 10000

    def update(self, label, output):
        if len(output.shape) > 1 and output.shape[1] > 1:
            output = output[:,1]
        elif len(output.shape) > 1 and output.shape[1] == 1:
            output = output[:,0]

        self.label = np.hstack([self.label, label])
        self.output = np.hstack([self.output, output])

    def get_tpr(self, fixed_fpr):
        fpr, tpr, thr = roc_curve(self.label, self.output)
        tpr_filtered = tpr[fpr <= fixed_fpr]
        if len(tpr_filtered) == 0:
            self.tpr = 0.0
        self.tpr = tpr_filtered[-1]

    def eval_stat(self,thr):

        pred = self.output >= thr
        TN = np.sum((self.label == 0) & (pred == False))
        FN = np.sum((self.label == 1) & (pred == False))
        FP = np.sum((self.label == 0) & (pred == True))
        TP = np.sum((self.label == 1) & (pred == True))
        if TN + FP==0:
            TN+=0.0001
        if TP + FN==0:
            TP+=0.0001
        return TN, FN, FP, TP




    def get_eer_and_thr(self):

        thresholds = []
        Min, Max = min(self.output), max(self.output)
        for i in range(self.grid_density + 1):
            thresholds.append(Min + i * (Max - Min) / float(self.grid_density))
        min_dist = 1.0
        min_dist_stats = []
        for thr in thresholds:
            TN, FN, FP, TP = self.eval_stat(thr)
            far = FP / float(TN + FP)
            frr = FN / float(TP + FN)
            dist = math.fabs(far - frr)
            if dist < min_dist:
                min_dist = dist
                min_dist_stats = [far, frr, thr]

        self.eer = (min_dist_stats[0] + min_dist_stats[1]) / 2.0
        self.threshold = min_dist_stats[2]




    def get_hter_apcer_etal_at_thr(self, thr=None): 
        if thr==None:
            self.get_eer_and_thr()
            thr = self.threshold
        TN, FN, FP, TP = self.eval_stat( thr)

        far = FP / float(TN + FP)
        frr = FN / float(TP + FN)
        self.apcer = far
        self.bpcer = frr
        self.acer = (self.apcer + self.bpcer) /2.0
        self.hter = (far + frr) / 2.0
        self.auc = roc_auc_score(self.label, self.output)


    def get_accuracy(self,thr=None):
        if thr==None:
            self.get_eer_and_thr()
            thr = self.threshold
        TN, FN, FP, TP = self.eval_stat(thr)
        self.accuracy = accuracy = float(TP + TN) / len(self.output)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count