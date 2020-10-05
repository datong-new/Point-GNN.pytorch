import torch
import numpy as np

def recall_precisions(labels, predictions, num_classes):
    recalls, precisions = {}, {}

    for class_idx in range(num_classes):
        gt = (labels==class_idx)
        pred = (predictions==class_idx)

        TP = float(torch.logical_and(gt.squeeze(), pred.squeeze()).sum())

        recalls[class_idx] = TP / gt.sum().item() if gt.sum().item()!=0 else 0
        precisions[class_idx] = TP / pred.sum().item() if pred.sum().item()!=0 else 0

    return recalls, precisions

def mAP(lables, logits, num_classes):
    mAPs = {}

    for class_idx in range(num_classes):
        pred = logits[:, class_idx]
        threshs = sorted(pred.tolist())
        threshs = threshs[::len(threshs)//30]
        gt = (lables==class_idx)

        precisions = []
        for thresh in threshs:
            _pred = (pred>thresh).bool()
            TP = float(torch.logical_and(gt.squeeze(), _pred.squeeze()).sum())
            if _pred.sum().float().item()==0:
                precisions += [0]
            else: precisions += [TP / _pred.sum().float().item()]
        mAPs[class_idx] = np.mean(precisions)
    return mAPs

if __name__ == "__main__":
    labels = torch.randint(0, 4, (100,))
    predictions = torch.randint(0,4, (100,))
    recalls, precisions = recall_precisions(labels, predictions, 4)

    logits = torch.rand(100, 4)
    mAPs = mAP(labels, logits, 4)
    print("recall: ", recalls)
    print("precision: ", precisions)
    print("mAPs: ", mAPs)






