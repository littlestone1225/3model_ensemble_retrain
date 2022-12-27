import numpy as np

# https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
# bbox_list = [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...]
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    bbox_array = np.array(boxes)
    error_type = bbox_array[:,1].astype(str)
    x1 = bbox_array[:,2].astype(int)
    y1 = bbox_array[:,3].astype(int)
    x2 = bbox_array[:,4].astype(int)
    y2 = bbox_array[:,5].astype(int)
    score = bbox_array[:,6].astype(float)
    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sort the bounding boxes by the score (high -> low)
    idxs = np.argsort(score)[::-1]

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the first index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        i = idxs[0]
        pick.append(i)
        suppress = [0]
        # loop over all indexes in the indexes list
        for pos in range(1, len(idxs)):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            # overlap = float(w * h) / (area[i] + area[j] - float(w * h))
            overlap_area = float(w * h)
            overlap = max(overlap_area/area[i], overlap_area/area[j])
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                # x1[i] = min(x1[i], x1[j])
                # y1[i] = min(y1[i], y1[j])
                # x2[i] = max(x2[i], x2[j])
                # y2[i] = max(y2[i], y2[j])
                # bbox_array[i][2:6] = x1[i], y1[i], x2[i], y2[i]
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return [ bbox[0:2] + [int(i) for i in bbox[2:6]] + [float(bbox[6])] for bbox in bbox_array[pick].tolist() ]

def non_max_suppression_no_score(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    bbox_array = np.array(boxes)
    error_type = bbox_array[:,1].astype(str)
    x1 = bbox_array[:,2].astype(int)
    y1 = bbox_array[:,3].astype(int)
    x2 = bbox_array[:,4].astype(int)
    y2 = bbox_array[:,5].astype(int)

    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sort the bounding boxes by the score (high -> low)
    idxs = np.arange(len(boxes))

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the first index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        i = idxs[0]
        pick.append(i)
        suppress = [0]
        if error_type[i] == 'solder_ball':
            idxs = np.delete(idxs, suppress)
            continue
        # loop over all indexes in the indexes list
        for pos in range(1, len(idxs)):
            # grab the current index
            j = idxs[pos]
            if error_type[j] == 'solder_ball':
                continue
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            # overlap = float(w * h) / (area[i] + area[j] - float(w * h))
            overlap_area = float(w * h)
            #overlap = max(overlap_area/area[i], overlap_area/area[j])
            union_area = area[i] + area[j] - overlap_area
            overlap = overlap_area/union_area
            # if there is sufficient overlap, suppress the
            # current bounding box
            # if overlap > overlapThresh and error_type[i] == error_type[j]:
            if overlap > overlapThresh:
                # x1[i] = min(x1[i], x1[j])
                # y1[i] = min(y1[i], y1[j])
                # x2[i] = max(x2[i], x2[j])
                # y2[i] = max(y2[i], y2[j])
                # bbox_array[i][2:6] = x1[i], y1[i], x2[i], y2[i]
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return [ bbox[0:2] + [int(i) for i in bbox[2:6]] for bbox in bbox_array[pick].tolist() ]

# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlap_thresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    bbox_array = np.array(boxes)
    error_type = bbox_array[:,1].astype(str)
    x1 = bbox_array[:,2].astype(int)
    y1 = bbox_array[:,3].astype(int)
    x2 = bbox_array[:,4].astype(int)
    y2 = bbox_array[:,5].astype(int)
    score = bbox_array[:,6].astype(float)
    # compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sort the bounding boxes by the score (high -> low)
    idxs = np.argsort(score)[::-1]
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
    return [boxes[i] for i in pick]