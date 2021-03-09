"""
This file is meant to contain functions which are
specific to region proposal networks.
"""
import re
from lib.core import *
from lib.cpu_nms import cpu_nms

def flatten_tensor(input):
    """
    Flattens and permutes a tensor from size
    [B x C x W x H] --> [B x (W x H) x C]
    """

    bsize = input.shape[0]
    csize = input.shape[1]

    return fluid.layers.reshape(fluid.layers.transpose(x=input, perm=[0, 2, 3, 1]), [bsize, -1, csize])

def calc_output_size(res, stride):
    """
    Approximate the output size of a network

    Args:
        res (ndarray): input resolution
        stride (int): stride of a network

    Returns:
         ndarray: output resolution
    """

    return np.ceil(np.array(res)/stride).astype(int)


def locate_anchors(anchors, feat_size, stride, convert_tensor=False, input_tensor=False):
    """
    Spreads each anchor shape across a feature map of size feat_size spaced by a known stride.

    Args:
        anchors (ndarray): N x 4 array describing [x1, y1, x2, y2] displacements for N anchors
        feat_size (ndarray): the downsampled resolution W x H to spread anchors across [feat_h, feat_w]
        stride (int): stride of a network
        convert_tensor (bool, optional): whether to return a torch tensor, otherwise ndarray [default=False]

    Returns:
         ndarray: 2D array = [(W x H) x 5] array consisting of [x1, y1, x2, y2, anchor_index]
    """

    # compute rois
    shift_x = np.array(range(0, feat_size[1], 1)) * float(stride)
    shift_y = np.array(range(0, feat_size[0], 1)) * float(stride)
    [shift_x, shift_y] = np.meshgrid(shift_x, shift_y)

    rois = np.expand_dims(anchors[:, 0:4], axis=1)
    shift_x = np.expand_dims(shift_x, axis=0)
    shift_y = np.expand_dims(shift_y, axis=0)

    shift_x1 = shift_x + np.expand_dims(rois[:, :, 0], axis=2)
    shift_y1 = shift_y + np.expand_dims(rois[:, :, 1], axis=2)
    shift_x2 = shift_x + np.expand_dims(rois[:, :, 2], axis=2)
    shift_y2 = shift_y + np.expand_dims(rois[:, :, 3], axis=2)

    # compute anchor tracker
    anchor_tracker = np.zeros(shift_x1.shape, dtype=float)
    for aind in range(0, rois.shape[0]): anchor_tracker[aind, :, :] = aind

    stack_size = feat_size[0] * anchors.shape[0]

    # torch and numpy MAY have different calls for reshaping, although
    # it is not very important which is used as long as it is CONSISTENT
    if convert_tensor:

        if input_tensor:
            stack_size = stack_size.numpy()
            feat_size = feat_size.numpy()

        # important to unroll according to pytorch
        shift_x1 = fluid.layers.reshape(fluid.dygraph.to_variable(shift_x1), [1, stack_size, feat_size[1]])
        shift_y1 = fluid.layers.reshape(fluid.dygraph.to_variable(shift_y1), [1, stack_size, feat_size[1]])
        shift_x2 = fluid.layers.reshape(fluid.dygraph.to_variable(shift_x2), [1, stack_size, feat_size[1]])
        shift_y2 = fluid.layers.reshape(fluid.dygraph.to_variable(shift_y2), [1, stack_size, feat_size[1]])
        anchor_tracker = fluid.layers.reshape(fluid.dygraph.to_variable(anchor_tracker), [1, stack_size, feat_size[1]])

        shift_x1.stop_gradient = False
        shift_y1.stop_gradient = False
        shift_x2.stop_gradient = False
        shift_y2.stop_gradient = False
        anchor_tracker.stop_gradient = False

        shift_x1 = fluid.layers.reshape(fluid.layers.transpose(shift_x1, [1, 2, 0]), [-1, 1])
        shift_y1 = fluid.layers.reshape(fluid.layers.transpose(shift_y1, [1, 2, 0]), [-1, 1])
        shift_x2 = fluid.layers.reshape(fluid.layers.transpose(shift_x2, [1, 2, 0]), [-1, 1])
        shift_y2 = fluid.layers.reshape(fluid.layers.transpose(shift_y2, [1, 2, 0]), [-1, 1])
        anchor_tracker = fluid.layers.reshape(fluid.layers.transpose(anchor_tracker, [1, 2, 0]), [-1, 1])

        rois = fluid.layers.concat([shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker], axis=1)

    else:

        shift_x1 = shift_x1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y1 = shift_y1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_x2 = shift_x2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y2 = shift_y2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        anchor_tracker = anchor_tracker.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)

        rois = np.concatenate((shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)

    return rois

def bbox_transform_inv(boxes, deltas, means=None, stds=None):
    """
    Compute the bbox target transforms in 3D.
    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    deltas = fluid.layers.cast(deltas, 'float32')

    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    # boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if stds is not None:
        dx *= stds[0]
        dy *= stds[1]
        dw *= stds[2]
        dh *= stds[3]

    if means is not None:
        dx += means[0]
        dy += means[1]
        dw += means[2]
        dh += means[3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = fluid.layers.exp(dw) * widths
    pred_h = fluid.layers.exp(dh) * heights

    pred_boxes = fluid.layers.concat([
        fluid.layers.reshape(pred_ctr_x - 0.5 * pred_w, [-1, 1]),
        fluid.layers.reshape(pred_ctr_y - 0.5 * pred_h, [-1, 1]),
        fluid.layers.reshape(pred_ctr_x + 0.5 * pred_w, [-1, 1]),
        fluid.layers.reshape(pred_ctr_y + 0.5 * pred_h, [-1, 1])
    ], axis=1)

    return pred_boxes


def determine_ignores(gts, lbls, ilbls, min_gt_vis=0.99, min_gt_h=0, max_gt_h=10e10, scale_factor=1):
    """
    Given various configuration settings, determine which ground truths
    are ignored and which are relevant.
    """

    igns = np.zeros([len(gts)], dtype=bool)
    rmvs = np.zeros([len(gts)], dtype=bool)

    for gtind, gt in enumerate(gts):

        ign = gt.ign
        ign |= gt.visibility < min_gt_vis
        ign |= gt.bbox_full[3] * scale_factor < min_gt_h
        ign |= gt.bbox_full[3] * scale_factor > max_gt_h
        ign |= gt.cls in ilbls

        rmv = not gt.cls in (lbls + ilbls)

        igns[gtind] = ign
        rmvs[gtind] = rmv

    return igns, rmvs


def bbXYWH2Coords(box):
    """
    Convert from [x,y,w,h] to [x1, y1, x2, y2]
    """

    if box.shape[0] == 0: return np.empty([0,4], dtype=float)

    box[:, 2] += box[:, 0] - 1
    box[:, 3] += box[:, 1] - 1

    return box


def clsName2Ind(lbls, cls):
    """
    Converts a cls name to an ind
    """
    if cls in lbls:
        return lbls.index(cls) + 1
    else:
        raise ValueError('unknown class')


# valid, ignore, inds -> index
def compute_targets(gts_val, gts_ign, box_lbls, rois, fg_thresh, ign_thresh, bg_thresh_lo, bg_thresh_hi, best_thresh,
                    gts_3d=None, gts_vertices=None, gts_corners_3d=None, anchors=[], tracker=[]):
    """
    Computes the bbox targets of a set of rois and a set
    of ground truth boxes, provided various ignore
    settings in configuration
    """

    ols = None
    has_3d = gts_3d is not None
    use_corner = gts_vertices is not None

    # init transforms which respectively hold [dx, dy, dw, dh, label]
    # for labels bg=-1, ign=0, fg>=1
    transforms = np.zeros([len(rois), 5], dtype=np.float32)
    raw_gt = np.zeros([len(rois), 5], dtype=np.float32)

    # if 3d, then init other terms after
    if use_corner:
        transforms = np.pad(transforms, [(0, 0), (0, gts_3d.shape[1] + gts_vertices.shape[1] + gts_corners_3d.shape[1])], 'constant')
        raw_gt = np.pad(raw_gt, [(0, 0), (0, gts_3d.shape[1] + gts_vertices.shape[1] + gts_corners_3d.shape[1])], 'constant')
    elif has_3d:
        transforms = np.pad(transforms, [(0, 0), (0, gts_3d.shape[1])], 'constant')
        raw_gt = np.pad(raw_gt, [(0, 0), (0, gts_3d.shape[1])], 'constant')

    if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

        if gts_ign.shape[0] > 0:

            # compute overlaps ign
            ols_ign = iou_ign(rois, gts_ign)
            ols_ign_max = np.amax(ols_ign, axis=1)

        else:
            ols_ign_max = np.zeros([rois.shape[0]], dtype=np.float32)

        if gts_val.shape[0] > 0:

            # compute overlaps valid
            ols = iou(rois, gts_val)
            ols_max = np.amax(ols, axis=1)
            targets = np.argmax(ols, axis=1)

            # find best matches for each ground truth
            gt_best_rois = np.argmax(ols, axis=0)
            gt_best_ols = np.amax(ols, axis=0)

            gt_best_rois = gt_best_rois[gt_best_ols >= best_thresh]
            gt_best_ols = gt_best_ols[gt_best_ols >= best_thresh]

            fg_inds = np.flatnonzero(ols_max >= fg_thresh)
            fg_inds = np.concatenate((fg_inds, gt_best_rois))
            fg_inds = np.unique(fg_inds)

            target_rois = gts_val[targets[fg_inds], :]
            src_rois = rois[fg_inds, :]

            if len(fg_inds) > 0:

                # compute 2d transform
                transforms[fg_inds, 0:4] = bbox_transform(src_rois, target_rois)

                raw_gt[fg_inds, 0:4] = target_rois

                if use_corner:
                    tracker = tracker.astype(np.int64)
                    src_3d = anchors[tracker[fg_inds], 4:]
                    target_3d = gts_3d[targets[fg_inds]]
                    target_vertices = gts_vertices[targets[fg_inds]]
                    target_corners_3d = gts_corners_3d[targets[fg_inds]]
                    dim_3d = target_3d.shape[1]

                    transforms[fg_inds, 5 + dim_3d:5 + dim_3d+2 *8] = bbox_transform_vertices(src_rois, target_vertices)
                    transforms[fg_inds, 5:5 + dim_3d] = bbox_transform_3d(src_rois, src_3d, target_3d)
                    transforms[fg_inds, 5 + dim_3d+2*8:5 + dim_3d+5*8] = bbox_transform_corners(src_3d, target_corners_3d)

                    raw_gt[fg_inds, 5 + dim_3d:5 + dim_3d+2*8] = target_vertices
                    raw_gt[fg_inds, 5 + dim_3d+2*8:5 + dim_3d+5*8] = target_corners_3d
                    raw_gt[fg_inds, 5:5 + dim_3d] = target_3d

                elif has_3d:

                    tracker = tracker.astype(np.int64)
                    src_3d = anchors[tracker[fg_inds], 4:]
                    target_3d = gts_3d[targets[fg_inds]]

                    raw_gt[fg_inds, 5:] = target_3d

                    # compute 3d transform
                    transforms[fg_inds, 5:] = bbox_transform_3d(src_rois, src_3d, target_3d)


                # store labels
                transforms[fg_inds, 4] = [box_lbls[x] for x in targets[fg_inds]]
                assert (all(transforms[fg_inds, 4] >= 1))

        else:

            ols_max = np.zeros(rois.shape[0], dtype=int)
            fg_inds = np.empty(shape=[0])
            gt_best_rois = np.empty(shape=[0])

        # determine ignores
        ign_inds = np.flatnonzero(ols_ign_max >= ign_thresh)

        # determine background
        bg_inds = np.flatnonzero((ols_max >= bg_thresh_lo) & (ols_max < bg_thresh_hi))

        # subtract fg and igns from background
        bg_inds = np.setdiff1d(bg_inds, ign_inds)
        bg_inds = np.setdiff1d(bg_inds, fg_inds)
        bg_inds = np.setdiff1d(bg_inds, gt_best_rois)

        # mark background
        transforms[bg_inds, 4] = -1

    else:

        # all background
        transforms[:, 4] = -1


    return transforms, ols, raw_gt


def bbox_transform(ex_rois, gt_rois):
    """
    Compute the bbox target transforms in 2D.
    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets


def bbox_transform_vertices(ex_rois, gt_vertices):
    """
    Compute the bbox target transforms in 2D.
    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1)

    targets_x = (gt_vertices[:, :8].T - ex_ctr_x) / ex_widths
    targets_y = (gt_vertices[:, 8:].T - ex_ctr_y) / ex_heights

    targets = np.vstack((targets_x, targets_y)).T

    return targets

def bbox_transform_3d(ex_rois_2d, ex_rois_3d, gt_rois):
    """
    Compute the bbox target transforms in 3D.
    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois_2d[:, 2] - ex_rois_2d[:, 0] + 1.0
    ex_heights = ex_rois_2d[:, 3] - ex_rois_2d[:, 1] + 1.0
    ex_ctr_x = ex_rois_2d[:, 0] + 0.5 * (ex_widths - 1)
    ex_ctr_y = ex_rois_2d[:, 1] + 0.5 * (ex_heights - 1)

    gt_ctr_x = gt_rois[:, 0]
    gt_ctr_y = gt_rois[:, 1]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights

    delta_z = gt_rois[:, 2] - ex_rois_3d[:, 0]
    scale_w = np.log(gt_rois[:, 3] / ex_rois_3d[:, 1])
    scale_h = np.log(gt_rois[:, 4] / ex_rois_3d[:, 2])
    scale_l = np.log(gt_rois[:, 5] / ex_rois_3d[:, 3])
    deltaRotY = gt_rois[:, 6] - ex_rois_3d[:, 4]

    targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY)).transpose()
    # axis = 0, [1,2] [1,2] to [[1,2],[1,2]], transpose to [[1,1],[2,2]]
    targets = np.hstack((targets, gt_rois[:, 7:]))
    # axis = 1, tp [[1,1],[2,2],[3,3]]



    return targets

def bbox_transform_corners(ex_rois_3d, gt_corners):
    """
    Compute the bbox target transforms in 3D.
    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """


    delta_z = gt_corners[:, 2*8:3*8].T - ex_rois_3d[:, 0]
    targets = np.hstack((delta_z.T, gt_corners[:, :2*8]))


    return targets

def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d,   0,   0,   0])
    y_corners = np.array([0, 0,   h3d, h3d,   0,   0, h3d, h3d])
    z_corners = np.array([0, 0,     0, w3d, w3d, w3d, w3d,   0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate object coordinate to camera coordinate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]  # normalize dim 3 -> 2, image coordinate

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d  # 2d corners in image coordinate and 3d corners in camera coordinate
    else:
        return verts3d

def parse_kitti_result(respath, mode='new'):

    text_file = open(respath, 'r')

    acc = np.zeros([3, 41], dtype=float)

    lind = 0
    for line in text_file:

        parsed = re.findall('([\d]+\.?[\d]*)', line)

        for i, num in enumerate(parsed):
            acc[lind, i] = float(num)

        lind += 1

    text_file.close()

    if mode == 'old':
        easy = np.mean(acc[0, 0:41:4])
        mod = np.mean(acc[1, 0:41:4])
        hard = np.mean(acc[2, 0:41:4])
    else:
        easy = np.mean(acc[0, 1:41:1])
        mod = np.mean(acc[1, 1:41:1])
        hard = np.mean(acc[2, 1:41:1])

    return easy, mod, hard


# box_2d = [x1, y1, width, height], only use r in Exp. x2d, y2d, z2d -> 2d center, depth. rY is converted from alpha.
def hill_climb(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d, step_z_init=0, step_r_init=0, z_lim=0, r_lim=0, min_ol_dif=0.0):

    step_z = step_z_init
    step_r = step_r_init

    ol_best, verts_best, _, invalid = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d)

    if invalid: return z2d, ry3d, verts_best

    # attempt to fit z/rot more properly
    while (step_z > z_lim or step_r > r_lim):

        if step_z > z_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d - step_z, w3d, h3d, l3d, ry3d)
            ol_pos, verts_pos, _, invalid_pos = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d + step_z, w3d, h3d, l3d, ry3d)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_z = step_z * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                z2d += step_z
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                z2d -= step_z
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_z = step_z * 0.5

        if step_r > r_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d - step_r)
            ol_pos, verts_pos, _, invalid_pos = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d + step_r)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_r = step_r * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                ry3d += step_r
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                ry3d -= step_r
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_r = step_r * 0.5

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return z2d, ry3d, verts_best


def im_detect_3d(cls, prob, bbox_2d, bbox_3d, feat_size, rois, rpn_conf, p2, scale_factor=1):
    """
    Object detection in 3D
    """
    synced = False

    # compute feature resolution
    anchors = np.array(rpn_conf.anchors)
    num_anchors = anchors.shape[0]

    bbox_x = bbox_2d[:, :, 0]
    bbox_y = bbox_2d[:, :, 1]
    bbox_w = bbox_2d[:, :, 2]
    bbox_h = bbox_2d[:, :, 3]

    bbox_x3d = bbox_3d[:, :, 0]
    bbox_y3d = bbox_3d[:, :, 1]
    bbox_z3d = bbox_3d[:, :, 2]
    bbox_w3d = bbox_3d[:, :, 3]
    bbox_h3d = bbox_3d[:, :, 4]
    bbox_l3d = bbox_3d[:, :, 5]
    bbox_ry3d = bbox_3d[:, :, 6]

    bbox_stds = np.array(rpn_conf.bbox_stds)
    bbox_means = np.array(rpn_conf.bbox_means)
    # detransform 3d
    bbox_x3d = bbox_x3d * bbox_stds[:, 4][0] + bbox_means[:, 4][0]
    bbox_y3d = bbox_y3d * bbox_stds[:, 5][0] + bbox_means[:, 5][0]
    bbox_z3d = bbox_z3d * bbox_stds[:, 6][0] + bbox_means[:, 6][0]
    bbox_w3d = bbox_w3d * bbox_stds[:, 7][0] + bbox_means[:, 7][0]
    bbox_h3d = bbox_h3d * bbox_stds[:, 8][0] + bbox_means[:, 8][0]
    bbox_l3d = bbox_l3d * bbox_stds[:, 9][0] + bbox_means[:, 9][0]
    bbox_ry3d = bbox_ry3d * bbox_stds[:, 10][0] + bbox_means[:, 10][0]

    # find 3d source
    tracker = rois[:, 4].detach().numpy().astype(np.int64)
    # TODO src_3d = torch.from_numpy(rpn_conf.anchors[tracker, 4:]).cuda().type(torch.cuda.FloatTensor)
    src_3d = fluid.dygraph.to_variable(anchors[tracker, 4:])
    src_3d = fluid.layers.cast(src_3d, 'float32')

    #tracker_sca = rois_sca[:, 4].cpu().detach().numpy().astype(np.int64)
    #src_3d_sca = torch.from_numpy(rpn_conf.anchors[tracker_sca, 4:]).cuda().type(torch.cuda.FloatTensor)

    # compute 3d transform
    widths = rois[:, 2] - rois[:, 0] + 1.0
    heights = rois[:, 3] - rois[:, 1] + 1.0
    ctr_x = rois[:, 0] + 0.5 * widths
    ctr_y = rois[:, 1] + 0.5 * heights

    bbox_x3d = bbox_x3d[0, :] * widths + ctr_x
    bbox_y3d = bbox_y3d[0, :] * heights + ctr_y
    bbox_z3d = src_3d[:, 0] + bbox_z3d[0, :]
    bbox_w3d = fluid.layers.exp(bbox_w3d[0, :]) * src_3d[:, 1]
    bbox_h3d = fluid.layers.exp(bbox_h3d[0, :]) * src_3d[:, 2]
    bbox_l3d = fluid.layers.exp(bbox_l3d[0, :]) * src_3d[:, 3]
    bbox_ry3d = src_3d[:, 4] + bbox_ry3d[0, :]

    # bundle
    coords_3d = fluid.layers.stack((bbox_x3d, bbox_y3d, bbox_z3d[:bbox_x3d.shape[0]], bbox_w3d[:bbox_x3d.shape[0]], bbox_h3d[:bbox_x3d.shape[0]], bbox_l3d[:bbox_x3d.shape[0]], bbox_ry3d[:bbox_x3d.shape[0]]), axis=1)

    # compile deltas pred
    a = fluid.layers.unsqueeze(bbox_x[0, :], axes=[1])
    deltas_2d = fluid.layers.concat([fluid.layers.unsqueeze(bbox_x[0, :], axes=[1]),
                                     fluid.layers.unsqueeze(bbox_y[0, :], axes=[1]),
                                     fluid.layers.unsqueeze(bbox_w[0, :], axes=[1]),
                                     fluid.layers.unsqueeze(bbox_h[0, :], axes=[1])], axis=1)
    coords_2d = bbox_transform_inv(rois, deltas_2d, means=bbox_means[0, :], stds=bbox_stds[0, :])

    # detach onto cpu
    coords_2d = coords_2d.detach().numpy()
    coords_3d = coords_3d.detach().numpy()
    prob = prob[0, :, :].detach().numpy()

    # scale coords
    coords_2d[:, 0:4] /= scale_factor
    coords_3d[:, 0:2] /= scale_factor

    cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
    scores = np.amax(prob[:, 1:], axis=1)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))

    sorted_inds = (-aboxes[:, 4]).argsort()
    original_inds = (sorted_inds).argsort()
    aboxes = aboxes[sorted_inds, :]
    coords_3d = coords_3d[sorted_inds, :]
    cls_pred = cls_pred[sorted_inds]
    tracker = tracker[sorted_inds]

    if synced:

        # nms
        keep_inds = gpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres, device_id=gpu)

        # convert to bool
        keep = np.zeros([aboxes.shape[0], 1], dtype=bool)
        keep[keep_inds, :] = True

        # stack the keep array,
        # sync to the original order
        aboxes = np.hstack((aboxes, keep))
        aboxes[original_inds, :]

    else:

        # pre-nms
        cls_pred = cls_pred[0:min(rpn_conf.nms_topN_pre, cls_pred.shape[0])]
        tracker = tracker[0:min(rpn_conf.nms_topN_pre, tracker.shape[0])]
        aboxes = aboxes[0:min(rpn_conf.nms_topN_pre, aboxes.shape[0]), :]
        coords_3d = coords_3d[0:min(rpn_conf.nms_topN_pre, coords_3d.shape[0])]

        # nms gpu_nms or cpu_nms
        # TODO gpu_nms(aboxes, rpn_conf.nms_thres, device_id=gpu)
        keep_inds = cpu_nms(aboxes[:, 0:5].astype(np.float32), rpn_conf.nms_thres)

        # stack cls prediction
        aboxes = np.hstack((aboxes, cls_pred[:, np.newaxis], coords_3d, tracker[:, np.newaxis]))

        # suppress boxes
        aboxes = aboxes[keep_inds, :]

    # clip boxes
    if rpn_conf.clip_boxes:
        aboxes[:, 0] = np.clip(aboxes[:, 0], 0, imW_orig - 1)
        aboxes[:, 1] = np.clip(aboxes[:, 1], 0, imH_orig - 1)
        aboxes[:, 2] = np.clip(aboxes[:, 2], 0, imW_orig - 1)
        aboxes[:, 3] = np.clip(aboxes[:, 3], 0, imH_orig - 1)

    return aboxes


def test_projection(p2, p2_inv, box_2d, cx, cy, z, w3d, h3d, l3d, rotY):
    """
    Tests the consistency of a 3D projection compared to a 2D box
    """

    x = box_2d[0]
    y = box_2d[1]
    x2 = x + box_2d[2] - 1
    y2 = y + box_2d[3] - 1

    coord3d = p2_inv.dot(np.array([cx * z, cy * z, z, 1]))  # camera coordinate

    cx3d = coord3d[0]
    cy3d = coord3d[1]
    cz3d = coord3d[2]

    # put back on ground first
    #cy3d += h3d/2

    # re-compute the 2D box using 3D (finally, avoids clipped boxes)
    verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    invalid = np.any(corners_3d[2, :] <= 0)

    x_new = min(verts3d[:, 0])
    y_new = min(verts3d[:, 1])
    x2_new = max(verts3d[:, 0])
    y2_new = max(verts3d[:, 1])

    b1 = np.array([x, y, x2, y2])[np.newaxis, :]
    b2 = np.array([x_new, y_new, x2_new, y2_new])[np.newaxis, :]

    #ol = iou(b1, b2)[0][0]
    ol = -(np.abs(x - x_new) + np.abs(y - y_new) + np.abs(x2 - x2_new) + np.abs(y2 - y2_new))  # L1 norm

    return ol, verts3d, b2, invalid

