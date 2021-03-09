import paddle.fluid as fluid
import sys

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *
from lib.core import *
from lib.util import *

class RPN_3D_loss(fluid.dygraph.Layer):

    def __init__(self, conf):

        super(RPN_3D_loss, self).__init__()

        self.num_classes = len(conf.lbls) + 1
        self.anchors = np.array(conf.anchors)
        self.num_anchors = self.anchors.shape[0]
        self.bbox_means = np.array(conf.bbox_means)
        self.bbox_stds = np.array(conf.bbox_stds)
        self.feat_stride = conf.feat_stride
        self.fg_fraction = conf.fg_fraction
        self.box_samples = conf.box_samples
        self.ign_thresh = conf.ign_thresh
        self.nms_thres = conf.nms_thres
        self.fg_thresh = conf.fg_thresh
        self.bg_thresh_lo = conf.bg_thresh_lo
        self.bg_thresh_hi = conf.bg_thresh_hi
        self.best_thresh = conf.best_thresh
        self.hard_negatives = conf.hard_negatives
        self.focal_loss = conf.focal_loss

        self.crop_size = conf.crop_size

        self.cls_2d_lambda = conf.cls_2d_lambda
        self.iou_2d_lambda = conf.iou_2d_lambda
        self.bbox_2d_lambda = conf.bbox_2d_lambda
        self.bbox_3d_lambda = conf.bbox_3d_lambda
        self.bbox_3d_proj_lambda = conf.bbox_3d_proj_lambda

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h

        self.use_corner = conf.use_corner
        self.corner_in_3d = conf.corner_in_3d
        self.use_hill_loss = conf.use_hill_loss


    def forward(self, cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, bbox_vertices=None, corners_3d=None, is_cpu=False):

        stats = []
        # TODO loss = torch.tensor(0).type(torch.cuda.FloatTensor)
        loss = fluid.layers.fill_constant(shape=[1], value=0, dtype='float32', force_cpu=is_cpu)

        FG_ENC = 1000
        BG_ENC = 2000

        IGN_FLAG = 3000

        batch_size = cls.shape[0]

        prob_detach = prob.detach().numpy()

        # cls : [B x (W x H) x (Class_num * Anchor_num)] 144
        bbox_x = bbox_2d[:, :, 0]  # [B x (W x H) x Anchor_num] 36
        bbox_y = bbox_2d[:, :, 1]
        bbox_w = bbox_2d[:, :, 2]
        bbox_h = bbox_2d[:, :, 3]

        bbox_x3d = bbox_3d[:, :, 0]  # 3d_proj center
        bbox_y3d = bbox_3d[:, :, 1]
        bbox_z3d = bbox_3d[:, :, 2]  # depth
        bbox_w3d = bbox_3d[:, :, 3]
        bbox_h3d = bbox_3d[:, :, 4]
        bbox_l3d = bbox_3d[:, :, 5]
        bbox_ry3d = bbox_3d[:, :, 6]

        # TODO
        bbox_x3d_proj = np.zeros(bbox_x3d.shape, 'float32')
        bbox_y3d_proj = np.zeros(bbox_x3d.shape, 'float32')
        bbox_z3d_proj = np.zeros(bbox_x3d.shape, 'float32')

        labels = np.zeros(cls.shape[0:2])  # B x (W x H)
        labels_weight = np.zeros(cls.shape[0:2])

        labels_scores = np.zeros(cls.shape[0:2])

        bbox_x_tar = np.zeros(cls.shape[0:2])
        bbox_y_tar = np.zeros(cls.shape[0:2])
        bbox_w_tar = np.zeros(cls.shape[0:2])
        bbox_h_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_tar = np.zeros(cls.shape[0:2])
        bbox_w3d_tar = np.zeros(cls.shape[0:2])
        bbox_h3d_tar = np.zeros(cls.shape[0:2])
        bbox_l3d_tar = np.zeros(cls.shape[0:2])
        bbox_ry3d_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_proj_tar = np.zeros(cls.shape[0:2])

        bbox_weights = np.zeros(cls.shape[0:2])

        # TODO ious_2d = fluid.layers.zeros(cls.shape[0:2], 'float32')
        ious_2d = np.zeros(cls.shape[0:2], 'float32')
        ious_3d = np.zeros(cls.shape[0:2], 'float32')
        coords_abs_z = np.zeros(cls.shape[0:2], 'float32')
        coords_abs_ry = np.zeros(cls.shape[0:2], 'float32')

        # get all rois
        rois = locate_anchors(self.anchors, feat_size, self.feat_stride, convert_tensor=True, input_tensor=True)
        rois = fluid.layers.cast(rois, 'float32')

        # de-mean std
        bbox_x3d_dn = bbox_x3d * self.bbox_stds[:, 4][0] + self.bbox_means[:, 4][0]
        bbox_y3d_dn = bbox_y3d * self.bbox_stds[:, 5][0] + self.bbox_means[:, 5][0]
        bbox_z3d_dn = bbox_z3d * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
        bbox_w3d_dn = bbox_w3d * self.bbox_stds[:, 7][0] + self.bbox_means[:, 7][0]
        bbox_h3d_dn = bbox_h3d * self.bbox_stds[:, 8][0] + self.bbox_means[:, 8][0]
        bbox_l3d_dn = bbox_l3d * self.bbox_stds[:, 9][0] + self.bbox_means[:, 9][0]
        bbox_ry3d_dn = bbox_ry3d * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]

        # bbox_x3d_dn = fluid.layers.cast(bbox_x3d_dn, 'float64')
        # bbox_y3d_dn = fluid.layers.cast(bbox_y3d_dn, 'float64')
        # bbox_z3d_dn = fluid.layers.cast(bbox_z3d_dn, 'float64')
        # bbox_w3d_dn = fluid.layers.cast(bbox_w3d_dn, 'float64')
        # bbox_h3d_dn = fluid.layers.cast(bbox_h3d_dn, 'float64')
        # bbox_l3d_dn = fluid.layers.cast(bbox_l3d_dn, 'float64')
        # bbox_ry3d_dn = fluid.layers.cast(bbox_ry3d_dn, 'float64')

        src_anchors = self.anchors[fluid.layers.cast(rois[:, 4], 'int64').numpy(), :]
        src_anchors = fluid.dygraph.to_variable(src_anchors)
        src_anchors = fluid.layers.cast(src_anchors, 'float32')
        if len(src_anchors.shape) == 1: src_anchors = fluid.layers.unsqueeze(src_anchors, 0)

        # compute 3d transform
        widths = rois[:, 2] - rois[:, 0] + 1.0
        heights = rois[:, 3] - rois[:, 1] + 1.0
        ctr_x = rois[:, 0] + 0.5 * widths
        ctr_y = rois[:, 1] + 0.5 * heights

        # de-normalization
        bbox_x3d_dn = bbox_x3d_dn * fluid.layers.unsqueeze(widths, 0) + fluid.layers.unsqueeze(ctr_x, 0)
        bbox_y3d_dn = bbox_y3d_dn * fluid.layers.unsqueeze(heights, 0) + fluid.layers.unsqueeze(ctr_y, 0)
        bbox_z3d_dn = fluid.layers.unsqueeze(src_anchors[:, 4], 0) + bbox_z3d_dn
        bbox_w3d_dn = fluid.layers.exp(bbox_w3d_dn) * fluid.layers.unsqueeze(src_anchors[:, 5], 0)
        bbox_h3d_dn = fluid.layers.exp(bbox_h3d_dn) * fluid.layers.unsqueeze(src_anchors[:, 6], 0)
        bbox_l3d_dn = fluid.layers.exp(bbox_l3d_dn) * fluid.layers.unsqueeze(src_anchors[:, 7], 0)
        bbox_ry3d_dn = fluid.layers.unsqueeze(src_anchors[:, 8], 0) + bbox_ry3d_dn

        if self.use_hill_loss:
            hill_coords_2d = np.zeros(cls.shape[0:2] + (4,))  # (4, 126720, 4)
            hill_p2 = fluid.layers.zeros(shape=[cls.shape[0], cls.shape[1], 4, 4])
            hill_3d = fluid.layers.zeros(shape=[cls.shape[0], cls.shape[1], 7])

        for bind in range(0, batch_size):

            imobj = imobjs[bind]
            gts = imobj.gts

            p2_inv = fluid.layers.cast(fluid.dygraph.to_variable(imobj.p2_inv), 'float32')
            p2 = fluid.layers.cast(fluid.dygraph.to_variable(imobj.p2), 'float32')

            # filter gts
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

            # accumulate boxes, Convert from [x,y,w,h] to [x1, y1, x2, y2]
            gts_all = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts]))
            # [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY]
            gts_3d = np.array([gt.bbox_3d for gt in gts])

            if not ((rmvs == False) & (igns == False)).any():
                continue

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            # accumulate labels
            box_lbls = np.array([gt.cls for gt in gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array([clsName2Ind(self.lbls, cls) for cls in box_lbls])

            if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

                # bbox regression
                if self.use_corner:
                    gts_vertices = np.array([gt.vertices for gt in imobj.gts])
                    gts_vertices = gts_vertices[(rmvs == False) & (igns == False), :]
                    gts_corners_3d = np.array([gt.corners_3d for gt in imobj.gts])
                    gts_corners_3d = gts_corners_3d[(rmvs == False) & (igns == False), :]
                    transforms, ols, raw_gt = compute_targets(gts_val, gts_ign, box_lbls, rois.numpy(), self.fg_thresh,
                                                              self.ign_thresh, self.bg_thresh_lo, self.bg_thresh_hi,
                                                              self.best_thresh, anchors=self.anchors, gts_3d=gts_3d,
                                                              gts_vertices=gts_vertices, gts_corners_3d=gts_corners_3d,
                                                              tracker=rois[:, 4].numpy())

                else:
                    transforms, ols, raw_gt = compute_targets(gts_val, gts_ign, box_lbls, rois.numpy(), self.fg_thresh,
                                                      self.ign_thresh, self.bg_thresh_lo, self.bg_thresh_hi,
                                                      self.best_thresh, anchors=self.anchors,  gts_3d=gts_3d,
                                                      tracker=rois[:, 4].numpy())

                if self.use_hill_loss:
                    hill_deltas_2d = transforms[:, 0:4]
                    hill_coords_2d[bind, :, :] = bbox_transform_inv(rois, fluid.dygraph.to_variable(hill_deltas_2d), means=self.bbox_means[0, :], stds=self.bbox_stds[0, :]).cpu().numpy() / imobj['scale_factor']

                    hill_p2[bind, :, :, :] = p2

                    hill_x3d = fluid.layers.unsqueeze(bbox_x3d_dn[bind], 0) / imobj['scale_factor']
                    hill_y3d = fluid.layers.unsqueeze(bbox_y3d_dn[bind], 0) / imobj['scale_factor']
                    hill_z3d = fluid.layers.unsqueeze(bbox_z3d_dn[bind], 0)
                    hill_w3d = bbox_w3d_dn[bind]
                    hill_h3d = bbox_h3d_dn[bind]
                    hill_l3d = bbox_l3d_dn[bind]
                    hill_ry3d = bbox_ry3d_dn[bind]
                    hill_coord3d = p2_inv.mm(fluid.layers.concat([hill_x3d * hill_z3d, hill_y3d * hill_z3d, hill_z3d,
                                                                  fluid.layers.ones(shape=hill_x3d.shape, dtype=hill_x3d.dtype)], axis=0))  # # (4, 126720) # 36 * 110 * 32
                    hill_cx3d = hill_coord3d[0]
                    hill_cy3d = hill_coord3d[1]
                    hill_cz3d = hill_coord3d[2]
                    hill_ry3d = convertAlpha2Rot_torch(hill_ry3d, hill_cz3d, hill_cx3d)

                    hill_3d_all = fluid.layers.concat([fluid.layers.unsqueeze(hill_cx3d, 1), fluid.layers.unsqueeze(hill_cy3d, 1), fluid.layers.unsqueeze(hill_cz3d, 1), fluid.layers.unsqueeze(hill_w3d, 1), fluid.layers.unsqueeze(hill_h3d, 1), fluid.layers.unsqueeze(hill_l3d, 1), fluid.layers.unsqueeze(hill_ry3d, 1)], axis=1)
                    hill_3d[bind, :, :] = hill_3d_all

                # normalize 2d
                transforms[:, 0:4] -= self.bbox_means[:, 0:4]
                transforms[:, 0:4] /= self.bbox_stds[:, 0:4]

                # normalize 3d
                transforms[:, 5:12] -= self.bbox_means[:, 4:11]
                transforms[:, 5:12] /= self.bbox_stds[:, 4:11]

                if self.use_corner:
                    transforms[:, 16:32] -= self.bbox_means[:, 11:27]
                    transforms[:, 16:32] /= self.bbox_stds[:, 11:27]
                    transforms[:, 32:40] -= self.bbox_means[:, 27:35]
                    transforms[:, 32:40] /= self.bbox_stds[:, 27:35]
                if self.corner_in_3d:
                    transforms[:, 40:56] -= self.bbox_means[:, 35:51]
                    transforms[:, 40:56] /= self.bbox_stds[:, 35:51]
                    transforms[:, 12:14] -= self.bbox_means[:, 51:53]
                    transforms[:, 12:14] /= self.bbox_stds[:, 51:53]

                labels_fg = transforms[:, 4] > 0
                labels_bg = transforms[:, 4] < 0
                labels_ign = transforms[:, 4] == 0

                fg_inds = np.flatnonzero(labels_fg)
                bg_inds = np.flatnonzero(labels_bg)
                ign_inds = np.flatnonzero(labels_ign)

                labels[bind, fg_inds] = transforms[fg_inds, 4]
                labels[bind, ign_inds] = IGN_FLAG
                labels[bind, bg_inds] = 0

                # GT
                bbox_x_tar[bind, :] = transforms[:, 0]
                bbox_y_tar[bind, :] = transforms[:, 1]
                bbox_w_tar[bind, :] = transforms[:, 2]
                bbox_h_tar[bind, :] = transforms[:, 3]

                bbox_x3d_tar[bind, :] = transforms[:, 5]
                bbox_y3d_tar[bind, :] = transforms[:, 6]
                bbox_z3d_tar[bind, :] = transforms[:, 7]
                bbox_w3d_tar[bind, :] = transforms[:, 8]
                bbox_h3d_tar[bind, :] = transforms[:, 9]
                bbox_l3d_tar[bind, :] = transforms[:, 10]
                bbox_ry3d_tar[bind, :] = transforms[:, 11]

                bbox_x3d_proj_tar[bind, :] = raw_gt[:, 12]
                bbox_y3d_proj_tar[bind, :] = raw_gt[:, 13]
                bbox_z3d_proj_tar[bind, :] = raw_gt[:, 14]

                if self.use_corner:
                    bbox_vertices_depth_tar = np.zeros(cls.shape[0:2] + (24,))
                    bbox_vertices_depth_tar[bind, :, :] = transforms[:, 16:40]
                if self.corner_in_3d:
                    bbox_3d_corners_tar = np.zeros(cls.shape[0:2] + (18,))
                    bbox_3d_corners_tar[bind, :, :16] = transforms[:, 40:56]
                    bbox_3d_corners_tar[bind, :, 16:] = transforms[:, 12:14]
                # ----------------------------------------
                # box sampling
                # ----------------------------------------

                if self.box_samples == np.inf:
                    fg_num = len(fg_inds)
                    bg_num = len(bg_inds)

                else:
                    fg_num = min(round(rois.shape[0]*self.box_samples * self.fg_fraction), len(fg_inds))
                    bg_num = min(round(rois.shape[0]*self.box_samples - fg_num), len(bg_inds))

                if self.hard_negatives:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        scores = prob_detach[bind, fg_inds, labels[bind, fg_inds].astype(int)]
                        fg_score_ascend = (scores).argsort()
                        fg_inds = fg_inds[fg_score_ascend]
                        fg_inds = fg_inds[0:fg_num]

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        fg_inds = np.random.choice(fg_inds, fg_num, replace=False)

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)

                labels_weight[bind, bg_inds] = BG_ENC  # TODO: set label-weight
                labels_weight[bind, fg_inds] = FG_ENC
                bbox_weights[bind, fg_inds] = 1

                # ----------------------------------------
                # compute IoU stats
                # ----------------------------------------

                if fg_num > 0:

                    # compile deltas pred
                    deltas_2d = fluid.layers.concat([
                        fluid.layers.reshape(bbox_x[bind, :], [-1, 1]),
                        fluid.layers.reshape(bbox_y[bind, :], [-1, 1]),
                        fluid.layers.reshape(bbox_w[bind, :], [-1, 1]),
                        fluid.layers.reshape(bbox_h[bind, :], [-1, 1])], axis=1)

                    # compile deltas targets
                    deltas_2d_tar = np.concatenate((bbox_x_tar[bind, :, np.newaxis], bbox_y_tar[bind, :, np.newaxis],
                                                    bbox_w_tar[bind, :, np.newaxis], bbox_h_tar[bind, :, np.newaxis]),
                                                   axis=1)

                    # move to gpu
                    deltas_2d_tar = fluid.dygraph.to_variable(deltas_2d_tar)
                    deltas_2d_tar = fluid.layers.cast(deltas_2d_tar, 'float32')

                    means = self.bbox_means[0, :]
                    stds = self.bbox_stds[0, :]

                    # TODO rois = rois.cuda()

                    coords_2d = bbox_transform_inv(rois, deltas_2d, means=means, stds=stds)  # convert to x1, x2, y1, y2
                    coords_2d_tar = bbox_transform_inv(rois, deltas_2d_tar, means=means, stds=stds)

                    # cal IOU
                    # TODO ious_2d[bind, fg_inds] = iou(coords_2d[fg_inds, :], coords_2d_tar[fg_inds, :], mode='list')
                    out = iou(fluid.layers.gather(coords_2d, fluid.dygraph.to_variable(fg_inds)),
                                                 fluid.layers.gather(coords_2d_tar, fluid.dygraph.to_variable(fg_inds)),
                                                 mode='list')
                    ious_2d[bind, fg_inds] = out.numpy()


                    # TODO src_anchors = self.anchors[rois[fg_inds, 4], :]
                    src_anchors = fluid.layers.gather(fluid.dygraph.to_variable(self.anchors),
                                                      fluid.layers.cast(fluid.layers.gather(rois[:, 4],
                                                      fluid.dygraph.to_variable(fg_inds)), 'int64'))
                    src_anchors = fluid.layers.cast(src_anchors, 'float32')

                    if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

                    # Prediction
                    bbox_x3d_dn_fg = fluid.layers.gather(bbox_x3d_dn[bind], fluid.dygraph.to_variable(fg_inds))
                    bbox_y3d_dn_fg = fluid.layers.gather(bbox_y3d_dn[bind], fluid.dygraph.to_variable(fg_inds))
                    bbox_z3d_dn_fg = fluid.layers.gather(bbox_z3d_dn[bind], fluid.dygraph.to_variable(fg_inds))
                    bbox_w3d_dn_fg = fluid.layers.gather(bbox_w3d_dn[bind], fluid.dygraph.to_variable(fg_inds))
                    bbox_h3d_dn_fg = fluid.layers.gather(bbox_h3d_dn[bind], fluid.dygraph.to_variable(fg_inds))
                    bbox_l3d_dn_fg = fluid.layers.gather(bbox_l3d_dn[bind], fluid.dygraph.to_variable(fg_inds))
                    bbox_ry3d_dn_fg = fluid.layers.gather(bbox_ry3d_dn[bind], fluid.dygraph.to_variable(fg_inds))

                    # re-scale all 2D back to original
                    # TODO bbox_x3d_dn_fg /= imobj['scale_factor']
                    # TODO bbox_y3d_dn_fg /= imobj['scale_factor']

                    coords_2d = fluid.layers.concat([fluid.layers.reshape(bbox_x3d_dn_fg * bbox_z3d_dn_fg, [1, -1]),
                                                    fluid.layers.reshape(bbox_y3d_dn_fg * bbox_z3d_dn_fg, [1, -1]),
                                                     fluid.layers.reshape(bbox_z3d_dn_fg, [1, -1])], axis=0)
                    coords_2d = fluid.layers.concat([coords_2d, fluid.layers.ones([1, coords_2d.shape[1]], dtype='float32')], axis=0)

                    coords_3d = fluid.layers.matmul(p2_inv, coords_2d)
                    # project center to 3d

                    bbox_x3d_proj[bind, fg_inds] = coords_3d[0, :].numpy()
                    bbox_y3d_proj[bind, fg_inds] = coords_3d[1, :].numpy()
                    bbox_z3d_proj[bind, fg_inds] = coords_3d[2, :].numpy()

                    # absolute targets
                    bbox_z3d_dn_tar = bbox_z3d_tar[bind, fg_inds] * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
                    bbox_z3d_dn_tar = fluid.layers.cast(fluid.dygraph.to_variable(bbox_z3d_dn_tar), 'float32')
                    bbox_z3d_dn_tar = src_anchors[:, 4] + bbox_z3d_dn_tar

                    bbox_ry3d_dn_tar = bbox_ry3d_tar[bind, fg_inds] * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]
                    bbox_ry3d_dn_tar = fluid.layers.cast(fluid.dygraph.to_variable(bbox_ry3d_dn_tar), 'float32')
                    bbox_ry3d_dn_tar = src_anchors[:, 8] + bbox_ry3d_dn_tar

                    coords_abs_z[bind, fg_inds] = fluid.layers.abs(bbox_z3d_dn_tar - bbox_z3d_dn_fg).numpy()
                    coords_abs_ry[bind, fg_inds] = fluid.layers.abs(bbox_ry3d_dn_tar - bbox_ry3d_dn_fg).numpy()

            else:  # No GT.

                bg_inds = np.arange(0, rois.shape[0])

                if self.box_samples == np.inf: bg_num = len(bg_inds)
                else: bg_num = min(round(self.box_samples * (1 - self.fg_fraction)), len(bg_inds))

                if self.hard_negatives:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        # Use probability prediction and sort
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)


                labels[bind, :] = 0
                labels_weight[bind, bg_inds] = BG_ENC


            # grab label predictions (for weighing purposes)
            active = labels[bind, :] != IGN_FLAG
            labels_scores[bind, active] = prob_detach[bind, active, labels[bind, active].astype(int)]

        # ----------------------------------------
        # useful statistics
        # ----------------------------------------

        fg_inds_all = np.flatnonzero((labels > 0) & (labels != IGN_FLAG))
        bg_inds_all = np.flatnonzero((labels == 0) & (labels != IGN_FLAG))

        fg_inds_unravel = np.unravel_index(fg_inds_all, prob_detach.shape[0:2])  # get position
        bg_inds_unravel = np.unravel_index(bg_inds_all, prob_detach.shape[0:2])

        cls_pred = fluid.layers.argmax(cls, axis=2).detach().numpy()

        # class prediction acc
        if self.cls_2d_lambda and len(fg_inds_all) > 0:
            acc_fg = np.mean(cls_pred[fg_inds_unravel] == labels[fg_inds_unravel])
            stats.append({'name': 'fg', 'val': acc_fg, 'format': '{:0.2f}', 'group': 'acc'})

        if self.cls_2d_lambda and len(bg_inds_all) > 0:
            acc_bg = np.mean(cls_pred[bg_inds_unravel] == labels[bg_inds_unravel])
            stats.append({'name': 'bg', 'val': acc_bg, 'format': '{:0.2f}', 'group': 'acc'})

        # ----------------------------------------
        # box weighting
        # ----------------------------------------

        fg_inds = np.flatnonzero(labels_weight == FG_ENC)
        bg_inds = np.flatnonzero(labels_weight == BG_ENC)
        active_inds = np.concatenate((fg_inds, bg_inds), axis=0)

        fg_num = len(fg_inds)
        bg_num = len(bg_inds)

        labels_weight[...] = 0.0
        box_samples = fg_num + bg_num

        fg_inds_unravel = np.unravel_index(fg_inds, labels_weight.shape)
        bg_inds_unravel = np.unravel_index(bg_inds, labels_weight.shape)
        active_inds_unravel = np.unravel_index(active_inds, labels_weight.shape)

        labels_weight[active_inds_unravel] = 1.0

        if self.fg_fraction is not None:

            if fg_num > 0:

                fg_weight = (self.fg_fraction /(1 - self.fg_fraction)) * (bg_num / fg_num)
                labels_weight[fg_inds_unravel] = fg_weight  # large for foreground
                labels_weight[bg_inds_unravel] = 1.0

            else:
                labels_weight[bg_inds_unravel] = 1.0

        # different method of doing hard negative mining
        # use the scores to normalize the importance of each sample
        # hence, encourages the network to get all "correct" rather than
        # becoming more correct at a decision it is already good at
        # this method is equivelent to the focal loss with additional mean scaling
        if self.focal_loss:

            weights_sum = 0

            # re-weight bg
            if bg_num > 0:
                bg_scores = labels_scores[bg_inds_unravel]  # prob_detach
                bg_weights = (1 - bg_scores) ** self.focal_loss
                weights_sum += np.sum(bg_weights)
                labels_weight[bg_inds_unravel] *= bg_weights

            # re-weight fg
            if fg_num > 0:
                fg_scores = labels_scores[fg_inds_unravel]
                fg_weights = (1 - fg_scores) ** self.focal_loss
                weights_sum += np.sum(fg_weights)
                labels_weight[fg_inds_unravel] *= fg_weights


        # ----------------------------------------
        # classification loss
        # ----------------------------------------
        labels = fluid.dygraph.to_variable(labels)
        labels = fluid.layers.reshape(labels, [-1])
        labels = fluid.layers.cast(labels, 'int64')

        labels_weight = fluid.dygraph.to_variable(labels_weight)
        labels_weight = fluid.layers.reshape(labels_weight, [-1])
        labels_weight = fluid.layers.cast(labels_weight, 'float32')

        cls = fluid.layers.reshape(cls, shape=[-1, cls.shape[2]])

        if self.cls_2d_lambda:

            # cls loss
            active = labels_weight > 0
            active = fluid.layers.cast(active, 'int64')
            index_active = fluid.layers.nonzero(active)

            if np.any(active.numpy()):
                #
                # loss_cls = fluid.layers.cross_entropy(
                #     fluid.layers.softmax(fluid.layers.gather(cls, index_active)),
                #     fluid.layers.reshape(fluid.layers.gather(labels, index_active), [-1, 1]), ignore_index=IGN_FLAG)

                loss_cls = fluid.layers.softmax_with_cross_entropy(
                    fluid.layers.gather(cls, index_active), fluid.layers.reshape(fluid.layers.gather(labels, index_active), [-1, 1]), ignore_index=IGN_FLAG)
                loss_cls = fluid.layers.reshape(loss_cls, [-1])
                loss_cls = (loss_cls * fluid.layers.gather(labels_weight, index_active))
                # simple gradient clipping
                loss_cls = fluid.layers.clip(loss_cls, min=0, max=2000)

                # take mean and scale lambda
                loss_cls = fluid.layers.mean(loss_cls)
                loss_cls *= self.cls_2d_lambda

                # print("loss_cls: ", loss_cls.numpy())
                loss = fluid.layers.sum([loss, loss_cls])

                stats.append({'name': 'cls', 'val': loss_cls.numpy(), 'format': '{:0.4f}', 'group': 'loss'})

        # ----------------------------------------
        # bbox regression loss
        # ----------------------------------------

        if np.sum(bbox_weights) > 0:

            # TODO
            bbox_weights = fluid.layers.reshape(fluid.dygraph.to_variable(bbox_weights), [-1])
            bbox_weights = fluid.layers.cast(bbox_weights, 'float32')

            active = bbox_weights > 0
            # TODO index = fluid.layers.nonzero(active)
            index_active = fluid.layers.reshape(fluid.layers.nonzero(active), [-1])

            if self.bbox_2d_lambda:

                # bbox loss 2d
                bbox_x_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_x_tar), 'float32'), [-1])
                bbox_y_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_y_tar), 'float32'), [-1])
                bbox_w_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_w_tar), 'float32'), [-1])
                bbox_h_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_h_tar), 'float32'), [-1])

                bbox_x = fluid.layers.reshape(fluid.layers.cast(bbox_x, 'float32'), [-1])
                bbox_y = fluid.layers.reshape(fluid.layers.cast(bbox_y, 'float32'), [-1])
                bbox_w = fluid.layers.reshape(fluid.layers.cast(bbox_w, 'float32'), [-1])
                bbox_h = fluid.layers.reshape(fluid.layers.cast(bbox_h, 'float32'), [-1])

                loss_bbox_x = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_x, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_x_tar, index_active), [-1, 1]))
                loss_bbox_y = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_y, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_y_tar, index_active), [-1, 1]))
                loss_bbox_w = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_w, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_w_tar, index_active), [-1, 1]))
                loss_bbox_h = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_h, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_h_tar, index_active), [-1, 1]))

                loss_bbox_x = fluid.layers.mean( loss_bbox_x * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_y = fluid.layers.mean( loss_bbox_y * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_w = fluid.layers.mean( loss_bbox_w * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_h = fluid.layers.mean( loss_bbox_h * fluid.layers.gather(bbox_weights, index_active))

                bbox_2d_loss = (loss_bbox_x + loss_bbox_y + loss_bbox_w + loss_bbox_h)
                bbox_2d_loss *= self.bbox_2d_lambda

                # print("bbox_2d_loss: ", bbox_2d_loss.numpy())
                loss = fluid.layers.sum([loss, bbox_2d_loss])
                stats.append({'name': 'bbox_2d', 'val': bbox_2d_loss.numpy(), 'format': '{:0.4f}', 'group': 'loss'})

            if self.use_hill_loss:
                hill_loss = 0
                hill_coords_2d = fluid.layers.reshape(fluid.dygraph.to_variable(hill_coords_2d), [-1, 4])[active]
                hill_p2 = hill_p2.view(-1, 4, 4)[active]
                hill_3d = hill_3d.view(-1, 7)[active]

                for index in range(hill_3d.shape[0]):
                    p2 = hill_p2[index]
                    c3d = hill_3d[index]
                    R = fluid.layers.zeros(shape=[3, 3])
                    R[0, 0] += fluid.layers.cos(c3d[6])
                    R[0, 2] += fluid.layers.sin(c3d[6])
                    R[2, 0] -= fluid.layers.sin(c3d[6])
                    R[2, 2] += fluid.layers.cos(c3d[6])
                    R[1, 1] += 1
                    # print(R)
                    corners = fluid.layers.zeros(shape=[3, 8])
                    corners[0, 1:5] += c3d[5]/2
                    corners[0, [0, 5, 6, 7]] -= c3d[5]/2
                    corners[1, [2, 3, 6, 7]] += c3d[4]/2
                    corners[1, [0, 1, 4, 5]] -= c3d[4]/2
                    corners[2, 3:7] += c3d[3]/2
                    corners[2, [0, 1, 2, 7]] -= c3d[3]/2
                    corners = R.mm(corners)
                    corners[0, :] += c3d[0]
                    corners[1, :] += c3d[1]
                    corners[2, :] += c3d[2]
                    # print(corners)
                    corners = fluid.layers.concat([corners, fluid.layers.ones(shape=[1, 8])], axis=0)
                    corners_2d = p2.mm(corners)
                    corners_2d = corners_2d / corners_2d[2]
                    x_new = fluid.layers.sum(fluid.layers.softmax(-corners_2d[0] * 100, axis=0) * corners_2d[0])
                    y_new = fluid.layers.sum(fluid.layers.softmax(-corners_2d[1] * 100, axis=0) * corners_2d[1])
                    x2_new = fluid.layers.sum(fluid.layers.softmax(corners_2d[0] * 100, axis=0) * corners_2d[0])
                    y2_new = fluid.layers.sum(fluid.layers.softmax(corners_2d[1] * 100, axis=0) * corners_2d[1])
                    # print(x_new, y_new, x2_new, y2_new)
                    # print(hill_coords_2d[index])

                    hill_loss += (fluid.layers.smooth_l1(x_new, hill_coords_2d[index][0]) + \
                                 fluid.layers.smooth_l1(y_new, hill_coords_2d[index][1]) +\
                                 fluid.layers.smooth_l1(x2_new, hill_coords_2d[index][2]) +\
                                 fluid.layers.smooth_l1(y2_new, hill_coords_2d[index][3]))
                hill_loss = hill_loss / hill_3d.size()[0] / 1000  # 2.5 pixel, 0.01 loss
                hill_loss = hill_loss * self.use_hill_loss

                loss = fluid.layers.sum([loss, hill_loss])
                stats.append({'name': 'hill_loss', 'val': hill_loss.numpy(), 'format': '{:0.4f}', 'group': 'loss'})

            if self.use_corner:
                bbox_vertices_depth_tar = fluid.layers.reshape(fluid.dygraph.to_variable(bbox_vertices_depth_tar), [-1, 24])
                bbox_vertices = fluid.layers.reshape(bbox_vertices, shape=[-1, 24])
                loss_vertices = fluid.layers.smooth_l1(bbox_vertices[active], bbox_vertices_depth_tar[active])
                loss_vertices = (loss_vertices * bbox_weights[active].view(-1, 1)).mean()
                loss_vertices = loss_vertices * self.use_corner

                loss = fluid.layers.sum([loss, loss_vertices])

                stats.append({'name': 'loss_vertices', 'val': loss_vertices.numpy(), 'format': '{:0.4f}', 'group': 'loss'})

            if self.corner_in_3d:
                bbox_3d_corners_tar = fluid.layers.reshape(fluid.dygraph.to_variable(bbox_3d_corners_tar), [-1, 18])
                corners_3d = fluid.layers.reshape(corners_3d, shape=[-1, 18])

                loss_corners_3d = fluid.layers.smooth_l1(corners_3d[active], bbox_3d_corners_tar[active])
                loss_corners_3d = fluid.layers.mean(loss_corners_3d * fluid.layers.reshape(bbox_weights[active], shape=[-1, 1]))
                loss_corners_3d = loss_corners_3d * self.corner_in_3d

                loss = fluid.layers.sum([loss, loss_corners_3d])
                stats.append({'name': 'loss_corners_3d', 'val': loss_corners_3d.numpy(), 'format': '{:0.4f}', 'group': 'loss'})

            if self.bbox_3d_lambda:

                # bbox loss 3d
                bbox_x3d_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_x3d_tar), 'float32'), [-1])
                bbox_y3d_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_y3d_tar), 'float32'), [-1])
                bbox_z3d_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_z3d_tar), 'float32'), [-1])
                bbox_w3d_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_w3d_tar), 'float32'), [-1])
                bbox_h3d_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_h3d_tar), 'float32'), [-1])
                bbox_l3d_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_l3d_tar), 'float32'), [-1])
                bbox_ry3d_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_ry3d_tar), 'float32'), [-1])

                bbox_x3d = fluid.layers.reshape(fluid.layers.cast(bbox_x3d, 'float32'), [-1])
                bbox_y3d = fluid.layers.reshape(fluid.layers.cast(bbox_y3d, 'float32'), [-1])
                bbox_z3d = fluid.layers.reshape(fluid.layers.cast(bbox_z3d, 'float32'), [-1])
                bbox_w3d = fluid.layers.reshape(fluid.layers.cast(bbox_w3d, 'float32'), [-1])
                bbox_h3d = fluid.layers.reshape(fluid.layers.cast(bbox_h3d, 'float32'), [-1])
                bbox_l3d = fluid.layers.reshape(fluid.layers.cast(bbox_l3d, 'float32'), [-1])
                bbox_ry3d = fluid.layers.reshape(fluid.layers.cast(bbox_ry3d, 'float32'), [-1])

                loss_bbox_x3d = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_x3d, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_x3d_tar, index_active), [-1, 1]))
                loss_bbox_y3d = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_y3d, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_y3d_tar, index_active), [-1, 1]))
                loss_bbox_z3d = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_z3d, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_z3d_tar, index_active), [-1, 1]))
                loss_bbox_w3d = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_w3d, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_w3d_tar, index_active), [-1, 1]))
                loss_bbox_h3d = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_h3d, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_h3d_tar, index_active), [-1, 1]))
                loss_bbox_l3d = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_l3d, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_l3d_tar, index_active), [-1, 1]))
                loss_bbox_ry3d = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_ry3d, index_active), [-1, 1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_ry3d_tar, index_active), [-1, 1]))

                loss_bbox_x3d = fluid.layers.mean( loss_bbox_x3d * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_y3d = fluid.layers.mean( loss_bbox_y3d * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_z3d = fluid.layers.mean( loss_bbox_z3d * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_w3d = fluid.layers.mean( loss_bbox_w3d * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_h3d = fluid.layers.mean( loss_bbox_h3d * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_l3d = fluid.layers.mean( loss_bbox_l3d * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_ry3d = fluid.layers.mean( loss_bbox_ry3d * fluid.layers.gather(bbox_weights, index_active))

                bbox_3d_loss = (loss_bbox_x3d + loss_bbox_y3d + loss_bbox_z3d)
                bbox_3d_loss += (loss_bbox_w3d + loss_bbox_h3d + loss_bbox_l3d + loss_bbox_ry3d)

                bbox_3d_loss *= self.bbox_3d_lambda

                # print("bbox_3d_loss: ", bbox_3d_loss.numpy())
                loss = fluid.layers.sum([loss, bbox_3d_loss])
                stats.append({'name': 'bbox_3d', 'val': bbox_3d_loss.numpy(), 'format': '{:0.4f}', 'group': 'loss'})

            if self.bbox_3d_proj_lambda:

                # bbox loss 3d
                bbox_x3d_proj_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_x3d_proj_tar), 'float32'), [-1])
                bbox_y3d_proj_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_y3d_proj_tar), 'float32'), [-1])
                bbox_z3d_proj_tar = fluid.layers.reshape(fluid.layers.cast(fluid.dygraph.to_variable(bbox_z3d_proj_tar), 'float32'), [-1])

                bbox_x3d_proj = fluid.layers.reshape(fluid.layers.cast(bbox_x3d_proj, 'float32'), [-1])
                bbox_y3d_proj = fluid.layers.reshape(fluid.layers.cast(bbox_y3d_proj, 'float32'), [-1])
                bbox_z3d_proj = fluid.layers.reshape(fluid.layers.cast(bbox_z3d_proj, 'float32'), [-1])

                loss_bbox_x3d_proj = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_x3d_proj, index_active), [1, -1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_x3d_proj_tar, index_active), [1, -1]))
                loss_bbox_y3d_proj = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_y3d_proj, index_active), [1, -1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_y3d_proj_tar, index_active), [1, -1]))
                loss_bbox_z3d_proj = fluid.layers.smooth_l1(fluid.layers.reshape(fluid.layers.gather(bbox_z3d_proj, index_active), [1, -1]),
                                                     fluid.layers.reshape(fluid.layers.gather(bbox_z3d_proj_tar, index_active), [1, -1]))

                loss_bbox_x3d_proj = fluid.layers.mean( fluid.layers.cast(loss_bbox_x3d_proj, 'float64') * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_y3d_proj = fluid.layers.mean( fluid.layers.cast(loss_bbox_y3d_proj, 'float64') * fluid.layers.gather(bbox_weights, index_active))
                loss_bbox_z3d_proj = fluid.layers.mean( fluid.layers.cast(loss_bbox_z3d_proj, 'float64') * fluid.layers.gather(bbox_weights, index_active))

                bbox_3d_proj_loss = (loss_bbox_x3d_proj + loss_bbox_y3d_proj + loss_bbox_z3d_proj)

                bbox_3d_proj_loss *= self.bbox_3d_proj_lambda

                loss = fluid.layers.sum([loss, bbox_3d_proj_loss])

                stats.append({'name': 'bbox_3d_proj', 'val': bbox_3d_proj_loss.numpy(), 'format': '{:0.4f}', 'group': 'loss'})

            coords_abs_z = fluid.layers.reshape(fluid.dygraph.to_variable(coords_abs_z), [-1])
            stats.append({'name': 'z', 'val': fluid.layers.mean(fluid.layers.gather(coords_abs_z, index_active)).numpy(),
                          'format': '{:0.2f}', 'group': 'misc'})

            coords_abs_ry = fluid.layers.reshape(fluid.dygraph.to_variable(coords_abs_ry), [-1])
            stats.append({'name': 'ry', 'val': fluid.layers.mean(fluid.layers.gather(coords_abs_ry, index_active)).numpy(),
                          'format': '{:0.2f}', 'group': 'misc'})

            ious_2d = fluid.layers.reshape(fluid.dygraph.to_variable(ious_2d), [-1])
            stats.append({'name': 'iou', 'val': fluid.layers.mean(fluid.layers.gather(ious_2d, index_active)).numpy(),
                          'format': '{:0.2f}', 'group': 'acc'})

            # use a 2d IoU based log loss
            if self.iou_2d_lambda:
                iou_2d_loss = -fluid.layers.log(fluid.layers.gather(ious_2d, index_active))
                iou_2d_loss = iou_2d_loss * fluid.layers.gather(bbox_weights, index_active)
                iou_2d_loss = fluid.layers.mean(iou_2d_loss)

                iou_2d_loss *= self.iou_2d_lambda

                loss = fluid.layers.sum([loss, iou_2d_loss])

                stats.append({'name': 'iou', 'val': iou_2d_loss.numpy(), 'format': '{:0.4f}', 'group': 'loss'})


        return loss, stats
