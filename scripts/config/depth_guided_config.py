from easydict import EasyDict as edict
import numpy as np
import datetime

def Config():

    conf = edict()

    # ----------------------------------------
    #  general
    # ----------------------------------------

    conf.model = 'resnet_dilate'
    conf.lr = 0.01
    conf.max_iter = 40000
    conf.snapshot_iter = 4000
    conf.display = 1
    conf.use_dropout = True
    conf.drop_channel = True
    conf.dropout_rate = 0.5
    conf.dropout_position = 'early'  # 'early'  'late' 'adaptive'
    conf.do_test = True
    conf.lr_policy = 'onecycle'  # 'onecycle'  # 'cosinePoly'  # 'cosineRestart'  # 'poly'
    conf.restart_iters = 5000
    conf.batch_size = 1
    conf.base_model = 50
    conf.depth_channel = 1
    conf.adaptive_diated = True
    conf.use_seg = False
    conf.use_corner = False
    conf.corner_in_3d = False
    conf.use_hill_loss = False
    conf.use_rcnn_pretrain = False # False
    conf.deformable = False

    # scale sampling
    # conf.test_scale = 80
    # conf.crop_size = [80, 240]
    conf.test_scale = 370
    conf.crop_size = [370, 1224]    # here is the size
    # conf.test_scale = 512
    # conf.crop_size = [512, 1760]    # here is the size
    conf.mirror_prob = 0.50
    conf.distort_prob = -1

    conf.alias = 'Adaptive_block2'

    conf.result_dir = '_'.join([conf.alias, conf.model + str(conf.base_model), 'batch' + str(conf.batch_size),
                                'dropout' + conf.dropout_position + str(conf.dropout_rate), 'lr' + str(conf.lr),
                                conf.lr_policy, 'iter' + str(conf.max_iter), 'scale' + str(conf.test_scale),
                                datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")]).replace('.', '_').replace(':', '_').replace('-', '_')

    # solver settings
    conf.solver_type = 'sgd'

    conf.momentum = 0.9
    conf.weight_decay = 0.0005

    # sgd parameters
    conf.lr_steps = None
    conf.lr_target = conf.lr * 0.00001
    
    # random
    conf.rng_seed = 2
    conf.cuda_seed = 2
    
    # misc network
    conf.image_means = [0.485, 0.456, 0.406]
    conf.image_stds = [0.229, 0.224, 0.225]
    if conf.use_rcnn_pretrain:
        conf.image_means = [102.9801, 115.9465, 122.7717]  # conf.image_means[::-1]
        conf.image_stds = [1, 1, 1]  #conf.image_stds[::-1]
    if conf.use_seg:
        conf.depth_mean = [4413.160626995486, 4413.160626995486, 5.426258330316642]
        conf.depth_std = [3270.0158918863494, 3270.0158918863494, 0.5365540402943388]
    else:
        conf.depth_mean = [4413.160626995486, 4413.160626995486, 4413.160626995486]  # DORN
        conf.depth_std = [3270.0158918863494, 3270.0158918863494, 3270.0158918863494]
        # conf.depth_mean = [8295.013626842678, 8295.013626842678, 8295.013626842678]  # PSM
        # conf.depth_std = [5134.9781439128665, 5134.9781439128665, 5134.9781439128665]
        # conf.depth_mean = [30.83664619525601, 30.83664619525601, 30.83664619525601]  # DISP
        # conf.depth_std = [19.992999492848206, 19.992999492848206, 19.992999492848206]
    if conf.depth_channel == 3:
        conf.depth_mean = [137.39162828, 40.58310471, 140.70854621]  # MONO1
        conf.depth_std = [33.75859339, 51.479677, 65.254889]
        conf.depth_mean = [107.0805491, 68.26778312, 133.50751215]  # MONO2
        conf.depth_std = [38.65614623, 73.59464917, 88.24401221]

    conf.feat_stride = 16
    
    conf.has_3d = True

    # ----------------------------------------
    #  image sampling and datasets
    # ----------------------------------------
    
    # datasets
    conf.dataset_test = 'kitti_split1'
    conf.datasets_train = [{'name': 'kitti_split1', 'anno_fmt': 'kitti_det', 'im_ext': '.png', 'scale': 1}]
    conf.use_3d_for_2d = True
    
    # percent expected height ranges based on test_scale
    # used for anchor selection 
    conf.percent_anc_h = [0.0625, 0.75]
    
    # labels settings
    conf.min_gt_h = conf.test_scale*conf.percent_anc_h[0]
    conf.max_gt_h = conf.test_scale*conf.percent_anc_h[1]
    conf.min_gt_vis = 0.65
    conf.ilbls = ['Van', 'ignore']
    conf.lbls = ['Car', 'Pedestrian', 'Cyclist']
    
    # ----------------------------------------
    #  detection sampling
    # ----------------------------------------
    
    # detection sampling

    conf.fg_image_ratio = 1.0
    conf.box_samples = 0.20
    conf.fg_fraction = 0.20
    conf.bg_thresh_lo = 0
    conf.bg_thresh_hi = 0.5
    conf.fg_thresh = 0.5
    conf.ign_thresh = 0.5
    conf.best_thresh = 0.35

    # ----------------------------------------
    #  inference and testing
    # ----------------------------------------

    # nms
    conf.nms_topN_pre = 3000
    conf.nms_topN_post = 40
    conf.nms_thres = 0.4
    conf.clip_boxes = False

    conf.test_protocol = 'kitti'
    conf.test_db = 'kitti'
    conf.test_min_h = 0
    conf.min_det_scales = [0, 0]

    # ----------------------------------------
    #  anchor settings
    # ----------------------------------------
    
    # clustering settings
    conf.cluster_anchors = 0
    conf.even_anchors = 0
    conf.expand_anchors = 0
                             
    conf.anchors = [
                            [-0.5, -8.5, 15.5, 23.5, 51.969, 0.531,
                             1.713, 1.025, -0.799],
                            [-8.5, -8.5, 23.5, 23.5, 52.176, 1.618,
                             1.6, 3.811, -0.453],
                            [-16.5, -8.5, 31.5, 23.5, 48.334,
                             1.644, 1.529, 3.966, 0.673],
                            [-2.528, -12.555, 17.528, 27.555,
                             44.781, 0.534, 1.771, 0.971, 0.093],
                            [-12.555, -12.555, 27.555, 27.555,
                             44.704, 1.599, 1.569, 3.814, -0.187],
                            [-22.583, -12.555, 37.583, 27.555,
                             43.492, 1.621, 1.536, 3.91, 0.719],
                            [-5.069, -17.638, 20.069, 32.638,
                             34.666, 0.561, 1.752, 0.967, -0.384],
                            [-17.638, -17.638, 32.638, 32.638,
                             35.35, 1.567, 1.591, 3.81, -0.511],
                            [-30.207, -17.638, 45.207, 32.638,
                             37.128, 1.602, 1.529, 3.904, 0.452],
                            [-8.255, -24.01, 23.255, 39.01, 28.771,
                             0.613, 1.76, 0.98, 0.067],
                            [-24.01, -24.01, 39.01, 39.01, 28.331,
                             1.543, 1.592, 3.66, -0.811],
                            [-39.764, -24.01, 54.764, 39.01,
                             30.541, 1.626, 1.524, 3.908, 0.312],
                            [-12.248, -31.996, 27.248, 46.996,
                             23.011, 0.606, 1.758, 0.996, 0.208],
                            [-31.996, -31.996, 46.996, 46.996,
                             22.948, 1.51, 1.599, 3.419, -1.076],
                            [-51.744, -31.996, 66.744, 46.996,
                             25.0, 1.628, 1.527, 3.917, 0.334],
                            [-17.253, -42.006, 32.253, 57.006,
                             18.479, 0.601, 1.747, 1.007, 0.347],
                            [-42.006, -42.006, 57.006, 57.006,
                             18.815, 1.487, 1.599, 3.337, -0.862],
                            [-66.759, -42.006, 81.759, 57.006,
                             20.576, 1.623, 1.532, 3.942, 0.323],
                            [-23.527, -54.553, 38.527, 69.553,
                             15.035, 0.625, 1.744, 0.917, 0.41],
                            [-54.553, -54.553, 69.553, 69.553,
                             15.346, 1.29, 1.659, 3.083, -0.275],
                            [-85.58, -54.553, 100.58, 69.553,
                             16.326, 1.613, 1.527, 3.934, 0.268],
                            [-31.39, -70.281, 46.39, 85.281,
                             12.265, 0.631, 1.747, 0.954, 0.317],
                            [-70.281, -70.281, 85.281, 85.281,
                             11.878, 1.044, 1.67, 2.415, -0.211],
                            [-109.171, -70.281, 124.171, 85.281,
                             13.58, 1.621, 1.539, 3.961, 0.189],
                            [-41.247, -89.994, 56.247, 104.994,
                             9.932, 0.61, 1.771, 0.934, 0.486],
                            [-89.994, -89.994, 104.994, 104.994,
                             8.949, 0.811, 1.766, 1.662, 0.08],
                            [-138.741, -89.994, 153.741, 104.994,
                             11.043, 1.61, 1.533, 3.899, 0.04],
                            [-53.602, -114.704, 68.602, 129.704,
                             8.389, 0.604, 1.793, 0.95, 0.806],
                            [-114.704, -114.704, 129.704, 129.704,
                             8.071, 1.01, 1.751, 2.19, -0.076],
                            [-175.806, -114.704, 190.806, 129.704,
                             9.184, 1.606, 1.526, 3.869, -0.066],
                            [-69.089, -145.677, 84.089, 160.677,
                             6.923, 0.627, 1.791, 0.96, 0.784],
                            [-145.677, -145.677, 160.677, 160.677,
                             6.784, 1.384, 1.615, 2.862, -1.035],
                            [-222.266, -145.677, 237.266, 160.677,
                             7.863, 1.617, 1.55, 3.948, -0.071],
                            [-88.5, -184.5, 103.5, 199.5, 5.189,
                             0.66, 1.755, 0.841, 0.173],
                            [-184.5, -184.5, 199.5, 199.5, 4.388,
                             0.743, 1.728, 1.381, 0.642],
                            [-280.5, -184.5, 295.5, 199.5, 5.583,
                             1.583, 1.547, 3.862, -0.072]]

    conf.bbox_means = [[-0.0, 0.002, 0.064, -0.093, 0.011,
                         -0.067, 0.192, 0.059, -0.021, 0.069,
                         -0.004]]
    conf.bbox_stds = [[0.14, 0.126, 0.247, 0.239, 0.163,
                     0.132, 3.621, 0.382, 0.102, 0.503,
                     1.855]]
    
    # initialize anchors
    base = (conf.max_gt_h / conf.min_gt_h) ** (1 / (12 - 1))
    conf.anchor_scales = np.array([conf.min_gt_h * (base ** i) for i in range(0, 12)])
    conf.anchor_ratios = np.array([0.5, 1.0, 1.5])
    
    # loss logic
    conf.hard_negatives = True
    conf.focal_loss = 1
    conf.cls_2d_lambda = 1
    conf.iou_2d_lambda = 0
    conf.bbox_2d_lambda = 1
    conf.bbox_3d_lambda = 1
    conf.bbox_3d_proj_lambda = 0.0
    
    conf.hill_climbing = True
    
    # visdom
    conf.visdom_port = 9891

    return conf