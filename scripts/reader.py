from PIL import Image
import pickle

from lib.rpn_util import *
from lib.augmentations import *
from scripts.config.depth_guided_config import *
from copy import deepcopy

logger = logging.getLogger(__name__)
python_ver = sys.version_info

class Transform(object):
    def __init__(self, height=375, length=1242):
        self.height = height
        self.length = length

    def __call__(self, rgb, depth):
        rgb = cv2.resize(rgb, (self.height, self.length), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.height, self.length), interpolation=cv2.INTER_NEAREST)

        return rgb, depth

class Reader(object):

    def __init__(self,
                 conf,
                 folder,
                 list_file,
                 cache_folder=None,
                 transform=None,
                 shuffle=True):

        self.conf = conf
        self.folder = folder
        self.list_file = list_file
        self.cache_folder = cache_folder
        self.shuffle = shuffle
        self.data_list = self.read_list()
        self.batch_size = conf.batch_size
        self.transform = transform
        # if self.transform is None:
        #     self.transform = Transform(conf.crop_size[0], conf.crop_size[1])

        self.video_det = False if not ('video_det' in conf) else conf.video_det
        self.video_count = 1 if not ('video_count' in conf) else conf.video_count
        self.use_3d_for_2d = ('use_3d_for_2d' in conf) and conf.use_3d_for_2d
        self.use_seg = conf.use_seg
        self.use_rcnn_pretrain = conf.use_rcnn_pretrain
        self.depth_channel = conf.depth_channel

        # store more information
        self.datasets_train = conf.datasets_train
        self.len = len(self.data_list)
        self.imdb = []

    def read_list(self):
        data_list = []
        with open(self.list_file) as infile:
            for line in infile:
                data_list.append(line.split()[0])
        return data_list

    def preprocess(self, rgb, depth):
        h, w, c = rgb.shape
        h_gt, w_gt, c = depth.shape
        assert h == h_gt, "Error"
        assert w == w_gt, "Error"

        trans = Transform()
        rgb, depth = trans(rgb, depth)

        # depth = depth[:, :, np.newaxis]
        return rgb, depth


    def create_reader(self):

        labelreader = self._reader_creator()

        def _batch_reader():
            batch_img = []
            batch_depth = []
            batch_label = []

            for index, imobj in enumerate(labelreader):
                if imobj is None:
                    continue

                if not self.video_det: # self.video_det = False
                    # read image
                    rgb = cv2.imread(imobj.path, cv2.IMREAD_COLOR).astype(np.float32)
                    if not self.use_seg:    # use_seg = False
                        if self.depth_channel == 3:
                            # baseline baseline baseline
                            depth = cv2.imread(imobj.path_depth, cv2.IMREAD_COLOR).astype(np.float32)
                        else:
                            depth = cv2.imread(imobj.path_depth, cv2.IMREAD_UNCHANGED).astype(np.float32)
                            depth = depth[:, :, np.newaxis]
                            depth = np.tile(depth, (1, 1, 3))
                    else:
                        depth = cv2.imread(imobj.path_depth, cv2.IMREAD_UNCHANGED).astype(np.float32)
                        seg = cv2.imread(imobj.segpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                        depth = depth[:, :, np.newaxis]
                        seg = seg[:, :, np.newaxis]
                        depth = np.tile(depth, (1, 1, 2))
                        depth = np.concatenate((depth, seg), axis=2)
                else:

                    # read images
                    im = cv2.imread(imobj.path)

                    video_count = 1 if self.video_count is None else self.video_count

                    if video_count >= 2:
                        im_pre = cv2.imread(imobj.path_pre)

                        if not im_pre.shape == im.shape:
                            im_pre = cv2.resize(im_pre, (im.shape[1], im.shape[0]))

                        im = np.concatenate((im, im_pre), axis=2)

                    if video_count >= 3:

                        im_pre2 = cv2.imread(imobj.path_pre2)

                        if im_pre2 is None:
                            im_pre2 = im_pre

                        if not im_pre2.shape == im.shape:
                            im_pre2 = cv2.resize(im_pre2, (im.shape[1], im.shape[0]))

                        im = np.concatenate((im, im_pre2), axis=2)

                    if video_count >= 4:

                        im_pre3 = cv2.imread(imobj.path_pre3)

                        if im_pre3 is None:
                            im_pre3 = im_pre2

                        if not im_pre3.shape == im.shape:
                            im_pre3 = cv2.resize(im_pre3, (im.shape[1], im.shape[0]))

                        im = np.concatenate((im, im_pre3), axis=2)

                if not self.use_rcnn_pretrain:
                    for i in range(int(rgb.shape[2] / 3)):
                        # convert to RGB then permute to be [B C H W]
                        rgb[:, :, (i * 3):(i * 3) + 3] = rgb[:, :, (i * 3 + 2, i * 3 + 1, i * 3)]

                rgb, depth, label = self.transform(rgb, depth, deepcopy(imobj))

                rgb = np.transpose(rgb, [2, 0, 1])
                depth = np.transpose(depth, [2, 0, 1])

                batch_img.append(rgb)
                batch_depth.append(depth)
                batch_label.append(label)

                if len(batch_img) == self.batch_size:
                    yield batch_img, batch_depth, batch_label
                    batch_img, batch_depth, batch_label = [], [], []

        return _batch_reader

    def _reader_creator(self):
        # use cache?
        if (self.cache_folder is not None) and os.path.exists(os.path.join(self.cache_folder, 'imdb.pkl')):
            logging.info('Preloading imdb from {}'.format(os.path.join(self.cache_folder, 'imdb.pkl')))
            with open(os.path.join(self.cache_folder, 'imdb.pkl'), 'rb') as file:
                imdb = pickle.load(file)

        else:
            imdb = []
            # cycle through each dataset
            for dbind, db in enumerate(self.conf.datasets_train):

                logging.info('Loading imdb {}'.format(db['name']))

                # single imdb
                imdb_single_db = []

                # kitti formatting
                if db['anno_fmt'].lower() == 'kitti_det':

                    for id in self.data_list:

                        # TODO the path problem in win '/', in os.join '\'
                        impath = os.path.join(self.folder, 'image_2', id + '.png')
                        depthpath = os.path.join(self.folder, 'depth_2', id + '.png')
                        calpath = os.path.join(self.folder, 'calib', id + '.txt')
                        labelpath = os.path.join(self.folder, 'label_2', id + '.txt')
                        impath_pre = os.path.join(self.folder, 'prev_2', id + '_01' + '.png')
                        impath_pre2 = os.path.join(self.folder, 'prev_2', id + '_02' + '.png')
                        impath_pre3 = os.path.join(self.folder, 'prev_2', id + '_03' + '.png')
                        segpath = os.path.join(self.folder, 'seg', id + '.png')

                        p2 = read_kitti_cal(calpath)
                        p2_inv = np.linalg.inv(p2)
                        gts = read_kitti_label(labelpath, p2, self.use_3d_for_2d)

                        obj = edict()
                        # store gts
                        obj.id = id
                        obj.p2 = p2
                        obj.p2_inv = p2_inv
                        obj.gts = gts

                        # im properties
                        im = Image.open(impath)
                        obj.path = impath
                        obj.path_depth = depthpath
                        obj.path_seg = segpath
                        obj.path_pre = impath_pre
                        obj.path_pre2 = impath_pre2
                        obj.path_pre3 = impath_pre3
                        obj.imW, obj.imH = im.size

                        imdb_single_db.append(obj)

                imdb += imdb_single_db

            imdb = np.array(imdb)

        self.imdb = imdb

        # cache off the imdb?
        if self.cache_folder is not None:
            if not os.path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)
            with open(os.path.join(self.cache_folder, 'imdb.pkl'), 'wb') as file:
                pickle.dump(imdb, file)

        return imdb

def read_kitti_cal(calfile):
    """
    Reads the kitti calibration projection matrix (p2) file from disc.

    Args:
        calfile (str): path to single calibration file
    """

    text_file = open(calfile, 'r')

    p2pat = re.compile(('(P2:)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                        '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    for line in text_file:

        parsed = p2pat.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed.group(2)
            p2[0, 1] = parsed.group(3)
            p2[0, 2] = parsed.group(4)
            p2[0, 3] = parsed.group(5)
            p2[1, 0] = parsed.group(6)
            p2[1, 1] = parsed.group(7)
            p2[1, 2] = parsed.group(8)
            p2[1, 3] = parsed.group(9)
            p2[2, 0] = parsed.group(10)
            p2[2, 1] = parsed.group(11)
            p2[2, 2] = parsed.group(12)
            p2[2, 3] = parsed.group(13)

            p2[3, 3] = 1

    text_file.close()

    return p2


def read_kitti_poses(posefile):
    text_file = open(posefile, 'r')

    ppat1 = re.compile(('(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                        '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    ppat2 = re.compile(('(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                        '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*'));

    ps = []

    for line in text_file:

        parsed1 = ppat1.fullmatch(line)
        parsed2 = ppat2.fullmatch(line)

        if parsed1 is not None:
            p = np.zeros([4, 4], dtype=float)
            p[0, 0] = parsed1.group(1)
            p[0, 1] = parsed1.group(2)
            p[0, 2] = parsed1.group(3)
            p[0, 3] = parsed1.group(4)
            p[1, 0] = parsed1.group(5)
            p[1, 1] = parsed1.group(6)
            p[1, 2] = parsed1.group(7)
            p[1, 3] = parsed1.group(8)
            p[2, 0] = parsed1.group(9)
            p[2, 1] = parsed1.group(10)
            p[2, 2] = parsed1.group(11)
            p[2, 3] = parsed1.group(12)

            p[3, 3] = 1

            ps.append(p)

        elif parsed2 is not None:

            p = np.zeros([4, 4], dtype=float)
            p[0, 0] = parsed2.group(1)
            p[0, 1] = parsed2.group(2)
            p[0, 2] = parsed2.group(3)
            p[0, 3] = parsed2.group(4)
            p[1, 0] = parsed2.group(5)
            p[1, 1] = parsed2.group(6)
            p[1, 2] = parsed2.group(7)
            p[1, 3] = parsed2.group(8)
            p[2, 0] = parsed2.group(9)
            p[2, 1] = parsed2.group(10)
            p[2, 2] = parsed2.group(11)
            p[2, 3] = parsed2.group(12)

            p[3, 3] = 1

            ps.append(p)

    text_file.close()

    return ps


def read_kitti_label(file, p2, use_3d_for_2d=False):
    """
    Reads the kitti label file from disc.

    Args:
        file (str): path to single label file for an image
        p2 (ndarray): projection matrix for the given image
    """

    gts = []

    text_file = open(file, 'r')

    '''
     Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.
    '''

    pattern = re.compile(('([a-zA-Z\-\?\_]+)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+'
                          + '(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s*((fpat)?)\n')
                         .replace('fpat', '[-+]?\d*\.\d+|[-+]?\d+'))

    for line in text_file:

        parsed = pattern.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:

            obj = edict()

            ign = False

            cls = parsed.group(1)  # type
            trunc = float(parsed.group(2))
            occ = float(parsed.group(3))
            alpha = float(parsed.group(4))

            x = float(parsed.group(5))  # left
            y = float(parsed.group(6))  # top
            x2 = float(parsed.group(7))  # right
            y2 = float(parsed.group(8))  # bottom

            width = x2 - x + 1
            height = y2 - y + 1

            h3d = float(parsed.group(9))
            w3d = float(parsed.group(10))
            l3d = float(parsed.group(11))

            cx3d = float(parsed.group(12))  # center of car in 3d
            cy3d = float(parsed.group(13))  # bottom of car in 3d
            cz3d = float(parsed.group(14))  # center of car in 3d
            rotY = float(parsed.group(15))

            # actually center the box
            cy3d -= (h3d / 2)

            elevation = (1.65 - cy3d)  # height above sea level

            if use_3d_for_2d and h3d > 0 and w3d > 0 and l3d > 0:

                # re-compute the 2D box using 3D (finally, avoids clipped boxes)
                verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

                # any boxes behind camera plane?
                if np.any(corners_3d[2, :] <= 0):
                    ign = True

                else:  # 3d for 2d
                    x = min(verts3d[:, 0])
                    y = min(verts3d[:, 1])
                    x2 = max(verts3d[:, 0])
                    y2 = max(verts3d[:, 1])

                    width = x2 - x + 1
                    height = y2 - y + 1

            else:
                verts3d, corners_3d = np.zeros((8, 2)), np.zeros((3, 8))

            # project cx, cy, cz
            coord3d = p2.dot(np.array([cx3d, cy3d, cz3d, 1]))

            # store the projected instead
            cx3d_2d = coord3d[0]
            cy3d_2d = coord3d[1]
            cz3d_2d = coord3d[2]  # TODO: depth?

            # 3d center to 2d, image coordinate
            cx = cx3d_2d / cz3d_2d
            cy = cy3d_2d / cz3d_2d

            # encode occlusion with range estimation
            # 0 = fully visible, 1 = partly occluded
            # 2 = largely occluded, 3 = unknown
            if occ == 0:
                vis = 1
            elif occ == 1:
                vis = 0.66
            elif occ == 2:
                vis = 0.33
            else:
                vis = 0.0

            while rotY > math.pi: rotY -= math.pi * 2
            while rotY < (-math.pi): rotY += math.pi * 2

            # recompute alpha
            alpha = convertRot2Alpha(rotY, cz3d, cx3d)  # TODO: why don't use alpha in Kitti directly?

            obj.elevation = elevation
            obj.cls = cls
            obj.occ = occ > 0
            obj.ign = ign
            obj.visibility = vis
            obj.trunc = trunc
            obj.alpha = alpha
            obj.rotY = rotY

            # is there an extra field? (assume to be track)
            if len(parsed.groups()) >= 16 and parsed.group(16).isdigit(): obj.track = int(parsed.group(16))

            obj.bbox_full = np.array([x, y, width, height])
            obj.bbox_3d = [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY]
            # 2d center, depth, 3d shape, alpha, 3d center, rY
            obj.center_3d = [cx3d, cy3d, cz3d]
            # print(verts3d[:8], corners_3d)
            # 8 * 2 x, y
            # [[716.2700834 144.0556177]
            # [820.29305993 144.00207322]
            # [820.29305993 307.58688203]
            # [808.68674867 300.53454034]
            # [808.68674867 146.02789809]
            # [710.44462716 146.07566844]
            # [710.44462716 300.36824124]
            # [716.2700834  307.40048192]]

            # 3 * 8, x, y (height), z (depth)
            # [[1.23763004  2.43757004  2.43757004  2.44236996  2.44236996  1.24242996 1.24242996  1.23763004]
            # [-0.42   -0.42   1.47   1.47   -0.42   -0.42   1.47   1.47]
            # [8.1760119 8.1640121 8.1640121 8.6439881 8.6439881 8.6559879 8.6559879 8.1760119]]
            obj.vertices = verts3d[:8].T.flatten()
            obj.corners_3d = corners_3d.flatten()

            gts.append(obj)

    text_file.close()

    return gts


def balance_samples(conf, imdb):
    """
    Balances the samples in an image dataset according to the given configuration.
    Basically we check which images have relevant foreground samples and which are empty,
    then we compute the sampling weights according to a desired fg_image_ratio.

    This is primarily useful in datasets which have a lot of empty (background) images, which may
    cause instability during training if not properly balanced against.
    """

    sample_weights = np.ones(len(imdb))

    if conf.fg_image_ratio >= 0:

        empty_inds = []
        valid_inds = []

        for imind, imobj in enumerate(imdb):

            valid = 0

            scale = conf.test_scale / imobj.imH
            igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                           conf.min_gt_h, conf.max_gt_h, scale)

            for gtind, gt in enumerate(imobj.gts):

                if (not igns[gtind]) and (not rmvs[gtind]):
                    valid += 1

            sample_weights[imind] = valid

            if valid > 0:
                valid_inds.append(imind)
            else:
                empty_inds.append(imind)

        if not (conf.fg_image_ratio == 2):
            fg_weight = len(imdb) * conf.fg_image_ratio / len(valid_inds)
            bg_weight = len(imdb) * (1 - conf.fg_image_ratio) / len(empty_inds)
            sample_weights[valid_inds] = fg_weight
            sample_weights[empty_inds] = bg_weight

            logging.info('weighted respectively as {:.2f} and {:.2f}'.format(fg_weight, bg_weight))

        logging.info(
            'Found {} foreground and {} empty images'.format(np.sum(sample_weights > 0), np.sum(sample_weights <= 0)))

    # force sampling weights to sum to 1
    sample_weights /= np.sum(sample_weights)

    return sample_weights

if __name__ == '__main__':
    train_reader = Reader(
            conf=Config(),
            folder="data/",
            list_file="../data/train.txt",
            cache_folder='output',
            transform=Augmentation(Config()),
            shuffle=True
        ).create_reader()

    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        for i in range(2):
            print(i)
            for batch_id, (img, label) in enumerate(train_reader()):
                rgb, depth = np.array([(x[0], x[1]) for x in img])

                rgb = fluid.dygraph.to_variable(rgb)
                depth = fluid.dygraph.to_variable(depth)

                out = fluid.layers.relu(rgb)
                print(out.shape)