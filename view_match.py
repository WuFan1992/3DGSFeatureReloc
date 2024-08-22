import argparse, os, imageio, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import pickle

from utils.visual_common import MultiFigure

parser = argparse.ArgumentParser(
    description='Script for viewing the keypoints.h5 and matches.h5 contents',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--pkl_path', type=str, help='Path to pkl file where save the matching points')
parser.add_argument("--img_query_path", type=str, default = None)
parser.add_argument("--img_ref_path", type=str, default = None)
#parser.add_argument('image_path', help='Path to corresponding images')
#parser.add_argument(
#    '--image-extension', default='jpg', type=str,
#    help='Extension of the images'
#)
parser.add_argument(
    '--save', default=None, type=str,
    help=('If give a path, saves the visualizations rather than displaying '
          'them interactively')
)
#parser.add_argument(
#    'mode', choices=['keypoints', 'matches'],
#    help=('Whether to dispay the keypoints (in a single image) or matches '
#          '(across pairs)')
#)

args = parser.parse_args()

save_i = 1
def show_or_save():
    global save_i

    if args.save is None:
        plt.show()
        return
    else:
        path = os.path.join(os.path.expanduser(args.save), f'{save_i}.png')
        plt.savefig(path)
        print(f'Saved to {path}')
        save_i += 1
        plt.close()
"""
def view_keypoints(h5_path, image_path):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]

        fname_with_ext = filename + '.' + args.image_extension
        path = os.path.join(image_path, fname_with_ext)
        if not os.path.isfile(path):
            raise IOError(f'Invalid image path {path}')

        image = imageio.imread(path)
        scale = 10 / max(image.shape)
        fig, ax = plt.subplots(figsize=(scale * image.shape[1], scale * image.shape[0]), constrained_layout=True)
        ax.axis('off')
        ax.imshow(image)
        ax.scatter(keypoints[:, 0], keypoints[:, 1], s=7, marker='o', color='white', edgecolors='black', linewidths=0.5)

        show_or_save()
"""

def view_matches(pkl_path, img_query_path, img_ref_path):

    with open(pkl_path, 'rb') as f:
        match_p = pickle.load(f)


    bm_1 = torch.from_numpy(imageio.imread(img_query_path))
    bm_2 = torch.from_numpy(imageio.imread(img_ref_path))

    bigger_x = max(bm_1.shape[0], bm_2.shape[0])
    bigger_y = max(bm_1.shape[1], bm_2.shape[1])

    padded_1 = F.pad(bm_1, (
                0, 0,
                0, bigger_y - bm_1.shape[1],
                0, bigger_x - bm_1.shape[0]
            ))
    padded_2 = F.pad(bm_2, (
                0, 0,
                0, bigger_y - bm_2.shape[1],
                0, bigger_x - bm_2.shape[0]
            ))
    
    fig = MultiFigure(padded_1, padded_2)

    ref_p = torch.tensor([list(e) for e in match_p["ref"] if e is not None])
    query_p = torch.tensor([list(e) for e in match_p["query"] if e is not None])

    fig.mark_xy(ref_p, query_p)

    show_or_save()

#if args.mode == 'keypoints':
#    pass
#    #view_keypoints(args.h5_path, args.image_path)
#elif args.mode == 'matches':
view_matches(args.pkl_path, args.img_query_path, args.img_ref_path)
