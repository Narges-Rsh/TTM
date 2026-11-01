import json
import math
import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import normalize_pose
from utils.pose_utils import gen_clip_seg_data_np, get_ab_labels
from torch.utils.data import DataLoader
#import cv2
from tsmoothie.smoother import *

SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]

# ============================================================
# Fixed-length TTM person-sequence dataset
# ============================================================

class PersonSequenceDataset(Dataset):
    """
    Wrapper around the existing PoseSegDataset.

    Existing PoseSegDataset item:
        [C, W, V]

    We first group windows by tracked person and chronological
    order, then split every person into fixed-length TTM chunks.

        pose:
            [sequence_length, C, W, V]

        valid_mask:
            [sequence_length]

    """

    def __init__(
        self,
        base_dataset,
        sequence_length=16,
    ):
        super().__init__()

        self.base_dataset = base_dataset

        self.sequence_length = int(
            sequence_length
        )

        if self.sequence_length < 1:

            raise ValueError(
                "sequence_length must be >= 1."
            )

        # ----------------------------------------------------
        # Do not allow augmentation copies to interfere with
        # mapping back to original segment indices.
        # ----------------------------------------------------

        if getattr(
            base_dataset,
            "num_transform",
            1,
        ) != 1:

            raise ValueError(
                "PersonSequenceDataset currently expects "
                "num_transform == 1."
            )

        # ====================================================
        # STEP 1
        #
        # Group original 24-frame segments by tracked person:
        #
        # (scene_id, clip_id, person_id)
        # ====================================================

        groups = {}

        for segment_index, meta in enumerate(
            base_dataset.metadata
        ):

            scene_id = meta[0]
            clip_id = meta[1]
            person_id = meta[2]

            start_frame = int(
                float(
                    meta[3]
                )
            )

            track_key = (
                str(scene_id),
                str(clip_id),
                str(person_id),
            )

            if track_key not in groups:

                groups[
                    track_key
                ] = []

            groups[
                track_key
            ].append(
                (
                    start_frame,
                    segment_index,
                )
            )

        # ====================================================
        # STEP 2
        #
        # Build complete person tracks first.
        #
        # We keep this information so we still know the
        # ORIGINAL number of windows for each person.
        # ====================================================

        self.tracks = []

        for track_key, windows in groups.items():

            # Chronological order
            windows = sorted(
                windows,
                key=lambda x: x[0],
            )

            start_frames = [
                item[0]
                for item in windows
            ]

            segment_indices = [
                item[1]
                for item in windows
            ]

            self.tracks.append(
                {
                    "track_key":
                        track_key,

                    "segment_indices":
                        segment_indices,

                    "start_frames":
                        start_frames,

                    "original_num_steps":
                        len(
                            segment_indices
                        ),
                }
            )

        # ----------------------------------------------------
        # Deterministic ordering of person tracks
        # ----------------------------------------------------

        self.tracks = sorted(
            self.tracks,
            key=lambda x: (
                x["track_key"][0],
                x["track_key"][1],
                x["track_key"][2],
            ),
        )

        # ====================================================
        # STEP 3
        #
        # Split every person track into fixed-size chunks.
        #
        # No windows are discarded.
        # ====================================================

        self.chunks = []

        for track in self.tracks:

            segment_indices = track[
                "segment_indices"
            ]

            start_frames = track[
                "start_frames"
            ]

            original_num_steps = len(
                segment_indices
            )

            chunk_index = 0

            for chunk_start in range(
                0,
                original_num_steps,
                self.sequence_length,
            ):

                chunk_end = min(
                    chunk_start
                    +
                    self.sequence_length,

                    original_num_steps,
                )

                chunk_segment_indices = (
                    segment_indices[
                        chunk_start:
                        chunk_end
                    ]
                )

                chunk_start_frames = (
                    start_frames[
                        chunk_start:
                        chunk_end
                    ]
                )

                valid_length = len(
                    chunk_segment_indices
                )

                self.chunks.append(
                    {
                        "track_key":
                            track[
                                "track_key"
                            ],

                        "segment_indices":
                            chunk_segment_indices,

                        "start_frames":
                            chunk_start_frames,

                        # Number of REAL windows
                        # in this chunk.
                        "num_steps":
                            valid_length,

                        # Number of windows in the
                        # original complete person track.
                        "original_num_steps":
                            original_num_steps,

                        # Which fixed chunk of this person?
                        "chunk_index":
                            chunk_index,

                        # Position in original person sequence.
                        "chunk_start_step":
                            chunk_start,
                    }
                )

                chunk_index += 1

        # ====================================================
        # Metadata needed later by evaluation
        # ====================================================

        self.metadata = (
            self.base_dataset.metadata
        )

        # ====================================================
        # Dataset statistics
        # ====================================================

        total_valid_windows = sum(

            chunk["num_steps"]

            for chunk
            in self.chunks
        )

        total_capacity = (
            len(self.chunks)
            *
            self.sequence_length
        )

        total_padding = (
            total_capacity
            -
            total_valid_windows
        )

        print()
        print("=" * 70)
        print("FIXED-LENGTH TTM DATASET")
        print("=" * 70)

        print(
            "Original 24-frame segments:",
            len(
                self.base_dataset.metadata
            )
        )

        print(
            "Original tracked-person sequences:",
            len(
                self.tracks
            )
        )

        print(
            "Fixed sequence length:",
            self.sequence_length
        )

        print(
            "Generated fixed chunks:",
            len(
                self.chunks
            )
        )

        print(
            "Valid windows:",
            total_valid_windows
        )

        print(
            "Padding windows:",
            total_padding
        )

        if total_capacity > 0:

            print(
                "Padding percentage:",
                (
                    100.0
                    *
                    total_padding
                    /
                    total_capacity
                )
            )


    # ========================================================
    # Length
    # ========================================================

    def __len__(
        self,
    ):

        # One dataset item = one fixed-length chunk

        return len(
            self.chunks
        )


    # ========================================================
    # Get one fixed-length chunk
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        chunk = self.chunks[
            index
        ]

        segment_indices = chunk[
            "segment_indices"
        ]

        valid_length = chunk[
            "num_steps"
        ]

        pose_segments = []


        for segment_index in segment_indices:

            original_item = (
                self.base_dataset[
                    segment_index
                ]
            )


            pose = original_item[
                0
            ]

            pose = torch.as_tensor(
                pose,
                dtype=torch.float32,
            )

            pose_segments.append(
                pose
            )


        pose_sequence = torch.stack(
            pose_segments,
            dim=0,
        )

        # ====================================================
        # Add padding at END
       
        padding_length = (
            self.sequence_length
            -
            valid_length
        )

        if padding_length > 0:

            padding = torch.zeros(

                (
                    padding_length,
                    pose_sequence.shape[1],
                    pose_sequence.shape[2],
                    pose_sequence.shape[3],
                ),

                dtype=
                    pose_sequence.dtype,
            )

            pose_sequence = torch.cat(

                [
                    pose_sequence,
                    padding,
                ],

                dim=0,
            )

        
        # Validity mask

        valid_mask = torch.zeros(

            self.sequence_length,

            dtype=torch.bool,
        )

        valid_mask[
            :valid_length
        ] = True

        # ====================================================
        # Padded segment indices
        #
        # -1 means this position is padding.
        # ====================================================

        padded_segment_indices = torch.full(

            (
                self.sequence_length,
            ),

            fill_value=-1,

            dtype=torch.long,
        )

        padded_segment_indices[
            :valid_length
        ] = torch.tensor(

            segment_indices,

            dtype=torch.long,
        )

        padded_start_frames = torch.full(

            (
                self.sequence_length,
            ),

            fill_value=-1,

            dtype=torch.long,
        )

        padded_start_frames[
            :valid_length
        ] = torch.tensor(

            chunk[
                "start_frames"
            ],

            dtype=torch.long,
        )

        if (
            pose_sequence.shape[0]
            !=
            self.sequence_length
        ):

            raise RuntimeError(
                "Fixed sequence construction failed. "
                f"Expected {self.sequence_length} steps, "
                f"received {pose_sequence.shape[0]}."
            )


        return {

        
            "pose":
                pose_sequence,


            "valid_mask":
                valid_mask,

            # ------------------------------------------------
            # Number of REAL windows in this chunk.
            #
            # Between 1 and 16.
            # ------------------------------------------------

            "num_steps":
                valid_length,

            # ------------------------------------------------
            # Number of windows in complete original track.
            # ------------------------------------------------

            "original_num_steps":
                chunk[
                    "original_num_steps"
                ],

            # ------------------------------------------------
            # Person identity
            # ------------------------------------------------

            "track_key":
                chunk[
                    "track_key"
                ],

            # ------------------------------------------------
            # Which chunk of this person?
            #
            # 0,1,2,...
            # ------------------------------------------------

            "chunk_index":
                chunk[
                    "chunk_index"
                ],

            # ------------------------------------------------
            # Original temporal position where this chunk begins
            # ------------------------------------------------

            "chunk_start_step":
                chunk[
                    "chunk_start_step"
                ],

            # ------------------------------------------------
            # Original segment indices.
            #
            # Padding positions = -1
            #
            # [16]
            # ------------------------------------------------

            "segment_indices":
                padded_segment_indices,

            # ------------------------------------------------
            # Original frame starts.
            #
            # Padding positions = -1
            #
            # [16]
            # ------------------------------------------------

            "start_frames":
                padded_start_frames,
        }






    
class PoseSegDataset(Dataset):
    """
    Generates a dataset with two objects, a np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced seq


    If path_to_patches is provided uses pre-extracted patches. If lmdb_file or vid_dir are
    provided extracts patches from them, while hurting performance.
    """

    def __init__(self, path_to_json_dir, path_to_vid_dir=None, normalize_pose_segs=True, return_indices=False,
                 return_metadata=False, debug=False, return_global=True, evaluate=False, abnormal_train_path=None,
                 **dataset_args):
        super().__init__()
        self.args = dataset_args
        self.path_to_json = path_to_json_dir
        self.patches_db = None
        self.use_patches = False
        self.normalize_pose_segs = normalize_pose_segs
        self.headless = dataset_args.get('headless', False)
        self.path_to_vid_dir = path_to_vid_dir
        self.eval = evaluate
        self.debug = debug
        num_clips = dataset_args.get('specific_clip', None)
        self.return_indices = return_indices
        self.return_metadata = return_metadata
        self.return_global = return_global
        self.transform_list = dataset_args.get('trans_list', None)
        self.k_filter = KalmanSmoother(component='level_trend', 
                          component_noise={'level':0.1, 'trend':0.1})
        # self.k_filter = KeypointsKalmanFilter(n_keypoints=18-1)
        self.apply_k_filter = dataset_args.get('Kalman_filter', False)
        if self.transform_list is None:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(self.transform_list)
        self.train_seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        self.segs_data_np, self.segs_meta, self.person_keys, self.global_data_np, \
        self.global_data, self.segs_score_np = \
            gen_dataset(path_to_json_dir, num_clips=num_clips, ret_keys=True,
                        ret_global_data=return_global, **dataset_args)
        self.segs_meta = np.array(self.segs_meta)
        if abnormal_train_path is not None:
            self.segs_data_np_ab, self.segs_meta_ab, self.person_keys_ab, self.global_data_np_ab, \
            self.global_data_ab, self.segs_score_np_ab = \
                gen_dataset(abnormal_train_path, num_clips=num_clips, ret_keys=True,
                            ret_global_data=return_global, **dataset_args)
            self.segs_meta_ab = np.array(self.segs_meta_ab)
            ab_labels = get_ab_labels(self.segs_data_np_ab, self.segs_meta_ab, path_to_vid_dir, abnormal_train_path)
            num_normal_samp = self.segs_data_np.shape[0]
            num_abnormal_samp = (ab_labels == -1).sum()
            total_num_normal_samp = num_normal_samp + (ab_labels == 1).sum()
            print("Num of abnormal sapmles: {}  | Num of normal samples: {}  |  Precent: {}".format(
                num_abnormal_samp, total_num_normal_samp, num_abnormal_samp / total_num_normal_samp))
            self.labels = np.concatenate((np.ones(num_normal_samp), ab_labels),
                                         axis=0).astype(int)
            self.segs_data_np = np.concatenate((self.segs_data_np, self.segs_data_np_ab), axis=0)
            self.segs_meta = np.concatenate((self.segs_meta, self.segs_meta_ab), axis=0)
            self.global_data_np = np.concatenate((self.global_data_np, self.global_data_np_ab), axis=0)
            self.segs_score_np = np.concatenate(
                (self.segs_score_np, self.segs_score_np_ab), axis=0)
            self.global_data += self.global_data_ab
            self.person_keys.update(self.person_keys_ab)
        else:
            self.labels = np.ones(self.segs_data_np.shape[0])
        # Convert person keys to ints
        self.person_keys = {k: [int(float(i)) for i in v] for k, v in self.person_keys.items()}
        self.metadata = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    # def filter_data(self, data):
    #     filtered_data = np.zeros_like(data)

    #     for i in range(data.shape[0]):
    #         # Apply the Kalman Filter on the concatenated data and velocity
    #         filtered_data[i] = self.k_filter.apply(data[i])[:, :2]

    #     return np.concatenate((data[:10], filtered_data[10:]), axis=0)
    
    def visualize_segment(self, tensor, output_video_path):
        tensor = tensor[:2, :, :]
        num_frames = tensor.shape[1]
        frame_size = (856, 480)  # Adjust the frame size as needed
        fps = 24.0  # Frames per second
        limbs = [(1, 18), (18, 6), (18, 7), (6, 8),
         (7, 9), (8, 10), (9, 11), (10, 12), (18, 3),
         (3, 4), (4, 5), (3, 13), (13, 14), (14, 15), (6, 12), (7, 13)]


        # Define the codec for the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed

        # Create a VideoWriter object to write the video
        out = cv2.VideoWriter(output_video_path+".mp4", fourcc, fps, frame_size)
        tensor = np.transpose(tensor, (1, 2, 0))
        for frame in range(num_frames):
            # keypoints = tensor[:, frame, :]
            
            # keypoints = keypoints.reshape(15,2)
            keypoints = tensor[frame]
            # Create an image for visualization
            image = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255  # White background

            # Plot keypoints
            for x, y in keypoints:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red circles for keypoints

            # Plot skeletal limbs
            for limb in limbs:
                start_idx, end_idx = limb
                start_point = keypoints[start_idx - 1]
                end_point = keypoints[end_idx - 1]
                cv2.line(image, (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])), (0, 255, 0), 2)  # Green lines for limbs

            # Write the frame to the video
            out.write(image)

        # Release the VideoWriter
        out.release()
    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = math.floor(index / self.num_samples)
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
        else:
            sample_index = index
            data_transformed = np.array(self.segs_data_np[index])
            trans_index = 0  # No transformations
        if self.apply_k_filter:
         
            kps = data_transformed[:2, :, :]
            channel, window, kps_n = kps.shape
            kps = np.transpose(kps, (2, 0, 1))
            kps = np.reshape(kps, (channel*kps_n, window))
            self.k_filter.smooth(kps)
            kps = self.k_filter.smooth_data
            kps = np.reshape(kps, (kps_n, channel, window))
            kps = np.transpose(kps, (1, 2, 0))
            data_transformed [:2, :, :] = kps
            # for kp_n in range(data_transformed.shape[2]):
            #     kps =  data_transformed [:, :, kp_n]
            #     self.k_filter.smooth(kps[:2, :])
            #     kps[:2, :] = self.k_filter.smooth_data
            #     data_transformed [:, :, kp_n] = kps
           
        
        if self.normalize_pose_segs:    
            data_transformed = normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...],
                                              **self.args).squeeze(axis=0).transpose(2, 0, 1)
        ret_arr = [data_transformed, trans_index]

        ret_arr += [self.segs_score_np[sample_index]]
        ret_arr += [self.labels[sample_index]]
   
        
        if self.args['relative']:
            relative_movement = np.zeros_like(ret_arr[0])
            for i in range(1, ret_arr[0].shape[1]):
                relative_movement[:, i, :] = ret_arr[0][:, i,:] - ret_arr[0][:, 0, :]
            ret_arr[0] = np.concatenate((ret_arr[0], relative_movement), axis=2)
        return ret_arr

    def get_all_data(self, normalize_pose_segs=True):
        if normalize_pose_segs:
            segs_data_np = normalize_pose(self.segs_data_np.transpose((0, 2, 3, 1)), **self.args).transpose(
                (0, 3, 1, 2))
        else:
            segs_data_np = self.segs_data_np
        if self.num_transform == 1 or self.eval:
            return list(segs_data_np)
        return segs_data_np

    def __len__(self):
        return self.num_transform * self.num_samples


#Data_preparatino for TTM
# ============================================================
# Fixed-length TTM collate function
# ============================================================

def person_sequence_collate(
    batch,
):
    """
    Collate fixed-length TTM chunks.

    Dataset item:

        pose:
            [S_fixed,C,W,V]

        valid_mask:
            [S_fixed]

    DataLoader output:

        pose:
            [B,S_fixed,C,W,V]

        valid_mask:
            [B,S_fixed]

    Example:

        [8,16,3,24,18]

    This now allows batch_size > 1.
    """

    # ========================================================
    # Pose
    #
    # list of:
    # [16,C,W,V]
    #
    # ->
    #
    # [B,16,C,W,V]
    # ========================================================

    pose = torch.stack(

        [
            sample[
                "pose"
            ]

            for sample
            in batch
        ],

        dim=0,
    )

    # ========================================================
    # Validity masks
    #
    # list of:
    # [16]
    #
    # ->
    #
    # [B,16]
    # ========================================================

    valid_mask = torch.stack(

        [
            sample[
                "valid_mask"
            ]

            for sample
            in batch
        ],

        dim=0,
    )

    # ========================================================
    # Original segment indices
    #
    # [B,16]
    # ========================================================

    segment_indices = torch.stack(

        [
            sample[
                "segment_indices"
            ]

            for sample
            in batch
        ],

        dim=0,
    )

    # ========================================================
    # Start frames
    #
    # [B,16]
    # ========================================================

    start_frames = torch.stack(

        [
            sample[
                "start_frames"
            ]

            for sample
            in batch
        ],

        dim=0,
    )

    # ========================================================
    # Number of REAL windows in each sample
    #
    # [B]
    # ========================================================

    num_steps = torch.tensor(

        [
            sample[
                "num_steps"
            ]

            for sample
            in batch
        ],

        dtype=torch.long,
    )

    # ========================================================
    # Original complete person lengths
    #
    # [B]
    # ========================================================

    original_num_steps = torch.tensor(

        [
            sample[
                "original_num_steps"
            ]

            for sample
            in batch
        ],

        dtype=torch.long,
    )

    # ========================================================
    # Chunk indices
    #
    # [B]
    # ========================================================

    chunk_index = torch.tensor(

        [
            sample[
                "chunk_index"
            ]

            for sample
            in batch
        ],

        dtype=torch.long,
    )

    chunk_start_step = torch.tensor(

        [
            sample[
                "chunk_start_step"
            ]

            for sample
            in batch
        ],

        dtype=torch.long,
    )

    # ========================================================
    # Track keys
    #
    # Keep as Python list because they are strings.
    #
    # Length = B
    # ========================================================

    track_key = [

        sample[
            "track_key"
        ]

        for sample
        in batch
    ]

    # ========================================================
    # Return batch
    # ========================================================

    return {

        "pose":
            pose,

        "valid_mask":
            valid_mask,

        "num_steps":
            num_steps,

        "original_num_steps":
            original_num_steps,

        "track_key":
            track_key,

        "chunk_index":
            chunk_index,

        "chunk_start_step":
            chunk_start_step,

        "segment_indices":
            segment_indices,

        "start_frames":
            start_frames,
    }




# ============================================================
# TTM dataset + loader
# ============================================================

def get_ttm_dataset_and_loader(
    args,
    trans_list,
    only_test=False,
):
    """
    Build original PoseSegDataset first and then convert
    tracked people into fixed-length TTM chunks.

    Original segment:
        [C,W,V]

    Fixed TTM sample:
        [S_fixed,C,W,V]

    Example:
        [16,3,24,18]

    DataLoader:
        [B,16,3,24,18]
    """

    # ========================================================
    # Original dataset
    #
    # Preserves ALL existing preprocessing.
    # ========================================================

    base_dataset, _ = get_dataset_and_loader(

        args,

        trans_list=trans_list,

        only_test=only_test,
    )

    sequence_length = getattr(

        args,

        "ttm_sequence_length",

        8,
    )


    ttm_dataset = {}

    ttm_loader = {}

    splits = (

        ["test"]

        if only_test

        else [
            "train",
            "test",
        ]
    )

    for split in splits:

        # ====================================================
        # Create fixed-length chunk dataset
        # ====================================================

        ttm_dataset[
            split
        ] = PersonSequenceDataset(

            base_dataset[
                split
            ],

            sequence_length=
                sequence_length,
        )


        ttm_loader[
            split
        ] = DataLoader(

            ttm_dataset[
                split
            ],

            batch_size=
                args.batch_size,

            # Shuffle fixed chunks during training.
            shuffle=(
                split
                ==
                "train"
            ),

            num_workers=
                args.num_workers,

            pin_memory=
                True,

            collate_fn=
                person_sequence_collate,
        )

    # ========================================================
    # Test-only compatibility
    # ========================================================

    if only_test:

        ttm_dataset[
            "train"
        ] = None

        ttm_loader[
            "train"
        ] = None

    return (
        base_dataset,
        ttm_dataset,
        ttm_loader,
    )




def get_dataset_and_loader(args, trans_list, only_test=False):
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset_args = {'headless': args.headless, 'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale,
                    'seg_len': args.seg_len, 'return_indices': True, 'return_metadata': True, "dataset": args.dataset,
                    'train_seg_conf_th': args.train_seg_conf_th, 'specific_clip': args.specific_clip, 'relative': args.relative, 'vid_res': args.vid_res, 'Kalman_filter': args.Kalman_filter, 'traj': args.traj}
    dataset, loader = dict(), dict()
    splits = ['train', 'test'] if not only_test else ['test']
    for split in splits:
        evaluate = split == 'test'
        abnormal_train_path = args.pose_path_train_abnormal if split == 'train' else None
        normalize_pose_segs = args.global_pose_segs
        # dataset_args['trans_list'] = trans_list[:args.num_transform] if split == 'train' else None
        dataset_args['trans_list'] = None
        #dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 12 # No strides for test set   #for previous VAD models
        dataset_args['seg_stride'] = args.seg_stride 
        dataset_args['vid_path'] = args.vid_path[split]
        dataset[split] = PoseSegDataset(args.pose_path[split], path_to_vid_dir=args.vid_path[split],
                                        normalize_pose_segs=normalize_pose_segs, 
                                        evaluate=evaluate,
                                        abnormal_train_path=abnormal_train_path,
                                        **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    if only_test:
        loader['train'] = None
    return dataset, loader


def shanghaitech_hr_skip(shanghaitech_hr, scene_id, clip_id):
    if not shanghaitech_hr:
        return shanghaitech_hr
    if (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
        return True
    return False


def gen_dataset(person_json_root, num_clips=None, kp18_format=True, ret_keys=False, ret_global_data=True,
                **dataset_args):
    segs_data_np = []
    segs_score_np = []
    segs_meta = []
    global_data = []
    person_keys = dict()
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 24)
    headless = dataset_args.get('headless', False)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
    dataset = dataset_args.get('dataset', 'ShanghaiTech')

    dir_list = os.listdir(person_json_root)
    if dataset=='c1' or dataset=='c2' or dataset=='c3' or dataset=='c4' or dataset=='CHAD' or dataset=='combined' or dataset=='corridor' or dataset=='NWPUC' or dataset=='HuVAD'or dataset=='ABC' or dataset=='Avenue' or dataset=='CPCC' or dataset=='CPCC0' or dataset=='CPCC1' or dataset=='CPCC2' or dataset=='CPCC3' or dataset=='CPCC4' or dataset=='CPCC5' or dataset=='CPCC6':
        json_list = sorted([fn for fn in dir_list])
    else:
        json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')])
    if num_clips is not None:
        json_list = [json_list[num_clips]]  # For debugging purposes
    for person_dict_fn in tqdm(json_list):
        if dataset == "UBnormal":
            type, scene_id, clip_id = \
                re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*', person_dict_fn)[0]
            clip_id = type + "_" + clip_id
        else:
            if dataset == 'corridor' or dataset == 'Avenue' or dataset=='NWPUC' or dataset=='ABC' or dataset=='CHAD' or dataset=='HuVAD'or dataset=='CPCC' or dataset=='CPCC0' or dataset=='CPCC1' or dataset=='CPCC2' or dataset=='CPCC3' or dataset=='CPCC4' or dataset=='CPCC5' or dataset=='CPCC6':
                scene_id, clip_id = (person_dict_fn.split('.')[0]).split('_')[:2]
            else:
                scene_id, clip_id = person_dict_fn.split('_')[:2]
            if shanghaitech_hr_skip(dataset=="ShanghaiTech-HR", scene_id, clip_id):
                continue
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
        clip_segs_data_np, clip_segs_meta, clip_keys, single_pos_np, _, score_segs_data_np = gen_clip_seg_data_np(
            clip_dict, start_ofst,
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            ret_keys=ret_keys,
            dataset=dataset)

        _, _, _, global_data_np, global_data, _ = gen_clip_seg_data_np(clip_dict, start_ofst, 1, 1, scene_id=scene_id,
                                                                       clip_id=clip_id,
                                                                       ret_keys=ret_keys,
                                                                       global_pose_data=global_data,
                                                                       dataset=dataset)
        segs_data_np.append(clip_segs_data_np)
        segs_score_np.append(score_segs_data_np)
        segs_meta += clip_segs_meta
        person_keys = {**person_keys, **clip_keys}

    # Global data
    global_data_np = np.expand_dims(np.concatenate(global_data, axis=0), axis=1)
    segs_data_np = np.concatenate(segs_data_np, axis=0)
    segs_score_np = np.concatenate(segs_score_np, axis=0)

    # if normalize_pose_segs:
    #     segs_data_np = normalize_pose(segs_data_np, vid_res=vid_res, **dataset_args)
    #     global_data_np = normalize_pose(global_data_np, vid_res=vid_res, **dataset_args)
    #     global_data = [normalize_pose(np.expand_dims(data, axis=0), **dataset_args).squeeze() for data
    #                    in global_data]
    if kp18_format and segs_data_np.shape[-2] == 17:
        segs_data_np = keypoints17_to_coco18(segs_data_np)
        global_data_np = keypoints17_to_coco18(global_data_np)
        global_data = [keypoints17_to_coco18(data) for data in global_data]
    if headless:
        segs_data_np = segs_data_np[:, :, 5:]
        global_data_np = global_data_np[:, :, 5:]
        global_data = [data[:, 5:, :] for data in global_data]

    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
    global_data_np = np.transpose(global_data_np, (0, 3, 1, 2)).astype(np.float32)

    if seg_conf_th > 0.0:
        segs_data_np, segs_meta, segs_score_np = \
            seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th)
    if ret_global_data:
        if ret_keys:
            return segs_data_np, segs_meta, person_keys, global_data_np, global_data, segs_score_np
        else:
            return segs_data_np, segs_meta, global_data_np, global_data, segs_score_np
    if ret_keys:
        return segs_data_np, segs_meta, person_keys, segs_score_np
    else:
        return segs_data_np, segs_meta, segs_score_np


def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int_)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


def seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th=2.0):
    # seg_len = segs_data_np.shape[2]
    # conf_vals = segs_data_np[:, 2]
    # sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
    sum_confs = segs_score_np.mean(axis=1)
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    segs_score_np = segs_score_np[sum_confs > seg_conf_th]

    return seg_data_filt, seg_meta_filt, segs_score_np
