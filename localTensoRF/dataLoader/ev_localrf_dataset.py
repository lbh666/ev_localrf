# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os, logging
import random

import numpy as np
import torch
import cv2
import re
import glob
from joblib import delayed, Parallel
from torch.utils.data import Dataset
from utils.utils import decode_flow
import json

def concatenate_append(old, new):
    # new = np.concatenate(new, 0).reshape(-1, dim)
    if old is not None:
        new = np.concatenate([old, *new], 0)
    else:
        new = np.concatenate(new, 0)
    return new

def get_digit(path):
    filename = os.path.basename(path)
    name_without_extension = filename.split(".")[0]

    return int(name_without_extension)

class LocalRFDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        frames_chunk=20,
        downsampling=-1,
        load_depth=False,
        load_flow=False,
        with_preprocessed_poses=False,
        n_init_frames=7,
        subsequence=[0, -1],
        test_frame_every=10,
        frame_step=1,
        events_in_imgs = 2,
        eventdir= None
    ):
        self.root_dir = datadir
        self.split = split
        self.frames_chunk = max(frames_chunk, n_init_frames)
        self.downsampling = downsampling
        self.load_depth = load_depth
        self.load_flow = load_flow
        self.frame_step = frame_step
        self.events_in_imgs = events_in_imgs

        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, "images", "*")))
        self.image_idx = [get_digit(x) for x in self.image_paths]
        event_step = (get_digit(self.image_paths[1]) - get_digit(self.image_paths[0])) / (self.events_in_imgs + 1)
        self.event_step = int(event_step)
        tmp_ = sorted(glob.glob(os.path.join(eventdir, "*")))
        self.event_map_paths = [self.image_paths[0]]
        for path in tmp_:
            if get_digit(path) not in self.image_idx and (get_digit(path) - get_digit(self.event_map_paths[-1])) % self.event_step == 0:
                self.event_map_paths.append(path)
        self.event_map_paths = sorted(self.event_map_paths[1:])
        self.all_paths = sorted(self.event_map_paths + self.image_paths, key=get_digit)
        logger = logging.getLogger('train')
        n_events = len([x for x in self.all_paths if "events" in x])
        logger.info(f"Init Dataset split:{split}, total {len(self.all_paths)} samples, {n_events} events, {len(self.all_paths) - n_events} images")
            
        self.all_image_paths = self.image_paths

        self.test_mask = []
        self.test_paths = []
        self.event_mask = np.ones(len(self.all_paths))
        idx = 0
        for i, image_path in enumerate(self.all_paths):
            if test_frame_every > 0 and "events" not in image_path and idx % test_frame_every == 0:
                self.test_paths.append(image_path)
                self.test_mask.append(1)
            else:
                self.test_mask.append(0)
            if "events" not in image_path:
                self.event_mask[i] = 0
                idx += 1
        self.test_mask = np.array(self.test_mask)

        if split=="test":
            self.all_paths = self.test_paths
            self.frames_chunk = len(self.image_paths)
        self.num_images = len(self.all_paths)
        self.all_fbases = {os.path.splitext(os.path.basename(image_path))[0]: idx for idx, image_path in enumerate(self.all_paths)}


        self.white_bg = False

        self.near_far = [0.1, 1e3] # Dummi
        self.scene_bbox = 2 * torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        self.all_rgbs = None
        self.all_invdepths = None
        self.all_fwd_flow, self.all_fwd_mask, self.all_bwd_flow, self.all_bwd_mask = None, None, None, None
        self.all_loss_weights = None
        self.all_events = None

        self.active_frames_bounds = [0, 0]
        self.event_set = []
        self.loaded_frames = 0
        self.activate_frames(n_init_frames)


    def activate_frames(self, n_frames=1):
        self.active_frames_bounds[1] += n_frames
        self.active_frames_bounds[1] = min(
            self.active_frames_bounds[1], self.num_images
        )

        if self.active_frames_bounds[1] > self.loaded_frames:
            self.read_meta()
        



    def has_left_frames(self):
        return self.active_frames_bounds[1] < self.num_images

    def deactivate_frames(self, first_frame):
        # n_frames = first_frame - self.active_frames_bounds[0]
        n_frames = int((1 - self.event_mask[self.active_frames_bounds[0]: first_frame]).sum())
        # n_events = n_frames * self.events_in_imgs
        n_events =int(self.event_mask[self.active_frames_bounds[0]: first_frame].sum())
        self.active_frames_bounds[0] = first_frame


        self.all_rgbs = self.all_rgbs[n_frames:] 
        self.all_events = self.all_events[n_events:] 
        if self.load_depth:
            self.all_invdepths = self.all_invdepths[n_frames:]
        if self.load_flow:
            self.all_fwd_flow = self.all_fwd_flow[n_frames:]
            self.all_fwd_mask = self.all_fwd_mask[n_frames:]
            self.all_bwd_flow = self.all_bwd_flow[n_frames:]
            self.all_bwd_mask = self.all_bwd_mask[n_frames:]
            # self.all_edges = self.all_edges[n_events * self.n_px_per_frame:]
        self.all_loss_weights = self.all_loss_weights[n_frames:]



    def read_meta(self):
        def read_image(i):
            image_path = self.all_paths[i]
            # print(f'load image: {image_path}')
            motion_mask_path = os.path.join(self.root_dir, "masks", 
                f"{os.path.splitext(os.path.basename(image_path))[0]}.png")
            if not os.path.isfile(motion_mask_path):
                motion_mask_path = os.path.join(self.root_dir, "masks/all.png")


            img = cv2.imread(image_path)[..., ::-1]
            img = img.astype(np.float32) / 255
            if self.downsampling != -1:
                scale = 1 / self.downsampling
                img = cv2.resize(img, None, 
                    fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            if self.load_depth:
                invdepth_path = os.path.join(self.root_dir, "depth", 
                    f"{os.path.splitext(os.path.basename(image_path))[0]}.png")
                invdepth = cv2.imread(invdepth_path, -1).astype(np.float32)
                invdepth = cv2.resize(
                    invdepth, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)
            else:
                invdepth = None

            if self.load_flow:
                glob_idx = self.all_image_paths.index(self.all_paths[i])
                if glob_idx+1 < len(self.all_image_paths):
                    fwd_flow_path = self.all_image_paths[glob_idx+1]
                else:
                    fwd_flow_path = self.all_image_paths[0]
                if self.frame_step != 1:
                    fwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"fwd_step{self.frame_step}_{os.path.splitext(os.path.basename(fwd_flow_path))[0]}.png")
                    bwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"bwd_step{self.frame_step}_{os.path.splitext(os.path.basename(image_path))[0]}.png")
                else:
                    fwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"fwd_{os.path.splitext(os.path.basename(fwd_flow_path))[0]}.png")
                    bwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"bwd_{os.path.splitext(os.path.basename(image_path))[0]}.png")
                encoded_fwd_flow = cv2.imread(fwd_flow_path, cv2.IMREAD_UNCHANGED)
                encoded_bwd_flow = cv2.imread(bwd_flow_path, cv2.IMREAD_UNCHANGED)
                flow_scale = img.shape[0] / encoded_fwd_flow.shape[0] 
                encoded_fwd_flow = cv2.resize(
                    encoded_fwd_flow, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)
                encoded_bwd_flow = cv2.resize(
                    encoded_bwd_flow, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)            
                fwd_flow, fwd_mask = decode_flow(encoded_fwd_flow)
                bwd_flow, bwd_mask = decode_flow(encoded_bwd_flow)
                fwd_flow = fwd_flow * flow_scale
                bwd_flow = bwd_flow * flow_scale
            else:
                fwd_flow, fwd_mask, bwd_flow, bwd_mask = None, None, None, None

            if os.path.isfile(motion_mask_path):
                mask = cv2.imread(motion_mask_path, cv2.IMREAD_UNCHANGED)
                if len(mask.shape) != 2:
                    mask = mask[..., 0]
                mask = cv2.resize(mask, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA) > 0
            else:
                mask = None

            return {
                "img": img[np.newaxis], 
                "invdepth": invdepth[np.newaxis] if invdepth is not None else None,
                "fwd_flow": fwd_flow[np.newaxis] if fwd_flow is not None else None,
                "fwd_mask": fwd_mask[np.newaxis] if fwd_mask is not None else None,
                "bwd_flow": bwd_flow[np.newaxis] if bwd_flow is not None else None,
                "bwd_mask": bwd_mask[np.newaxis] if bwd_mask is not None else None,
                "mask": mask[np.newaxis] if mask else None,
            }

        def read_event(i):
            tmp_t = i
            while not self.all_paths[tmp_t].endswith('jpg'):
                tmp_t -= 1
            event_path = self.all_paths[i]
            data = np.load(event_path)
            img = cv2.imread(self.all_paths[tmp_t])[..., ::-1]
            img = img.astype(np.float32) / 255
            if img.ndim == 3:
                img = img.mean(axis=-1)
            if self.downsampling != -1:
                scale = 1 / self.downsampling
                img = cv2.resize(img, None, 
                    fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                data = cv2.resize(data, None, 
                    fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            evdata = img * np.exp(data * 0.2)
            evdata = np.clip(evdata, 0, 1)
            return {"events": evdata[np.newaxis, ..., np.newaxis]}
        
        def read_image_and_event(i):
            image_path = self.all_paths[i]
            if "events" in image_path:
                return read_event(i)
            return read_image(i)
        n_frames_to_load = min(self.frames_chunk, self.num_images - self.loaded_frames)
        all_data = Parallel(n_jobs=-1, backend="threading")(
            delayed(read_image_and_event)(i) for i in range(self.loaded_frames, self.loaded_frames + n_frames_to_load) 
        )
        # all_data = [read_image_and_event(i) for i in range(self.loaded_frames, self.loaded_frames + n_frames_to_load) ]

        self.loaded_frames += n_frames_to_load
        all_rgbs = [data["img"] for data in all_data if data and "img" in data]
        all_invdepths = [data["invdepth"] for data in all_data if data and "invdepth" in data]
        all_fwd_flow = [data["fwd_flow"] for data in all_data if data and "fwd_flow" in data]
        all_fwd_mask = [data["fwd_mask"] for data in all_data if data and "fwd_mask" in data]
        all_bwd_flow = [data["bwd_flow"] for data in all_data if data and "bwd_flow" in data]
        all_bwd_mask = [data["bwd_mask"] for data in all_data if data and "bwd_mask" in data]
        all_mask = [data["mask"] for data in all_data if data and "mask" in data]
        all_events = [data["events"] for data in all_data if data and "events" in data]

        all_laplacian = [
                np.ones_like(img[0][..., 0]) * cv2.Laplacian(
                            cv2.cvtColor((img[0]*255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F
                        ).var()
            for img in all_rgbs
        ]
        all_loss_weights = [laplacian if mask is None else laplacian * mask for laplacian, mask in zip(all_laplacian, all_mask)]

        self.img_wh = list(all_rgbs[0].shape[2:0:-1]) if all_rgbs else list(all_events[0].shape[1:3][::-1])
        self.n_px_per_frame = self.img_wh[0] * self.img_wh[1]
        self.H, self.W = 1080, 1920

        if self.split != "train":
            self.all_rgbs = np.concatenate(all_rgbs, 0)
            if self.load_depth:
                self.all_invdepths = np.concatenate(all_invdepths, 0)
            if self.load_flow:
                self.all_fwd_flow = np.concatenate(all_fwd_flow, 0)
                self.all_fwd_mask = np.concatenate(all_fwd_mask, 0)
                self.all_bwd_flow = np.concatenate(all_bwd_flow, 0)
                self.all_bwd_mask = np.concatenate(all_bwd_mask, 0)
        else:
            self.all_rgbs = concatenate_append(self.all_rgbs, all_rgbs) if all_rgbs else self.all_rgbs
            self.all_events = concatenate_append(self.all_events, all_events) if all_events else self.all_events
            if self.load_depth:
                self.all_invdepths = concatenate_append(self.all_invdepths, all_invdepths) if all_invdepths else self.all_invdepths
            if self.load_flow:
                self.all_fwd_flow = concatenate_append(self.all_fwd_flow, all_fwd_flow)
                self.all_fwd_mask = concatenate_append(self.all_fwd_mask, all_fwd_mask)
                self.all_bwd_flow = concatenate_append(self.all_bwd_flow, all_bwd_flow)
                self.all_bwd_mask = concatenate_append(self.all_bwd_mask, all_bwd_mask)
            self.all_loss_weights = concatenate_append(self.all_loss_weights, all_loss_weights) if all_loss_weights else self.all_loss_weights


    def __len__(self):
        return int(1e10)

    def __getitem__(self, i):
        raise NotImplementedError
        idx = np.random.randint(self.sampling_bound[0], self.sampling_bound[1])

        return {"rgbs": self.all_rgbs[idx], "idx": idx}

    def get_frame_fbase(self, view_id):
        return list(self.all_fbases.keys())[view_id]
    
    def get_can_sample_img(self, optimize_poses):
        active_test_mask = self.test_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]
        test_ratio = active_test_mask.mean()
        if optimize_poses:
            train_test_poses = test_ratio > random.uniform(0, 1)
        else:
            train_test_poses = False

        # test or train sample mode 
        inclusion_mask = active_test_mask if train_test_poses else 1 - active_test_mask

        # where can we sample, excluding the event frame
        can_sample = (1 - self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]> 0 ) & (inclusion_mask > 0)

        return can_sample, train_test_poses

    def sample_img(self, batch_size, is_refining, optimize_poses, n_views=16):
        can_sample, train_test_poses = self.get_can_sample_img(optimize_poses)

        # map raw idx to pose idx 
        sample_map = np.nonzero(can_sample)[0] + self.active_frames_bounds[0]
        
        # image raw idx based on number of frame where can sample 
        raw_samples = np.random.randint(0, can_sample.sum(), n_views, dtype=np.int64)

        # map pose idx to image idx
        img_id_map = np.cumsum(1 - self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]) - 1
        img_id_map = img_id_map.astype(np.int64)

        # Force having the last views during coarse optimization
        if not is_refining and can_sample.sum() > 4:
            raw_samples[:2] = can_sample.sum() - 1
            raw_samples[2:4] = can_sample.sum() - 2
            raw_samples[4] = can_sample.sum() - 3
            raw_samples[5] = can_sample.sum() - 4

        view_ids = sample_map[raw_samples]

        idx = np.random.randint(0, self.n_px_per_frame, batch_size, dtype=np.int64)
        idx = idx.reshape(n_views, -1)

        # pose idx -> image idx
        idx = idx + img_id_map[(view_ids - self.active_frames_bounds[0]).astype(np.int64)][..., None] * self.n_px_per_frame
        idx = idx.reshape(-1)

        idx_sample = idx

        return {
            "rgbs": self.all_rgbs.reshape(-1,3)[idx_sample], 
            "loss_weights": self.all_loss_weights.reshape(-1,1)[idx_sample], 
            "invdepths": self.all_invdepths.reshape(-1,1)[idx_sample] if self.load_depth else None,
            "fwd_flow": self.all_fwd_flow.reshape(-1,2)[idx_sample] if self.load_flow else None,
            "fwd_mask": self.all_fwd_mask.reshape(-1,1)[idx_sample] if self.load_flow else None,
            "bwd_flow": self.all_bwd_flow.reshape(-1,2)[idx_sample] if self.load_flow else None,
            "bwd_mask": self.all_bwd_mask.reshape(-1,1)[idx_sample] if self.load_flow else None,
            "idx": idx, # pixel idx
            "view_ids": view_ids, # pose idx
            "train_test_poses": train_test_poses,
            "mode": "rgb"
        }

    def sample_event(self, batch_size, is_refining, optimize_poses, n_views=16):
        inclusion_mask = self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]
        sample_map = np.nonzero(inclusion_mask)[0] + self.active_frames_bounds[0]

        event_id_map = np.cumsum(self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]) - 1
        event_id_map = event_id_map.astype(np.int64)
        
        raw_samples = np.random.randint(0, inclusion_mask.sum(), size=(n_views,), dtype=np.int64)
        # Force having the last views during coarse optimization
        if not is_refining and inclusion_mask.sum() > 4:
            raw_samples[:2] = inclusion_mask.sum() - 1
            raw_samples[2:4] = inclusion_mask.sum() - 2
            raw_samples[4] = inclusion_mask.sum() - 3
            raw_samples[5] = inclusion_mask.sum() - 4
        view_ids = sample_map[raw_samples]

        idx = np.random.randint(0, self.n_px_per_frame, batch_size, dtype=np.int64)
        idx = idx.reshape(n_views, -1)
        idx += event_id_map[(view_ids - self.active_frames_bounds[0]).astype(np.int64)][..., None] * self.n_px_per_frame
        idx = idx.reshape(-1)
        idx_sample = idx

        return {
            "events": self.all_events.reshape(-1,1)[idx_sample], 
            "idx": idx, # pixel idx
            "view_ids": view_ids, # pose idx
            "mode": "event"
        }
    
    def sample(self, batch_size, is_refining, optimize_poses, n_views=16):
        event_ratio = self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]].mean()
        
        if event_ratio > random.uniform(0, 1):
            return self.sample_event(batch_size, is_refining, optimize_poses, n_views)
        else:
            return self.sample_img(batch_size, is_refining, optimize_poses, n_views)
    

    def sample_event_patch(self, batch_size, is_refining, optimize_poses, n_views=16):

        # test or train sample mode 
        inclusion_mask = self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]

        sample_map = np.nonzero(inclusion_mask)[0] + self.active_frames_bounds[0]

        event_id_map = np.cumsum(self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]) - 1
        event_id_map = event_id_map.astype(np.int64)

        raw_samples = np.random.randint(0, inclusion_mask.sum(), size=(n_views,), dtype=np.int64)
        view_ids = sample_map[raw_samples]
        patch_size = int((batch_size // n_views)**(1/2))

        i, j = np.random.randint(0, self.H  - patch_size + 1, size=[n_views,1,1]), np.random.randint(0, self.W  - patch_size + 1, size=[n_views,1,1])
        di, dj = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='ij')
        di, dj = di[None].repeat(n_views, 0), dj[None].repeat(n_views, 0)
        di, dj = di + i, dj + j
        idx = dj + di * self.W + event_id_map[(view_ids - self.active_frames_bounds[0]).astype(np.int64)][..., None] * self.n_px_per_frame
        idx = idx.reshape(-1)

        idx_sample = idx

        return {
            "events": self.all_events.reshape(-1,1)[idx_sample], 
            "idx": idx, # pixel idx
            "view_ids": view_ids, # pose idx
            "mode": "event"
        }
    
    def sample_image_patch(self, batch_size, is_refining, optimize_poses, n_views=16):
        active_test_mask = self.test_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]
        test_ratio = active_test_mask.mean()
        if optimize_poses:
            train_test_poses = test_ratio > random.uniform(0, 1)
        else:
            train_test_poses = False

        # test or train sample mode 
        inclusion_mask = active_test_mask if train_test_poses else 1 - active_test_mask

        # where can we sample, excluding the event frame
        can_sample = (1 - self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]> 0 ) & (inclusion_mask > 0)

        # map raw idx to pose idx 
        sample_map = np.nonzero(can_sample)[0] + self.active_frames_bounds[0]
        
        # image raw idx based on number of frame where can sample 
        raw_samples = np.random.randint(0, can_sample.sum(), n_views, dtype=np.int64)

        # map pose idx to image idx
        img_id_map = np.cumsum(1 - self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]) - 1
        img_id_map = img_id_map.astype(np.int64)

        raw_samples = np.random.randint(0, can_sample.sum(), size=(n_views,), dtype=np.int64)
        view_ids = sample_map[raw_samples]
        patch_size = int((batch_size // n_views)**(1/2))

        i, j = np.random.randint(0, self.H  - patch_size + 1, size=[n_views,1,1]), np.random.randint(0, self.W  - patch_size + 1, size=[n_views,1,1])
        di, dj = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='ij')
        di, dj = di[None].repeat(n_views, 0), dj[None].repeat(n_views, 0)
        di, dj = di + i, dj + j
        idx = dj + di * self.W + img_id_map[(view_ids - self.active_frames_bounds[0]).astype(np.int64)][..., None] * self.n_px_per_frame
        idx = idx.reshape(-1)

        idx_sample = idx

        return {
            "rgbs": self.all_rgbs.reshape(-1,3)[idx_sample], 
            "loss_weights": self.all_loss_weights.reshape(-1,1)[idx_sample], 
            "invdepths": self.all_invdepths.reshape(-1,1)[idx_sample] if self.load_depth else None,
            "fwd_flow": self.all_fwd_flow.reshape(-1,2)[idx_sample] if self.load_flow else None,
            "fwd_mask": self.all_fwd_mask.reshape(-1,1)[idx_sample] if self.load_flow else None,
            "bwd_flow": self.all_bwd_flow.reshape(-1,2)[idx_sample] if self.load_flow else None,
            "bwd_mask": self.all_bwd_mask.reshape(-1,1)[idx_sample] if self.load_flow else None,
            "idx": idx, # pixel idx
            "view_ids": view_ids, # pose idx
            "train_test_poses": train_test_poses,
            "mode": "rgb"
        }
    
    def sample_patch(self, batch_size, is_refining, optimize_poses, n_views=16):
        event_ratio = self.event_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]].mean()
        
        if event_ratio > random.uniform(0, 1):
            return self.sample_event_patch(batch_size, is_refining, optimize_poses, n_views)
        else:
            return self.sample_image_patch(batch_size, is_refining, optimize_poses, n_views)