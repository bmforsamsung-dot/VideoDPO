"""
modified from @author xiaowei MACVID dataset 
"""

import os
import random
import json
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import glob
import pandas as pd
import yaml


class TextVideo(Dataset):
    """
    Data is structured as follows.
        |video_dataset_0
            |clip1.mp4
            |clip2.mp4
            |...
            |metadata.json
    """

    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        frame_stride=4,
        subset_split="all",
        clip_length=1.0,
    ):
        self.data_root = data_root
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.frame_stride = frame_stride
        self.clip_length = clip_length
        assert self.subset_split in ["train", "test", "all"]
        self.exts = ["avi", "mp4", "webm"]

        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]
        # assert(isinstance(self.resolution, list) and len(self.resolution) == 2)

        self._make_dataset()

    def _make_dataset(self):
        with open(self.data_root, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        self.videos = []
        for meta_path in self.config["META"]:
            metadata_path = os.path.join(meta_path, "metadata.json")
            with open(metadata_path, "r") as f:
                videos = json.load(f)
                for item in videos:
                    if item["basic"]["clip_duration"] < self.clip_length:
                        continue
                    item["basic"]["clip_path"] = os.path.join(
                        meta_path, item["basic"]["clip_path"]
                    )
                    self.videos.append(item)

        print(f"Number of videos = {len(self.videos)}")

    def __getitem__(self, index):
        while True:
            # video_path = os.path.join(self.data_root, f"videos/{self.videos.loc[index]['page_dir']}") + f"/{self.videos.loc[index]['videoid']}.mp4"
            # video_path = os.path.join(self.data_root, '.'+self.videos[index]['basic']['clip_path'])
            video_path = os.path.join(
                self.data_root, self.videos[index]["basic"]["clip_path"]
            )
            try:
                video_reader = VideoReader(
                    video_path,
                    ctx=cpu(0),
                    width=self.resolution[1],
                    height=self.resolution[0],
                )
                if len(video_reader) < self.video_length:
                    index += 1
                    continue
                else:
                    break
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                return self.__getitem__(index)

        all_frames = list(range(0, len(video_reader), self.frame_stride))
        if len(all_frames) < self.video_length:
            all_frames = list(range(0, len(video_reader), 1))

        # select random clip
        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = all_frames[rand_idx : rand_idx + self.video_length]

        frames = video_reader.get_batch(frame_indices)
        assert (
            frames.shape[0] == self.video_length
        ), f"{len(frames)}, self.video_length={self.video_length}"

        frames = (
            torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        )  # [t,h,w,c] -> [c,t,h,w]
        assert (
            frames.shape[2] == self.resolution[0]
            and frames.shape[3] == self.resolution[1]
        ), f"frames={frames.shape}, self.resolution={self.resolution}"
        frames = (frames / 255 - 0.5) * 2
        data = {
            "video": frames,
            "caption": self.videos[index]["misc"]["frame_caption"][0],
        }
        return data

    def __len__(self):
        return len(self.videos)


"""
    A tipical item for DPO contain: [video1,video2,label0,caption]
    For better scalbility: we add a dataframe to choose video and leave the metadata.json still. 
"""


class TextVideoDPO(Dataset):
    """
    Data is structured as follows.
        |video_dataset_0
            |clip1.mp4
            |clip2.mp4
            |...
            |metadata.json
    """

    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        frame_stride=4,
        subset_split="all",
        clip_length=1.0,
        dupbeta=1.0, # scale up factor
    ):
        self.data = TextVideo(
            data_root, resolution, video_length, frame_stride, subset_split, clip_length
        )
        self.pairs = []
        self.data_root = data_root
        with open(self.data_root, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        label_key = "label"
        self.dupbeta = dupbeta
        for meta_path in self.config["META"]:
            # 我意识到这个pairdata可以放到metadata.json里面 再加一个字典就行了
            pairdata_path = os.path.join(meta_path, "pair.json")
            with open(pairdata_path, "r") as f:
                pairs = json.load(f)
                for item in pairs:
                    # under the pair.json after 0601,label_key has no use
                    if dupbeta:
                        if 'score' not in item:
                            item['score']=1
                        item = [item["video1"], item["video2"], item["frame_caption"],item['score']]
                    else:
                        item = [item["video1"], item["video2"], item["frame_caption"]]
                    self.pairs.append(item)
        print(f"DPO dataset has {self.__len__()} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        if self.dupbeta:
            videowidx, videolidx, frame_caption,prob_score = self.pairs[index]
            dupfactor = (0.72 / prob_score )**self.dupbeta # scale up factor 
        else:
            videowidx, videolidx, frame_caption = self.pairs[index]
        videow = self.data[videowidx]["video"]
        videol = self.data[videolidx]["video"]
        # print(f"video idx {videowidx} {videolidx}")
        # print("in dataloader getitem",videow.shape,videol.shape)
        if videow.dim()==5:
            combined_frames = torch.cat([videow, videol], dim=0)
        else:
            combined_frames = torch.cat([videow, videol], dim=0)
        if isinstance(frame_caption, list):
            frame_caption = frame_caption[0]

        
        # print("in dataloader getitem",combined_frames.shape)
        if self.dupbeta:
            return {"video": combined_frames, "caption": frame_caption,"dupfactor":dupfactor}
        else:
            return {"video": combined_frames, "caption": frame_caption,"dupfactor":1.0}


class TextVideoGroupDPO(Dataset):
    """
    Group-wise DPO dataset.
    Each item in group.json is expected to include:
      - frame_caption
      - winner: {video_idx}
      - losers: [{video_idx, sa_score, pc_score}, ...]
    """

    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        frame_stride=4,
        subset_split="all",
        clip_length=1.0,
    ):
        self.data = TextVideo(
            data_root, resolution, video_length, frame_stride, subset_split, clip_length
        )
        self.groups = []
        self.data_root = data_root
        with open(self.data_root, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        for meta_path in self.config["META"]:
            group_path = os.path.join(meta_path, "group.json")
            if not os.path.exists(group_path):
                continue
            with open(group_path, "r") as f:
                groups = json.load(f)
                self.groups.extend(groups)
        print(f"GroupDPO dataset has {self.__len__()} groups")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        item = self.groups[index]
        frame_caption = item.get("frame_caption", "")
        if isinstance(frame_caption, list):
            frame_caption = frame_caption[0]

        winner_idx = int(item["winner"]["video_idx"])
        losers = item["losers"]
        loser_indices = [int(x["video_idx"]) for x in losers]
        all_indices = [winner_idx] + loser_indices

        videos = [self.data[idx]["video"] for idx in all_indices]
        combined_frames = torch.cat(videos, dim=0)

        sa_scores = [1.0] + [float(x.get("sa_score", 0.0)) for x in losers]
        pc_scores = [1.0] + [float(x.get("pc_score", 0.0)) for x in losers]

        return {
            "video": combined_frames,
            "caption": frame_caption,
            "dupfactor": 1.0,
            "group_size": len(all_indices),
            "winner_index": 0,
            "sa_scores": torch.tensor(sa_scores, dtype=torch.float32),
            "pc_scores": torch.tensor(pc_scores, dtype=torch.float32),
        }
