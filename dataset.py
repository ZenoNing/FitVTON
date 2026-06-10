"""PyTorch datasets for FitVTON training and benchmark inference.

Sections:
  - DressCode datasets for stage-2 texture LoRA training
  - VITON benchmark dataset
  - GarmentCodeVTON dataset for stage-1 fitting LoRA / mask-head training
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import torch
import yaml
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms

def _rgb_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _random_crop(image, crop_box, orig_size, *, mask: bool = False):
    image = image.crop(crop_box)
    resample = Image.NEAREST if mask else Image.BILINEAR
    return image.resize(orig_size, resample)


def _shrink_and_pad(
    image,
    shrink_ratio: float = 0.7,
    pad_color=255,
    left=None,
    top=None,
    *,
    mask: bool = False,
):
    width, height = image.size
    new_w, new_h = int(width * shrink_ratio), int(height * shrink_ratio)
    resample = Image.NEAREST if mask else Image.BILINEAR
    small_img = image.resize((new_w, new_h), resample)
    if mask:
        background = Image.new("L", (width, height), int(pad_color) if isinstance(pad_color, int) else 0)
    else:
        color = pad_color if isinstance(pad_color, tuple) else (pad_color, pad_color, pad_color)
        background = Image.new("RGB", (width, height), color)
    if left is None:
        left = random.randint(0, width - new_w) if width > new_w else 0
    if top is None:
        top = random.randint(0, height - new_h) if height > new_h else 0
    background.paste(small_img, (left, top))
    return background


def _augment_reference_centered(reference_img):
    ref_aug_type = random.choices(["original", "crop", "shrink_pad"], weights=[6, 2, 2])[0]
    if ref_aug_type == "crop":
        width, height = reference_img.size
        crop_ratio = random.uniform(0.8, 1.0)
        crop_w, crop_h = int(width * crop_ratio), int(height * crop_ratio)
        left = max((width - crop_w) // 2, 0)
        top = max((height - crop_h) // 2, 0)
        crop_box = (left, top, left + crop_w, top + crop_h)
        return _random_crop(reference_img, crop_box, (width, height))
    if ref_aug_type == "shrink_pad":
        shrink_ratio = random.uniform(0.8, 1.0)
        width, height = reference_img.size
        new_w, new_h = int(width * shrink_ratio), int(height * shrink_ratio)
        left = max((width - new_w) // 2, 0)
        top = max((height - new_h) // 2, 0)
        return _shrink_and_pad(reference_img, shrink_ratio, left=left, top=top)
    return reference_img


def _augment_source_gt_masks(source_img, gt_img, body_mask_img=None, garment_mask_img=None):
    aug_type = random.choices(["original", "crop", "shrink_pad"], weights=[8, 1, 1])[0]
    if aug_type == "crop":
        width, height = source_img.size
        crop_ratio = random.uniform(0.7, 1.0)
        crop_w, crop_h = int(width * crop_ratio), int(height * crop_ratio)
        left = random.randint(0, width - crop_w) if width > crop_w else 0
        top = random.randint(0, height - crop_h) if height > crop_h else 0
        crop_box = (left, top, left + crop_w, top + crop_h)
        orig_size = (width, height)
        source_img = _random_crop(source_img, crop_box, orig_size)
        gt_img = _random_crop(gt_img, crop_box, orig_size)
        if body_mask_img is not None:
            body_mask_img = _random_crop(body_mask_img, crop_box, orig_size, mask=True)
        if garment_mask_img is not None:
            garment_mask_img = _random_crop(garment_mask_img, crop_box, orig_size, mask=True)
    elif aug_type == "shrink_pad":
        shrink_ratio = random.uniform(0.7, 1.0)
        width, height = source_img.size
        new_w, new_h = int(width * shrink_ratio), int(height * shrink_ratio)
        left = random.randint(0, width - new_w) if width > new_w else 0
        top = random.randint(0, height - new_h) if height > new_h else 0
        source_img = _shrink_and_pad(source_img, shrink_ratio, left=left, top=top)
        gt_img = _shrink_and_pad(gt_img, shrink_ratio, left=left, top=top)
        if body_mask_img is not None:
            body_mask_img = _shrink_and_pad(body_mask_img, shrink_ratio, pad_color=0, left=left, top=top, mask=True)
        if garment_mask_img is not None:
            garment_mask_img = _shrink_and_pad(garment_mask_img, shrink_ratio, pad_color=0, left=left, top=top, mask=True)
    return source_img, gt_img, body_mask_img, garment_mask_img


def _normalize_garment_length(length: str) -> str:
    length = length.strip()
    if length in {"short", "short-length"}:
        return "short-length"
    if length in {"long", "long-length"}:
        return "long-length"
    return length


def _wearing_style_word(wearing_style: str) -> str:
    if wearing_style in {"tucked", "tucked_in"}:
        return "tucked in"
    if wearing_style == "untucked":
        return "untucked"
    if wearing_style == "one_piece":
        return "one-piece"
    return wearing_style.replace("_", " ")


def build_long_vton_prompt(gender: str, shape_height: str, length_word: str, garment_type: str, wear_style: str) -> str:
    if shape_height == "unknown body" or " " not in shape_height:
        person_sentence = f"The person is an unknown body {gender}."
    else:
        shape, height = shape_height.split(" ", 1)
        person_sentence = f"The person is a {height} and {shape} {gender}."
    cloth_sentence = f"The cloth is a {length_word} {garment_type} garment."
    wear_sentence = f"The wearing style is {_wearing_style_word(wear_style)}."
    return f"{person_sentence} {cloth_sentence} {wear_sentence}"


class DressCodeDataset(Dataset):
    """DressCode pseudo/reference/gt triples with long_vton prompts from a 7-column TSV."""

    TYPE_TO_DIR = {
        "0": "upper_body",
        "1": "lower_body",
        "2": "dresses",
    }
    TYPE_TO_PROMPT = {
        "0": "upper",
        "1": "lower",
        "2": "dress",
    }

    def __init__(
        self,
        dresscode_root,
        pseudo_root,
        pairs_file,
        transform=False,
        disable_augmentation=False,
    ):
        self.dresscode_root = Path(dresscode_root)
        self.pseudo_root = Path(pseudo_root)
        self.pairs_file = Path(pairs_file)
        self.disable_augmentation = disable_augmentation
        self.samples = []

        self.transforms = _rgb_transform() if transform else None

        if not self.pairs_file.exists():
            raise FileNotFoundError(f"pairs_file not found: {self.pairs_file}")

        with self.pairs_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) != 7:
                    raise ValueError(
                        f"{self.pairs_file}:{line_no}: expected 7 tab-separated columns, got {len(cols)}"
                    )

                (
                    pseudo_image,
                    reference_image,
                    gender,
                    wearing_style,
                    length,
                    type_id,
                    gt_image,
                ) = cols
                category = self.TYPE_TO_DIR.get(type_id)
                if category is None:
                    raise ValueError(f"{self.pairs_file}:{line_no}: unknown type_id {type_id!r}")

                pseudo_path = self._resolve_pseudo_path(pseudo_image, category)
                reference_path = self.dresscode_root / category / "images" / reference_image
                gt_path = self.dresscode_root / category / "images" / gt_image

                missing = [
                    str(path)
                    for path in (pseudo_path, reference_path, gt_path)
                    if not path.exists()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"{self.pairs_file}:{line_no}: missing file(s): {', '.join(missing)}"
                    )

                self.samples.append(
                    {
                        "pseudo_path": pseudo_path,
                        "reference_path": reference_path,
                        "gt_path": gt_path,
                        "prompt": self._build_prompt(
                            gender=gender,
                            wearing_style=wearing_style,
                            length=length,
                            type_id=type_id,
                        ),
                    }
                )

        self._length = len(self.samples)
        if self._length == 0:
            raise ValueError(f"No valid samples found in {self.pairs_file}")

    def _resolve_pseudo_path(self, pseudo_image, category):
        flat_path = self.pseudo_root / pseudo_image
        if flat_path.exists():
            return flat_path
        return self.pseudo_root / category / pseudo_image

    @classmethod
    def _build_prompt(cls, gender, wearing_style, length, type_id):
        return build_long_vton_prompt(
            gender=gender,
            shape_height="slim tall",
            length_word=_normalize_garment_length(length),
            garment_type=cls.TYPE_TO_PROMPT[type_id],
            wear_style=wearing_style,
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pseudo_img = exif_transpose(Image.open(sample["pseudo_path"])).convert("RGB")
        reference_img = exif_transpose(Image.open(sample["reference_path"])).convert("RGB")
        gt_img = exif_transpose(Image.open(sample["gt_path"])).convert("RGB")

        if not self.disable_augmentation:
            pseudo_img, gt_img, _, _ = _augment_source_gt_masks(pseudo_img, gt_img)

        if self.transforms:
            pseudo_img = self.transforms(pseudo_img)
            reference_img = self.transforms(reference_img)
            gt_img = self.transforms(gt_img)

        return {
            "pseudo_image": pseudo_img,
            "reference_image": reference_img,
            "gt_image": gt_img,
            "prompt": sample["prompt"],
        }



class VITONDataset(Dataset):

    def __init__(
        self,
        data_root,
        split: str = "train",
        pairs_file=None,
        transform=False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.custom_instance_prompts = True

        self.triples = []

        if transform == True:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transforms = None

        if pairs_file is None:
            pairs_file = self.data_root / split / "train_first_step_3col.txt"
        self.pairs_file = Path(pairs_file)
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"pairs_file not found: {self.pairs_file}")

        instance_dir = self.data_root / "second_training_data"
        cloth_dir = self.data_root / split / "cloth"
        image_dir = self.data_root / split / "image"

        with open(self.pairs_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) < 3:
                    cols = line.split()
                if len(cols) < 3:
                    continue

                src_name, _ref_name, pseudo_name = cols[0], cols[1], cols[2]

                pseudo_path = instance_dir / pseudo_name
                reference_path = cloth_dir / src_name
                gt_path = image_dir / src_name

                if pseudo_path.exists() and reference_path.exists() and gt_path.exists():
                    self.triples.append((str(pseudo_path), str(reference_path), str(gt_path)))

        self._length = len(self.triples)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        pseudo_path, reference_path, gt_path = self.triples[idx]

        pseudo_img = exif_transpose(Image.open(pseudo_path)).convert("RGB")
        reference_img = exif_transpose(Image.open(reference_path)).convert("RGB")
        gt_img = exif_transpose(Image.open(gt_path)).convert("RGB")

        if self.transforms:
            pseudo_img = self.transforms(pseudo_img)
            reference_img = self.transforms(reference_img)
            gt_img = self.transforms(gt_img)

        return {
            "pseudo_image": pseudo_img,
            "reference_image": reference_img,
            "gt_image": gt_img,
            "prompt": "",
        }



def _make_body_shape_prompts(num_body_types=16):
    widths = ("slim", "average", "plus-size", "heavy")
    heights = ("short", "medium-short", "medium-tall", "tall")
    return {i: f"{widths[i % 4]} {heights[i // 4]}" for i in range(num_body_types)}


class GarmentCodeVTONDataset(Dataset):
    """
    GarmentCode simulation dataset for stage-1 fitting LoRA / mask-head training.

    Layout::

        Ref/<garment>/render_front.png
        female|male/<person>/<outfit>/<tucked_in|untucked|one_piece>/poseN/
            render_front.png
            body_mask_front.png
            garment_mask_front.png

    Returns instance / reference / gt images, body & garment masks, long_vton prompt,
    and per-sample metadata for curriculum-style subsampling.
    """

    TWO_PIECE_STYLES = {"tucked_in", "untucked"}
    ONE_PIECE_STYLE = "one_piece"
    NUM_BODY_TYPES = 16
    BODY_SHAPE_PROMPTS = _make_body_shape_prompts(NUM_BODY_TYPES)
    FEMALE_BODY_SHAPE_PROMPTS = BODY_SHAPE_PROMPTS
    MALE_BODY_SHAPE_PROMPTS = BODY_SHAPE_PROMPTS
    REF_GARMENT_LENGTH_PROMPTS = {
        "upper1": "short-length upper",
        "upper2": "short-length upper",
        "upper3": "long-length upper",
        "pants1": "short-length lower",
        "pants2": "short-length lower",
        "pants3": "long-length lower",
        "pants4": "long-length lower",
        "circleskirt1": "short-length lower",
        "circleskirt2": "long-length lower",
        "pencilskirt1": "short-length lower",
        "pencilskirt2": "long-length lower",
        "dress1": "long-length one-piece",
        "dress2": "long-length one-piece",
        "dress3": "long-length one-piece",
        "dress4": "short-length one-piece",
        "dress5": "long-length one-piece",
        "dress6": "short-length one-piece",
        "dress7": "long-length one-piece",
        "dress8": "long-length one-piece",
    }

    def __init__(
        self,
        data_root,
        transform=False,
        include_genders=("female", "male"),
        max_pairs_per_group=None,
        cap_seed=0,
        pair_mode="all",
        prompt_style="long_vton",
        training_stage="all",
        curriculum_stage=None,
    ):
        """
        Args:
            max_pairs_per_group: Cap (instance, reference, gt) pairs per (gender, person, pose).
            cap_seed: Deterministic seed for per-group subsampling.
            pair_mode: "all" for mixed try-on triplets; "dress_to_dress" for one-piece only.
            prompt_style: Only "long_vton" is supported.
            training_stage: Stage filter / stratified capping. One of "all", "cloth_balanced",
                "wearing_two_piece", "shape_balanced", "wearing_all_style".
            curriculum_stage: Deprecated alias for training_stage.
        """
        self.data_root = Path(data_root)
        self.custom_instance_prompts = True
        if isinstance(include_genders, str):
            include_genders = tuple(g.strip() for g in include_genders.split(",") if g.strip())
        self.include_genders = tuple(include_genders)
        self.max_pairs_per_group = max_pairs_per_group
        self.cap_seed = cap_seed
        self.pair_mode = pair_mode
        self.prompt_style = prompt_style
        self.training_stage = training_stage or curriculum_stage or "all"
        self.samples = []

        self.transforms = _rgb_transform() if transform else None

        records_by_identity_pose = self._scan_records()
        self._build_samples(records_by_identity_pose)
        self._apply_training_stage()
        if self.max_pairs_per_group is not None:
            self._cap_samples_per_group()
        self._length = len(self.samples)

    def _apply_training_stage(self):
        if self.training_stage == "all":
            return
        if self.training_stage == "shape_balanced":
            # Shape balance is primarily provided by equal per-person/pose caps.
            return
        if self.training_stage == "cloth_balanced":
            return
        if self.training_stage == "wearing_two_piece":
            self.samples = [
                s for s in self.samples if s["metadata"]["target_wear_style"] in self.TWO_PIECE_STYLES
            ]
            return
        if self.training_stage == "wearing_all_style":
            return
        raise ValueError(f"Unsupported training_stage: {self.training_stage}")

    def _cap_samples_per_group(self):
        """
        Per (gender, person, pose) group, randomly retain at most `self.max_pairs_per_group`
        samples. Sampling is deterministic given `(cap_seed, group_key)` so the subset is
        stable across processes (DDP) and across resumes. A stable presort is applied
        before shuffle to guarantee reproducibility regardless of OS file iteration order.
        """
        groups = {}
        for s in self.samples:
            key = (s["metadata"]["gender"], s["metadata"]["person"], s["metadata"]["pose"])
            groups.setdefault(key, []).append(s)
        capped = []
        for key in sorted(groups.keys()):
            bucket = groups[key]
            if len(bucket) > self.max_pairs_per_group:
                if self.training_stage == "cloth_balanced":
                    bucket = self._stratified_cap(
                        bucket,
                        key,
                        lambda s: s["metadata"]["change_part"],
                        ["upper", "lower", "one_piece"],
                    )
                elif self.training_stage == "wearing_two_piece":
                    bucket = self._stratified_cap(
                        bucket,
                        key,
                        lambda s: (s["metadata"]["target_wear_style"], s["metadata"]["change_part"]),
                        [
                            ("tucked_in", "upper"),
                            ("tucked_in", "lower"),
                            ("untucked", "upper"),
                            ("untucked", "lower"),
                        ],
                    )
                elif self.training_stage == "wearing_all_style":
                    bucket = self._weighted_stratified_cap(
                        bucket,
                        key,
                        lambda s: s["metadata"]["target_wear_style"],
                        {"tucked_in": 2, "untucked": 2, "one_piece": 1},
                    )
                else:
                    bucket = sorted(
                        bucket,
                        key=lambda s: (str(s["source_path"]), str(s["gt_path"]), str(s["reference_path"])),
                    )
                    seed_bytes = "|".join([str(self.cap_seed), *key]).encode("utf-8")
                    stable_seed = int(hashlib.sha256(seed_bytes).hexdigest()[:16], 16)
                    rng_local = random.Random(stable_seed)
                    rng_local.shuffle(bucket)
                    bucket = bucket[: self.max_pairs_per_group]
            capped.extend(bucket)
        self.samples = capped

    def _stable_shuffle(self, bucket, group_key, salt):
        bucket = sorted(
            bucket,
            key=lambda s: (str(s["source_path"]), str(s["gt_path"]), str(s["reference_path"])),
        )
        seed_bytes = "|".join([str(self.cap_seed), *group_key, str(salt)]).encode("utf-8")
        stable_seed = int(hashlib.sha256(seed_bytes).hexdigest()[:16], 16)
        rng_local = random.Random(stable_seed)
        rng_local.shuffle(bucket)
        return bucket

    def _stratified_cap(self, bucket, group_key, key_fn, strata):
        buckets = {stratum: [] for stratum in strata}
        overflow = []
        for sample in bucket:
            stratum = key_fn(sample)
            if stratum in buckets:
                buckets[stratum].append(sample)
            else:
                overflow.append(sample)

        target = self.max_pairs_per_group
        per_stratum = max(1, target // max(1, len(strata)))
        selected = []
        remainder = []
        for stratum in strata:
            shuffled = self._stable_shuffle(buckets[stratum], group_key, stratum)
            selected.extend(shuffled[:per_stratum])
            remainder.extend(shuffled[per_stratum:])
        remainder.extend(self._stable_shuffle(overflow, group_key, "overflow"))

        if len(selected) < target:
            selected.extend(self._stable_shuffle(remainder, group_key, "remainder")[: target - len(selected)])
        return selected[:target]

    def _weighted_stratified_cap(self, bucket, group_key, key_fn, weights):
        buckets = {stratum: [] for stratum in weights}
        overflow = []
        for sample in bucket:
            stratum = key_fn(sample)
            if stratum in buckets:
                buckets[stratum].append(sample)
            else:
                overflow.append(sample)

        target = self.max_pairs_per_group
        total_weight = max(1, sum(weights.values()))
        selected = []
        remainder = []
        for stratum, weight in weights.items():
            quota = max(1, target * weight // total_weight)
            shuffled = self._stable_shuffle(buckets[stratum], group_key, stratum)
            selected.extend(shuffled[:quota])
            remainder.extend(shuffled[quota:])
        remainder.extend(self._stable_shuffle(overflow, group_key, "overflow"))

        if len(selected) < target:
            selected.extend(self._stable_shuffle(remainder, group_key, "remainder")[: target - len(selected)])
        return selected[:target]

    def __len__(self):
        return self._length

    @staticmethod
    def _natural_key(path_or_name):
        name = path_or_name.name if hasattr(path_or_name, "name") else str(path_or_name)
        # Split into alternating non-digit / digit chunks so compound names like
        # "upper1_pants10" sort as ("upper", 1, "_pants", 10) rather than
        # ("upper_pants", 110) under a naive digit-concat scheme.
        parts = []
        buf = ""
        is_digit = name[:1].isdigit() if name else False
        for ch in name:
            if ch.isdigit() == is_digit:
                buf += ch
            else:
                parts.append(int(buf) if is_digit else buf)
                buf = ch
                is_digit = ch.isdigit()
        if buf:
            parts.append(int(buf) if is_digit else buf)
        return tuple(parts)

    @staticmethod
    def _is_two_piece(outfit_name):
        return "_" in outfit_name

    @staticmethod
    def _split_two_piece(outfit_name):
        upper, bottom = outfit_name.split("_", 1)
        return upper, bottom

    @classmethod
    def _body_shape_prompt(cls, person_name, gender=None):
        del gender  # shared index-to-body-shape mapping
        digits = "".join(ch for ch in person_name if ch.isdigit())
        if not digits:
            return "unknown body"
        body_idx = int(digits)
        if body_idx < 0 or body_idx >= cls.NUM_BODY_TYPES:
            return "unknown body"
        return cls.BODY_SHAPE_PROMPTS[body_idx]

    @classmethod
    def _length_prompt(cls, garment_name):
        if garment_name in cls.REF_GARMENT_LENGTH_PROMPTS:
            return cls.REF_GARMENT_LENGTH_PROMPTS[garment_name]
        if garment_name.startswith("upper"):
            return "unspecified-length upper"
        if garment_name.startswith(("pants", "circleskirt", "pencilskirt")):
            return "unspecified-length lower"
        if garment_name.startswith("dress"):
            return "unspecified-length one-piece"
        return "unspecified-length garment"

    def _reference_path(self, garment_name):
        return self.data_root / "Ref" / garment_name / "render_front.png"

    def _scan_records(self):
        records_by_identity_pose = {}
        for gender in self.include_genders:
            gender_dir = self.data_root / gender
            if not gender_dir.exists():
                continue

            for person_dir in sorted((p for p in gender_dir.iterdir() if p.is_dir()), key=self._natural_key):
                for outfit_dir in sorted((p for p in person_dir.iterdir() if p.is_dir()), key=self._natural_key):
                    for style_dir in sorted((p for p in outfit_dir.iterdir() if p.is_dir()), key=self._natural_key):
                        wear_style = style_dir.name
                        if wear_style not in self.TWO_PIECE_STYLES and wear_style != self.ONE_PIECE_STYLE:
                            continue

                        for pose_dir in sorted((p for p in style_dir.iterdir() if p.is_dir()), key=self._natural_key):
                            image_path = pose_dir / "render_front.png"
                            body_mask_path = pose_dir / "body_mask_front.png"
                            garment_mask_path = pose_dir / "garment_mask_front.png"
                            if not (image_path.exists() and body_mask_path.exists() and garment_mask_path.exists()):
                                continue

                            record = {
                                "gender": gender,
                                "person": person_dir.name,
                                "outfit": outfit_dir.name,
                                "wear_style": wear_style,
                                "pose": pose_dir.name,
                                "image_path": image_path,
                                "body_mask_path": body_mask_path,
                                "garment_mask_path": garment_mask_path,
                            }
                            key = (gender, person_dir.name, pose_dir.name)
                            records_by_identity_pose.setdefault(key, []).append(record)

        return records_by_identity_pose

    def _build_samples(self, records_by_identity_pose):
        for records in records_by_identity_pose.values():
            if self.pair_mode == "dress_to_dress":
                one_piece_records = [
                    r for r in records if not self._is_two_piece(r["outfit"]) and r["wear_style"] == self.ONE_PIECE_STYLE
                ]
                for target in one_piece_records:
                    reference_path = self._reference_path(target["outfit"])
                    if not reference_path.exists():
                        continue
                    for source in one_piece_records:
                        if source["image_path"] == target["image_path"]:
                            continue
                        self.samples.append(self._make_sample(source, target, reference_path, "one_piece"))
                continue

            two_piece_records = [r for r in records if self._is_two_piece(r["outfit"]) and r["wear_style"] in self.TWO_PIECE_STYLES]

            for target in records:
                target_style = target["wear_style"]
                target_outfit = target["outfit"]

                if target_style == self.ONE_PIECE_STYLE:
                    reference_path = self._reference_path(target_outfit)
                    if not reference_path.exists():
                        continue
                    sources = [source for source in two_piece_records if source["image_path"] != target["image_path"]]
                    self.samples.extend(
                        self._make_sample(source, target, reference_path, "one_piece") for source in sources
                    )
                    continue

                if not self._is_two_piece(target_outfit):
                    continue

                target_upper, target_bottom = self._split_two_piece(target_outfit)
                for source in two_piece_records:
                    if source["image_path"] == target["image_path"]:
                        continue
                    source_upper, source_bottom = self._split_two_piece(source["outfit"])

                    if source_bottom == target_bottom and source_upper != target_upper:
                        reference_path = self._reference_path(target_upper)
                        if reference_path.exists():
                            self.samples.append(self._make_sample(source, target, reference_path, "upper"))

                    if source_upper == target_upper and source_bottom != target_bottom:
                        reference_path = self._reference_path(target_bottom)
                        if reference_path.exists():
                            self.samples.append(self._make_sample(source, target, reference_path, "lower"))

    def _make_sample(self, source, target, reference_path, change_part):
        return {
            "source_path": source["image_path"],
            "reference_path": reference_path,
            "gt_path": target["image_path"],
            "body_mask_path": target["body_mask_path"],
            "garment_mask_path": target["garment_mask_path"],
            "prompt": self._build_prompt(
                target, reference_path.parent.name, change_part
            ),
            "metadata": {
                "gender": target["gender"],
                "person": target["person"],
                "source_outfit": source["outfit"],
                "target_outfit": target["outfit"],
                "reference_garment": reference_path.parent.name,
                "target_wear_style": target["wear_style"],
                "change_part": change_part,
                "pose": target["pose"],
            },
        }

    def _build_prompt(self, target, reference_garment, change_part):
        if self.prompt_style != "long_vton":
            raise ValueError(f"Unsupported prompt_style: {self.prompt_style!r}; only 'long_vton' is supported.")
        return self._long_vton_prompt(target, reference_garment, change_part)

    @classmethod
    def _garment_length_word(cls, garment_name):
        label = cls._length_prompt(garment_name)
        if label.startswith("short-length"):
            return "short-length"
        if label.startswith("long-length"):
            return "long-length"
        return "unspecified-length"

    @classmethod
    def _garment_type_word(cls, change_part):
        if change_part == "upper":
            return "upper"
        if change_part == "lower":
            return "lower"
        return "dress"

    @classmethod
    def _wearing_style_word(cls, wear_style):
        if wear_style == "tucked_in":
            return "tucked in"
        if wear_style == "untucked":
            return "untucked"
        if wear_style == "one_piece":
            return "one-piece"
        return wear_style.replace("_", " ")

    @classmethod
    def _long_vton_prompt(cls, target, reference_garment, change_part):
        label = cls._body_shape_prompt(target["person"], target.get("gender"))
        gender = target["gender"]
        if label == "unknown body" or " " not in label:
            person_sentence = f"The person is an unknown body {gender}."
        else:
            shape, height = label.split(" ", 1)
            person_sentence = f"The person is a {height} and {shape} {gender}."
        length_word = cls._garment_length_word(reference_garment)
        garment_type = cls._garment_type_word(change_part)
        cloth_sentence = f"The cloth is a {length_word} {garment_type} garment."
        wear_sentence = f"The wearing style is {cls._wearing_style_word(target['wear_style'])}."
        return f"{person_sentence} {cloth_sentence} {wear_sentence}"

    def __getitem__(self, idx):
        sample = self.samples[idx]

        source_img = exif_transpose(Image.open(sample["source_path"])).convert("RGB")
        reference_img = exif_transpose(Image.open(sample["reference_path"])).convert("RGB")
        gt_img = exif_transpose(Image.open(sample["gt_path"])).convert("RGB")
        body_mask_img = Image.open(sample["body_mask_path"]).convert("L")
        garment_mask_img = Image.open(sample["garment_mask_path"]).convert("L")

        reference_img = _augment_reference_centered(reference_img)
        source_img, gt_img, body_mask_img, garment_mask_img = _augment_source_gt_masks(
            source_img, gt_img, body_mask_img, garment_mask_img
        )

        if self.transforms:
            source_img = self.transforms(source_img)
            reference_img = self.transforms(reference_img)
            gt_img = self.transforms(gt_img)
            body_mask_img = transforms.ToTensor()(body_mask_img)
            garment_mask_img = transforms.ToTensor()(garment_mask_img)

        return {
            "source_image": source_img,
            "reference_image": reference_img,
            "gt_image": gt_img,
            "body_mask_image": body_mask_img,
            "garment_mask_image": garment_mask_img,
            "prompt": sample["prompt"],
            "metadata": sample["metadata"],
        }

__all__ = [
    "DressCodeDataset",
    "VITONDataset",
    "GarmentCodeVTONDataset",
    "GarmentCodeVTONV3Dataset",
]

DressCodeSecondStageDataset = DressCodeDataset
GarmentCodeVTONV3Dataset = GarmentCodeVTONDataset


if __name__ == "__main__":
    from system_config import cfg_path

    ds = GarmentCodeVTONDataset(data_root=cfg_path("datasets", "garmentcode_root"), transform=True)
    print(len(ds), ds[0]["prompt"][:80])