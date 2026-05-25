#!/usr/bin/env python

# Copyright 2025 Martin Junius
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ChangeLog
# Version 0.1 / 2025-07-13
#       Lightroom Classic XMP manipulation tool, based on imgexiftool 0.2
# Version 0.2 / 2025-07-14
#       Interpolate and change numeric meta data

VERSION     = "0.2 / 2025-07-14"
AUTHOR      = "Martin Junius"
NAME        = "lr-xmp-tool"
DESCRIPTION = "Lightroom Classic XMP manipulation tool"

EXIFTOOL_EXE = "c:/Tools/exiftool/exiftool.exe"

import sys
import argparse
import os
import re
from itertools import pairwise

# The following libs must be installed with pip
from exiftool import ExifToolHelper

from icecream import ic
# Disable debugging
ic.disable()
# Local modules
from verbose import message, verbose, warning, error



# Relevant keywords
KEYWORD_KEY_FRAME = "XMP:Label"

KEYWORDS_CROP = [ "XMP:CropTop", "XMP:CropLeft", "XMP:CropBottom", "XMP:CropRight", "XMP:CropRight", "XMP:CropAngle" ]



# Command line options
class Options:
    keys = [ "XMP:Exposure2012" ]
                                    # -K --keywords
    list = False                    # -l --list
    all = False                     # -a --all
    match = None                    # -m --match
    key_frame_label = "Yellow"      # -L --key-frame-label
    interpolate = False             # -I --interpolate
    no_change = False               # -n --no-change
    float2 = False                  # -2 --float2



def get_seq(filename: str) -> str:
    """
    Get sequence number from filename (last min 2 digits match)
    e.g. 2023-19namibia-207-9628.xmp -> 9628

    :param filename: name
    :type filename: str
    :return: sequence number
    :rtype: str
    """
    m = re.findall(r'(\d\d\d+)', filename)
    return m[-1] if m else None



def sort_file_list(files: list) -> list:
    """
    Sort file list based on sequence number

    :param files: list of filenames
    :type files: list
    :return: list of sorted filenames
    :rtype: list
    """
    files_dict = { get_seq(f): f for f in files }
    sorted_seq = sorted(files_dict.keys())
    if "9999" in sorted_seq:
        # Handle wrap around: 9999 -> 0001
        last_seq = 0
        jump_idx = 0
        for idx, seq in enumerate(sorted_seq):
            seq = sorted_seq[idx]
            if int(seq) == last_seq + 1:
                last_seq = int(seq)
                continue
            if jump_idx:
                error(f"2nd jump in sequence at [{idx}] {last_seq}->{seq}, previous [{jump_idx}]")
            verbose(f"jump in sequence at [{idx}] {last_seq}->{seq}, reshuffle needed")
            jump_idx = idx
            last_seq = int(seq)
        # Reshuffle list
        sorted_seq = sorted_seq[jump_idx: ] + sorted_seq[0:jump_idx]

    verbose(f"{len(sorted_seq)} items in sorted file list")
    return [ files_dict[seq] for seq in sorted_seq ]



def find_key_frames(exiftool: ExifToolHelper, files: list, key: str=KEYWORD_KEY_FRAME) -> list:
    """
    Find key frames (images with "XMP:Label" == "Yellow")

    :param exiftool: ExifTool instance
    :type exiftool: ExifToolHelper
    :param files: sorted list of file names
    :type files: list
    :param key: meta data key, defaults to KEYWORD_KEY_FRAME ("XMP:Label")
    :type key: str, optional
    :return: list of key frames indices
    :rtype: list
    """
    key_frames = []

    for idx, filename in enumerate(files):
        for metadata in exiftool.get_tags(filename, KEYWORD_KEY_FRAME):
            ic(metadata)
            label = metadata.get(KEYWORD_KEY_FRAME)
            if label and label == Options.key_frame_label:
                verbose(f"key frame at [{idx}] {KEYWORD_KEY_FRAME}={Options.key_frame_label}")
                key_frames.append(idx)

    return key_frames



def _2float(v) -> float:
    try:
        v = float(v)
    except ValueError:
        pass
    return v


def _2fstr(v: float) -> str:
    return f"{v:+.2f}" if Options.float2 else f"{v:f}"


def _interpolate(idx1: int, idx2: int, i: int, v1: float, v2: float) -> float:
    return v1 + (v2 - v1)*(i - idx1)/(idx2 - idx1)


def process_key_frames(exiftool: ExifToolHelper, files: list, idx1: int, idx2: int) -> None:
    verbose(f"processing key frame pair [{idx1}:{idx2}]")

    frame1 = {}
    frame2 = {}

    # Interpolate meta data for keywords
    for metadata in exiftool.get_tags(files[idx1], Options.keys):
        for k, v in metadata.items():
            frame1[k] = _2float(v)
    for metadata in exiftool.get_tags(files[idx2], Options.keys):
        for k, v in metadata.items():
            frame2[k] = _2float(v)
    ic(frame1, frame2)    

    for k in frame1.keys():
        v1 = frame1[k]
        v2 = frame2[k]
        if isinstance(v1, float) and isinstance(v2, float):
            for i in range(idx1, idx2):
                v = _2fstr(_interpolate(idx1, idx2, i, v1, v2))
                ic(idx1, idx2, i, v1, v2, k, v)
                verbose(f"setting [{i}] {k}={v}")
                if not Options.no_change:
                    exiftool.set_tags(files[i], tags={k: v})



def process_dir(exiftool: ExifToolHelper, dir: str) -> None:
    verbose(f"processing {dir=}")
    img_files = sort_file_list([f for f in os.listdir(dir) 
                                if  f.lower().endswith(".jpg") or 
                                    f.lower().endswith(".tif") or
                                    f.lower().endswith(".xmp")    ])
    img_files_full_path = [ os.path.join(dir, f) for f in img_files]
    # print(img_files_full_path)

    # List meta data only
    if Options.list:
        for filename in img_files_full_path:
            process_image(exiftool, filename)

    # Process meta data and interpolate numeric values
    if Options.interpolate:
        key_frames = find_key_frames(exiftool, img_files_full_path)
        ic(key_frames)

        for idx1, idx2 in pairwise(key_frames):
            ic(idx1, idx2)
            process_key_frames(exiftool, img_files_full_path, idx1, idx2)



def process_image(exiftool: ExifToolHelper, filename: str) -> None:
    seq = get_seq(filename)
    verbose(f"[{seq}] _FILE: {filename}")

    # Meta data
    for metadata in exiftool.get_metadata(filename):
        for k, v in metadata.items():
            if Options.all or (Options.match and Options.match in k) or k in Options.keys:
                verbose(f"{k}: {v}")
                print(f"{seq}\t{v}")



def main():
    arg = argparse.ArgumentParser(
        prog        = NAME,
        description = DESCRIPTION,
        epilog      = "Version " + VERSION + " / " + AUTHOR)
    arg.add_argument("-v", "--verbose", action="store_true", help="verbose messages")
    arg.add_argument("-d", "--debug", action="store_true", help="more debug messages")
    arg.add_argument("-l", "--list", action="store_true", help="list meta data, default: "+", ".join(Options.keys))
    arg.add_argument("-a", "--all", action="store_true", help="output all meta data")
    arg.add_argument("-m", "--match", help="output meta data keywords containing MATCH")
    arg.add_argument("-k", "--keywords", help=f"use meta data KEYWORDS, \"+\" adds")
    arg.add_argument("--crop", action="store_true", help="use CropTop/Left/Bottom/Right/Angle keywords")
    arg.add_argument("-2", "--float2", action="store_true", help="output float numbers as +/-D.DD")
    arg.add_argument("-i", "--interpolate", action="store_true", help="interpolate numeric meta data values for KEYWORDS")
    arg.add_argument("-n", "--no-change", action="store_true", help="dry run, no change to meta data")
    arg.add_argument("image", nargs="+", help="image file or directory")

    args = arg.parse_args()

    if args.debug:
        ic.enable()
        ic(sys.version_info, sys.path, args)
    if args.verbose:
        verbose.set_prog(NAME)
        verbose.enable()
    Options.list = args.list
    Options.all = args.all
    Options.match = args.match
    Options.interpolate = args.interpolate
    Options.no_change = args.no_change
    Options.float2 = args.float2
    if args.crop:
        Options.keys = KEYWORDS_CROP
    if args.keywords:
        h = args.keywords
        if h.startswith("+"):
            h = h.lstrip("+")
            h = h.lstrip(",")
            Options.keys.extend(h.split(","))
        else:
            Options.keys = h.split(",")
            verbose(f"keywords {",".join(Options.keys)}")

    try:
        with ExifToolHelper(executable=EXIFTOOL_EXE) as exiftool:
            for file in args.image:
                if os.path.isfile(file):
                    process_image(exiftool, file)
                elif os.path.isdir(file):
                    process_dir(exiftool, file)
                else:
                    warning(f"{file}: no such file or directory")
    except KeyboardInterrupt:
        message("Cancelled by ^C")


if __name__ == "__main__":
    main()
