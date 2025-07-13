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

VERSION     = "0.1 / 2025-07-13"
AUTHOR      = "Martin Junius"
NAME        = "lr-xmp-tool"
DESCRIPTION = "Lightroom Classic XMP manipulation tool"

EXIFTOOL_EXE = "c:/Tools/exiftool/exiftool.exe"

import sys
import argparse
import os
import re

# The following libs must be installed with pip
from exiftool import ExifToolHelper

from icecream import ic
# Disable debugging
ic.disable()
# Local modules
from verbose import message, verbose, warning, error



# Relevant keywords
KEYWORD_KEY_FRAME = "XMP:Label"



# Command line options
class Options:
    keys = [ "XMP:CountryCode", "XMP:State", "XMP:City", "XMP:Location" ]
                                    # -K --keywords
    get_list = False                # --get-list
    get_title = False               # --get-title
    all = False                     # -a --all
    match = None                    # -m --match
    key_frame_label = "Yellow"      # -L --key-frame-label



def get_seq(filename: str) -> str:
    """
    Get sequence number from filename (last min 2 digits match)
    e.g. 2023-19namibia-207-9628.xmp -> 9628

    :param filename: name
    :type filename: str
    :return: sequence number
    :rtype: str
    """
    m = re.findall(r'(\d\d+)', filename)
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
            verbose(f"jump in sequence at [{idx}] {last_seq}->{seq}")
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
        for metadata in exiftool.get_metadata(filename):
            label = metadata.get(KEYWORD_KEY_FRAME)
            if label and label == Options.key_frame_label:
                verbose(f"key frame at [{idx}] {KEYWORD_KEY_FRAME}={Options.key_frame_label}")
                key_frames.append(idx)

    return key_frames



def process_dir(exiftool: ExifToolHelper, dir: str) -> None:
    verbose(f"processing {dir=}")
    img_files = sort_file_list([f for f in os.listdir(dir) 
                                if  f.lower().endswith(".jpg") or 
                                    f.lower().endswith(".tif") or
                                    f.lower().endswith(".xmp")    ])
    img_files_full_path = [ os.path.join(dir, f) for f in img_files]
    # print(img_files_full_path)
    key_frames = find_key_frames(exiftool, img_files_full_path)
    ic(key_frames)



def process_image(exiftool: ExifToolHelper, filename: str) -> None:
    verbose(f"processing image {filename}")

    # Meta data
    for metadata in exiftool.get_metadata(filename):
        for k, v in metadata.items():
            if Options.all or (Options.match and Options.match in k) or k in Options.keys:
                verbose(f"  {k} = {v}")

        if Options.get_title:
            title = metadata.get("XMP:Title")
            if title:
                print(f"{title}")

        if Options.get_list:
            data = [ metadata.get(k) or "n/a" for k in Options.keys ]
            if data:
                print(", ".join(data), sep="\t")



def main():
    arg = argparse.ArgumentParser(
        prog        = NAME,
        description = DESCRIPTION,
        epilog      = "Version " + VERSION + " / " + AUTHOR)
    arg.add_argument("-v", "--verbose", action="store_true", help="verbose messages")
    arg.add_argument("-d", "--debug", action="store_true", help="more debug messages")
    arg.add_argument("-K", "--keywords", help=f"show meta data for KEYWORDS, \"+\" adds")
    arg.add_argument("-a", "--all", action="store_true", help="output all meta data")
    arg.add_argument("-m", "--match", help="output meta data keywords containing MATCH")
    arg.add_argument("--get-list", action="store_true", help="get meta data list, default: "+", ".join(Options.keys))
    arg.add_argument("--get-title", action="store_true", help="get meta data: XMP:Title")
    arg.add_argument("image", nargs="+", help="image file or directory")

    args = arg.parse_args()

    if args.debug:
        ic.enable()
        ic(sys.version_info, sys.path, args)
    if args.verbose:
        verbose.set_prog(NAME)
        verbose.enable()
    Options.get_list = args.get_list
    Options.get_title = args.get_title
    Options.all = args.all
    Options.match = args.match
    if args.keywords:
        h = args.keywords
        if h.startswith("+"):
            h = h.lstrip("+")
            h = h.lstrip(",")
            Options.keys.extend(h.split(","))
        else:
            Options.keys = h.split(",")

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
