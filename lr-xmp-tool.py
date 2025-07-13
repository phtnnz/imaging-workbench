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
NAME        = "imgexiftool"
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
from verbose import verbose, warning, error



# Command line options
class Options:
    keys = [ "XMP:CountryCode", "XMP:State", "XMP:City", "XMP:Location" ]
                                    # -K --keywords
    get_list = False                # --get-list
    get_title = False               # --get-title
    all = False                     # -a --all
    match = None                    # -m --match
    key_frame_label = "Yellow"      # -L --key-frame-label

def process_dir(exiftool: ExifToolHelper, dir: str) -> None:
    verbose(f"processing {dir=}")
    img_files = [f for f in os.listdir(dir) if  f.lower().endswith(".jpg") or 
                                                f.lower().endswith(".tif") or
                                                f.lower().endswith(".xmp")    ]
    ic(img_files)

    for f in img_files:
        filename = os.path.join(dir, f)
        process_image(exiftool, filename)



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

    with ExifToolHelper(executable=EXIFTOOL_EXE) as exiftool:
        for file in args.image:
            if os.path.isfile(file):
                process_image(exiftool, file)
            elif os.path.isdir(file):
                process_dir(exiftool, file)
            else:
                warning(f"{file}: no such file or directory")


if __name__ == "__main__":
    main()
