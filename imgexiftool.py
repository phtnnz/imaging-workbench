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
# Version 0.0 / 2025-07-07
#       Test PyExifTool handling

VERSION = "0.0 / 2025-07-07"
AUTHOR  = "Martin Junius"
NAME    = "imgexiftool"

import sys
import argparse
import os
import re

# The following libs must be installed with pip
from exiftool import ExifTool, ExifToolHelper

from icecream import ic
# Disable debugging
ic.disable()
# Local modules
from verbose import verbose, warning, error



# Command line options
class Options:
    img_list = None             # -i --img-list
    output   = "tmp"            # -o --output



# Global exiftool object
EXIFTOOL_EXE = "c:/tools/exiftool/exiftool.exe"



def process_dir(exiftool: ExifToolHelper, dir: str) -> None:
    verbose(f"processing {dir=}")
    img_files = [f for f in os.listdir(dir) if f.lower().endswith(".jpg") or f.lower().endswith(".tif")]
    ic(img_files)

    for f in img_files:
        filename = os.path.join(dir, f)
        process_image(exiftool, filename)



def process_image(exiftool: ExifToolHelper, filename: str) -> None:
    verbose(f"processing image {filename}")

    # Sequence/page number
    m = re.search(r'(\d\d+)', filename)
    seq = int(m.group(1)) if m else 0
    verbose(f"{seq=}")

    # Meta data
    for metadata in exiftool.get_metadata(filename):
        for k, v in metadata.items():
            verbose(f"  {k} = {v}")
        title = metadata.get("XMP:Title")
        if title:
            print(f"{seq}\t{title}")



def main():
    arg = argparse.ArgumentParser(
        prog        = NAME,
        description = "PyExifTool test script",
        epilog      = "Version " + VERSION + " / " + AUTHOR)
    arg.add_argument("-v", "--verbose", action="store_true", help="verbose messages")
    arg.add_argument("-d", "--debug", action="store_true", help="more debug messages")
    arg.add_argument("image", nargs="+", help="image file or directory")

    args = arg.parse_args()

    if args.debug:
        ic.enable()
        ic(sys.version_info, sys.path, args)
    if args.verbose:
        verbose.set_prog(NAME)
        verbose.enable()

    with ExifToolHelper(executable=EXIFTOOL_EXE) as exiftool:
        for file in args.image:
            if os.path.isfile(file):
                process_image(exiftool, file)
            elif os.path.isdir(file):
                process_dir(exiftool, file)



if __name__ == "__main__":
    main()
