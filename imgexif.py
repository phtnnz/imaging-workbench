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
#       Test PIL EXIF handling

VERSION = "0.0 / 2025-07-07"
AUTHOR  = "Martin Junius"
NAME    = "imgexif"

import sys
import argparse
import os

# The following libs must be installed with pip
from PIL import Image, ExifTags
from PIL import IptcImagePlugin

from icecream import ic
# Disable debugging
ic.disable()
# Local modules
from verbose import verbose, warning, error



# Command line options
class Options:
    img_list = None             # -i --img-list
    output   = "tmp"            # -o --output



def process_dir(dir: str):
    img_files = [f for f in os.listdir(dir) if f.endswith(".jpg") or f.endswith(".tif")]
    ic(img_files)



def process_image(filename: str) -> None:
    verbose(f"processing image {filename}")

    with Image.open(filename) as img:
        ic(img)
        exif = img.getexif()
        ic(exif)

        if exif:
            for k, v in exif.items():
                tag = ExifTags.TAGS.get(k)
                ic(tag, v)

            for id in ExifTags.IFD:
                ic(id)
                try:
                    ifd = exif.get_ifd(id)
                    if id == ExifTags.IFD.GPSInfo:
                        resolve = ExifTags.GPSTAGS
                    else:
                        resolve = ExifTags.TAGS
                    for k, v in ifd.items():
                        tag = resolve.get(k)
                        ic(tag, v)
                except KeyError:
                    pass
        else:
            warning(f"no EXIF information")

        xmp = img.getxmp()
        # xmp["xmpmeta"]["RDF"]["Description"][<FIELD>]
        country = xmp["xmpmeta"]["RDF"]["Description"]["Country"]
        state = xmp["xmpmeta"]["RDF"]["Description"]["State"]
        city = xmp["xmpmeta"]["RDF"]["Description"]["City"]
        # sub = xmp["xmpmeta"]["RDF"]["Description"]["Sub-location"]
        verbose("fields:")
        verbose(f"{country=} {state=} {city=}")

        # IPTC???
        iptc = IptcImagePlugin.getiptcinfo(img)
        ic(iptc)
        # Decode???



# Hack from https://stackoverflow.com/questions/6405208/how-to-convert-numeric-string-ranges-to-a-list-in-python
def str_to_list(s):
    return sum(((list(range(*[int(j) + k for k,j in enumerate(i.split('-'))]))
         if '-' in i else [int(i)]) for i in s.split(',')), [])



def main():
    arg = argparse.ArgumentParser(
        prog        = NAME,
        description = "PIL EXIF test script",
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

    for file in args.image:
        if os.path.isfile(file):
            process_image(file)
        elif os.path.isdir(file):
            process_dir(file)



if __name__ == "__main__":
    main()
