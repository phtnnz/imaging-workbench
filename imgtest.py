#!/usr/bin/env python

# Copyright 2024 Martin Junius
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
# Version 0.0 / 2024-08-19
#       TEXT

import sys
import argparse
import os

# The following libs must be installed with pip
from PIL import Image
import numpy as np
from numpy.polynomial import Polynomial

from icecream import ic
# Disable debugging
ic.disable()
# Local modules
from verbose import verbose, warning, error

VERSION = "0.0 / 2024-08-19"
AUTHOR  = "Martin Junius"
NAME    = "imgtest"



# Command line options
class Options:
    img_list = None             # -i --img-list



def process_dir(dir: str):
    img_files = [f for f in os.listdir(dir) if f.endswith(".jpg") or f.endswith(".tif")]
    ic(img_files)

    # For linear regression of mean/median value
    x_list = []
    y_list = []

    # Get mean values from all images
    ic("--- mean values ---")
    for idx, img_file in enumerate(img_files):
        idx += 1    # 1 ... n
        if Options.img_list and not idx in Options.img_list:
            continue

        file = os.path.join(dir, img_file)
        verbose(f"loading [{idx}] {file=}")
        with Image.open(file) as img:
            img.load()
            ic(img)
            # img.show()
            data = np.array(img)
            # ic(data)
            ic(data.mean(), np.median(data), data.min(), data.max())
            # x = index, y = mean value
            x_list.append(idx)
            y_list.append(np.median(data))

    # Compute regression and target mean values
    ic("--- regression ---")
    ic(x_list, y_list)
    x = np.array(x_list)
    y = np.array(y_list)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    ic(m, c)
    med_target = m*x + c
    ic(med_target)

    # Poly fit see
    # https://stackoverflow.com/questions/18767523/fitting-data-with-numpy
    # https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html

    # Process all images
    ic("--- adjust image values ---")
    for idx, img_file in enumerate(img_files):
        idx += 1    # 1 ... n
        if Options.img_list and not idx in Options.img_list:
            continue

        file = os.path.join(dir, img_file)
        verbose(f"loading [{idx}] {file=}")
        with Image.open(file) as img:
            img.load()
            ic(img)
            data = np.array(img)
            med  = np.median(data)
            med1 = med_target[idx-1]
            ic(med, med1)
            # data to fit
            x = [0, med, 255]
            y = [0, med1, 255]
            poly = Polynomial.fit(x, y, deg=2)
            ic(poly)
            ic(poly(med))
            data = poly(data).astype(np.uint8)
            med2 = np.median(data)
            ic(med2)
            verbose(f"  median: orig={med:.0f} target={med1:.3f} new={med2:.0f}")
            # break

    ic("--- end ---")



# Hack from https://stackoverflow.com/questions/6405208/how-to-convert-numeric-string-ranges-to-a-list-in-python
def str_to_list(s):
    return sum(((list(range(*[int(j) + k for k,j in enumerate(i.split('-'))]))
         if '-' in i else [int(i)]) for i in s.split(',')), [])



def main():
    arg = argparse.ArgumentParser(
        prog        = NAME,
        description = "Generic python script template",
        epilog      = "Version " + VERSION + " / " + AUTHOR)
    arg.add_argument("-v", "--verbose", action="store_true", help="verbose messages")
    arg.add_argument("-d", "--debug", action="store_true", help="more debug messages")
    arg.add_argument("-i", "--img-list", help="index list of images to process")
    arg.add_argument("directory", nargs="+", help="image directory")

    args = arg.parse_args()

    if args.debug:
        ic.enable()
        ic(sys.version_info, sys.path, args)
    if args.verbose:
        verbose.set_prog(NAME)
        verbose.enable()
    # ... more options ...
    if args.img_list:
        Options.img_list = str_to_list(args.img_list)
    verbose("img list =", Options.img_list)
        
    for dir in args.directory:
        process_dir(dir)



if __name__ == "__main__":
    main()
