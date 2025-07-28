# Astropy-Workbench

Python Image Processing Scripts

Copyright 2024-2025 Martin Junius

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## About

This is just my personal playground for working with the Python imaging libraries and numpy,
not necessarily usable for anyone else.


## Lightroom XMP Tool

List and manipulate Lightroom Classic XMP sidecar data

```
usage: lr-xmp-tool [-h] [-v] [-d] [-l] [-a] [-m MATCH] [-k KEYWORDS] [-i] [-n] image [image ...]

Lightroom Classic XMP manipulation tool

positional arguments:
  image                 image file or directory

options:
  -h, --help            show this help message and exit
  -v, --verbose         verbose messages
  -d, --debug           more debug messages
  -l, --list            list meta data, default: XMP:Exposure2012
  -a, --all             output all meta data
  -m MATCH, --match MATCH
                        output meta data keywords containing MATCH
  -k KEYWORDS, --keywords KEYWORDS
                        use meta data KEYWORDS, "+" adds
  -i, --interpolate     interpolate numeric meta data values for KEYWORDS
  -n, --no-change       dry run, no change to meta data

Version 0.2 / 2025-07-14 / Martin Junius
```
