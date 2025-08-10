# NijiGPen: Blender Grease Pencil Add-on

[[ 📖 Documentation ](https://chsh2.github.io/nijigp/)] | [[ 🎥 YouTube Demos ](https://www.youtube.com/playlist?list=PLEgTVZ2uBvPMM0sGzzQTyoV0or8_PTs6t)]

NijiGPen is a [Blender](https://www.blender.org/) add-on that brings new features to Grease Pencil for creating 2D graphic design and illustrations. It provides with the following functions:

- 2D algorithms for vector shapes and strokes (e.g., Boolean and Offset)
- Refinement and cleanup of hand-drawn 2D strokes
- 3D mesh/shading generation from 2D shapes
- Data exchange with other painting/designing tools

## Installation

Requirement: 

- Blender 3.3 ~ 4.2 (Grease Pencil 2), or
- Blender 4.3 ~ 4.5 (Grease Pencil 3)

Please follow the steps in [this page](https://chsh2.github.io/nijigp/docs/get_started/installation/) to finish the installation.

This add-on requires **third-party Python packages**. Please check [these details](https://chsh2.github.io/nijigp/docs/get_started/installation/dependency/) if you have any questions during installation.

## Features

![intro1](https://github.com/chsh2/nijiGPen/assets/110356534/82e82ae5-611e-48c1-8ba7-d75a319dde71)
![intro2](https://github.com/chsh2/nijiGPen/assets/110356534/336eeab5-93dd-468a-9c23-8ead9ad741d3)

## Credits

The functions of this add-on are implemented with the following packages:

- [Pyclipper](https://github.com/fonttools/pyclipper) wrapper by [Maxime Chalton](https://sites.google.com/site/maxelsbackyard/home/pyclipper) and the [Clipper](http://www.angusj.com/delphi/clipper.php) library by [Angus Johnson](http://www.angusj.com/delphi/clipper.php)
- [Triangle](https://github.com/drufat/triangle) by Dzhelil Rufat and [Jonathan Richard Shewchuk](http://www.cs.berkeley.edu/~jrs)
- [SciPy](https://scipy.org/)
- [Scikit-image](https://scikit-image.org/) 

Besides, although not using the codes directly or implementing the same algorithm, the functions of this add-on are largely inspired by the following works:

 - Lee, In-Kwon. "Curve reconstruction from unorganized points." Computer aided geometric design 17, no. 2 (2000): 161-177.
 - Liu, Chenxi, Enrique Rosales, and Alla Sheffer. "Strokeaggregator: Consolidating raw sketches into artist-intended curve drawings." ACM Transactions on Graphics (TOG) 37, no. 4 (2018): 1-15.
 - Dvorožňák, Marek, Daniel Sýkora, Cassidy Curtis, Brian Curless, Olga Sorkine-Hornung, and David Salesin. "Monster mash: a single-view approach to casual 3D modeling and animation." ACM Transactions on Graphics (TOG) 39, no. 6 (2020): 1-12.
 - Johnston, Scott F. "Lumo: illumination for cel animation." In Proceedings of the 2nd international symposium on Non-photorealistic animation and rendering, pp. 45-ff. 2002.
 - Hudon, Matis, Sebastian Lutz, Rafael Pagés, and Aljosa Smolic. "Augmenting hand-drawn art with global illumination effects through surface inflation." In Proceedings of the 16th ACM SIGGRAPH European Conference on Visual Media Production, pp. 1-9. 2019.
 - Sýkora, Daniel, John Dingliana, and Steven Collins. "Lazybrush: Flexible painting tool for hand‐drawn cartoons." In Computer Graphics Forum, vol. 28, no. 2, pp. 599-608. Oxford, UK: Blackwell Publishing Ltd, 2009.
 - Parakkat, Amal Dev, Pooran Memari, and Marie‐Paule Cani. "Delaunay Painting: Perceptual Image Colouring from Raster Contours with Gaps." In Computer Graphics Forum, vol. 41, no. 6, pp. 166-181. 2022.
