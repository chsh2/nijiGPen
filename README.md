# NijiGPen

[[English]](README.md) [[中文]](README_zh.md)

NijiGPen is a [Blender](https://www.blender.org/) add-on that brings new features to Grease Pencil for creating 2D graphic design and illustrations. It provides with the following functions:

- 2D polygon operations (Boolean and Offset)
- Refinement and cleanup of hand-drawn 2D strokes
- Conversion from 2D shapes to 3D meshes
- Data exchange with other painting/designing tools

Please note that the development of this add-on is still in an early stage. Bugs may exist and UI/operators may change frequently.

## Requirements

Blender 3.3 or later

(At the current stage, the add-on focuses on stable verions only. It is not guaranteed that all functions can work normally in alpha or beta verions of Blender.)

## Installation

Different from most add-ons, NijiGPen heavily relies on third-party Python packages. Therefore, the installation process has an extra step.

1. Download the archive file from the Release page.

2. In Blender, open [Edit]->[Preferences]->[Add-ons] and click the Install button to load the archive.

3. **Enable the NijiGPen add-on, and a new panel "Dependency Management" will appear. Click "Check" to check if packages are installed, and click "Install" button for those missing. This step requires the Internet connection.**

<img src="https://user-images.githubusercontent.com/110356534/229329621-ccfd0407-af1d-442d-b2f7-05ba7dc54ade.png" width=50% height=50%>

It is recommended to use this add-on with the portable version of Blender, or Blender installed in a non-system folder through the official installer. Otherwise, additional actions may be required. Please refer to the [Wiki page](https://github.com/chsh2/nijiGPen/wiki/Dependency-Installation) for more information.

## Upgrade
**Blender Upgrade:** After upgrading Blender to a new version, the Installation Step 3 (Dependency Management) should be executed again.

**Add-on Upgrade:** If you want to replace the installed add-on with a newer version, it is recommended to remove the old version first and perform each installation step again.

## Usage

In **Draw** and **Edit** modes of a Grease Pencil object, a label titled “NijiGP” will appear in the sidebar of 3D Viewport window. Most of this add-on’s operators can be found here.

In **Draw**, **Edit** and **Sculpt** modes of a Grease Pencil object, a group of shortcut buttons will appear at the bottom of the viewport. This feature can be disabled in Preferences setting.

Some of the operations are demonstrated in the following videos:

https://www.youtube.com/watch?v=xRzwWkjkBUY

https://www.bilibili.com/video/bv1tg411C77g

## List of Functions

NijiGPen provides with the following functions:

- 2D Polygon Algorithms
    - Boolean Operation: Union, Difference and Intersection
    - Offset/Inset Operation on either fill, outline or corners of a stroke
- 2D Line Algorithms
    - Sketch Cleanup: fitting multiple strokes into a single smooth one
- 3D Mesh Generation
    - Different types of meshes can be converted from 2D strokes: triangle or grid, 3D depth or normal maps
    - A series of stylized shaders
- Import
    - Paste SVG Shapes: extend the built-in SVG module with clipboard reading and hole detection
    - Paste XML Palette: convert XML codes from services such as Adobe Color to a Blender palette
    - Raster Image Tracing: line art and multi-color support
- UI
    - A group of shortcut buttons for better touchscreen control

New functions may be added in future releases.

## Limitations

Please refer to the [Wiki page](https://github.com/chsh2/nijiGPen/wiki/Known-Issues).

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

