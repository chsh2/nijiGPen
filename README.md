# NijiGPen

[[English]](README.md) [[中文]](README_zh.md)

NijiGPen is a [Blender](https://www.blender.org/) add-on that brings new features to Grease Pencil for creating 2D graphic design and illustrations. It provides with the following functions:

- Boolean and Offset operations for 2D shapes
- Conversion from 2D shapes to 3D meshes
- Data exchange with other painting/designing tools

Please note that the development of this add-on is still in an early stage. Bugs may exist and UI/operators may change frequently.

## Requirements

Blender 3.x (3.3 for full features)

(At the current stage, the add-on focuses on stable verions only. It is not guaranteed that all functions can work normally in alpha or beta verions of Blender.)

## Installation

Different from most add-ons, NijiGPen heavily relies on third-party Python packages. Therefore, the installation process has an extra step.

1. Download the archive file from the Release page.

2. In Blender, open [Edit]->[Preferences]->[Add-ons] and click the Install button to load the archive.

3. **Enable the NijiGPen add-on, and a new panel "Dependency Management" will appear. Click "Refresh" to check if packages are installed, and click "Install" button for those missing. This step requires the Internet connection.**

![image](https://user-images.githubusercontent.com/110356534/199868050-60927e38-88fe-422c-9495-aae62986f9c5.png)

It is recommended to use this add-on with the portable version of Blender, or Blender installed in a non-system folder through the official installer. Otherwise, additional actions may be required. Please refer to the [Wiki page](https://github.com/chsh2/nijiGPen/wiki/Dependency-Installation) for more information.

## Upgrade
**Blender Upgrade:** After upgrading Blender to a new version, the Installation Step 3 (Dependency Management) should be executed again.

**Add-on Upgrade:** If you want to replace the installed add-on with a newer version, it is recommended to remove the old version first and perform each installation step again.

## Usage

In **Draw** and **Edit** modes of a Grease Pencil object, a label titled “NijiGP” will appear in the sidebar of 3D Viewport window. Most of this add-on’s operators can be found here.

In **Draw** and **Sculpt** modes of a Grease Pencil object, extra undo/redo shortcut buttons will appear in the Properties panel. This feature can be disabled in Preferences setting.

Some of the operations are demonstrated in the following videos:

https://www.youtube.com/watch?v=xRzwWkjkBUY

https://www.bilibili.com/video/bv1tg411C77g

## List of Functions

NijiGPen provides with the following functions:

- 2D Polygon Algorithms
    - Boolean Operation: Union, Difference and Intersection
        - Perform Boolean in Edit mode, on selected strokes
        - Perform Boolean in Draw mode, with the last drawn stroke
        - Both fill and line width can be used as the clip shape
    - Offset/Inset Operation
        - Fill Mode: handle self-overlapping correctly
        - Line Mode: turn a line into a shape
        - Corner Mode: a bevel-like effect
        - Color tint
- 3D Mesh Generation
    - Frustum: multiple slope/corner styles available
    - Planar: with a normal map and therefore 3D shading
- Import
    - Paste SVG Shapes: extend the built-in SVG module with clipboard reading and hole detection
    - Paste XML Palette: convert XML codes from services such as Adobe Color to a Blender palette
    - Extract Line Art from Image
- UI
    - Extra Undo/Redo Button: for the convenience of Windows touchscreen users; can be disabled if not needed

New functions may be added in future releases.

## Limitations

Currently, a number of limitations exist in different aspects. Some of them may be improved in future releases.

### General

- All 2D operators assume that the user is painting in either XY, XZ or YZ plane. To use them in other planes, one way is to rotate the Grease Pencil object (without applying the rotation).

### Offset/Boolean Operators

- Vertex groups will not be kept after either an Offset or a Boolean operation.
- If a Boolean operation leads to holes, corresponding strokes will generate but do not appear transparent. You can apply Holdout materials manually.

### SVG Pasting

- Hole detection may fail for complex shapes.

### Mesh Generation

- Vertex color layer is not generated in Blender 3.1 or versions below. 

## Credits

- [Pyclipper](https://github.com/fonttools/pyclipper) wrapper by [Maxime Chalton](https://sites.google.com/site/maxelsbackyard/home/pyclipper) and the [Clipper](http://www.angusj.com/delphi/clipper.php) library by [Angus Johnson](http://www.angusj.com/delphi/clipper.php)
- [Triangle](https://github.com/drufat/triangle) by Dzhelil Rufat and [Jonathan Richard Shewchuk](http://www.cs.berkeley.edu/~jrs)
- [Scikit-image](https://scikit-image.org/) 
