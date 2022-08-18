# NijiGPen

[[English]](README.md) [[中文]](README_zh.md)

NijiGPen is a [Blender](https://www.blender.org/) add-on that brings new features to Grease Pencil for 2D graphic design and illustrations, including:

- Boolean and Offset operations for 2D shapes
- Conversion from 2D shapes to 3D meshes
- Data exchange with other painting/designing tools

Please note that the development of this add-on is still in an early stage. Bugs may exist and UI/operators may change frequently.

## Requirements

Blender 3.x

(At the current stage, the add-on focuses on stable verions only. It is not guaranteed that all functions can work normally in alpha or beta verions of Blender.)

## Installation

This add-on requires additional Python packages ([Pyclipper](https://pypi.org/project/pyclipper/)). Therefore, the installation process has an extra step.

1. Download the archive file from the Release page.

2. In Blender, open [Edit]->[Preferences]->[Add-ons] and click the Install button to load the archive.

3. Enable the NijiGPen add-on, and click "Install Dependencies". This step requires the Internet connection.

Note: The third step enables pip and installs Python packages. It executes the following commands:
```
python -m ensurepip --upgrade
python -m pip install pyclipper
```
If this step stucks or fails, you can open a terminal in Blender's Python directory (e.g., `blender/3.xx/python/bin`) and execute the commands above manually.

## Upgrade
**Blender Upgrade:** After upgrading Blender to a new version, the Installation Step 3 (Install Dependencies) should be executed again.

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
    - Offset Method: multiple slope/corner styles available
- Import
    - Paste SVG Shapes: extend the built-in SVG module with clipboard reading and hole detection
    - Paste XML Palette: convert XML codes from services such as Adobe Color to a Blender palette
- UI
    - Extra Undo/Redo Button: for the convenience of Windows touchscreen users; can be disabled if not needed

New functions may be added in future releases.

## Limitations

Currently, a number of limitations exist in different aspects. Some of them may be improved in future releases.

### General

- All 2D operators assume that the user is painting in the XZ-plane. To use them in other planes, one way is to rotate the Grease Pencil object (without applying the rotation).

### Offset/Boolean Operators

- Vertex groups will not be kept after either an Offset or a Boolean operation.
- If a Boolean operation leads to holes, corresponding strokes will generate but do not appear transparent. You can apply Holdout materials manually.
- Point radius (pressure) is currently ignored in the Line mode of Offset/Boolean operations.

### SVG Pasting

- Hole detection may fail for complex shapes.

## Credits

- [Pyclipper](https://github.com/fonttools/pyclipper) wrapper by [Maxime Chalton](https://sites.google.com/site/maxelsbackyard/home/pyclipper) and the [Clipper](http://www.angusj.com/delphi/clipper.php) library by [Angus Johnson](http://www.angusj.com/delphi/clipper.php)