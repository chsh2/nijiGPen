# NijiGPen

[[English]](README.md) [[中文]](README_zh.md)

NijiGPen是作用于Grease Pencil（蜡笔）的Blender插件，关注Blender在2D平面设计与绘画领域的应用。它的功能如下：

- 补全一些基础的2D功能，如Grease Pencil的布尔运算和偏移运算。
- 将手绘2D图形转换为3D网格。
- 与其它设计/绘图工具的数据互通。

目前，本插件尚处于早期开发阶段，性能有可能不稳定，功能与UI也还在频繁修改中，使用时请注意。

## 需求

Blender 3.0以上版本

## 安装说明

NijiGPen的功能基于额外的Python包（[Pyclipper](https://pypi.org/project/pyclipper/)），与通常插件相比多出一个安装步骤：

1. 在Releases页面下载zip文件。
2. 在Blender的【偏好设置】面板中安装该zip文件。
3. 启用插件，并点击“Install Dependencies”按钮。

第三步实际上执行了以下操作：
```
python -m ensurepip --upgrade
python -m pip install pyclipper
```
如果第三步卡住或失败，可以尝试在命令行工具中进入Blender的Python目录（该目录通常是`blender/3.xx/python/bin`这样的命名）并手动输入这些命令。

注意：这一过程中要求网络畅通并且能够访问到pip的源服务器。如有困难，请考虑使用VPN或修改pip源等方法。

## 使用说明

当处于Grease Pencil的**编辑模式**或**绘制模式**时，3D视图的侧边栏会出现名为“NijiGP”的面板，在其中可使用本插件的全部功能。

另外，处于绘制或雕刻模式时，属性面板中会出现额外的撤销/重做按钮。该功能可在偏好设置中关闭。

## 功能列表

目前插件有如下功能：

- 2D图形运算
    - 布尔：并集、交集、相减
    - 偏移
        - 图形膨胀与收缩
        - 将线条变为图形
- 3D生成
    - 基于偏移的算法：生成多种样式的3D网格物体（斜面、圆角和等高线图）
- 导入
    - 从剪贴板粘贴XML色卡：将Adobe Color等工具生成的XML格式色卡转换为Blender调色板
    - 从剪贴板粘贴SVG代码：在Blender自带SVG模块的基础上增加了检测孔洞的功能

其它功能将陆续添加。

## 限制

现阶段，插件的某些功能受到限制，其中一部分可能在将来的版本中修复：

- 插件假设使用者在XZ平面绘制2D图形，并将Y轴视为深度。如果想要在其它平面中使用功能，请旋转对应的Grease Pencil物体，但不要应用变换。
- 执行布尔或偏移运算后，笔画的顶点组数据不会保持。
- 在布尔或偏移运算的线条模式中，压感暂时不会被考虑（线条的每个点被视作同样宽）。
- 如果布尔运算使图形出现孔洞，孔洞不会显示为透明。可手动将其改为具有“阻隔”属性的材质来解决问题。
- 粘贴SVG代码时，如果图形较为复杂，孔洞自动检测功能有可能失效。

若发现其它问题，可在Issues页面提出。

## 致谢

- [Pyclipper](https://github.com/fonttools/pyclipper) wrapper by [Maxime Chalton](https://sites.google.com/site/maxelsbackyard/home/pyclipper) and the [Clipper](http://www.angusj.com/delphi/clipper.php) library by [Angus Johnson](http://www.angusj.com/delphi/clipper.php)