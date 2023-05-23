# NijiGPen

[[English]](README.md) [[中文]](README_zh.md)

NijiGPen是作用于Grease Pencil（蜡笔）的Blender插件，关注Blender在2D平面设计与绘画领域的应用。它的功能如下：

- 补全一些基础2D几何操作，如布尔运算和偏移运算。
- 修饰与清理手绘线条。
- 将手绘2D图形转换为3D网格。
- 与其它设计/绘图工具的数据互通。

目前，本插件尚处于早期开发阶段，性能有可能不稳定，功能与UI也还在频繁修改中，使用时请注意。

## 需求

Blender 3.3以上版本

（建议使用稳定版本的Blender，对于alpha或beta版本，无法保证所有功能仍可正常使用。）

## 安装说明

与多数插件不同，NijiGPen的功能大量依赖于额外的Python包。因此除通常的安装流程外，还需要一个额外的步骤：

1. 在Releases页面下载zip文件。
2. 在Blender的【偏好设置】面板中安装该zip文件。
3. **启用插件，面板中将出现如下图所示的Python包管理器。点击按钮以检测状态，或安装缺失的包。**

<img src="https://user-images.githubusercontent.com/110356534/229329621-ccfd0407-af1d-442d-b2f7-05ba7dc54ade.png" width=50% height=50%>

注意：

 - 这一过程中要求网络畅通并且能够访问到pip的源服务器。如有困难，请考虑使用VPN或修改pip源等方法。
 - 推荐将Blender安装于非系统目录，否则可能需要使用管理员权限运行Blender才能成功安装Python包。详情请查看[Wiki页面](https://github.com/chsh2/nijiGPen/wiki/Dependency-Installation)。

## 升级说明

如果更换Blender版本，需要重新执行上述安装说明的第三步（Python包管理）。

如果需要安装新版本的此插件，建议完全移除旧版本，之后重新执行每一个安装步骤。

## 使用说明

当处于Grease Pencil的**编辑模式**或**绘制模式**时，3D视图的侧边栏会出现名为“NijiGP”的面板，在其中可使用本插件的全部功能。

另外，处于绘制、编辑或雕刻模式时，屏幕下方将出现一组快捷按钮。该功能可在偏好设置中关闭。

演示视频：

https://www.youtube.com/playlist?list=PLEgTVZ2uBvPMM0sGzzQTyoV0or8_PTs6t

https://www.bilibili.com/video/bv1tg411C77g

## 功能列表

目前插件有如下功能：

- 2D图形运算
    - 布尔：并集、交集、相减
    - 偏移：线条的形状、轮廓线或拐角均可外扩/内缩
- 2D线条运算
    - 线稿清理：从多条手绘线条中提取出单一平滑线条
    - 智能填充：根据标注的颜色提示，自动对线稿上色
- 3D生成
    - 生成多种样式的3D网格物体（可选三角或方格、法线贴图或3D高度）
    - 内置卡通渲染材质
- 导入
    - 从剪贴板粘贴XML色卡：将Adobe Color等工具生成的XML格式色卡或Hex代码转换为Blender调色板
    - 从剪贴板粘贴SVG代码：在Blender自带SVG模块的基础上增加了检测孔洞的功能
    - 图片转换为矢量图：在Blender自带模块的基础上增加了线稿导入与彩色图片导入
    - GBR/ABR笔刷：提取GIMP/Adobe Photoshop笔刷中的贴图并转为Blender可用的格式
- 导出
    - 渲染为PSD：保持图层结构与混合模式

其它功能将陆续添加。

## 限制

现阶段，插件的某些功能受到限制，详情请见[Wiki页面](https://github.com/chsh2/nijiGPen/wiki/Dependency-Installation).

若发现其它问题，可在Issues页面提出。

## 致谢

本插件使用了下列Python库：

- [Pyclipper](https://github.com/fonttools/pyclipper) wrapper by [Maxime Chalton](https://sites.google.com/site/maxelsbackyard/home/pyclipper) and the [Clipper](http://www.angusj.com/delphi/clipper.php) library by [Angus Johnson](http://www.angusj.com/delphi/clipper.php)
- [Triangle](https://github.com/drufat/triangle) by Dzhelil Rufat and [Jonathan Richard Shewchuk](http://www.cs.berkeley.edu/~jrs)
- [SciPy](https://scipy.org/)
- [Scikit-image](https://scikit-image.org/) 

另外，虽然没有直接使用代码或算法，本插件的功能受到以下研究工作的启发：

- Lee, In-Kwon. "Curve reconstruction from unorganized points." Computer aided geometric design 17, no. 2 (2000): 161-177.
- Liu, Chenxi, Enrique Rosales, and Alla Sheffer. "Strokeaggregator: Consolidating raw sketches into artist-intended curve drawings." ACM Transactions on Graphics (TOG) 37, no. 4 (2018): 1-15.
- Dvorožňák, Marek, Daniel Sýkora, Cassidy Curtis, Brian Curless, Olga Sorkine-Hornung, and David Salesin. "Monster mash: a single-view approach to casual 3D modeling and animation." ACM Transactions on Graphics (TOG) 39, no. 6 (2020): 1-12.
- Johnston, Scott F. "Lumo: illumination for cel animation." In Proceedings of the 2nd international symposium on Non-photorealistic animation and rendering, pp. 45-ff. 2002.
- Sýkora, Daniel, John Dingliana, and Steven Collins. "Lazybrush: Flexible painting tool for hand‐drawn cartoons." In Computer Graphics Forum, vol. 28, no. 2, pp. 599-608. Oxford, UK: Blackwell Publishing Ltd, 2009.
- Parakkat, Amal Dev, Pooran Memari, and Marie‐Paule Cani. "Delaunay Painting: Perceptual Image Colouring from Raster Contours with Gaps." In Computer Graphics Forum, vol. 41, no. 6, pp. 166-181. 2022.
