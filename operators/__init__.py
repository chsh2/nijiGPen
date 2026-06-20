_needs_reload = "bpy" in locals()

import bpy
from . import (
    common,
    operator_io_assets,
    operator_io_paste,
    operator_io_raster,
    operator_io_render,
    operator_color,
    operator_fill,
    operator_line,
    operator_mesh,
    operator_polygon,
    operator_polygon_shading,
    operator_rig,
)

if _needs_reload:
    import importlib

    common = importlib.reload(common)
    operator_io_assets = importlib.reload(operator_io_assets)
    operator_io_paste = importlib.reload(operator_io_paste)
    operator_io_raster = importlib.reload(operator_io_raster)
    operator_io_render = importlib.reload(operator_io_render)
    operator_color = importlib.reload(operator_color)
    operator_fill = importlib.reload(operator_fill)
    operator_line = importlib.reload(operator_line)
    operator_mesh = importlib.reload(operator_mesh)
    operator_polygon = importlib.reload(operator_polygon)
    operator_polygon_shading = importlib.reload(operator_polygon_shading)
    operator_rig = importlib.reload(operator_rig)

_all_classes = [
    # common
    common.ColorTintPropertyGroup,
    # operator_io_assets
    operator_io_assets.ImportBrushOperator,
    operator_io_assets.ImportSwatchOperator,
    operator_io_assets.AppendSVGOperator,
    # operator_io_paste
    operator_io_paste.PasteSVGOperator,
    operator_io_paste.PasteSwatchOperator,
    # operator_io_raster
    operator_io_raster.ImportLineImageOperator,
    operator_io_raster.ImportColorImageOperator,
    operator_io_raster.RenderAndVectorizeOperator,
    # operator_io_render
    operator_io_render.MultiLayerRenderOperator,
    # operator_color
    operator_color.TintSelectedOperator,
    operator_color.RecolorSelectedOperator,
    # operator_fill
    operator_fill.SmartFillOperator,
    operator_fill.HatchFillOperator,
    # operator_line
    operator_line.FitSelectedOperator,
    operator_line.SelectSimilarOperator,
    operator_line.ClusterAndFitOperator,
    operator_line.FitLastOperator,
    operator_line.PinchSelectedOperator,
    operator_line.TaperSelectedOperator,
    # operator_mesh
    operator_mesh.MeshManagement,
    operator_mesh.MeshGenerationByNormal,
    operator_mesh.MeshGenerationByOffsetting,
    # operator_polygon
    operator_polygon.HoleProcessingOperator,
    operator_polygon.OffsetSelectedOperator,
    operator_polygon.FractureSelectedOperator,
    operator_polygon.BoolSelectedOperator,
    operator_polygon.BoolLastOperator,
    operator_polygon.SweepSelectedOperator,
    # operator_polygon_shading
    operator_polygon_shading.ShadeSelectedOperator,
    # operator_rig
    operator_rig.VertexGroupClearOperator,
    operator_rig.PinRigOperator,
    operator_rig.MeshFromArmOperator,
    operator_rig.TransferWeightOperator,
    operator_rig.LayersToGroupsOperator,
    operator_rig.BakeRiggingOperator,
]


def register_classes():
    for cls in _all_classes:
        bpy.utils.register_class(cls)


def unregister_classes():
    for cls in reversed(_all_classes):
        bpy.utils.unregister_class(cls)
