# -*- coding: utf-8 -*-
"""
Mesh Best-Fit Plane tool for FreeCAD.

Workflow:
1) Activate command from toolbar/menu.
2) Left-click points on a mesh (or any 3D geometry) in the 3D view.
3) Right-click to finish and create a best-fit plane.
4) Press ESC to cancel.

Requirements:
- FreeCAD with Gui
- numpy
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

import FreeCAD as App
import FreeCADGui as Gui
import Part


# Draft import is optional for this script, but requested by the tool constraints.
try:
    import Draft  # noqa: F401
except Exception:
    Draft = None


def fit_plane(points: List[App.Vector]) -> Tuple[App.Vector, App.Vector]:
    """Fit a plane to 3D points using SVD on centered coordinates.

    Returns:
        (normal, center)
        normal: App.Vector unit normal of the fitted plane
        center: App.Vector centroid of input points
    """
    if len(points) < 3:
        raise ValueError("At least 3 points are required to fit a plane.")

    arr = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
    center_np = arr.mean(axis=0)
    centered = arr - center_np

    # SVD: normal is the right-singular vector corresponding to smallest singular value.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal_np = vh[-1, :]

    norm = np.linalg.norm(normal_np)
    if norm < 1e-12:
        raise ValueError("Degenerate point configuration; cannot determine plane normal.")
    normal_np /= norm

    normal = App.Vector(float(normal_np[0]), float(normal_np[1]), float(normal_np[2]))
    center = App.Vector(float(center_np[0]), float(center_np[1]), float(center_np[2]))

    return normal, center


def _estimate_plane_size(points: List[App.Vector], center: App.Vector) -> float:
    """Estimate a visible plane size from point spread."""
    if not points:
        return 10.0
    max_dist = max((p.sub(center)).Length for p in points)
    return max(10.0, 2.0 * max_dist)


def create_plane_object(
    doc: App.Document,
    normal: App.Vector,
    center: App.Vector,
    size: float,
    name: str = "BestFitPlane",
):
    """Create a Part::Plane (fallback Part::Feature if needed) aligned to normal/center."""
    z_axis = App.Vector(0, 0, 1)

    # Flip for stable orientation (optional): keep positive Z when possible.
    if normal.dot(z_axis) < 0:
        normal = normal.negative()

    rot = App.Rotation(z_axis, normal)
    placement = App.Placement(center, rot)

    obj = None
    try:
        obj = doc.addObject("Part::Plane", name)
        obj.Length = size
        obj.Width = size
        obj.Placement = placement
    except Exception:
        # Generic fallback: create a geometric plane face.
        plane = Part.Plane(App.Vector(0, 0, 0), App.Vector(0, 0, 1))
        shape = plane.toShape(-size / 2.0, size / 2.0, -size / 2.0, size / 2.0)
        shape.Placement = placement
        obj = doc.addObject("Part::Feature", name)
        obj.Shape = shape

    return obj


class _MeshBestFitPlaneTool:
    """Interactive picker for best-fit plane from clicked points."""

    def __init__(self):
        self.points: List[App.Vector] = []
        self.marker_objs = []
        self.view = None
        self.callback = None

    def start(self):
        doc = App.ActiveDocument
        if doc is None:
            doc = App.newDocument("Unnamed")

        self.view = Gui.ActiveDocument.ActiveView
        self.callback = self.view.addEventCallback("SoEvent", self._on_event)

        App.Console.PrintMessage(
            "[Mesh Best-Fit Plane] Left click: add point | Right click: finish | ESC: cancel\n"
        )

    def stop(self):
        if self.view and self.callback:
            try:
                self.view.removeEventCallback("SoEvent", self.callback)
            except Exception:
                pass
        self.callback = None

    def _on_event(self, info):
        event = info.get("Type")
        if event == "SoKeyboardEvent":
            key = info.get("Key")
            state = info.get("State")
            if key == "ESCAPE" and state == "DOWN":
                self.cancel()
            return

        if event != "SoMouseButtonEvent":
            return

        state = info.get("State")
        button = info.get("Button")
        if state != "DOWN":
            return

        # Finish with right mouse button.
        if button == "BUTTON2":
            self.finish()
            return

        # Add point with left mouse button.
        if button != "BUTTON1":
            return

        pos = info.get("Position")
        if not pos:
            return

        x, y = int(pos[0]), int(pos[1])

        # Try getting exact picked point from object info (works well for mesh/shape picks).
        picked_point = None
        try:
            obj_info = self.view.getObjectInfo((x, y))
            if obj_info and all(k in obj_info for k in ("x", "y", "z")):
                picked_point = App.Vector(obj_info["x"], obj_info["y"], obj_info["z"])
        except Exception:
            pass

        # Fallback: project screen to 3D world point.
        if picked_point is None:
            try:
                picked_point = self.view.getPoint(x, y)
            except Exception:
                picked_point = None

        if picked_point is None:
            App.Console.PrintWarning("[Mesh Best-Fit Plane] Could not pick point.\n")
            return

        self.points.append(picked_point)
        App.Console.PrintMessage(
            f"[Mesh Best-Fit Plane] Point {len(self.points)}: "
            f"({picked_point.x:.3f}, {picked_point.y:.3f}, {picked_point.z:.3f})\n"
        )

        self._add_marker(picked_point, len(self.points))

    def _add_marker(self, point: App.Vector, idx: int):
        """Create a tiny temporary marker for visual feedback."""
        doc = App.ActiveDocument
        if doc is None:
            return

        name = f"BFP_Point_{idx}"
        marker = doc.addObject("Part::Vertex", name)
        marker.X = point.x
        marker.Y = point.y
        marker.Z = point.z
        marker.ViewObject.PointSize = 6
        marker.ViewObject.PointColor = (1.0, 0.2, 0.2)
        self.marker_objs.append(marker)
        doc.recompute()

    def _clear_markers(self):
        doc = App.ActiveDocument
        if doc is None:
            return
        for obj in self.marker_objs:
            try:
                doc.removeObject(obj.Name)
            except Exception:
                pass
        self.marker_objs = []
        doc.recompute()

    def cancel(self):
        App.Console.PrintMessage("[Mesh Best-Fit Plane] Cancelled.\n")
        self._clear_markers()
        self.stop()

    def finish(self):
        try:
            if len(self.points) < 3:
                App.Console.PrintError("[Mesh Best-Fit Plane] Need at least 3 points.\n")
                return

            normal, center = fit_plane(self.points)
            size = _estimate_plane_size(self.points, center)
            obj = create_plane_object(App.ActiveDocument, normal, center, size)
            App.ActiveDocument.recompute()

            App.Console.PrintMessage(
                "[Mesh Best-Fit Plane] Created plane '\n"
                f"{obj.Name}' with normal=({normal.x:.5f}, {normal.y:.5f}, {normal.z:.5f}).\n"
            )
        except Exception as exc:
            App.Console.PrintError(f"[Mesh Best-Fit Plane] Failed: {exc}\n")
        finally:
            self._clear_markers()
            self.stop()


class MeshBestFitPlaneCommand:
    """GUI command wrapper for FreeCAD."""

    def GetResources(self):
        return {
            "MenuText": "Mesh Best-Fit Plane",
            "ToolTip": "Pick points on a mesh and create a least-squares best-fit plane",
            "Pixmap": "Draft_Point",
        }

    def Activated(self):
        if Gui.ActiveDocument is None:
            App.newDocument("Unnamed")
        tool = _MeshBestFitPlaneTool()
        tool.start()

    def IsActive(self):
        return App.ActiveDocument is not None


def register_command():
    Gui.addCommand("Mesh_BestFitPlane", MeshBestFitPlaneCommand())


# Auto-register when script is run as macro.
register_command()
App.Console.PrintMessage("[Mesh Best-Fit Plane] Command registered: Mesh_BestFitPlane\n")
