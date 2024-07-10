import pathlib
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import bpy  # type: ignore
import bpy_types  # type: ignore
import bmesh
from mathutils import Vector, Matrix

bl_info = {
    'name': 'Convex Decomposition',
    'blender': (4, 1, 0),
    'category': 'Object',
    'version': (0, 3, 0),
    'author': 'Oliver Nagy',
    'description': 'Create collision shapes and Export to FBX (Unreal) or GLTF/GLB (Godot)',
    'warning': 'WIP',
}


class ConvexDecompositionPreferences(bpy.types.AddonPreferences):
    """Addon preferences menu."""
    bl_idname = "convex_decomposition"

    vhacd_binary: bpy.props.StringProperty(  # type: ignore
        name="V-HACD Binary",
        subtype='FILE_PATH',
    )
    coacd_binary: bpy.props.StringProperty(  # type: ignore
        name="CoACD Binary",
        subtype='FILE_PATH',
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "coacd_binary")
        layout.prop(self, "vhacd_binary")


class SelectionGuard():
    """Ensure the same objects are selected at the end."""
    def __init__(self, clear: bool = False):
        self.clear = clear
        self.selected = None
        self.active = None

    def __enter__(self, clear=False):
        self.selected = bpy.context.selected_objects
        self.active = bpy.context.view_layer.objects.active

        if self.clear:
            bpy.ops.object.select_all(action='DESELECT')
        return self

    def __exit__(self, *args, **kwargs):
        bpy.ops.object.select_all(action='DESELECT')
        assert self.selected is not None
        assert self.active is not None
        for obj in self.selected:
            obj.select_set(True)

        # Restore the active object.
        bpy.context.view_layer.objects.active = self.active


class ConvexDecompositionBaseOperator(bpy.types.Operator):
    """Base class for the operators with common utility methods."""

    bl_idname = 'opr.convex_decomposition_base'
    bl_label = 'Convex Decomposition Base Class'

    def get_selected_object(self) -> Tuple[bpy_types.Object, bool]:
        """Return the selected object.

        Set the error flag if more or less than one object is currently
        selected or if we are not in OBJECT mode.

        """
        # User must be in OBJECT mode.
        if bpy.context.mode != 'OBJECT':
            self.report({'ERROR'}, "Must be in OBJECT mode")
            return None, True

        # User must have exactly one object selected.
        selected = bpy.context.selected_objects
        if len(selected) != 1:
            self.report({'ERROR'}, "Must have exactly one object selected")
            return None, True

        return selected[0], False

    def remove_stale_hulls(self, root_obj: bpy_types.Object) -> None:
        """Remove the convex decomposition results from previous runs for `root_obj`."""
        # Store current mode
        current_mode = bpy.context.object.mode

        # Switch to object mode if not already
        if current_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        with SelectionGuard(clear=True):
            for obj in bpy.data.objects:
                if obj.name.startswith(f"UCX_{root_obj.name}_"):
                    obj.select_set(True)
            bpy.ops.object.delete()

        # Restore original mode
        if current_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=current_mode)

    def rename_hulls(self, parent: bpy_types.Object, collection_name: str) -> None:
        """Rename all convex hulls in the specified collection to Unreal Engine format.

        Renames hulls to the format "UCX_{parent.name}_{seq-number}".
        This ensures that Unreal Engine can load the object and automatically
        recognize all its collision shapes.
        """
        hull_collection = bpy.data.collections.get(collection_name)
        if not hull_collection:
            self.report({'WARNING'}, f"Collection '{collection_name}' not found. Skipping hull renaming.")
            return

        # Step 1: Rename to temporary names
        temp_hulls = []
        for obj in hull_collection.objects:
            if obj.parent == parent:
                obj.name = "TEMP_HULL"
                temp_hulls.append(obj)

        # Step 2: Rename to final names
        for i, hull_obj in enumerate(temp_hulls):
            hull_obj.name = f"UCX_{parent.name}_{i}"

    def upsert_collection(self, name: str) -> bpy_types.Collection:
        """Create a dedicated collection` `name` for the convex hulls.

        Does nothing if the collection already exists.

        """
        try:
            collection = bpy.data.collections[name]
        except KeyError:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    def randomise_colour(self, obj: bpy_types.Object, transparency: int) -> None:
        """Assign a random colour to `obj`."""
        red, green, blue = [random.random() for _ in range(3)]

        material = bpy.data.materials.new("random material")
        material.diffuse_color = (red, green, blue, (100 - transparency) / 100.0)
        obj.data.materials.clear()
        obj.data.materials.append(material)


class ConvexDecompositionClearOperator(ConvexDecompositionBaseOperator):
    """Select the children of all selected objects."""

    bl_idname = 'opr.convex_decomposition_select'
    bl_label = 'Select all children of all selected objects'

    def execute(self, context):
        if bpy.context.mode != 'OBJECT':
            self.report({'ERROR'}, "Must be in OBJECT mode")
            return None, True

        selected = bpy.context.selected_objects
        for parent in selected:
            for child in parent.children:
                child.select_set(True)

        return {'FINISHED'}


class ConvexDecompositionUnrealExportOperator(ConvexDecompositionBaseOperator):
    """Export object with collision shapes as FBX."""

    bl_idname = 'opr.convex_decomposition_unreal_export'
    bl_label = 'Export object with Unreal Engine compatible collision meshes as FBX'

    def unreal_export(self, obj: bpy_types.Object) -> None:
        """Export the object and its collision shapes to FBX.

        The function will automatically centre the object for the export.

        If the `obj` has collision shapes from a convex decomposition they will
        be exported as well and Unreal Engine should automatically recognise
        them on import.

        """
        # Output path.
        root_path = Path(bpy.path.abspath("//"))
        fname = root_path / f"{obj.name}.fbx"

        # Temporarily move object to the centre of the scene to ensure it is
        # centred for the export.
        bak_location = obj.location.copy()
        obj.location = (0, 0, 0)

        # Select all the children of this object.
        with SelectionGuard():
            for child in obj.children:
                if child.name.startswith("UCX_"):
                    child.select_set(True)

            bpy.ops.export_scene.fbx(
                filepath=str(fname),
                check_existing=True,
                use_selection=True,
                mesh_smooth_type="FACE",
                axis_forward='-Z',
                axis_up='Y',
            )

        # Restore the original position.
        obj.location = bak_location
        self.report({'INFO'}, f"Exported object as FBX to <{fname.absolute()}>")

    def execute(self, context):
        # User must have exactly one object selected in OBJECT mode.
        root_obj, err = self.get_selected_object()
        if err:
            return {'FINISHED'}

        self.unreal_export(root_obj)
        return {'FINISHED'}


class ConvexDecompositionGodotExportOperator(ConvexDecompositionBaseOperator):
    """Export object with collision shapes as GLB."""

    bl_idname = 'opr.convex_decomposition_godot_export'
    bl_label = 'Export object with Godot compatible collision meshes as GLB'

    def godot_export(self, obj: bpy_types.Object) -> None:
        """Export the object and its collision shapes to GLB.

        The function will automatically centre the object for the export.

        If the `obj` has collision shapes from a convex decomposition they will
        be exported as well and Unreal Engine should automatically recognise
        them on import.

        """
        # Output path.
        root_path = Path(bpy.path.abspath("//"))
        fname = root_path / f"{obj.name}.glb"

        # Temporarily move object to the centre of the scene to ensure it is
        # centred for the export.
        bak_location = obj.location.copy()
        obj.location = (0, 0, 0)

        # Select all the children of this object.
        with SelectionGuard():
            for child in obj.children:
                if child.name.startswith("UCX_"):
                    child.select_set(True)

            bpy.ops.export_scene.gltf(
                filepath=str(fname),
                check_existing=True,
                use_selection=True,
                export_format="GLB",
            )

        # Restore the original position.
        obj.location = bak_location
        self.report({'INFO'}, f"Exported object as GLB to <{fname.absolute()}>")

    def execute(self, context):
        # User must have exactly one object selected in OBJECT mode.
        root_obj, err = self.get_selected_object()
        if err:
            return {'FINISHED'}

        self.godot_export(root_obj)
        return {'FINISHED'}

class ConvexDecompositionRunOperator(ConvexDecompositionBaseOperator):
    """Use VHACD or CoACD to create a convex decomposition of objects."""
    bl_idname = 'opr.convex_decomposition_run'
    bl_label = 'Convex Decomposition Base Class'
    bl_description = "Run Solver"

    def export_mesh_for_solver(self, obj: bpy_types.Object, path: Path) -> Path:
        """Save a temporary copy of `obj` in OBJ format to a temporary folder.

        This is necessary because the various solvers all expect an OBJ file as input.
        """
        fname = path / "src.obj"

        # Store the current selection state
        original_selection = bpy.context.selected_objects
        original_active = bpy.context.view_layer.objects.active

        try:
            # Ensure only the desired object is selected
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            # Export the object
            bpy.ops.wm.obj_export(
                filepath=str(fname),
                check_existing=False,
                export_selected_objects=True,
                export_triangulated_mesh=True,
                export_materials=False,
                apply_modifiers=True
            )
        finally:
            # Restore the original selection state
            bpy.ops.object.select_all(action='DESELECT')
            for o in original_selection:
                o.select_set(True)
            bpy.context.view_layer.objects.active = original_active

        return fname

    def run_vhacd(self, obj_file: Path,
                  props: bpy.types.PropertyGroup,
                  binary: Path):
        """
        Compute convex decomposition for `obj_file` with VHACD.
        """
        cmd = [
            binary,
            str(obj_file),

            "-r", str(props.i_voxel_resolution),
            "-d", str(props.i_max_recursion_depth),
            "-v", str(props.i_max_hull_vert_count),
            "-l", str(props.i_min_edge_length),

            "-e", str(props.f_volume_error_percent),

            "-s", "true" if props.b_shrinkwrap else "false",
            "-p", "true" if props.b_split_location else "false",
            "-a", "true",       # Always run asynchronously.
            "-g", "true",       # Logging

            "-f", str(props.e_fill_mode),
        ]

        self.report({"INFO"}, f"Running command <{cmd}>")
        subprocess.run(cmd, cwd=obj_file.parent)

        fout = obj_file.parent / "decomp.obj"
        return fout

    def run_coacd(self, obj_file: Path,
                  props: bpy.types.PropertyGroup,
                  binary: Path) -> Path:
        """
        Compute convex decomposition for `obj_file` with CoACD.
        """
        result_file = obj_file.parent / "hulls.obj"
        preparg = "off" if props.b_no_preprocess else "auto"
        cmd = [
            binary,
            "--input", str(obj_file),
            "--output", str(result_file),

            "--threshold", str(props.f_threshold),
            "-k", str(props.f_k),

            "--mcts-iteration", str(props.i_mcts_iterations),
            "--mcts-depth", str(props.i_mcts_depth),
            "--mcts-node", str(props.i_mcts_node),
            "--prep-resolution", str(props.i_prep_resolution),
            "--resolution", str(props.i_resolution),
            "--preprocess-mode", preparg,
        ]
        cmd.append("--pca") if props.b_pca else None
        cmd.append("--no-merge") if not props.b_merge else None

        self.report({"INFO"}, f"Running command <{cmd}>")
        subprocess.run(cmd, cwd=obj_file.parent)
        return result_file

    def run_manual(self, obj, context):
        props = context.scene.ConvDecompPropertiesManual
        if props.method == "SPLIT_BY_FACE":
            return self.split_mesh_by_face(context, obj)
        else:
            self.report({'ERROR'}, f"Unknown Manual method <{props.method}>")
            return []

    def split_mesh_by_face(self, context, obj):
        if obj.type != 'MESH':
            raise ValueError("Object is not a mesh")

        # Store the current mode
        original_mode = context.object.mode

        # Enter edit mode if not already in it
        if original_mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        # Get the selected face
        selected_faces = [f for f in bm.faces if f.select]
        if len(selected_faces) != 1:
            bpy.ops.object.mode_set(mode=original_mode)
            raise ValueError("Please select exactly one face")
        face = selected_faces[0]

        # Store face information
        face_center = face.calc_center_median()
        face_normal = face.normal.copy()

        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create a cube
        longest_dimension = max(obj.dimensions)
        cube_size = longest_dimension * 2
        bpy.ops.mesh.primitive_cube_add(size=cube_size)
        cube = context.active_object

        # Align cube's top face with the selected face
        cube.rotation_euler = face_normal.to_track_quat('Z', 'Y').to_euler()
        offset = face_normal * (cube_size / 2)
        cube.location = obj.matrix_world @ face_center - offset

        # Make two duplicates of the original mesh
        new_obj1 = obj.copy()
        new_obj1.data = obj.data.copy()
        context.collection.objects.link(new_obj1)

        new_obj2 = obj.copy()
        new_obj2.data = obj.data.copy()
        context.collection.objects.link(new_obj2)

        # Boolean operations
        bool_intersect = new_obj1.modifiers.new(name="Boolean", type='BOOLEAN')
        bool_intersect.operation = 'INTERSECT'
        bool_intersect.object = cube

        bool_difference = new_obj2.modifiers.new(name="Boolean", type='BOOLEAN')
        bool_difference.operation = 'DIFFERENCE'
        bool_difference.object = cube

        # Apply modifiers
        context.view_layer.objects.active = new_obj1
        bpy.ops.object.modifier_apply(modifier="Boolean")

        context.view_layer.objects.active = new_obj2
        bpy.ops.object.modifier_apply(modifier="Boolean")

        # Remove the cube
        bpy.data.objects.remove(cube, do_unlink=True)

        # Ensure we're in object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        context.view_layer.update()

        return [new_obj1, new_obj2]

    def import_solver_results(self, fname: Path, hull_prefix: str):
        """Load the solver output `fname` (an OBJ file)."""
        # Replace all object names in the OBJ file with a solver independent
        # naming scheme.
        data = ""
        lines = fname.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith("o "):
                data += f"o {hull_prefix}{i}\n"
            else:
                data += line + "\n"
        fname.write_text(data)

        # Import the hulls back into Blender.
        with SelectionGuard():
            bpy.ops.wm.obj_import(
                filepath=str(fname),
                filter_glob='*.obj',
            )

    def process_hull_objects(self, hull_objs, parent_obj, props):
        hull_collection = self.upsert_collection(props.hull_collection_name)
        for obj in hull_objs:
            # Unlink the current object from all its collections.
            for coll in obj.users_collection:
                coll.objects.unlink(obj)

            # Link the object to our dedicated collection.
            hull_collection.objects.link(obj)

            # Assign a random colour to the hull.
            self.randomise_colour(obj, props.transparency)

            # Parent the hull to the parent object without changing the relative transform.
            if parent_obj:
                obj.parent = parent_obj
                obj.matrix_parent_inverse = parent_obj.matrix_world.inverted()

    def execute(self, context):
        # Convenience.
        prefs = context.preferences.addons["convex_decomposition"].preferences
        props = context.scene.ConvDecompProperties

        # Check if we're in the correct mode based on the solver
        if props.solver == "Manual":
            if context.object.mode != 'EDIT':
                self.report({'ERROR'}, "Must be in EDIT mode for Manual split by face")
                return {'CANCELLED'}

            # Check if exactly one face is selected for Manual method
            bm = bmesh.from_edit_mesh(context.edit_object.data)
            selected_faces = [f for f in bm.faces if f.select]
            if len(selected_faces) != 1:
                self.report({'ERROR'}, "Please select exactly one face for Manual split")
                return {'CANCELLED'}

            root_obj = context.edit_object
        else:
            if context.object.mode != 'OBJECT':
                self.report({'ERROR'}, "Must be in OBJECT mode for VHACD or CoACD")
                return {'CANCELLED'}

            # Check if exactly one object is selected for VHACD and CoACD
            selected_objects = context.selected_objects
            if len(selected_objects) != 1:
                self.report({'ERROR'}, "Please select exactly one object for VHACD or CoACD")
                return {'CANCELLED'}

            root_obj = selected_objects[0]

        if root_obj is None:
            self.report({'ERROR'}, "No active object selected")
            return {'CANCELLED'}

        # Check if the object is already a split part
        is_split_part = root_obj.name.startswith("UCX_")
        original_name = root_obj.name

        if is_split_part:
            parent_obj = root_obj.parent
            if not parent_obj:
                parent_obj = root_obj  # Fallback if no parent
        else:
            parent_obj = root_obj

        self.report({'INFO'}, f"Computing collision meshes for <{original_name}>")

        if not is_split_part:
            self.remove_stale_hulls(root_obj)

        hull_objs = []

        if props.solver == "Manual":
            hull_objs = self.run_manual(root_obj, context)
        else:
            # Save the selected root object to a temporary location for the solvers.
            tmp_path = Path(tempfile.mkdtemp(prefix="devcomp-"))
            self.report({"INFO"}, f"Created temporary directory for solver: {tmp_path}")
            obj_path = self.export_mesh_for_solver(root_obj, tmp_path)

            # Use the selected solver to compute the convex decomposition.
            if props.solver == "VHACD":
                hull_path = self.run_vhacd(obj_path, context.scene.ConvDecompPropertiesVHACD, Path(prefs.vhacd_binary))
            elif props.solver == "CoACD":
                hull_path = self.run_coacd(obj_path, context.scene.ConvDecompPropertiesCoACD, Path(prefs.coacd_binary))
            else:
                self.report({'ERROR'}, f"Unknown Solver <{props.solver}>")
                return {'CANCELLED'}

            # Import solver results
            self.import_solver_results(hull_path, props.tmp_hull_prefix)
            hull_objs = [obj for obj in bpy.data.objects if obj.name.startswith(props.tmp_hull_prefix)]

        # Process the resulting hull objects
        self.process_hull_objects(hull_objs, parent_obj, props)

        # If the original object was a split part, delete it
        if is_split_part:
            bpy.data.objects.remove(root_obj, do_unlink=True)

        # Rename Hulls
        self.rename_hulls(parent_obj, props.hull_collection_name)

        self.report({'INFO'}, f"Convex decomposition completed successfully for {original_name}")
        return {'FINISHED'}


class ConvexDecompositionPanel(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_ConvDec'
    bl_label = 'Convex Decomposition'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ConvDecomp"

    def draw(self, context):
        props = context.scene.ConvDecompProperties
        prefs = context.preferences.addons["convex_decomposition"].preferences
        layout = self.layout

        layout.prop(props, 'solver')

        is_valid_solver = True
        if props.solver == "VHACD":
            binary = Path(prefs.vhacd_binary)
            solver_props = context.scene.ConvDecompPropertiesVHACD
            is_valid_solver = binary.name != "" and binary.exists()
        elif props.solver == "CoACD":
            binary = Path(prefs.coacd_binary)
            solver_props = context.scene.ConvDecompPropertiesCoACD
            is_valid_solver = binary.name != "" and binary.exists()
        elif props.solver == "Manual":
            solver_props = context.scene.ConvDecompPropertiesManual
        else:
            self.report({'ERROR'}, f"Unknown Solver <{props.solver}>")
            return

        row = layout.row()
        row.operator('opr.convex_decomposition_run', text="Run")
        row.enabled = is_valid_solver or props.solver == "Manual"

        # Display <Clear> and <Export> buttons.
        row = layout.row()
        row.operator('opr.convex_decomposition_select', text="Select")
        row = layout.row()
        row.operator('opr.convex_decomposition_unreal_export', text="Export FBX")
        row.operator('opr.convex_decomposition_godot_export', text="Export GLB")

        # Display "Hull Transparency" slider.
        layout.separator()
        layout.box().row().prop(props, 'transparency')

        # Solver Specific parameters.
        layout.separator()
        box = layout.box()
        box.enabled = is_valid_solver or props.solver == "Manual"
        solver_specific = [_ for _ in solver_props.__annotations__]
        for name in solver_specific:
            box.row().prop(solver_props, name)

class ConvexDecompositionPropertiesManual(bpy.types.PropertyGroup):
    method: bpy.props.EnumProperty(
        name="Method",
        description="Manual decomposition method",
        items=[
            ('SPLIT_BY_FACE', 'Split by Face', 'Split the mesh by the active face'),
            # Add more manual methods here in the future
        ],
        default='SPLIT_BY_FACE',
    )

class ConvexDecompositionPropertiesVHACD(bpy.types.PropertyGroup):
    i_voxel_resolution: bpy.props.IntProperty(  # type: ignore
        name="Voxel Resolution",
        description="Total number of voxels to use.",
        default=100_000,
        min=1,
        subtype='UNSIGNED'
    )
    f_volume_error_percent: bpy.props.FloatProperty(  # type: ignore
        name="Volume Error (%)",
        description="Volume error allowed as a percentage.",
        default=10,
        min=0.001,
        max=10,
        subtype='UNSIGNED'
    )
    i_max_recursion_depth: bpy.props.IntProperty(  # type: ignore
        name="Max Recursion Depth",
        description="Maximum recursion depth.",
        default=10,
        min=1,
        subtype='UNSIGNED'
    )
    i_max_hull_vert_count: bpy.props.IntProperty(  # type: ignore
        name="Max Hull Vert Count",
        description="Maximum number of vertices in the output convex hull.",
        default=64,
        min=1,
        subtype='UNSIGNED'
    )
    i_min_edge_length: bpy.props.IntProperty(  # type: ignore
        name="Min Edge Length",
        description="Minimum size of a voxel edge.",
        default=2,
        min=1,
        subtype='UNSIGNED'
    )
    b_shrinkwrap: bpy.props.BoolProperty(  # type: ignore
        name="Shrink Wrap",
        description="Whether or not to shrinkwrap output to source mesh.",
        default=True,
    )
    b_split_location: bpy.props.BoolProperty(  # type: ignore
        name="Optimal Split Location",
        description=(
            "If false, splits hulls in the middle. "
            "If true, tries to find optimal split plane location."
        ),
        default=False,
    )
    e_fill_mode: bpy.props.EnumProperty(  # type: ignore
        name="Fill Mode",
        description="Select Convex Decomposition Solver.",
        items={
            ('flood', 'flood', 'Use Flood Fill'),
            ('surface', 'surface', 'Use Surface Method'),
            ('raycast', 'raycast', 'Use Raycast Method'),
        },
        default='flood',
    )


class ConvexDecompositionPropertiesCoACD(bpy.types.PropertyGroup):
    f_threshold: bpy.props.FloatProperty(  # type: ignore
        name="Concavity Threshold",
        description=(
            "This is primary parameter to control the quality of the decomposition."
        ),
        default=0.05,
        min=0.01,
        max=1,
        subtype='UNSIGNED'
    )
    i_mcts_iterations: bpy.props.IntProperty(  # type: ignore
        name="MCTS Iterations",
        description="Number of search iterations in MCTS.",
        default=100,
        min=60,
        max=2_000,
        subtype='UNSIGNED'
    )
    i_mcts_depth: bpy.props.IntProperty(  # type: ignore
        name="MCTS Depth",
        description="Max search depth in MCTS.",
        default=3,
        min=2,
        max=7,
        subtype='UNSIGNED'
    )
    i_mcts_node: bpy.props.IntProperty(  # type: ignore
        name="MCTS Node",
        description="Max number of child nodes in MCTS.",
        default=20,
        min=10,
        max=40,
        subtype='UNSIGNED'
    )
    i_prep_resolution: bpy.props.IntProperty(  # type: ignore
        name="Manifold Pre-Processing Resolution",
        description="Resolution for manifold pre-processing.",
        default=10_000,
        min=1_000,
        max=100_000,
        subtype='UNSIGNED'
    )
    i_resolution: bpy.props.IntProperty(  # type: ignore
        name="Sampling Resolution",
        description="Sampling resolution for Hausdorff distance.",
        default=2_000,
        min=1_000,
        max=10_000,
        subtype='UNSIGNED'
    )

    f_k: bpy.props.FloatProperty(  # type: ignore
        name="K",
        description="Value of K for R_v calculation.",
        default=0.3,
        min=0,
        max=1,
        subtype='UNSIGNED'
    )

    b_no_preprocess: bpy.props.BoolProperty(  # type: ignore
        name="Watertight Mesh",
        description=(
            "Enable this if your mesh is already watertight."
            "It will speed up the computation and reduce artefacts."
        ),
        default=True,
    )
    b_merge: bpy.props.BoolProperty(  # type: ignore
        name="Merge Post-Processing",
        description="",
        default=True,
    )
    b_pca: bpy.props.BoolProperty(  # type: ignore
        name="PCA Pre-Processing",
        description="",
        default=False,
    )


def update_transparency(self, context) -> None:
    """Change the hull transparencies of all selected objects."""
    # User must be in OBJECT mode.
    if bpy.context.mode != 'OBJECT':
        return

    for root_obj in bpy.context.selected_objects:
        props = context.scene.ConvDecompProperties
        transparency = (100 - props.transparency) / 100.0

        # Update the transparency of all children of the selected object that
        # are collision hulls.
        for obj in root_obj.children:
            if obj.name.startswith(f"UCX_{root_obj.name}_"):
                mat = obj.data.materials[0]
                mat.diffuse_color[3] = transparency


class ConvexDecompositionProperties(bpy.types.PropertyGroup):
    tmp_hull_prefix: bpy.props.StringProperty(  # type: ignore
        name="Hull Prefix",
        description="Name prefix for the temporary hull names created by the solvers.",
        default="_tmphull_",
    )
    hull_collection_name: bpy.props.StringProperty(  # type: ignore
        name="Hull Collection",
        description="The collection to hold all the convex hulls.",
        default="convex hulls",
    )
    solver: bpy.props.EnumProperty(
        name="Solver",
        description="Supported Convex Decomposition Solvers.",
        items={
            ('VHACD', 'VHACD', 'Use VHACD'),
            ('CoACD', 'CoACD', 'Use CoACD'),
            ('Manual', 'Manual', 'Manual decomposition'),
        },
        default='VHACD',
    )
    transparency: bpy.props.IntProperty(  # type: ignore
        name="Hull Transparency",
        description="Transparency of hulls in viewport",
        default=90,
        min=0,
        max=100,
        subtype='UNSIGNED',
        update=update_transparency,
    )


# ----------------------------------------------------------------------
# Addon registration.
# ----------------------------------------------------------------------


CLASSES = [
    ConvexDecompositionPanel,
    ConvexDecompositionProperties,
    ConvexDecompositionPropertiesVHACD,
    ConvexDecompositionPropertiesCoACD,
    ConvexDecompositionPropertiesManual,
    ConvexDecompositionRunOperator,
    ConvexDecompositionClearOperator,
    ConvexDecompositionUnrealExportOperator,
    ConvexDecompositionGodotExportOperator,
    ConvexDecompositionPreferences,
]

def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.ConvDecompProperties = bpy.props.PointerProperty(type=ConvexDecompositionProperties)
    bpy.types.Scene.ConvDecompPropertiesVHACD = bpy.props.PointerProperty(type=ConvexDecompositionPropertiesVHACD)
    bpy.types.Scene.ConvDecompPropertiesCoACD = bpy.props.PointerProperty(type=ConvexDecompositionPropertiesCoACD)
    bpy.types.Scene.ConvDecompPropertiesManual = bpy.props.PointerProperty(type=ConvexDecompositionPropertiesManual)

def unregister():
    for cls in CLASSES:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.ConvDecompProperties
    del bpy.types.Scene.ConvDecompPropertiesVHACD
    del bpy.types.Scene.ConvDecompPropertiesCoACD
    del bpy.types.Scene.ConvDecompPropertiesManual

if __name__ == '__main__':
    register()
