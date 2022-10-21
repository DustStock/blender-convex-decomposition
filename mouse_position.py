import pathlib
import random
import subprocess
from pathlib import Path
from typing import List

import bpy
import bpy_types


class ConvexDecompositionVHACD(bpy.types.Operator):
    """ This operator used VHACD to produce collision shapes for Unreal Engine.
    """
    bl_idname = "wm.vhacd"
    bl_label = "Convex Decomposition of Selected Object"

    def make_collection(self, collection_name: str) -> bpy_types.Collection:
        """ Upsert a dedicated outliner collection for the convex hulls."""
        try:
            collection = bpy.data.collections[collection_name]
        except KeyError:
            collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    def export_object(self) -> Path:
        fpath = Path("/tmp/foo")
        pathlib.Path.mkdir(fpath, exist_ok=True)
        fname = fpath / "src.obj"
        bpy.ops.export_scene.obj(filepath=str(fname), check_existing=False,
                                 use_selection=True, use_materials=False)
        return fname

    def remove_stale_hulls(self, name: str) -> None:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.name.startswith(f"UCX_{name}_"):
                obj.select_set(True)
        bpy.ops.object.delete()

    def randomise_colour(self, obj: bpy_types.Object) -> None:
        red, green, blue = [random.random() for _ in range(3)]
        alpha = 1.0
        material = bpy.data.materials.new("random material")
        material.diffuse_color = (red, green, blue, alpha)
        obj.data.materials.clear()
        obj.data.materials.append(material)

    def merge_obj_files(self, prefix: str, out_files: List[Path]) -> Path:
        data = ""
        vert_ofs = 0

        # Concatenate all OBJ files and assign each mesh a unique name.
        for i, fname in enumerate(out_files):
            data += f"o {prefix}{i}\n"

            vert_cnt = 0
            for line in fname.read_text().splitlines():
                if line.startswith("v "):
                    vert_cnt += 1
                    data += line + "\n"
                elif line.startswith("f "):
                    el = line.split()
                    vert_idx = [int(_) for _ in el[1:]]
                    vert_idx = [str(_ + vert_ofs) for _ in vert_idx]
                    data += "f " + str.join(" ", vert_idx) + "\n"
                else:
                    self.report({'ERROR'}, f"Unknown OBJ line entry <{line}>")
                    assert False
            vert_ofs += vert_cnt

        out = Path("/tmp/foo/merged.obj")
        out.write_text(data)
        return out

    def rename_hulls(self, hull_prefix: str, obj_name: str) -> List[str]:
        objs = [_ for _ in bpy.data.objects if _.name.startswith(hull_prefix)]
        names = []
        for i, obj in enumerate(objs):
            name = f"UCX_{obj_name}_{i}"
            names.append(name)
            obj.name = name
        return names

    def execute(self, context):
        # Abort if we are not in OBJECT mode.
        if bpy.context.object.mode != 'OBJECT':
            self.report({'ERROR'}, "Must be in OBJECT mode")
            return {'FINISHED'}

        # Get a handle to the selected object. Abort unless exactly one object
        # is selected.
        selected = bpy.context.selected_objects
        if len(selected) != 1:
            self.report({'INFO'}, "Must have exactly one object selected")
            return
        obj_name = selected[0].name
        self.report({'INFO'}, f"Computing Collision Meshes for <{obj_name}>")

        # Save selected object as an .obj file to a temporary location.
        fname = self.export_object()

        # Call VHACD to do the convex decomposition.
        subprocess.run(["vhacd", str(fname), "-o", "obj"])

        # Delete the original object from the temporary location and fetch the
        # list of all created collision shapes.
        fname.unlink()
        pattern = str(fname.stem) + "*.obj"
        out_files = list(fname.parent.glob(pattern))
        self.report({"INFO"}, f"Produced {len(out_files)} Convex Hulls")

        self.remove_stale_hulls(obj_name)

        hull_prefix = "_tmphull_"
        merged_obj_file = self.merge_obj_files(hull_prefix, out_files)
        bpy.ops.import_scene.obj(filepath=str(merged_obj_file), filter_glob='*.obj')
        del merged_obj_file

        self.rename_hulls(hull_prefix, obj_name)
        del hull_prefix

        return {'FINISHED'}


        collection_name = "vhacd"
        vhacd_collection = self.make_collection(collection_name)

        # Load each generated collision mesh into Blender, give it a name that
        # will work with Unreal Engine (eg 'UCX_objname_123') and also assign
        # it a random material.
        for fname in out_files:
            # Import the new object. Blender will automatically select it.
            bpy.ops.import_scene.obj(filepath=str(fname), filter_glob='*.obj')

            # Sanity check: Blender must have selected the just imported object.
            selected = bpy.context.selected_objects
            assert len(selected) == 1
            obj = selected[0]

            # Extract the numerical suffix, ie /tmp/src012.obj -> 012
            stem_name = str(fname.stem)  # eg /tmp/src012.obj -> src012
            suffix = stem_name.partition("src")[2]  # src012 -> 012

            # Rename the object to match Unreal's FBX convention for collision shapes.
            obj.name = f"UCX_{obj_name}_{suffix}"

            # Unlink the current object from all its collections.
            for coll in obj.users_collection:
                coll.objects.unlink(obj)

            # Assign the collision shape partly transparent object with a
            # random color.
            self.randomise_colour(obj)

            # Link the object to our dedicated collection.
            vhacd_collection.objects.link(obj)

        # Re-select the original object again for a consistent user experience.
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[obj_name].select_set(True)

        return {'FINISHED'}


# Only needed if you want to add into a dynamic menu.
def menu_func(self, context):
    self.layout.operator(ConvexDecompositionVHACD.bl_idname, text="Convex Decomposition for Unreal Engine")

# Register and add to the view menu. This will also make it appear in the
# Search menu (F3 key) under "Convex Decomposition for Unreal Engine".
bpy.utils.register_class(ConvexDecompositionVHACD)
bpy.types.VIEW3D_MT_view.append(menu_func)
