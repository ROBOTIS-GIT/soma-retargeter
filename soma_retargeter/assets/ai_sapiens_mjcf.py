# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and validate the AI Sapiens retargeting MJCF."""

from __future__ import annotations

import argparse
import copy
import json
import math
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import soma_retargeter.utils.io_utils as io_utils


DEFAULT_PATCH_PATH = io_utils.get_config_file("ai_sapiens", "ai_sapiens_retarget_mjcf_patch.json")
DEFAULT_OUTPUT_PATH = io_utils.get_config_file("ai_sapiens", "ai_sapiens_retarget.xml")
DEFAULT_RETARGET_CONFIG_PATH = io_utils.get_config_file("ai_sapiens", "soma_to_ai_sapiens_retargeter_config.json")
_MUJOCO_STL_FACE_LIMIT = 200_000
_DEFAULT_VISUAL_RGBA = "0.7 0.7 0.7 1"


class AiSapiensMJCFError(RuntimeError):
    """Raised when AI Sapiens retarget MJCF generation or validation fails."""


def get_repo_root() -> Path:
    return io_utils.get_package_root().parent


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def load_patch(path: str | Path = DEFAULT_PATCH_PATH) -> dict[str, Any]:
    patch_path = Path(path)
    try:
        with patch_path.open("r", encoding="utf-8") as f:
            patch = json.load(f)
    except FileNotFoundError as exc:
        raise AiSapiensMJCFError(f"AI Sapiens MJCF patch not found: {patch_path}") from exc
    if not isinstance(patch.get("retarget_bodies"), list):
        raise AiSapiensMJCFError(f"Invalid patch file, missing retarget_bodies: {patch_path}")
    for key in ("source_urdf", "source_meshdir", "meshdir"):
        if not patch.get(key):
            raise AiSapiensMJCFError(f"Invalid patch file, missing {key}: {patch_path}")
    return patch


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    return root.find(f".//body[@name='{name}']")


def _body_names(root: ET.Element) -> set[str]:
    return {body.get("name") for body in root.findall(".//body") if body.get("name")}


def _find_submodule_root(source_path: Path) -> Path | None:
    for parent in source_path.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _check_submodule_commit(source_path: Path, expected_commit: str | None) -> None:
    if not expected_commit:
        return

    submodule_root = _find_submodule_root(source_path)
    if submodule_root is None:
        return

    result = subprocess.run(
        ["git", "-C", str(submodule_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AiSapiensMJCFError(
            f"Could not read ai_sapiens submodule commit: {result.stderr.strip()}"
        )

    actual_commit = result.stdout.strip()
    if actual_commit != expected_commit:
        raise AiSapiensMJCFError(
            "AI Sapiens submodule commit mismatch. "
            f"Expected {expected_commit}, got {actual_commit}."
        )


def _replace_urdf_mesh_paths(
    root: ET.Element,
    *,
    package_mesh_prefix: str,
    source_meshdir: Path,
) -> None:
    source_meshdir_str = str(source_meshdir)
    for compiler in root.findall(".//compiler"):
        compiler.set("meshdir", source_meshdir_str)

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if filename.startswith(package_mesh_prefix):
            suffix = filename[len(package_mesh_prefix) :].lstrip("/")
            mesh.set("filename", str(source_meshdir / suffix))


def _read_binary_stl_bounds(path: Path) -> tuple[int, tuple[float, float, float]]:
    with path.open("rb") as f:
        data = f.read()
    if len(data) < 84:
        raise AiSapiensMJCFError(f"Invalid STL file: {path}")

    face_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + face_count * 50
    if expected_size > len(data):
        raise AiSapiensMJCFError(f"Invalid binary STL face table: {path}")

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    offset = 84
    for _ in range(face_count):
        vertices = struct.unpack_from("<9f", data, offset + 12)
        for idx in range(0, 9, 3):
            for axis in range(3):
                value = vertices[idx + axis]
                mins[axis] = min(mins[axis], value)
                maxs[axis] = max(maxs[axis], value)
        offset += 50

    return face_count, tuple(max(maxs[i] - mins[i], 1e-6) for i in range(3))


def _mesh_scale(mesh: ET.Element) -> tuple[float, float, float]:
    scale = mesh.get("scale")
    if not scale:
        return (1.0, 1.0, 1.0)
    parts = [float(part) for part in scale.split()]
    if len(parts) != 3:
        raise AiSapiensMJCFError(f"Invalid mesh scale: {scale}")
    return tuple(parts)


def _format_float(value: float) -> str:
    return f"{value:.9g}"


def _format_float_list(values: tuple[float, ...]) -> str:
    return " ".join(_format_float(value) for value in values)


def _parse_float_list(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    parts = tuple(float(part) for part in value.split())
    if len(parts) != len(default):
        raise AiSapiensMJCFError(f"Invalid numeric list: {value}")
    return parts


def _quat_from_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _material_rgba(root: ET.Element, visual: ET.Element) -> str:
    material = visual.find("material")
    if material is None:
        return _DEFAULT_VISUAL_RGBA

    color = material.find("color")
    if color is not None and color.get("rgba"):
        return color.get("rgba")

    material_name = material.get("name")
    if material_name:
        global_material = root.find(f"material[@name='{material_name}']")
        global_color = global_material.find("color") if global_material is not None else None
        if global_color is not None and global_color.get("rgba"):
            return global_color.get("rgba")

    return _DEFAULT_VISUAL_RGBA


def _collect_urdf_visual_meshes(root: ET.Element) -> list[dict[str, str]]:
    """Collect URDF visual mesh geoms before MuJoCo compile-only mesh reduction."""
    visual_meshes: list[dict[str, str]] = []
    used_names: set[str] = set()

    for link in root.findall("link"):
        body_name = link.get("name")
        if not body_name:
            continue

        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            filename = mesh.get("filename") if mesh is not None else None
            if not filename:
                continue

            mesh_path = Path(filename)
            mesh_name = mesh_path.stem
            unique_name = mesh_name
            suffix = 2
            while unique_name in used_names:
                unique_name = f"{mesh_name}_{suffix}"
                suffix += 1
            used_names.add(unique_name)

            origin = visual.find("origin")
            pos = _parse_float_list(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = _parse_float_list(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            quat = _quat_from_rpy(rpy)

            visual_spec = {
                "body": body_name,
                "mesh_name": unique_name,
                "file": mesh_path.name,
                "scale": mesh.get("scale", "1 1 1"),
                "rgba": _material_rgba(root, visual),
            }
            if any(abs(value) > 1e-12 for value in pos):
                visual_spec["pos"] = _format_float_list(pos)
            if any(abs(quat[index] - (1.0 if index == 0 else 0.0)) > 1e-12 for index in range(4)):
                visual_spec["quat"] = _format_float_list(quat)
            visual_meshes.append(visual_spec)

    return visual_meshes


def _is_visual_mesh_geom(geom: ET.Element) -> bool:
    return (
        geom.get("type") == "mesh"
        and geom.get("mesh") is not None
        and geom.get("contype") == "0"
        and geom.get("conaffinity") == "0"
        and geom.get("group") == "1"
        and geom.get("density") == "0"
    )


def _remove_collision_geoms(root: ET.Element) -> None:
    """Keep visual mesh geoms only so collision primitives do not hide the robot mesh."""
    for body in root.findall(".//body"):
        for geom in list(body.findall("geom")):
            if not _is_visual_mesh_geom(geom):
                body.remove(geom)


def _restore_visual_mesh_geoms(root: ET.Element, visual_meshes: list[dict[str, str]]) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)

    existing_meshes = {mesh.get("name") for mesh in asset.findall("mesh") if mesh.get("name")}
    for visual in visual_meshes:
        if visual["mesh_name"] not in existing_meshes:
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": visual["mesh_name"],
                    "content_type": "model/stl",
                    "file": visual["file"],
                    "scale": visual["scale"],
                },
            )
            existing_meshes.add(visual["mesh_name"])

        body = _find_body(root, visual["body"])
        if body is None:
            raise AiSapiensMJCFError(
                f"Cannot restore visual mesh {visual['mesh_name']}: body {visual['body']} not found."
            )
        if any(geom.get("mesh") == visual["mesh_name"] for geom in body.findall("geom")):
            continue

        geom_attributes = {
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "density": "0",
            "rgba": visual["rgba"],
            "mesh": visual["mesh_name"],
        }
        for key in ("pos", "quat"):
            if key in visual:
                geom_attributes[key] = visual[key]
        body.append(ET.Element("geom", geom_attributes))


def _replace_unsupported_urdf_meshes(root: ET.Element) -> None:
    """Avoid MuJoCo STL decoder failures for visual meshes over its face limit."""
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            geometry = visual.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None or not mesh.get("filename"):
                continue
            face_count, _ = _read_binary_stl_bounds(Path(mesh.get("filename")))
            if face_count > _MUJOCO_STL_FACE_LIMIT:
                link.remove(visual)

        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None or not mesh.get("filename"):
                continue
            face_count, bounds = _read_binary_stl_bounds(Path(mesh.get("filename")))
            if face_count <= _MUJOCO_STL_FACE_LIMIT:
                continue

            scale = _mesh_scale(mesh)
            size = " ".join(f"{bounds[i] * abs(scale[i]):.9g}" for i in range(3))
            geometry.remove(mesh)
            ET.SubElement(geometry, "box", {"size": size})


def _add_floating_root_to_urdf(root: ET.Element, floating_base: dict[str, str]) -> None:
    world_link = floating_base["world_link"]
    joint_name = floating_base["joint_name"]
    child_link = floating_base["child_link"]

    if root.find(f"link[@name='{world_link}']") is not None:
        raise AiSapiensMJCFError(f"URDF already contains link {world_link}.")
    if root.find(f"joint[@name='{joint_name}']") is not None:
        raise AiSapiensMJCFError(f"URDF already contains joint {joint_name}.")
    if root.find(f"link[@name='{child_link}']") is None:
        raise AiSapiensMJCFError(f"URDF floating root child link not found: {child_link}")

    root.append(ET.Element("link", {"name": world_link}))
    joint = ET.Element("joint", {"name": joint_name, "type": "floating"})
    ET.SubElement(joint, "parent", {"link": world_link})
    ET.SubElement(joint, "child", {"link": child_link})
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": floating_base["origin_xyz"],
            "rpy": floating_base.get("origin_rpy", "0 0 0"),
        },
    )
    root.insert(0, joint)


def _compile_urdf_to_mjcf(
    *,
    source_urdf: Path,
    source_meshdir: Path,
    patch: dict[str, Any],
    temp_dir: Path,
) -> ET.ElementTree:
    try:
        import mujoco
    except ImportError as exc:
        raise AiSapiensMJCFError("MuJoCo is required to generate AI Sapiens MJCF from URDF.") from exc

    root = ET.parse(source_urdf).getroot()
    _replace_urdf_mesh_paths(
        root,
        package_mesh_prefix=patch.get(
            "urdf_package_mesh_prefix",
            "package://ai_sapiens_description/meshes/k1_rev1",
        ),
        source_meshdir=source_meshdir,
    )
    visual_meshes = _collect_urdf_visual_meshes(root)
    _replace_unsupported_urdf_meshes(root)
    _add_floating_root_to_urdf(root, patch["floating_base"])

    prepared_urdf = temp_dir / "k1_rev1_floating_root.urdf"
    compiled_mjcf = temp_dir / "k1_rev1_compiled.xml"
    ET.ElementTree(root).write(prepared_urdf, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(prepared_urdf))
    if model.nq != 30 or model.nv != 29:
        raise AiSapiensMJCFError(
            f"Compiled AI Sapiens URDF has unexpected dimensions: nq={model.nq}, nv={model.nv}. "
            "Expected nq=30, nv=29."
        )
    mujoco.mj_saveLastXML(str(compiled_mjcf), model)
    tree = ET.parse(compiled_mjcf)
    _remove_collision_geoms(tree.getroot())
    _restore_visual_mesh_geoms(tree.getroot(), visual_meshes)
    return tree


def generate_retarget_mjcf(
    *,
    source_urdf: str | Path | None = None,
    output_mjcf: str | Path = DEFAULT_OUTPUT_PATH,
    patch_path: str | Path = DEFAULT_PATCH_PATH,
    repo_root: str | Path | None = None,
) -> Path:
    repo_root_path = Path(repo_root) if repo_root is not None else get_repo_root()
    patch = load_patch(patch_path)
    source_path = _resolve_repo_path(source_urdf or patch["source_urdf"], repo_root_path)
    source_meshdir = _resolve_repo_path(patch["source_meshdir"], repo_root_path)
    output_path = Path(output_mjcf)

    if not source_path.exists():
        raise AiSapiensMJCFError(
            "AI Sapiens source URDF not found: "
            f"{source_path}. Run `git submodule update --init --recursive`."
        )
    if not source_meshdir.is_dir():
        raise AiSapiensMJCFError(
            "AI Sapiens STL mesh directory not found: "
            f"{source_meshdir}. Run `git submodule sync --recursive`, "
            "`git submodule update --init --recursive`, and "
            "`git -C third_party/ai_sapiens lfs pull`."
        )

    _check_submodule_commit(source_path, patch.get("source_commit"))

    with tempfile.TemporaryDirectory(prefix="ai-sapiens-urdf-mjcf.") as temp_dir:
        tree = _compile_urdf_to_mjcf(
            source_urdf=source_path,
            source_meshdir=source_meshdir,
            patch=patch,
            temp_dir=Path(temp_dir),
        )
    root = tree.getroot()
    root.set("model", patch.get("model", root.get("model", "ai_sapiens")))

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", patch["meshdir"])

    existing_bodies = _body_names(root)
    for body_spec in patch["retarget_bodies"]:
        parent_name = body_spec["parent"]
        attributes = body_spec["attributes"]
        body_name = attributes["name"]
        if body_name in existing_bodies:
            continue
        parent = _find_body(root, parent_name)
        if parent is None:
            raise AiSapiensMJCFError(
                f"Cannot add retarget body {body_name}: parent body {parent_name} not found."
            )
        parent.append(ET.Element("body", attributes))
        existing_bodies.add(body_name)

    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _load_ik_map_bodies(config_path: Path) -> set[str]:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    required: set[str] = set()
    for mapping in config.get("ik_map", {}).values():
        for key in ("t_body", "r_body"):
            body_name = mapping.get(key)
            if body_name:
                required.add(body_name)
    return required


def _resolve_meshdir(output_mjcf: Path, meshdir: str) -> Path:
    mesh_path = Path(meshdir)
    if mesh_path.is_absolute():
        return mesh_path
    return (output_mjcf.parent / mesh_path).resolve()


def _write_mujoco_validation_copy(output_path: Path, root: ET.Element, mesh_root: Path) -> Path:
    """Write a temporary MJCF copy that excludes MuJoCo-incompatible visual meshes."""
    validation_root = copy.deepcopy(root)
    high_poly_meshes: set[str] = set()

    asset = validation_root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            mesh_name = mesh.get("name")
            mesh_file = mesh.get("file")
            if not mesh_name or not mesh_file:
                continue
            face_count, _ = _read_binary_stl_bounds(mesh_root / mesh_file)
            if face_count > _MUJOCO_STL_FACE_LIMIT:
                high_poly_meshes.add(mesh_name)
                asset.remove(mesh)

    if high_poly_meshes:
        for body in validation_root.findall(".//body"):
            for geom in list(body.findall("geom")):
                if geom.get("mesh") in high_poly_meshes:
                    body.remove(geom)

    with tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.mujoco-validation.",
        suffix=".xml",
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        validation_path = Path(temp_file.name)

    validation_tree = ET.ElementTree(validation_root)
    ET.indent(validation_tree, space="  ")
    validation_tree.write(validation_path, encoding="utf-8", xml_declaration=True)
    return validation_path


def validate_retarget_mjcf(
    output_mjcf: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    patch_path: str | Path = DEFAULT_PATCH_PATH,
    retarget_config_path: str | Path = DEFAULT_RETARGET_CONFIG_PATH,
    validate_newton: bool = True,
) -> None:
    output_path = Path(output_mjcf)
    if not output_path.exists():
        raise AiSapiensMJCFError(f"AI Sapiens retarget MJCF not found: {output_path}")

    patch = load_patch(patch_path)
    repo_root = get_repo_root()
    source_urdf = _resolve_repo_path(patch["source_urdf"], repo_root)
    source_meshdir = _resolve_repo_path(patch["source_meshdir"], repo_root)
    if not source_urdf.exists():
        raise AiSapiensMJCFError(
            "AI Sapiens source URDF not found: "
            f"{source_urdf}. Run `git submodule update --init --recursive`."
        )
    if not source_meshdir.is_dir():
        raise AiSapiensMJCFError(
            "AI Sapiens STL mesh directory not found: "
            f"{source_meshdir}. Run `git submodule update --init --recursive`."
        )
    _check_submodule_commit(source_urdf, patch.get("source_commit"))

    tree = ET.parse(output_path)
    root = tree.getroot()
    body_names = _body_names(root)

    patch_bodies = {body["attributes"]["name"] for body in patch["retarget_bodies"]}
    missing_patch_bodies = sorted(patch_bodies - body_names)
    if missing_patch_bodies:
        raise AiSapiensMJCFError(
            "Generated MJCF is missing retarget bodies: "
            + ", ".join(missing_patch_bodies)
        )

    required_ik_bodies = _load_ik_map_bodies(Path(retarget_config_path))
    missing_ik_bodies = sorted(required_ik_bodies - body_names)
    if missing_ik_bodies:
        raise AiSapiensMJCFError(
            "Generated MJCF is missing ik_map bodies: "
            + ", ".join(missing_ik_bodies)
        )

    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    if not meshdir:
        raise AiSapiensMJCFError("Generated MJCF has no compiler meshdir.")

    mesh_root = _resolve_meshdir(output_path, meshdir)
    missing_meshes = sorted(
        str(mesh_root / mesh.get("file"))
        for mesh in root.findall(".//mesh")
        if mesh.get("file") and not (mesh_root / mesh.get("file")).exists()
    )
    if missing_meshes:
        raise AiSapiensMJCFError(
            "Generated MJCF references missing mesh files. "
            "Run `git submodule sync --recursive`, "
            "`git submodule update --init --recursive`, and "
            "`git -C third_party/ai_sapiens lfs pull`. Missing: "
            + ", ".join(missing_meshes)
        )

    try:
        import mujoco
    except ImportError as exc:
        raise AiSapiensMJCFError("MuJoCo is required to validate AI Sapiens MJCF loading.") from exc

    validation_path = _write_mujoco_validation_copy(output_path, root, mesh_root)
    try:
        model = mujoco.MjModel.from_xml_path(str(validation_path))
    finally:
        validation_path.unlink(missing_ok=True)
    if model.nq != 30 or model.nv != 29:
        raise AiSapiensMJCFError(
            f"Generated AI Sapiens MJCF has unexpected dimensions: nq={model.nq}, nv={model.nv}. "
            "Expected nq=30, nv=29."
        )

    if validate_newton:
        try:
            import newton
        except ImportError as exc:
            raise AiSapiensMJCFError("Newton is required to validate MJCF loading.") from exc
        builder = newton.ModelBuilder()
        builder.add_mjcf(output_path)


def ensure_default_retarget_mjcf(
    *,
    force: bool = False,
    check_only: bool = False,
    validate_newton: bool = True,
) -> Path:
    output_path = DEFAULT_OUTPUT_PATH
    if check_only:
        validate_retarget_mjcf(output_path, validate_newton=validate_newton)
        return output_path

    if force or not output_path.exists():
        generate_retarget_mjcf(output_mjcf=output_path)

    validate_retarget_mjcf(output_path, validate_newton=validate_newton)
    return output_path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate the output MJCF even if it exists.")
    parser.add_argument("--check", action="store_true", help="Validate the output MJCF without generating it.")
    parser.add_argument("--skip-newton", action="store_true", help="Skip Newton add_mjcf validation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        output_path = ensure_default_retarget_mjcf(
            force=args.force,
            check_only=args.check,
            validate_newton=not args.skip_newton,
        )
    except AiSapiensMJCFError as exc:
        print(f"[ERROR]: {exc}", file=sys.stderr)
        return 1

    action = "Validated" if args.check else "Ensured"
    print(f"[INFO]: {action} AI Sapiens retarget MJCF: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
