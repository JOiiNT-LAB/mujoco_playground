import xml.etree.ElementTree as ET


def generate_mujoco_constants(xml_input: str) -> str:
    """Parses a MuJoCo XML string/filepath and generates the constants Python module code."""
    tree = (
        ET.parse(xml_input)
        if xml_input.endswith(".xml")
        else ET.fromstring(xml_input)
    )
    root = tree.getroot() if hasattr(tree, "getroot") else tree

    model_name = root.attrib.get("model", "robot")
    model_lower = model_name.lower()

    # 1. Root Body Extraction
    worldbody = root.find("worldbody")
    root_body = "pelvis"
    if worldbody is not None:
        first_body = worldbody.find("body")
        if first_body is not None:
            root_body = first_body.attrib.get("name", root_body)

    # 2. Extract Non-Free Joints & Ranges
    joint_ranges = []
    if worldbody is not None:
        for joint in worldbody.iter("joint"):
            j_type = joint.attrib.get("type", "revolute")
            if j_type == "free":
                continue
            name = joint.attrib.get("name", "")
            rng_str = joint.attrib.get("range")
            if rng_str:
                min_val, max_val = map(float, rng_str.split())
                joint_ranges.append((name, (min_val, max_val)))

    # 3. Format Commented Joint List
    joint_comments = "# Joints in order:\n" + "\n".join(
        f"#   {i:2d}: {name}" for i, (name, _) in enumerate(joint_ranges)
    )

    # 4. Formatter helper for tuple ranges with inline joint names
    ranges_code = "RESTRICTED_JOINT_RANGE = (\n"
    current_category = ""
    for name, (low, high) in joint_ranges:
        # Group comment prefix based on joint naming pattern
        prefix = name.rsplit("_", 2)[0] if "_" in name else name
        if prefix != current_category:
            current_category = prefix
            clean_category = (
                current_category.replace("_", " ").strip().capitalize()
            )
            ranges_code += f"    # {clean_category}.\n"
        ranges_code += f"    ({low}, {high}),  # {name}\n"
    ranges_code += ")"

    # 5. Construct Final Output File
    code_template = f"""# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

\"\"\"Constants for {model_name}.\"\"\"

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "{model_lower}"
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  return {{
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }}[task_name]


FEET_SITES = [
    "left_foot",
    "right_foot",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

LEFT_FEET_GEOMS = ["left_ankle_pitch_link"]
RIGHT_FEET_GEOMS = ["right_ankle_pitch_link"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "{root_body}"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "imu_acc"
GYRO_SENSOR = "imu_gyro"

{joint_comments}
{ranges_code}
"""
    return code_template


if __name__ == "__main__":
    xml_path = "/home/rl_h2_project_container/ros2_ws/src/mujoco_playground/mujoco_playground/_src/locomotion/h2/xmls/h2_mujoco.xml"
    out_path = "/home/rl_h2_project_container/ros2_ws/src/mujoco_playground/mujoco_playground/_src/locomotion/h2/h2_constants.py"

    with open(xml_path, "r") as f:
        xml_content = f.read()

    output_py = generate_mujoco_constants(xml_content)
    with open(out_path, "w") as f:
        f.write(output_py)