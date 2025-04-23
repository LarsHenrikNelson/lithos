import os
import re
import subprocess

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor, TagRemovePreprocessor
from traitlets import Unicode


def get_current_branch():
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip()


class ImageNamePreprocessor(Preprocessor):
    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type == "code" and "outputs" in cell:
            for output in cell.outputs:
                if "data" in output and any(
                    key.startswith("image/") for key in output.data.keys()
                ):
                    # Get image name from metadata
                    image_name = cell.metadata.get("filename", f"output_{index}")

                    # Store the image name for this cell in resources
                    if "image_names" not in resources:
                        resources["image_names"] = {}
                    resources["image_names"][index] = image_name

        return cell, resources


class CustomMarkdownExporter(MarkdownExporter):
    def from_notebook_node(self, nb, resources=None, **kw):
        resources = resources or {}
        resources["output_files_dir"] = "README_files"

        tag_preprocessor = TagRemovePreprocessor(remove_cell_tags=["hidden"])
        nb, resources = tag_preprocessor.preprocess(nb, resources)

        # Process the notebook with our preprocessor
        preprocessor = ImageNamePreprocessor()
        nb, resources = preprocessor.preprocess(nb, resources)

        # Run the standard export
        output, resources = super().from_notebook_node(nb, resources, **kw)

        # Replace the image paths with custom names
        if "image_names" in resources:
            for idx, name in resources["image_names"].items():
                # Replace standard output_X_Y.ext pattern with custom name
                pattern = rf"README_files/output_{idx}_\d+\.([a-zA-Z]+)"
                current_branch = get_current_branch()
                file = f"https://raw.githubusercontent.com/LarsHenrikNelson/lithos/refs/heads/{current_branch}/doc/_static/README_files"
                replacement = rf"{file}/{name}.\1"
                output = re.sub(pattern, replacement, output)

                # Also rename the actual files
                if "outputs" in resources:
                    new_outputs = {}
                    for key, val in resources["outputs"].items():
                        if f"output_{idx}_" in key:
                            ext = key.split(".")[-1]
                            new_key = f"doc/_static/README_files/{name}.{ext}"
                            new_outputs[new_key] = val
                        else:
                            new_outputs[key] = val
                    resources["outputs"] = new_outputs

        return output, resources


# Load the notebook
notebook_path = "README.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)

# Configure and run the exporter
exporter = CustomMarkdownExporter()
output, resources = exporter.from_notebook_node(notebook)

# Write the markdown file
with open("README.md", "w") as f:
    f.write(output)

# Write the image files
if "outputs" in resources:
    os.makedirs("doc/_static/README_files", exist_ok=True)
    for filename, data in resources["outputs"].items():
        with open(filename, "wb") as f:
            f.write(data)

print("Conversion complete. Markdown saved to README.md")
