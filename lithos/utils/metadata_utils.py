from pathlib import Path
import ast

from ..plotting.types import Group, Subgroup, UniqueGroups

def home_dir():
    p = Path.home()
    h = ".lithos"
    prog_dir = Path(p / h)
    if not prog_dir.exists():
        prog_dir.mkdir(parents=True, exist_ok=True)
    return prog_dir


def metadata_dir():
    h = home_dir()
    mdir = Path(h / "metadata_dir.txt")
    if mdir.exists():
        with open(mdir, "r") as f:
            mdir = f.read()
            mdir = Path(mdir)
        if not mdir.exists():
            set_metadata_dir(mdir)
    else:
        set_metadata_dir(mdir.parent / "metadata")
        mdir = metadata_dir()
    return Path(mdir)


def set_metadata_dir(directory: str | Path):
    mdir = Path(directory)
    if not mdir.exists():
        mdir.mkdir(parents=True, exist_ok=True)
    hdir = home_dir()
    hdir = hdir / "metadata_dir.txt"
    with open(hdir, "w") as f:
        f.write(str(mdir))


# Custom data serializer
def metadata_to_string(metadata, level=0):
    output = []
    if isinstance(metadata, dict):
        output.append(f"{' '*level*2}" + "{\n")
        for key in metadata.keys():
            if isinstance(key, str):
                temp_key = f"'{key}'"
            else:
                temp_key = key
            output.append(f"{' '*level*2}{temp_key}:\n")
            if isinstance(metadata[key], (list, tuple, dict, Group, Subgroup, UniqueGroups)):
                temp = metadata_to_string(metadata[key], level + 1)
                output.extend(temp)
            else:
                if isinstance(metadata[key], str):
                    temp = f'"{metadata[key]}"'
                else:
                    temp = str(metadata[key])
                output[-1] = output[-1][:-1] + f" {temp},\n"
        output.append(f"{' '*level*2}" + "},\n")
    elif isinstance(metadata, list):
        output.append(f"{' '*level*2}" + "[\n")
        for index, val in enumerate(metadata):
            if isinstance(val, (dict)):
                temp = metadata_to_string(val, level + 1)
                output.extend(temp)
            else:
                if isinstance(val, str):
                    val = f"'{val}'"
                if index == 0:
                    prepend = " " * level * 2
                    postend = ","
                elif index == (len(metadata) - 1):
                    postend = ",\n"
                    prepend = ""
                else:
                    postend = ","
                    prepend = ""
                temp = prepend + str(val) + postend
                output.append(temp)
        output.append(f"{' '*level*2}" + "],\n")
    elif isinstance(metadata, (Group, Subgroup, UniqueGroups)):
        temp = metadata_to_string(metadata._asdict(), level + 1)
        output.extend(temp)
    elif isinstance(metadata, tuple):
        output.append(f"{' '*level*2}" + "(\n")
        for index, val in enumerate(metadata):
            if isinstance(val, (dict)):
                temp = metadata_to_string(val, level + 1)
                output.extend(temp)
            else:
                if isinstance(val, str):
                    val = f"'{val}'"
                if index == 0:
                    prepend = " " * level * 2
                    postend = ","
                elif index == (len(metadata) - 1):
                    postend = ",\n"
                    prepend = ""
                else:
                    postend = ","
                    prepend = ""
                temp = prepend + str(val) + postend
                output.append(temp)
        output.append(f"{' '*level*2}" + "),\n")
    elif isinstance(metadata, callable):
        temp = (" " * level * 2) + "callable,\n"
        output.append(temp)
    if level == 0:
        output[-1] = "}\n"
    return output

def _process_metadata(metadata: dict):
    if "plot_prefs" in metadata:
        for plot_item in metadata["plot_prefs"]:
            for key, value in plot_item.items():
                if isinstance(value, dict):
                    if "group" in value:
                        value = Group(**value)
                        plot_item[key] = value
                    elif "subgroup" in value:
                        value = Subgroup(**value)
                    elif "unique_groups" in value:
                        value = UniqueGroups(**value)
    return metadata

def load_metadata(metadata_path: str | Path):
    if not isinstance(metadata_path, (str, dict, Path)):
        raise AttributeError("metadata_path must be a string, dict, or Path")
    if isinstance(metadata_path, str):
        if len(metadata_path.split(".")) == 1:
            file_path = metadata_dir()
            file_path = file_path / f"{metadata_path}"
        else:
            file_path = Path(metadata_path)
    else:
        file_path = Path(metadata_path)
    file_path = Path(file_path)
    file_path = file_path.with_suffix(".txt")
    if file_path.exists():
        with open(file_path, "r") as f:
            lines = f.read()
            lines = lines.replace("\n", "")
        output = ast.literal_eval(lines)
    else:
        raise FileNotFoundError("Metadata file does not exist")
    output = _process_metadata(output)
    return output


def save_metadata(metadata, file_path: str | Path):
    if isinstance(file_path, str):
        if len(file_path.split(".")) == 1:
            temp_path = metadata_dir()
            file_path = temp_path / file_path
    file_path = Path(file_path)
    with open(file_path.with_suffix(".txt"), "w") as f:
        f.writelines(metadata_to_string(metadata))
