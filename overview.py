import os


def document_project(root_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as doc:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Remove hidden directories from dirnames in-place
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            level = dirpath.replace(root_dir, "").count(os.sep)
            indent = "    " * level
            doc.write(f"{indent}{os.path.basename(dirpath)}/\n")
            subindent = "    " * (level + 1)
            for f in filenames:
                if f.startswith("."):  # Skip hidden files as well (optional)
                    continue
                file_path = os.path.join(dirpath, f)
                doc.write(f"{subindent}{f}\n")
                if f.endswith(".py"):
                    doc.write(f"{subindent}--- Code in {f} ---\n")
                    with open(file_path, "r", encoding="utf-8") as code_file:
                        for line in code_file:
                            doc.write(f"{subindent}{line}")
                    doc.write(f"{subindent}--- End of {f} ---\n\n")


# Usage
document_project("./wblbm/", "./LBM_code_base.txt")
