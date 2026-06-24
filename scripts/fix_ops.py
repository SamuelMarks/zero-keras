with open("src/zero_keras/ops.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = 0

for i, line in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue

    if "raise NotImplementedError(" in line:
        # The string is on the next line
        str_line = lines[i + 1]
        op_name = str_line.split('"ops.')[1].split(" is not yet")[0]

        if op_name == "newaxis":
            # Find the "def newaxis" line backwards and remove the function definition
            j = len(new_lines) - 1
            while not new_lines[j].startswith("def newaxis"):
                j -= 1
            new_lines = new_lines[:j]
            new_lines.append("newaxis = msc_ops.newaxis\n")
            skip = 2  # skip the string and closing paren
        elif op_name == "image":
            j = len(new_lines) - 1
            while not new_lines[j].startswith("def image"):
                j -= 1
            new_lines = new_lines[:j]
            new_lines.append("image = msc_ops.image\n")
            skip = 2
        else:
            indent = line.split("raise")[0]
            new_lines.append(
                f'{indent}return _wrap_op("{op_name}", getattr(msc_ops, "{op_name}"))(*args, **kwargs)\n'
            )
            skip = 2
    else:
        new_lines.append(line)

with open("src/zero_keras/ops.py", "w") as f:
    f.writelines(new_lines)
