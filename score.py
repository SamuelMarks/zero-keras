with open("KERAS_TODO.md") as f:
    lines = f.readlines()

total = 0
done = 0
for line in lines:
    if "|" in line and "keras." in line:
        total += 1
        if "[x]" in line:
            done += 1

print(f"{done}/{total} APIs checked in KERAS_TODO.md ({done / total * 100:.1f}%)")
