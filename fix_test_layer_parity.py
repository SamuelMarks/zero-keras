with open("tests/test_layer_parity.py") as f:
    text = f.read()

text = text.replace('\\"', '"')
with open("tests/test_layer_parity.py", "w") as f:
    f.write(text)
