import os
import re
import subprocess
import json
import ast


def get_color(pct):
    if pct >= 100:
        return "brightgreen"
    if pct >= 90:
        return "green"
    if pct >= 80:
        return "yellowgreen"
    if pct >= 70:
        return "yellow"
    if pct >= 60:
        return "orange"
    return "red"


def format_cov(cov):
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_test_coverage():
    try:
        subprocess.run(["coverage", "json", "-o", "coverage.json"], check=False)
        with open("coverage.json", "r") as f:
            data = json.load(f)
            return data["totals"]["percent_covered"]
    except Exception:
        return 0.0


def get_doc_coverage():
    total_nodes = 0
    nodes_with_doc = 0
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    tree = ast.parse(f.read())

                # Module
                total_nodes += 1
                if ast.get_docstring(tree):
                    nodes_with_doc += 1

                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        total_nodes += 1
                        if ast.get_docstring(node):
                            nodes_with_doc += 1
    if total_nodes == 0:
        return 100.0
    return (nodes_with_doc / total_nodes) * 100.0


def update_readme():
    if not os.path.exists("README.md"):
        return
    test_cov = get_test_coverage()
    doc_cov = get_doc_coverage()
    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)
    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)
    with open("README.md", "r") as f:
        content = f.read()
    test_re = re.compile(
        r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\([^)]*\))?"
    )
    content = test_re.sub(
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](https://github.com/SamuelMarks/zero-keras/actions/workflows/ci.yml)",
        content,
    )
    doc_re = re.compile(
        r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\([^)]*\))?"
    )
    content = doc_re.sub(
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](https://github.com/SamuelMarks/zero-keras/tree/master/docs)",
        content,
    )
    with open("README.md", "w") as f:
        f.write(content)


if __name__ == "__main__":
    update_readme()
