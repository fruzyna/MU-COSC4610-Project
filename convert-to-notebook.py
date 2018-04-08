import nbformat
import re
from nbformat.v4 import new_notebook, new_code_cell

with open('liquor.py') as f:
    src = f.read()

src = src.replace('# coding: utf-8', '')
cells = [new_code_cell(cell) for cell in re.split('\n\n\n# In\[[0-9]+\]:\n\n\n', src)]
cells.pop(0)

nb = new_notebook(cells=cells)
nbformat.write(nb, 'liquor.ipynb')