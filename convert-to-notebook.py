import nbformat
import re
import os
from nbformat.v4 import new_notebook, new_code_cell
       
# Removes blank lines at the ends of cells recursively
# This is an issue that notably occurs after the last cell
def removeExtraLines(cell):
    if cell.endswith('\n'):
        return removeExtraLines(cell[:-1])
    else:
        return cell;
    
# Converts a single .py file to a .ipynb given its name
def convertNotebook(name):
    with open(name + '.py') as f:
        src = f.read()

    src = src.replace('# coding: utf-8', '')
    cells = [new_code_cell(removeExtraLines(cell)) for cell in re.split('\n+# In\[.+\]:\n+', src)]
    cells.pop(0)

    nb = new_notebook(cells=cells)
    nbformat.write(nb, name + '.ipynb')
    print('Converted ' + name)

# Searches for .py files to convert in the current directory
for file in os.listdir('.'):
    name = os.path.splitext(file)[0]
    if file.endswith('.py') and name != 'convert-to-notebook':
        convertNotebook(name)