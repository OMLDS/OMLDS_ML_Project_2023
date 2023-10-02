# Notebooks folder
The purpose of files in this folder is rapid testing and exploring of ideas via  
interactive python methods.  This folder contains VSCode interactive 
python files (*.py) and Jupyter notebooks (*.ipynb) files.  These
files should not be used/referenced in any process or workflow outside this
folder.  The routines tested here may be the basis of python scripts used in a
solution and should be considered works in process (WIP).


### Jupyter Notebooks (*.ipynb)
Jupyter notebooks are ubiquitous with data science and prototyping, 
especially in python.

An advantage to Jupyter notebooks is they will render as static pages within
GitHub.

VSCode natively handles Jupyter notebook files.  If you choose to use 
VSCode as your editor you can use either Jupyter notebooks or the interactive 
of method of python files noted below.

## VSCode Interactive Python (*.py)
VSCode (Visual Studio Code) is an open source code editor/lightweight IDE
(Integrated Development Environment) by Microsoft.

A feature of VSCode is an interactive python method that enables running
python in blocks of code in the same way you interactively run in Jupyter 
notebook cells.  An advantage is these files use the standard *.py file 
extenstion and may be used as a regular python script file because the 
interactive cell designator is ignored as a python comment.

To designate an interactive code cell use the python comment symbol 
followed by two percent symbols ('# %%') as seen in this example code block.
Within the editor you'll see options for running the cell appear.

```
# %% - This line designates the start of a code cell
import pandas as pd
df = pd.DataFrame({"one": 1, "two":2}, {"one":1, "two":2})
print(df)
```

Simliarly to designate a Markdown cell add [markdown] after the double percent
symbol designator as in this example
```
# %% [markdown] - This line designates the start of a markdown cell
# # Markdown Cell Header 
# Example markdown cell
```


