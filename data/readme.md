# Data folder
The purpose of this folder is to contain data files for use in development and
testing of the project.

## Data files
Data files are not typically stored in the git repository.  This is managed 
by excluding them in the .gitignore file in the project's root directory.  More
about how to use the .gitignore file can be found at [git-scm.com/docs/gitignore](https://git-scm.com/docs/gitignore)

In this project the data files are in .csv (comma separated values) format. The
files are excluded from the git repository by the entry *.csv in .gitignore. 
The files are available here on [kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection?resource=download)


## Folder substructure
The data folder can also be organized into subfolders when it makes sense for 
the project.  

An example is when the data folder will contain both input and output data.  
Then, it may be useful to have subfolders like the following structure.  Name 
the subfolders as appropriate for your project.

```
project_root           <- Project's root folder/directory
├── README.md          <- The top-level README for developers using this project.
├── data
|   ├── readme.md      <- readme specific to this folder
|   ├── input          <- input data files 
|   ├── output         <- output data files

```
