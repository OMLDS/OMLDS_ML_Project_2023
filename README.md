## OMLDS End-to-End Machine Learning Challenge Project

Welcome to the 2023 OMLDS End-to-End Machine Learning Challenge Project.

### Objective

Expose and challenge OMLDS members to develop an end-to-end machine learning project from business insights to model deployment.

### Business problem

A regional bank has awarded your organization to develop a business solution to proactively detect credit card fraudulent transactions and achieve, at least, **15%** reduction in costs related to insurance and customer dissatisfaction.

### Dataset

The bank's IT team has assembled a dataset with the relevant factors extracted from their DBMS.

Dataset link: https://www.kaggle.com/datasets/kartik2112/fraud-detection

### Resources

The Cost of Credit Card Fraud https://www.tokenex.com/blog/vh-the-cost-of-credit-card-fraud/

Python for Data Analysis, 2nd edition https://www.amazon.com/Python-Data-Analysis-Wrangling-IPython-ebook/dp/B075X4LT6K

Data Science for Business, 1st edition https://a.co/d/50yScI0

Exploratory Data Analysis for Feature Selection in Machine Learning https://shorturl.at/aouO5

# Getting Started
Be a hands on participant in the project to learn new skills and collaborate 
with fellow OMLDS members.  
## You need the following for hands on participation:
* GitHub account ([signup](https://github.com/signup) for free account if you don't have one)
* Google and related Google services
  * [Google](https://accounts.google.com/signup)
  * [Colab](https://colab.research.google.com/) 
  * [Kaggle](https://www.kaggle.com/account/login)

Once you have a github account, fork this repository to your account.

* Some articles to familiarize yourself with using Colab with GitHub and Conda
  * [Clone a GitHub Repo to Colab](https://www.geeksforgeeks.org/how-to-clone-github-repository-and-push-changes-in-colaboratory/)
  * [Instal Conda in Colab](https://inside-machinelearning.com/en/how-to-install-use-conda-on-google-colab/)


## Folder structure
The project folder structure is setup to organize code for development and 
deployment.  The folder structure is a subjective choice of how to organize the 
code and should be altered to fit the needs of the project and team members.


```
omlds_ml_project_2023  <- project's root folder/directory
├── README.md          <- the top-level README for this project.
├── LICENSE            <- license associate with this project
├── .gitignore         <- defines any files which should not be tracked in the 
│                         git repository.
│
├── data               <- project data files 
|   ├── readme.md      <- readme specific to this folder
|   ├── <other>        <- subfolders as needed
│
├── docs               <- manuals and/or reference info for project 
|   ├── readme.md      <- readme specific to this folder 
|   ├── <other>        <- subfolders as needed
│
├── notebooks          <- interactive python files for rapid dev idead testing
|   ├── readme.md      <- readme specific to this folder
|   ├── <other>        <- subfolders as needed
│
├── src                <- source code 
|   ├── readme.md      <- readme specific to this folder 
|   ├── data_handling  <- scripts to ingest, write or transform data 
|   ├── features       <- scripts to defining ML model features (factors)
|   ├── models         <- scripts to train and use trained ML models
│
│   Note: typically use one of the following requirements files not both.  We 
|         include both for understanding/learning the process.
├── requirements.txt   <- file to reproducing the virtual environment with pip
│                         generated with `pip list --format=freeze > requirements.txt`
├── requirements.yml   <- file to reproducing the virtual environment with conda
│                         generated with `conda env export > requirements.yml`
│                         

```
