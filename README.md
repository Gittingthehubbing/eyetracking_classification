# eyetracking_classification
Uses machine learning to classify eye-tracking data

Installation
------------

```
  conda create -n eye python==3.8.5
  conda activate eye
  pip install -R requirements.txt
```
or
```
  conda env create -f environment.yml
```

Expects data in the following folder structure:

eyetracker_data
 ┣ data
 ┃ ┣ .ipynb_checkpoints
 ┃ ┃ ┣ N1-checkpoint.txt
 ┃ ┃ ┗ N999-checkpoint.txt
 ┃ ┣ N1.txt
 ┃ ┣ N10.txt
 ┃ ┣ N100.txt
 ┃ ┣ N1000.txt
 ┃ ┗ N999.txt
 ┣ Outcome.txt
 ┗ sub_num.txt
