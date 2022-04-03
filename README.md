# eyetracking_classification
Uses machine learning to classify eye-tracking data.

Installation
------------

```
  conda create -n eye python==3.8.5
  conda activate eye
  pip install -r requirements.txt
```
or
```
  conda env create -f environment.yml
```


Dataset
------------
Expects data in the following folder structure:
```
eyetracker_data
 ┣ data
 ┃ ┣ N1.txt
 ┃ ┣ N10.txt
 ┃ ┣ N100.txt
 ┃ ┣ N1000.txt
 ┃ ┗ N999.txt
 ┣ Outcome.txt
 ┗ sub_num.txt
```


Running it
------------
To run:
Modify config.py accordingly then:
```
  python main.py
```
or
```
  python config.py
```
