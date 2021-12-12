# MP-03
This mini-project experiments with <i>word embeddings</i> to solve a Synonym test automatically and see if the program does better than humans.
It also compares the performance of different models on different data sets and analyzes these results to draw insight.<br>

## Installation
*Assuming `python3.8` and `anaconda` platform installed*. <br>
1. Activate a conda environment inside the `MP-03` folder <br>
```shell
cd MP-03
conda activate
```
2. Install any non-built-in modules or packages<br>
```shell
pip install pandas
pip install gensim
```
4. Run script<br>
```shell
python3 mp03.py
```
## Imports
The script requires the following packages and libraries:
```python
import gensim.downloader as api
import os
import pandas as pd
import random
```
## Output
<b>`results/`</b> contains the following output files:
1. From Task 1:
- `word2vec-google-news-300-details.csv`
2. From Task 2:
- `fasttext-wiki-news-subwords-300-details.csv`
- `glove-wiki-gigaword-300-details.csv`
- `glove-twitter-200-details.csv`
- `glove-twitter-50-details.csv`
4. For analysis:
- `analysis.csv`
- `random_baseline-details.csv`

## Presentation and Discussion Files
The following file is for Demo presentation purposes:
1. `MP03-Presentation.pdf`
2. `MP03-Presentation.ipynb`

## Author
Alvira Konovalov<br>
40074264
