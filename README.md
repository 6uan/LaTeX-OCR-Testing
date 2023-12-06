# LaTeX OCR (pix2tex) - with Handwritten Recognition

[![GitHub](https://img.shields.io/github/license/lukas-blecher/LaTeX-OCR)](https://github.com/lukas-blecher/LaTeX-OCR) [![Documentation Status](https://readthedocs.org/projects/pix2tex/badge/?version=latest)](https://pix2tex.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/pix2tex?logo=pypi)](https://pypi.org/project/pix2tex) [![PyPI - Downloads](https://img.shields.io/pypi/dm/pix2tex?logo=pypi)](https://pypi.org/project/pix2tex) [![GitHub all releases](https://img.shields.io/github/downloads/lukas-blecher/LaTeX-OCR/total?color=blue&logo=github)](https://github.com/lukas-blecher/LaTeX-OCR/releases) 


## Preview from Lukas Blecher (Repo Creator)

![header](https://user-images.githubusercontent.com/55287601/109183599-69431f00-778e-11eb-9809-d42b9451e018.png)


We built upon the pix2tex project (https://github.com/lukas-blecher/LaTeX-OCR). It is a LaTeX OCR that takes the most recent image [of a formula or equation] in your clipboard and converts it into LaTeX code.

LaTeX is the preferred format when creating academic material. An OCR that specializes in converting image equation to LaTeX equations would allow students and academic professional to leverage the older material on more advanced mathematical subjects where most of the searchable resources out there are on subject closely related to algebra. Also, popular OCR tools like Socratic (made by Google) do not accurately read advanced calculus equations and creating the LaTeX equations from often lead to format mistakes.

**Our Goals:**

- Gain the competency of navigating, analyzing, and comprehending code bases; open-source contributing.

- Use additional data to expand upon the tool’s the handwritten scanning capabilities.


## Problem Statement <!--- do not change this line -->

When **academic professionals** use old documents, build upon the math found in scientific papers, or repurpose documents made by other academic professionals, they need the original LaTeX syntax to do their work. Currently, professors would manually create the equations or email the author assuming they can get a hold of them and they reply. This is an unreliable and tedious process.

When **students** do their homework or search for help online, they need a standard and reliable format to share their equations. Currently, students don’t search for help or hand write their equations. When using AI tools and search engines LaTeX is often the only format that computers can read large, complex, or calculus equations. This is an unrefined process when we have LaTeX, but LaTeX is hard to learn and not well known among students. When student do try LaTeX, it’s very easy to make mistakes.


## Key Results <!--- do not change this line -->

The main results was a 0.537 accuracy during epoch 2 when training the model with a dataset of hanwritten equation samples.


## Methodologies <!--- do not change this line -->

To accomplish this, we conducted research on Kaggle to find handwritten datasets and we did. We then forked the original pix2tex repository and looked through the files and code. We discovered a Google Collab workspace that detailed how to train the model with our own dataset. After some troubleshooting, we were successful in training the model and attained a 0.537 accuracy.


## Data Sources 

Kaggle Datasets: https://www.kaggle.com/datasets/aidapearson/ocr-data


## Technologies Used <!--- do not change this line -->

- Python
- Google Collaborate
- Wandb


## Authors <!--- do not change this line -->

Juan Gomez - Undergraduate - [GitHub](https://github.com/6uan) - [LinkedIn](https://www.linkedin.com/in/j-gomez-/) <br>
Enrique Torres - Undergraduate - [GitHub](https://github.com/EnriqueTMT) - [LinkedIn](https://www.linkedin.com/in/tenrique/) <br>
Lukas Blecher (original developer of pix2tex) - [GitHub](https://github.com/lukas-blecher) <br>
Contributing developers to pix2tex <br>


## Using the model
To run the model you need Python 3.7+ ✔

If you don't have PyTorch installed. Follow their instructions [here](https://pytorch.org/get-started/locally/). ✔

Install the package `pix2tex`: 

```
pip install "pix2tex[gui]" ✔
```

Model checkpoints will be downloaded automatically.

There are three ways to get a prediction from an image. 
1. You can use the command line tool by calling `pix2tex`. Here you can parse already existing images from the disk and images in your clipboard.

2. Thanks to [@katie-lim](https://github.com/katie-lim), you can use a nice user interface as a quick way to get the model prediction. Just call the GUI with `latexocr`. From here you can take a screenshot and the predicted latex code is rendered using [MathJax](https://www.mathjax.org/) and copied to your clipboard.

    If the model is unsure about the what's in the image it might output a different prediction every time you click "Retry". With the `temperature` parameter you can control this behavior (low temperature will produce the same result).

3. Use from within Python
    ```python
    from PIL import Image
    from pix2tex.cli import LatexOCR
    
    img = Image.open('path/to/image.png')
    model = LatexOCR()
    print(model(img))
    ```

The model works best with images of smaller resolution. That's why I added a preprocessing step where another neural network predicts the optimal resolution of the input image. This model will automatically resize the custom image to best resemble the training data and thus increase performance of images found in the wild. Still it's not perfect and might not be able to handle huge images optimally, so don't zoom in all the way before taking a picture. 

Always double check the result carefully. You can try to redo the prediction with an other resolution if the answer was wrong.

## Training the model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukas-blecher/LaTeX-OCR/blob/main/notebooks/LaTeX_OCR_training.ipynb)

Install a couple of dependencies `pip install "pix2tex[train]"`.
1. First we need to combine the images with their ground truth labels. I wrote a dataset class (which needs further improving) that saves the relative paths to the images with the LaTeX code they were rendered with. To generate the dataset pickle file run 

```
python -m pix2tex.dataset.dataset --equations path_to_textfile --images path_to_images --out dataset.pkl
```
To use your own tokenizer pass it via `--tokenizer` (See below).

You can find my generated training data on the [Google Drive](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO) as well (formulae.zip - images, math.txt - labels). Repeat the step for the validation and test data. All use the same label text file.

2. Edit the `data` (and `valdata`) entry in the config file to the newly generated `.pkl` file. Change other hyperparameters if you want to. See `pix2tex/model/settings/config.yaml` for a template.
3. Now for the actual training run 
```
python -m pix2tex.train --config path_to_config_file
```

If you want to use your own data you might be interested in creating your own tokenizer with
```
python -m pix2tex.dataset.dataset --equations path_to_textfile --vocab-size 8000 --out tokenizer.json
```
Don't forget to update the path to the tokenizer in the config file and set `num_tokens` to your vocabulary size.

## Model
The model consist of a ViT [[1](#References)] encoder with a ResNet backbone and a Transformer [[2](#References)] decoder.

### Performance
| BLEU score | normed edit distance | token accuracy |
| ---------- | -------------------- | -------------- |
| 0.88       | 0.10                 | 0.60           |


## Acknowledgment
Code taken and modified from [lucidrains](https://github.com/lucidrains), [rwightman](https://github.com/rwightman/pytorch-image-models), [im2markup](https://github.com/harvardnlp/im2markup), [arxiv_leaks](https://github.com/soskek/arxiv_leaks), [pkra: Mathjax](https://github.com/pkra/MathJax-single-file), [harupy: snipping tool](https://github.com/harupy/snipping-tool)

## References
[1] [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR)

