# UCD-TREC-2019-IS
This repository contains reproducible code of UCD at TREC 2019-B [Incident Streams track](http://dcs.gla.ac.uk/~richardm/TREC_IS/)

### Requirements

-  Python 3+
- `pip install allennlp`. For running AllenNLP on Windows, refer to [this repository](https://github.com/wangcongcong123/AllenNLPonWins). For unfarmilar with AllenNLP, refer to [the tutorials](https://github.com/allenai/allennlp/tree/master/tutorials).

### Steps of Usage

#### Baseline
About how to train, predict and submit, go through [the notebook script](UCDbaseline.ipynb).

---
The following gives instructions on how to train our three deep learning based models and make predictions by them.
Here only provides [a fixture of dataset](dataset/train_fixtures.json) (after simply being preprocessed) for both training and testing due to license limit. You can follow the instructions below to make it adaptive to the whole dataset. 

#### bilstmalpha
- Train model for information type categorization: `allennlp train experiments/category_crisis_classifier_bilstmalpha.json -s tmp/category_crisis_classifier_bilstmalpha --include-package allennlp_mylib` Change cuda_device to -1 in category_crisis_classifier_bilstmalpha.json if you want to run on CPU.
- Train model for priority estimation: `allennlp train experiments/priority_crisis_classifier_bilstmalpha.json -s tmp/priority_crisis_classifier_bilstmalpha --include-package allennlp_mylib`
- Make predictions for information types: `allennlp predict tmp/category_crisis_classifier_bilstmalpha/model.tar.gz dataset/train_fixtures.json --include-package allennlp_mylib --predictor category_crisis_predictor --output-file predictions/predicted-category-bilstmalpha`
- Make predictions for priority estimation: `allennlp predict tmp/priority_crisis_classifier_bilstmalpha/model.tar.gz dataset/train_fixtures.json --include-package allennlp_mylib --predictor priority_crisis_predictor --output-file predictions/predicted-priority-bilstmalpha`

#### bilstmbeta
- The commands are the same as in bilstmalpha except for changing `bilstmalpha` to `bilstmbeta`

#### bcnelmo
- The commands are the same as in bilstmalpha except for changing `bilstmalpha` to `bcnelmo`
---
Taking bilstmalpha as an example, here gives instructions on how to submit or evaluate (on previous dataset: benchmarking analysis) the predictions made by the trained models in bilstmalpha.
- Refer to [this page](http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/2019B_Submission.html), and combine the priority and information type predictions to output a submit file as required format. In our runs, we used linear combination to generate the priority importane score, you can have any other ways to do this.
- If you want to evaluate on previous dataset for benchmarking analysis, refer to the [script](http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/TREC-IS_V2_2018Events_2019AFormat_Evaluation_Notebook.ipynb) that you can apply to evaluate on the submit runs.

### Extras
- To quickly apply GPT-2 for data augmentation, here is a recommended repository for reference: [gpt-2-Pytorch](https://github.com/graykode/gpt-2-Pytorch)

### Reference
If you want to use the code, cite the following paper, please.
    [TBA]
