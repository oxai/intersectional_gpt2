# How True is GPT-2?

We include code and data for the paper [`How True is GPT-2? An Empirical Analysis of Intersectional Occupational Biases`](http://arxiv.org/abs/2102.04130) 

<p align="center">
<img src="data/_splash.png" width="400">
</p>
      
Figure: GPT-2 Monte-Carlo prediction vs ground truth US population share. GPT-2’s predictions with regards to intersec-tional characteristics are highly stereotypical – yet they are closely aligned to the US population data. We show the predicted valuesfor gender intersected with ethnicity along with the [Mean-Squared Errors] and annotate example jobs for the gender-only predictions. For details, please look at the paper.

## Code
We include the following iPython notebooks. Each of the iPython notebooks are organized 
* `NB1_Generate_Text_from_GPT-2.ipynb`: Generates the gender-occupation prefix templates and generates sentences using GPT-2 by calling the HuggingFace API
* `NB2_Generate_Name_Sentences_from_GPT-2.ipynb`: Generates the name-occupation prefix templates and generates sentences using GPT-2 by calling the HuggingFace API
* `NB3_Generate_Freq_Matrices.ipynb`: Generates frequency matrices for occupations associated for each gender-intersection and name generated by the prefix templates
* `NB4_Generate_Figures.ipynb`: Generates the following figures: distribution plots of occupations, Gini indices, Lorenz curves, gender parity bar and scatter plots 
* `NB5_Logistic_Regression.ipynb`: Performs logistic regression and generates heatmaps of regression coefficients 
* `NB6_XLNet.ipynb`: Generates frequency matrix, distribution plot of occupations, and gender parity bar plot for occupations generated using XLNet
* `NB7_Comparison_to_RealWorld.ipynb`: Compares occupational distribution generated by GPT-2 with real-world US Labor Bureau data

## Data
We include the following CSV files:
* `xlnet_gender_occupations.csv`: a list of occupations associated by XLNet, extracted by Stanford’s NER tool, for ‘man’/’woman’
* `gender_occupations_template.csv`: a list of occupations associated by GPT-2, extracted by Stanford’s NER tool, for ‘man’/’woman’ intersected with the categories for ethnicity, sexuality, political affiliation, and religion
* `top_names_country_processed.csv`: top names chosen for each geographic region, extracted from Wikipedia 
* `names_occupations_template.csv`: a list of occupations associated by GPT-2, extracted by Stanford’s NER tool, for geographic-specific names
* `job_replacements.csv`: renames duplicate occupations (i.e. nurse and nurse practitioner)
* `us_rowwise_data.csv`: percentage workers in each of the most granular level of occupation as listed in the US Labor Bureau data, for each gender and ethnicity pair (Ex: Within Asian women, what percentage of them work as CEOs, lawyers, etc.?)
* `gpt_vs_us_data.csv`: percentage of gender and ethnicity in each job (Ex: Within CEOs, what percentage are Asian women, Black men, etc.?)

## Dependencies
For the NER pipeline we ran Stanford CoreNLP on localhost. To setup Stanford CoreNLP we followed instructions listed here: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

## Citation
If you find the code in this repo useful, please consider citing:
```
@misc{Kirk2021How,
      title={How True is GPT-2? An Empirical Analysis of Intersectional Occupational Biases}, 
      author={Hannah Kirk and Yennie Jun and Haider Iqbal and Elias Benussi and Filippo Volpin and Frederic A. Dreyer and Aleksandar Shtedritski and Yuki M. Asano},
      year={2021},
      eprint={2102.04130},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
