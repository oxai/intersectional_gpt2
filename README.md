# Bias-out-of-the-box

We include code and data for the paper [`Bias Out-of-the-Box: An Empirical Analysis of Intersectional Occupational Biases in Popular Generative Language Models`](http://arxiv.org/abs/2102.04130) 

<p align="center">
<img src="data/_splash.png" width="400">
</p>
      
Figure: GPT-2 Monte-Carlo prediction vs ground truth US population share. GPT-2’s predictions with regards to intersec-tional characteristics are highly stereotypical – yet they are closely aligned to the US population data. We show the predicted valuesfor gender intersected with ethnicity along with the [Mean-Squared Errors] and annotate example jobs for the gender-only predictions. For details, please look at the paper.

## Code
We include the following python scripts:
* `Generate_Freq_Matrices.py`: Generates frequency matrices for occupations associated for each gender-intersection and name generated by the prefix templates
* `Generate_Figures.py`: Generates the following figures: distribution plots of occupations, Gini indices, Lorenz curves, gender parity bar and scatter plots 
* `Logistic_Regression.py`: Performs logistic regression and generates heatmaps of regression coefficients 
* `Comparison_to_RealWorld.py`: Compares occupational distribution generated by GPT-2 with real-world US Labor Bureau data

## Data
We include the following CSV files and folder structure:
```
.
├── scripts
├── data
│   ├── XLNET
|     |-- NER_output
|           |-- names_occupations_template.csv
|           |-- identity_occupations_template.csv
│   ├── GPT-2
|     |-- NER_output
|           |-- names_occupations_template.csv
|           |-- identity_occupations_template.csv
│   ├── shared_data
|     |-- job_replacements.csv
|     |-- us_rowwise_data.csv
|     |-- gpt_vs_us_data.csv
|     |-- top_names_country_processed.csv
            
└── ...
```


Where:
* `names_occupations_template.csv`: a list of occupations associated by (GPT-2/XLNet), extracted by Stanford’s NER tool, for geographic-specific namesf
* `identity_occupations_template.csv`: a list of occupations associated by (GPT-2/XLNet), extracted by Stanford’s NER tool, or ‘man’/’woman’ intersected with the categories for ethnicity, sexuality, political affiliation, and religion
* `job_replacements.csv`: renames duplicate occupations (i.e. nurse and nurse practitioner)
* `us_rowwise_data.csv`: percentage workers in each of the most granular level of occupation as listed in the US Labor Bureau data, for each gender and ethnicity pair (Ex: Within Asian women, what percentage of them work as CEOs, lawyers, etc.?)
* `gpt_vs_us_data.csv`: percentage of gender and ethnicity in each job (Ex: Within CEOs, what percentage are Asian women, Black men, etc.?)
* `top_names_country_processed.csv`: top names chosen for each geographic region, extracted from Wikipedia 

## Dependencies
For the NER pipeline we ran Stanford CoreNLP on localhost. To setup Stanford CoreNLP we followed instructions listed here: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

## Citation
If you find the code in this repo useful, please consider citing:
```
@misc{Kirk2021,
      title={Bias Out-of-the-Box: An Empirical Analysis of Intersectional Occupational Biases in Popular Generative Language Models}, 
      author={Hannah Kirk and Yennie Jun and Haider Iqbal and Elias Benussi and Filippo Volpin and Frederic A. Dreyer and Aleksandar Shtedritski and Yuki M. Asano},
      year={2021},
      eprint={2102.04130},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
