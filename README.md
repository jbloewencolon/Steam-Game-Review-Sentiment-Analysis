# The Road to Hades is Paved with Positive Reviews: 
## Creating a Better Game-Review Sentiment Analyzer
A Flatiron School Phase 4 Project

![machine_sentiment.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/machine%20sentiment.png)
    
By Jordan Loewen-Colón July 11th 2023

# The Business Problem

SuperGiant Games, a video game company, faces a challenge in understanding the specific aspects of their games that players enjoy. The primary hurdle is the lack of detailed information available in the reviews from Steam, the largest aggregator of game reviews. While Steam provides a binary recommendation status, it does not offer insights into the underlying reasons behind players' preferences. To address this, SuperGiant Games has assigned us the task of developing a model that can analyze game reviews and provide a more comprehensive and nuanced understanding of what players appreciate about their games. By leveraging advanced techniques, we aim to uncover valuable insights that go beyond simple recommendations, allowing SuperGiant Games to gain a deeper understanding of player preferences and further enhance their game development strategies.

# Recommendations:
We recommend SuperGiant Games to continue focusing on the strong storytelling elements of Hades, as players consistently highlighted this aspect. Additionally, efforts can be made to enhance players' ability to express their positive impressions of the 'music' and 'visuals' by potentially providing prompts or specific questions related to these aspects in reviews or feedback forms. This would help gather more detailed and insightful feedback on the game's audio and visual components.

# Data Understanding

To make our recommendations, we analyzed reviews from the video game Hades, found on Steam. There were initially:

* 228720 Reviews
* 26 Columns

Since the vast majority of reviews were rated positive ('voted_up) we kept our interest primarily to 'reviews' and 'author.playtime_forever.' 

# Step 2: Data Preperation

We dropped all non-English reviews, lemmatized, tokenized, and created new columns for length of review (small, medium, large) and a new binary column depicting above or below-average playtime. Finally, we created a pipeline to streamline our model production going forward and split the data into training and test sets.

# Data Modeling

Our first model was a simple logistic regression. Starting with a logistic regression model offers interpretability and simplicity, serving as an efficient method to establish baseline performance for binary classification, such as distinguishing participants willing to try psychedelics. Its probabilistic output and capability to highlight feature importance provides crucial insights into factors influencing willingness to participate in the trial, while setting a comparative standard for future, more complex models. The model resulted in a **precision score of 97%** which implies a lower rate of false positives, as precision is the ratio of true positives to the sum of true positives and false positives.

![confusionmatrix.png](https://github.com/jbloewencolon/Phase-3---Open-to-Psychedelic-Experience/blob/main/Images/log%20reg%20confusion%20matrix.png)

The model had a total of 9 false positives which isn't bad. We then looked to see what the coefficients with the highest magnitude were: 

![log reg featureimport.png](https://github.com/jbloewencolon/Phase-3---Open-to-Psychedelic-Experience/blob/main/Images/log%20reg%20feature%20importances.png)

The Oscore had the largest coefficient magnitude of all our personality traits. The coefficient value of 0.53 for "Oscore" means that for every one-unit increase, the log odds of the outcome "Psychedelics" being 'yes' (versus 'no') increase by 0.5, assuming all other variables in the model are held constant. To better understand this in terms of odds (rather than log odds), we can calculate the odds by taking the exponent of the coefficient: exp(0.5) ≈ 1.65. This means that for every one-unit increase in "Oscore", the odds of the outcome "Psychedelics" being 'yes' (versus 'no') increase by about 65%, assuming all other variables in the model are held constant. And since, as we saw above, people who have taken psychedelics have a higher Oscore, we can assume that a higher Oscore means a higher likelihood that a person has consumed a psychedelic (or perhaps will).

We continued with our modeling by using both a Random Tree Classifier (RFC) and a Gradient Boost classifier (GBC) which produced **precision ratings of 97%** as well. Looking at the feature importances of both revealed an agreement between the Log and GBC models that Oscore is the most important feature, while our RFC model though Oscore was second to SS.

**RFC Feature Importances:**
![RFC featureimport.png](https://github.com/jbloewencolon/Phase-3---Open-to-Psychedelic-Experience/blob/main/Images/RFC%20feature%20importance.png)

**GBC Feature Importances:**
![GBC featureimport.png](https://github.com/jbloewencolon/Phase-3---Open-to-Psychedelic-Experience/blob/main/Images/GBC%20feature%20importance.png)

When comparing all our models, it looks like our **Logistical Regression model scores highest on accuracy, and F1**. The RFC model scored highest on precision and recall. While the scores are close, we'll give the Log model the edge and choose it to draw understandings. 

**Model: Logistic Regression**

- accuracy: 0.8763326226012793
- precision: 0.9723926380368099
- recall: 0.8661202185792349
- F1-score: 0.9161849710982659

**Model: RFC**

- accuracy: 0.8571428571428571
- precision: 0.9746031746031746
- recall: 0.8387978142076503
- F1-score: 0.9016152716593245

**Model: GBC**

- accuracy: 0.8528784648187633
- precision: 0.9684542586750788
- recall: 0.8387978142076503
- F1-score: 0.8989751098096632

# Data Understanding

Interpretations:

Out of the 130 coefficients, our Oscore is in the top 35, but there is a signifigant difference between it and our leading coefficients: Never Having Taken a Legal Highs, Nicotine, and Cocaine. A question to ask is, do people who take psychedelics and have NEVER taken a Legal High score higher on openness than non-psychedelic consumers?

![top coefficients.png](https://github.com/jbloewencolon/Phase-3---Open-to-Psychedelic-Experience/blob/main/Images/top%20coefficients.png)

As it turns out:
The average Oscore for individuals who scored 'CL0' in the Legalh column and are categorized as 'Psychedelics' is: -0.13
The average Oscore for individuals who scored 'CL0' in the Legalh column and are categorized as non-psychedelics is: -0.6
The average Oscore for all other results in the Legalh column is: 0.41

Looks like **psychedelic users score higher on the Oscore than non-psychedelic users**! So that indicates that Oscore possibly contributes positively to Psychedelic use in conjunction with our most important coefficient. However, it looks like the average Legal High users scores even higher. That means we will want to filter them out. But what about Oscore more generally?

![oscore use.png](https://github.com/jbloewencolon/Phase-3---Open-to-Psychedelic-Experience/blob/main/Images/oscore%20use%20vs%20nonuse.png)

The average 'Oscore' for individuals who take psychedelics is: 0.15
The average 'Oscore' for individuals who do not take psychedelics is: -0.59

# Conclusion

Our first recommedation involves **Recruitment Strategy**. The precision of 97% achieved by the logistic regression model indicates that the model is effective in identifying potential trial participants who are genuinely likely to experiment with psychedelics. **The institute can focus on targeting individuals who exhibit characteristics associated with high precision, such as never having taken legal highs, nicotine, or cocaine.** These factors can be used as screening criteria during the recruitment process.

Our second recommendation involves the **importance of the Oscore**. The Oscore coefficient with a magnitude of 0.5 compared to the other personality traits indicates that it is one of the significant predictors of psychedelic use. This model indicates that individuals with higher Oscores tend to be more inclined towards using psychedelics. **Therefore, considering an individual's Oscore can contribute positively to the prediction of psychedelic usage.** The institute can incorporate the assessment of Oscore into the screening process to further refine the selection of potential participants.

Our final recommendation involves a **comparison of Oscore and psychedelic use**. The analysis of the average Oscore for individuals who take psychedelics and those who do not reveals a notable difference. Individuals who take psychedelics have an average Oscore of 0.152, while those who do not have an average Oscore of -0.593. This indicates that Oscore may be a relevant factor in understanding the inclination towards psychedelic use. **The institute can explore further research to investigate the relationship between Oscore and the therapeutic effects of psychedelic-assisted therapies**.


# Next Steps

Our logistic regression model may have scored so high for two reasons:
1) Because log regressions assume that there's a linear decision boundary between the classes, while decision trees (and by extension, Random Forests and Gradient Boosting models) do not, If the data indeed has a linear decision boundary, logistic regression might outperform more complex models.

2) Random Forest and Gradient Boosting models are more complex than logistic regression, and this complexity can lead them to overfit the training data, especially if the dataset is small, which ours is.

Therefore, getting more data might actually allow our more complex models to provide more precise predictions.

# Questions?
For a full analysis please check the Jupyter Notebook or slide presentation.
Further questions? Contact Jordan Loewen-Colón @ jbloewen@syr.edu

## Repository Structure


```
├── data : data used for modeling
├── images : images used in PPT and README
├── draft 1 : previous files from first draft of project
├── SMOTE version.ipynb : notebook used to pull from API
├── README.md : project information and repository structure
├── presentation.pdf : the powerpoint presentation used to present data analysis

