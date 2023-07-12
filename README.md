# The Road to Hades is Paved with Positive Reviews: 
## Creating a Better Game-Review Sentiment Analyzer
A Flatiron School Phase 4 Project

![machine_sentiment.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/machine%20sentiment.png)
    
By Jordan Loewen-Colón July 11th 2023

# The Business Problem

SuperGiant Games, a video game company, faces a challenge in understanding the specific aspects of their games that players enjoy. The primary hurdle is the lack of detailed information available in the reviews from Steam, the largest aggregator of game reviews. While Steam provides a binary recommendation status, it does not offer insights into the underlying reasons behind players' preferences. To address this, SuperGiant Games has assigned us the task of developing a model that can analyze game reviews and provide a more comprehensive and nuanced understanding of what players appreciate about their games. By leveraging advanced techniques, we aim to uncover valuable insights that go beyond simple recommendations, allowing SuperGiant Games to gain a deeper understanding of player preferences and further enhance their game development strategies.

# Recommendations:
We recommend SuperGiant Games to continue focusing on the strong storytelling elements of Hades, as players consistently highlighted this aspect. Additionally, efforts can be made to enhance players' ability to express their positive impressions of the 'music' and 'visuals' by potentially providing prompts or specific questions related to these aspects in reviews or feedback forms. This would help gather more detailed and insightful feedback on the game's audio and visual components.

# Step 1: Data Understanding

To make our recommendations, we analyzed reviews from the video game Hades, found on Steam. There were initially:

* 228720 Reviews
* 26 Columns
    1. query_summary
    2. cursors
    3. recommendationid
    4. language
    5. review
    6. timestamp_created
    7. timestamp_updated
    8. voted_up
    9. votes_up
    10. votes_funny
    11. weighted_vote_score
    12. comment_count
    13. steam_purchase
    14. received_for_free
    15. written_during_early_access
    16. hidden_in_steam_china
    17. steam_china_location
    18. author.steamid
    19. author.num_games_owned
    20. author.num_reviews
    21. author.playtime_forever
    22. author.playtime_last_two_weeks
    23. author.playtime_at_review
    24. author.last_played
    25. timestamp_dev_responded
    26. developer_response

Since the vast majority of reviews were rated positive ('voted_up) we kept our interest primarily to 'reviews' and 'author.playtime_forever.' 

# Step 2: Data Preparation

We dropped all non-English reviews, lemmatized, tokenized, and created new columns for length of review (small, medium, large) and a new binary column depicting above or below-average playtime. Finally, we created a pipeline to streamline our model production going forward and split the data into training and test sets.

# Step 3: Data Modeling

Our first model was a simple logistic regression. Starting with a logistic regression model offers interpretability and simplicity, serving as an efficient method to establish baseline performance for binary classification, such as 

**RFC Feature Importances:**
![RFC featureimport.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/theme%20appearance%20counts.PNG)


**Model: Logistic Regression**



**Model: RFC**



**Model: GBC**


# Step 4: Data Understanding



# Conclusion

1) The reviews for the game Hades generally expressed positive sentiment, although the overall level of positivity falls within the range of 0 to 0.25.

2) When discussing their experiences with the game, players frequently emphasized the importance of the game's story. This indicates that the narrative elements of Hades are a significant aspect of player enjoyment.

3) It appears that players may have limited vocabulary when describing their appreciation for the 'music' and 'visuals' in Hades. This suggests that while players find these aspects appealing, they may struggle to articulate their specific likes or preferences regarding the music and visual elements of the game.
   
Given the computational limitations, making confident predictions about the specific aspects of the game that received positive reviews remains challenging. However, we were successful in adding complexity to the analysis of reviews by incorporating sentiment analysis and exploring themes within the text. This approach has revealed potential insights and indicates the value of delving deeper into the analysis. Further investigation into the sentiment scores of specific themes and their impact on overall sentiment could provide valuable insights into the aspects of the game that resonate with reviewers. Despite the challenges, our findings suggest that there is merit in continuing to explore and refine our analysis methods to gain a deeper understanding of the factors contributing to positive reviews.

# Next Steps

I'd like to check the sentiment scores for each of our themes. So I need code that looks at the sentiment scores of the sentences of each review, determines whether or not the sentence is referring to a particular one of our 4 themes, and then adds that score to the proper theme column. For each review. Maybe check to see how my pre-selected themes did in terms of meaningful score using the LDA.

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

