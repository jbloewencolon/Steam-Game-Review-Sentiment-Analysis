# The Road to Hades is Paved with Positive Reviews: 
## Creating a Better Game-Review Sentiment Analyzer

![machine_sentiment.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/machine%20sentiment.png)
    
By Jordan Loewen-Colón July 11th 2023

# The "Business Problem"

SuperGiant Games faces a challenge in understanding the specific aspects of their games that players enjoy. The primary hurdle is the lack of detailed information available in the reviews from Steam, the largest aggregator of game reviews. While Steam provides a binary recommendation status, it does not offer insights into the underlying reasons behind players' preferences. To address this, SuperGiant Games has assigned us the task of developing a model that can analyze game reviews and provide a more comprehensive and nuanced understanding of what players appreciate about their games. By leveraging advanced techniques, we aim to uncover valuable insights beyond simple recommendations, allowing SuperGiant Games to better understand player preferences and further enhance their game development strategies.

# Recommendations:
We recommend SuperGiant Games continue to focus on their **storytelling**, as players consistently highlighted this aspect. Additionally, efforts can be made to enhance players' ability to express their positive impressions of the 'music' and 'visuals' by potentially providing in-game prompts, specific keywords or phrases, or specific questions related to these aspects in reviews or feedback forms. This would help gather more detailed and insightful feedback on the game's audio and visual components.

# Step 1: Data Understanding

To make our recommendations, we analyzed reviews from the video game Hades, found on Steam. We used a special jupyter model (steam_import) to pull the data using the Steam API. That code can be found [here](https://github.com/jbloewencolon/Game-Review-Sentiment-Analyzer/blob/main/Sandbox/API%20Import%20Steam%20and%20Preprocess%20Data.ipynb). Our initial data looked like this:

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

Since the vast majority of reviews were rated positive ('voted_up') we kept our interest primarily on 'reviews' and 'author.playtime_forever.' 

# Step 2: Data Preparation

To begin our data preparation, we dropped all non-English reviews, unnecessary columns, and NaNs. We then lemmatized and tokenized the text so that our models would have a smoother time gleaning information from the data. We created new columns for length of review (small, medium, large, extra large) to check on the spread of review length for our data set (which came out fairly equitable!) and a new column depicting low, average, or high playtime as our target variable. Finally, we created a pipeline to streamline our model production going forward and split the data into training and test sets.

# Step 3: Data Modeling

Our first model was a simple logistic regression. Starting with a logistic regression model offers interpretability and simplicity, serving as an efficient method to establish baseline performance for binary classification. We used it to verify that our model would overfit if we focused just on predicting whether a review was positive or not based on review content.

**Logistic Regression**
![f1_log_reg.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/f1%20for%20log%20reg.PNG)

As predicted, that low recall rate on the minority class, and perfect score on the majority class, does not end up telling us much about our data. So let's change tactics for our more complex models. Rather than trying to predict the positivity of a review based on its content, let's see if we can predict the length of a review by whether or not a player plays an above or below average amount. Because our data set is so large, we will only use a subset of the total data.

**Model: XGB**
![XGB_first.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/XGB%20first.PNG)

Not great! Our model isn't much better than a coinflip on the training data, and doing even worse on the test data. However, tuning the hyperparameters of our XGB model using GridSearchCV, did not actually results in better scores, so we will keep our model as is. Let's see if we can improve our model with some sentiment analysis:

**Model: TextBlob Sentiment Analysis**

We are going to use TextBlob's NaiveBayesAnalyzer (NBA) for our sentiment analysis. The NBA was trained on movie reviews, the closest we get to game reviews. To help it out, we are going to provide our model with 4 themes to look for in the data. We want to help our client figure out what it was exactly that people enjoyed about their games. Here are the themes:
![themes.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/themes.PNG)

We want to get sentiments on the general review level and on the sentence level. The more fine-grained the better! These functions take a review as input and calculate the sentiment scores for each sentence in the review and review at large using TextBlob's sentiment analysis. It returns a list of sentiment scores which we can then visualize:

![sentiment.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/general%20sentiment.PNG)

This histogram gives us more data than our logistic regression. We can see that rather than a simple binary of recommended or not, players had a range of sentiments concerning what they liked about the game. We then decided to create a little program that can pick a review at random and display its content, its polarity, and which words within the review are contributing to that polarity based on the themes we provided. Here is a random review sampling:

![sample_review.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/review%20sample.PNG)

With this review we can see that it was generally positive, and liked the voice acting (.55), writing (.47), and art (.59), with each receiving positive polarity. Our analyzer did not pick up that "movement" and "replayability" might be part of 'gameplay', but we can adjust that later. 

Unfortunately, adding the new sentiment data to our XGB model didn't improve its predictive capacity. Let's see if we can get any other addtional data that might help with refining for future analysis.

Now for some additional verification, we are going to run an unsupervised learning model to see if it covers similar topics. Specifically, we will use Gensim's Latent Dirichlet Allocation (LDA) model. We will prepare the reviews for LDA by removing the stopwords, lemmatizing them, and creating the dictionary and corpus needed for the topic modeling. When we have it show us the top 10 topics it found, we get this:

![sample_review.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/topics.PNG)

It's hard to get a clear theme from these. Lots of action words, so perhaps 'gameplay' is a good theme? Or perhaps it's too general. Let's check the top bigrams to see if they reveal anything else about the review topics:

![top_bigrams.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/top%20bigrams.PNG)

Some of these look helpful. We might categorize button_mashy, hack_slash, learning_curve, keyboard_mouse, and fishing_minigame as 'gameplay' topics, and greek_mythology as 'story.' Let's see if we get any more clarity by limiting our bigrams to our pre-selected themes:

![theme_bigrams.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/them%20bigrams.PNG)

That is definitely more useful! We we are able to see which of the words are associated with each them, and how often those pairs appeared. Now let's step back and see how often our themes appeared more generally. 

![theme_counts.png](https://github.com/jbloewencolon/Phase-4-Game-Sentiment-Analyzer/blob/main/Images/theme%20appearance%20counts.PNG)

# Step 4: Data Understanding and Conclusions

1) The reviews for the game Hades generally expressed positive sentiment, although the overall level of positivity falls within the range of 0 to 0.25.

2) When discussing their experiences with the game, players frequently emphasized the importance of the game's story. This indicates that the narrative elements of Hades are a significant aspect of player enjoyment.

3) It appears that players may have limited vocabulary when describing their appreciation for the 'music' and 'visuals' in Hades. This suggests that while players find these aspects appealing, they may struggle to articulate their specific likes or preferences regarding the music and visual elements of the game.
   
Given the computational limitations, making confident predictions about the specific aspects of the game that received positive reviews remains challenging. However, we were successful in adding complexity to the analysis of reviews by incorporating sentiment analysis and exploring themes within the text. This approach has revealed potential insights and indicates the value of delving deeper into the analysis. Further investigation into the sentiment scores of specific themes and their impact on overall sentiment could provide valuable insights into the aspects of the game that resonate with reviewers. Despite the challenges, our findings suggest that there is merit in continuing to explore and refine our analysis methods to gain a deeper understanding of the factors contributing to positive reviews.

# Recomendations

Based on these findings, I would recommend SuperGiant Games to continue focusing on the strong storytelling elements of Hades, as players consistently highlighted this aspect. Additionally, efforts can be made to enhance players' ability to express their positive impressions of the 'music' and 'visuals' by potentially providing prompts or specific questions related to these aspects in reviews or feedback forms. This would help gather more detailed and insightful feedback on the game's audio and visual components.

# Next Steps

I'd like to check the sentiment scores for each of our themes. So I need code that looks at the sentiment scores of the sentences of each review, determines whether or not the sentence is referring to a particular one of our 4 themes, and then adds that score to the proper theme column. For each review. Maybe check to see how my pre-selected themes did in terms of meaningful score using the LDA.

# Questions?
For a full analysis please check the Jupyter Notebook or slide presentation.
Further questions? Contact Jordan Loewen-Colón @ jbloewen@syr.edu


## Repository Structure


├── [data](https://github.com/jbloewencolon/Game-Review-Sentiment-Analyzer/tree/main/Data) : data used for modeling
├── [images](https://github.com/jbloewencolon/Game-Review-Sentiment-Analyzer/tree/main/Images) : images used in PPT and README
├── [Sandbox](https://github.com/jbloewencolon/Game-Review-Sentiment-Analyzer/tree/main/Sandbox) : previous files from earlier drafts of project
├── [game-review-sentiment-analysis.ipynb](https://github.com/jbloewencolon/Game-Review-Sentiment-Analyzer/blob/main/game-review-sentiment-analysis.ipynb) : notebook used to pull from API
├── [README.md](https://github.com/jbloewencolon/Game-Review-Sentiment-Analyzer/blob/main/README.md) : project information and repository structure
├── [presentation.pdf](https://github.com/jbloewencolon/Game-Review-Sentiment-Analyzer/blob/main/presentation.pdf) : the PowerPoint presentation used to present data analysis


