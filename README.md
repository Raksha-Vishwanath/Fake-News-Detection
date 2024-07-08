# Fake-News-Detection
This project uses the LIAR dataset which consists of 12.8k of Political data, to train 4 models - Random Forest, Logistic Regression, Decision Tree, and SVM. We start by preprocessing the data to remove any leading or trailing white spaces, removing any unwanted symbols, IP addresses, or links in data. Rows with missing data are computed and dropped. Preprocessing also involves converting the dataset into a binary classification for convenience. TF-IDF vectorization is applied to convert textual data into numerical data and this clean data is used for training on 4 models. The final ensemble model uses the three best performing models, i.e., Random Forest, Logistic Regression, and Decision Tree. A Hard Voting Ensemble is used in the goal to achieve higher performance. We also apply Correlation to data being passed to the final model, to ensure the model uses the appropriate number of attributes to make the prediction.

Form UI: The User Fills in the form with appropriate details and clicks on the classify button
![image](https://github.com/Raksha-Vishwanath/Fake-News-Detection/assets/111189940/2938554c-2269-484a-9b47-7c698af8ccfa)
![image](https://github.com/Raksha-Vishwanath/Fake-News-Detection/assets/111189940/25e5a9b9-dc99-403a-9eac-833ee6c7b0db)


The Model then classifies it to be Real 
![image](https://github.com/Raksha-Vishwanath/Fake-News-Detection/assets/111189940/215b34c8-ddf8-4eb0-8fa6-c7ddac685669)

Or The Model then classifies it to be Fake
![image](https://github.com/Raksha-Vishwanath/Fake-News-Detection/assets/111189940/4e7cae27-0e99-4f79-a77c-e7a74dea2760)
