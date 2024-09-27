Sentiment Analysis on Research Papers

Objective: The project aims to analyze the sentiment of research papers using machine learning techniques. The primary goal is to classify the sentiment of the text into categories such as positive, negative, or neutral, which can help in understanding the overall sentiment trends in academic literature.
Project Components

    Data Collection:
        Dataset: Research papers from various sources (e.g., online repositories, journals).
        Text Data: The text data is extracted from the research papers for sentiment analysis.

    Preprocessing:
        Text Cleaning: Removing irrelevant information, punctuation, and special characters from the text.
        Tokenization: Splitting the text into individual words or phrases.
        Stopword Removal: Filtering out common words (e.g., 'the', 'is', 'in') that do not contribute to sentiment analysis.
        Stemming/Lemmatization: Reducing words to their base or root forms.

    Feature Extraction:
        TF-IDF Vectorization: Transforming the text data into numerical format using Term Frequency-Inverse Document Frequency (TF-IDF) representation, which captures the importance of words in the context of the dataset.
        N-grams: Using combinations of words (bigrams, trigrams) to capture contextual information in the text.

    Model Development:
        Machine Learning Models:
            Naive Bayes Classifier: A probabilistic model based on Bayes' theorem used for text classification.
            Support Vector Machine (SVM): A supervised learning model that aims to find the hyperplane that best separates the classes in the feature space.

    Model Evaluation:
        Train-Test Split: Dividing the dataset into training and testing sets to evaluate model performance.
        Metrics: Using accuracy, precision, recall, and F1-score to assess the effectiveness of the models.

    Results:
        Presenting the performance of the models based on the evaluation metrics.
        Comparing the results of Naive Bayes and SVM models.

Files and Structure

    Jupyter Notebooks:
        naive_bayes.ipynb: Contains the implementation of the Naive Bayes model for sentiment analysis.
        tidf_with_ngrams_svm.ipynb: Contains the implementation of the SVM model using TF-IDF and N-grams.

    Model Files:
        naive_bayes_model.pkl: Serialized model file for the Naive Bayes classifier.
        svm_model.pkl: Serialized model file for the SVM classifier.
        tidf_vectorizer.pkl: Serialized TF-IDF vectorizer used for transforming text data.

    Text Files:
        polar_phrases.txt: Contains polar phrases used to help classify sentiment.

Conclusion

This project provides a framework for sentiment analysis of research papers, demonstrating the application of machine learning techniques in natural language processing. The results can aid researchers and academics in understanding sentiment trends within scientific literature, potentially leading to further insights into the impacts and implications of research findings.
