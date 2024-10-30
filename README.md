# 🎥 Movie Review Sentiment Analysis

This project focuses on performing **binary sentiment analysis** on movie reviews using the Kaggle 50k movie review dataset. The primary goal is to determine whether a given review is **positive** or **negative**. I plan to use BERT, a model designed to capture word context effectively, making it ideal for sentiment analysis.

## 🧠 Models Used

- **DistilBERT**: Lightweight version of BERT, providing fast training and good performance. Likely to be the primary model due to efficiency.
- **BERT**, **RoBERTa** and **BERT-Large**: Potential models for achieving higher accuracy if computational resources and time permit.

## 📝 Project Steps

1. **Data Preparation**: Load the dataset, clean the text, tokenize, and prepare inputs for the model.
2. **Model Training**: Fine-tune DistilBERT (and potentially other BERTs) on the movie review dataset to classify reviews as positive or negative.
3. **Evaluation**: Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.
4. **Optional Deployment**: Deploy the model via a small web interface to predict sentiment for new movie reviews.

## 🔮 Future Plans
- Possibly create a web interface for real-time sentiment prediction and visualizations.
- Add a feature to extract and display the most sentiment-rich sentence in each review to provide more context for the sentiment classification.
- Add highlighted sentiment words within the review text to help users understand the classification.

## 🗂️ Repository Structure

- **notebooks/**: Jupyter notebooks for data preprocessing, model training, and evaluation.
- **data/**: Dataset files for training and testing.
- **models/**: Saved trained models.
- **web/** (optional): Code for web deployment of the sentiment model.

## 📄 License & Dataset

This project is licensed under the MIT License.
The data is sourced from the Kaggle 50k Movie Dataset.
