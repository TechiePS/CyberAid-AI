# **CyberAid AI** 🌐🛡️

## **Multilingual Complaint Classification and Guidance System**

----------------------------------------------------------------------------------------------------------------------------


### **Overview**
CyberAid AI is a robust AI-powered system designed to classify cybercrime complaints into predefined categories and subcategories while offering tailored advice to users. The system supports multilingual input, allowing complaints to be processed in regional languages. Built with advanced NLP techniques and FastText embeddings, this solution ensures accurate classification and actionable guidance for users.

----------------------------------------------------------------------------------------------------------------------------

### 🚀 **Features**
- **Multilingual Complaint Processing**:
  - Detects the input language and translates complaints into English for analysis.
  - Supports regional languages, enhancing accessibility for users.
- **Real-Time Classification**:
  - Automatically classifies complaints into relevant **categories** and **subcategories**.
- **Actionable Advice Generation**:
  - Provides guidance to users based on complaint classifications.
- **Interactive Web Application**:
  - Built with Flask, users can submit complaints and view classifications and advice instantly.

----------------------------------------------------------------------------------------------------------------------------

### 🛠️ **Technologies Used**
- **Python**: Primary programming language for development.
- **TensorFlow & Keras**: Used for building and training the FFNN model.
- **Gensim**: For FastText embeddings to capture semantic meaning in complaints.
- **Flask**: For building the web application.
- **NLTK**: For text preprocessing, including tokenization, stopword removal, and lemmatization.
- **Google Translate API**: To handle multilingual input and output.
- **Pickle**: For saving and loading the trained models and label encoders.

----------------------------------------------------------------------------------------------------------------------------

### 📊 **Dataset**
The dataset used consists of cybercrime complaints and includes the following key fields:
- **CrimeAdditionalInfo**: Text description of the complaint.
- **Category**: High-level classification of the complaint (e.g., Financial Fraud, Hacking).
- **Subcategory**: Specific classification within the category (e.g., Phishing, Malware Attack).

The text data has been preprocessed by:
- Lowercasing.
- Removing special characters and numbers.
- Lemmatization and stopword removal.
- Converting complaints into FastText embeddings.

----------------------------------------------------------------------------------------------------------------------------

### ⚙️ **How to Run the Project**

1. **Clone the Repository**:
 
   git clone https://github.com/yourusername/cyberaid-ai.git
   cd cyberaid-ai

2. **Install Required Libraries**:

    pip install -r requirements.txt

3. **Prepare Data**:

    Ensure the training and testing datasets are available in the project directory.
    Run the preprocessing script to clean the data and train the FastText embeddings.

4.**Run the Flask App**:

    python app.py
    The web application will be accessible at http://127.0.0.1:5000/.

5.**Using the Web Interface**:

    Submit your complaint text (in any supported language).
    The app will classify the complaint into categories and subcategories and provide actionable advice.


----------------------------------------------------------------------------------------------------------------------------


### 🧑‍💻 **Model Details**

    Architecture:
        Input: FastText embeddings of complaint text.
        Hidden Layers: Dense layers with ReLU activation, BatchNormalization, and Dropout for regularization.
        Outputs:
        Category Classification: Softmax activation for multi-class classification.
        Subcategory Classification: Softmax activation for detailed classification.

    Training:
        Loss: S parse Categorical Crossentropy for both outputs.
        Optimizer: Adam.
        Evaluation Metrics: Accuracy, Precision, Recall, F1-Score.


----------------------------------------------------------------------------------------------------------------------------


### ⚡ **Example Workflow**
    Input:
    
        Complaint: "My account was hacked, and money was stolen."
        Language: Hindi

    Output:

        Predicted Category: Online Financial Fraud
        Predicted Subcategory: Phishing
        Actionable Advice: "Report the incident to your bank and law enforcement. Change your passwords and enable two-factor authentication."


----------------------------------------------------------------------------------------------------------------------------


## Screenshots
    Here are some screenshots of the application:

    #### Home Page 
    ![Home Page ](E:\hacki\CyberAid AI\output\Prediction Result - Brave 21_Nov_2024 21_07_53.png)
    *This is the main interface of the application, where users can navigate to various features.*

    #### Result Page
    ![Result Page ](E:\hacki\CyberAid AI\output\Prediction Result - Brave 21_Nov_2024 21_07_46.png)
    *This page displays the prediction results based on user input.*

    #### Home Page (Regional language)
    ![Home Page(complain in hindi) ](E:\hacki\CyberAid AI\output\Prediction Result - Brave 21_Nov_2024 21_08_21.png)
    *This version of the home page allows users to submit complaints in Hindi, making it accessible for Hindi-speaking users.*

    #### Result Page (Regional language)
    ![Result Page ](E:\hacki\CyberAid AI\output\Prediction Result - Brave 21_Nov_2024 21_08_31.png)
    *This page shows the results for complaints made in Hindi, ensuring that users receive feedback in their preferred language.*


----------------------------------------------------------------------------------------------------------------------------


### 📁**File Structure**

    .
    ├── app.py                 # Main Flask application
    ├── model.py               # Model training and prediction logic
    ├── train.csv              # Training dataset
    ├── test.csv               # Testing dataset
    ├── requirements.txt       # List of required libraries
    ├── ffnn_model.h5          # Saved FFNN model
    ├── category_encoder.pkl   # Label encoder for categories
    ├── subcategory_encoder.pkl # Label encoder for subcategories
    └── templates
        ├── index.html         # Web app homepage template
        └── result.html        # Web app result page template


----------------------------------------------------------------------------------------------------------------------------

### ⚠️ **Troubleshooting**

     Dataset Not Found:
        Ensure the train.csv and test.csv files are in the correct directory.
    Google Translate API Issues:
        Verify API key and internet connectivity.
    Long Processing Times:
        Reduce the model complexity or limit dataset size during testing.

----------------------------------------------------------------------------------------------------------------------------


### 🎯 **Future Improvements**

    Advanced Models:
        Use transformer-based models like BERT for improved accuracy.
    Multimodal Input:
        Enable processing of evidence in images or videos.
    Mobile App:
        Extend the functionality to a mobile application for on-the-go complaint submission.


----------------------------------------------------------------------------------------------------------------------------


### 📄 **License**
    This project is licensed under the MIT License. See the LICENSE file for more details.

----------------------------------------------------------------------------------------------------------------------------

### 🤝 **Contributing**
    Contributions are welcome! Feel free to fork this repository, propose changes, or submit pull requests.