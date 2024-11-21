from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import FastText
import pickle
from langdetect import detect  # To detect the language of user input
from googletrans import Translator  # To translate text to regional language
import json

# Initialize NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Load the trained neural network model (FFNN)
model = load_model(r'E:\hacki\CyberAid AI\models\ffnn_model.h5')

# Load encoders
with open(r'E:\hacki\CyberAid AI\models\category_encoder.pkl', 'rb') as f:
    category_encoder = pickle.load(f)

with open(r'E:\hacki\CyberAid AI\models\subcategory_encoder.pkl', 'rb') as f:
    subcategory_encoder = pickle.load(f)

# Load the FastText model (pretrained)
fasttext_model = FastText.load(r'E:\hacki\CyberAid AI\models\fasttext_model.bin')

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function for user input
def preprocess_input(text):
    text = text.lower()
    text = re.sub(r'/W', ' ', text)  # Fix regex typo
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    processed_text = ' '.join(tokens)
    return processed_text

# Function to generate FastText embeddings from user input
def get_fasttext_embeddings(text):
    tokens = text.split()
    embeddings = [fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(fasttext_model.vector_size)

# Function to generate advice based on category and subcategory
def get_fasttext_embeddings(text):
    tokens = text.split()
    embeddings = [fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(fasttext_model.vector_size)

# Function to generate advice based on category and subcategory
def generate_advice(category, subcategory):
    advice = ""
    
    # **Any Other Cyber Crime: Other**
    if category == "Any Other Cyber Crime" and subcategory == "Other":
        advice = "Report the crime to your local cybercrime cell and provide any relevant evidence or communication. Ensure your devices are secure and that any unauthorized access is blocked."
    
    # **Child Pornography (CP) / Child Sexual Abuse Material (CSAM)**
    elif category == "Child Pornography CPChild Sexual Abuse Material CSAM":
        advice = "Immediately report the content to the authorities and online platforms. This is a serious criminal offense, and the authorities will handle it. Avoid sharing or forwarding such materials."
    
    # **Cryptocurrency Crime: Cryptocurrency Fraud**
    elif category == "Cryptocurrency Crime" and subcategory == "Cryptocurrency Fraud":
        advice = "If you suspect cryptocurrency fraud, report the incident to the exchange platform and the authorities. Immediately freeze any affected account and avoid making further transactions."

    # **Cyber Attack/Dependent Crimes**
    elif category == "Cyber Attack/ Dependent Crimes":
        if subcategory == "Data Breach/Theft":
            advice = "If your data has been breached, report it to your organization and local authorities. Change your passwords immediately and monitor your accounts for suspicious activity."
        elif subcategory == "Denial of Service (DoS)/Distributed Denial of Service (DDOS) attacks":
            advice = "Contact your ISP and hosting provider to mitigate the attack. They can implement countermeasures like traffic filtering and blocking the malicious source."
        elif subcategory == "Hacking/Defacement":
            advice = "Report the incident to your hosting provider and cybersecurity experts. Revert your system and website to a secure backup and ensure your security protocols are up to date."
        elif subcategory == "Malware Attack":
            advice = "Run antivirus software to remove malware. Revert to a clean backup if possible, and monitor your system for unusual activity."
        elif subcategory == "Ransomware Attack":
            advice = "Do not pay the ransom. Instead, report the attack to law enforcement and consult cybersecurity experts for recovery options. Ensure you have recent backups."
        elif subcategory == "SQL Injection":
            advice = "Immediately fix any vulnerabilities in your web application's database layer. Report the attack to your server provider and security team."
        elif subcategory == "Tampering with computer source documents":
            advice = "Report the incident to the authorities. Ensure proper access controls and security are implemented to prevent further tampering."
    
    # **Cyber Terrorism**
    elif category == "Cyber Terrorism" and subcategory == "Cyber Terrorism":
        advice = "Cyber terrorism is a serious crime. Contact national security agencies and the police immediately. Ensure that sensitive data and systems are secured to prevent further attacks."

    # **Hacking/Damage to Computer Systems, etc.**
    elif category == "Hacking Damage to computercomputer system etc":
        if subcategory == "Damage to computer computer systems etc":
            advice = "Report the damage to the relevant authorities and ensure your system is repaired or replaced. Implement strict security measures to prevent future attacks."
        elif subcategory == "Email Hacking":
            advice = "Change your email password immediately. Inform your contacts about the breach and enable two-factor authentication (2FA)."
        elif subcategory == "Tampering with computer source documents":
            advice = "Report the tampering to your local authorities and forensic experts to recover and secure your documents."
        elif subcategory == "Unauthorized AccessData Breach":
            advice = "Report unauthorized access to the relevant authorities. Ensure your accounts and sensitive data are secure by changing passwords and enabling 2FA."
        elif subcategory == "Website DefacementHacking":
            advice = "Contact your hosting provider immediately to restore your website and secure it against further attacks. Review access logs to trace the attacker."

    # **Online Cyber Trafficking**
    elif category == "Online Cyber Trafficking" and subcategory == "Online Trafficking":
        advice = "If you suspect online trafficking, immediately report it to the police and online platforms. Avoid sharing or distributing the materials involved."

    # **Online Financial Fraud**
    elif category == "Online Financial Fraud":
        if subcategory == "Business Email CompromiseEmail Takeover":
            advice = "Report the incident to your email provider and your financial institution. Change all passwords and monitor your accounts closely."
        elif subcategory == "DebitCredit Card FraudSim Swap Fraud":
            advice = "Immediately contact your bank and credit card provider to block your cards and prevent further fraudulent transactions."
        elif subcategory == "DematDepository Fraud":
            advice = "Report fraudulent transactions to your depository service provider and the authorities. Block your demat account and file a police complaint."
        elif subcategory == "EWallet Related Fraud":
            advice = "Contact the e-wallet provider immediately to block the account and file a complaint with the authorities."
        elif subcategory == "Fraud CallVishing":
            advice = "Report the incident to your telecom provider and the authorities. Never share your personal or financial details over the phone."
        elif subcategory == "Internet Banking Related Fraud":
            advice = "Contact your bank immediately to report the fraud. Freeze your account and monitor for suspicious transactions."
        elif subcategory == "UPI Related Frauds":
            advice = "Report the fraud to your bank and the cybercrime department. Ensure that you never share your UPI credentials or PIN with anyone."
    
    # **Online Gambling & Betting**
    elif category == "Online Gambling Betting" and subcategory == "Online Gambling Betting":
        advice = "Report illegal gambling activities to local authorities. Use official channels to ensure safe and legal betting activities."

    # **Online and Social Media Related Crime**
    elif category == "Online and Social Media Related Crime":
        if subcategory == "Cheating by Impersonation":
            advice = "Report the incident to the social media platform and block the offender. If the impersonation is damaging, contact the authorities."
        elif subcategory == "Cyber Bullying Stalking Sexting":
            advice = "Report bullying and harassment to the platform and authorities. Document all evidence and block the offenders."
        elif subcategory == "EMail Phishing":
            advice = "Never click on suspicious links in emails. Report phishing emails to the provider and authorities. Change your email passwords."
        elif subcategory == "FakeImpersonating Profile":
            advice = "Report the fake profile to the platform and block the individual. Protect your online privacy by updating your security settings."
        elif subcategory == "Impersonating Email":
            advice = "Report the fake email to your email provider. Be cautious about any suspicious requests and verify their authenticity before responding."
        elif subcategory == "Intimidating Email":
            advice = "Report threatening emails to the authorities immediately. Document all communications and secure your email account."
        elif subcategory == "Online Job Fraud":
            advice = "Report fraudulent job offers to the platform and the authorities. Never share personal information or money upfront."
        elif subcategory == "Online Matrimonial Fraud":
            advice = "Be cautious while meeting someone online. Report fraudulent matrimonial profiles to the platform and authorities."
        elif subcategory == "Profile Hacking Identity Theft":
            advice = "Report the theft to the platform and authorities. Change your passwords and monitor your accounts for any unauthorized activity."
        elif subcategory == "Provocative Speech for unlawful acts":
            advice = "Report any provocative speech or threats to the authorities and the platform. Block the user and ensure your online safety."

    # **Ransomware**
    elif category == "Ransomware" and subcategory == "Ransomware":
        advice = "Never pay the ransom. Report the incident to law enforcement and cybersecurity professionals. If you have backups, restore them."
    
    # **Rape/Gang Rape (RGR) / Sexually Abusive Content**
    elif category == "RapeGang Rape RGRSexually Abusive Content":
        advice = "Report the crime to the authorities immediately. Secure any evidence and avoid deleting any related files. Seek counseling and legal support."
    
    # **Report Unlawful Content: Against Interest of Sovereignty or Integrity of India**
    elif category == "Report Unlawful Content" and subcategory == "Against Interest of sovereignty or integrity of India":
        advice = "Report any unlawful content that threatens national security to the Cyber Crime Cell and ensure your data and devices are secure."

    # **Sexually Explicit Act**
    elif category == "Sexually Explicit Act" and subcategory == "Sexually Explicit Act":
        advice = "If you encounter explicit content, report it to the authorities immediately. Avoid sharing or distributing such content."

    # **Sexually Obscene Material**
    elif category == "Sexually Obscene material" and subcategory == "Sexually Obscene material":
        advice = "Report obscene material to the platform and authorities. Ensure that your device is secure and that such content is not shared further."

    return advice
# Route for the home page with form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['complaint']
        
        # Detect the language of the input
        input_language = detect(user_input)
        
        # Preprocess the input complaint text
        processed_input = preprocess_input(user_input)
        
        # Generate FastText embeddings for the processed input
        embedding = get_fasttext_embeddings(processed_input).reshape(1, -1)
        
        # Predict category and subcategory using the trained FNN model
        predictions = model.predict(embedding)
        
        # Get predicted category and subcategory
        category_prediction_idx = np.argmax(predictions[0], axis=1)[0]
        subcategory_prediction_idx = np.argmax(predictions[1], axis=1)[0]
        
        # Convert numeric predictions to actual labels using the encoders
        category_result = category_encoder.inverse_transform([category_prediction_idx])[0]
        subcategory_result = subcategory_encoder.inverse_transform([subcategory_prediction_idx])[0]
        
        # Get probabilities for category and subcategory
        category_probabilities = predictions[0][0]
        subcategory_probabilities = predictions[1][0]
        
        # Get advice based on the category and subcategory
        advice = generate_advice(category_result, subcategory_result)
        
        # Initialize the translator for language translation
        translator = Translator()
        
        # Check if advice is empty and handle accordingly
        if not advice:
            advice = "No advice available."
        
        # Translate outputs to the detected language (if not English)
        try:
            if input_language != 'en':
                translated_category = translator.translate(category_result, src='en', dest=input_language).text
                translated_subcategory = translator.translate(subcategory_result, src='en', dest=input_language).text
                translated_advice = translator.translate(advice, src='en', dest=input_language).text
            else:
                translated_category = category_result
                translated_subcategory = subcategory_result
                translated_advice = advice
        except Exception as e:
            translated_advice = f"Translation failed: {str(e)}"
            print(f"Error during translation: {e}")
        
        # Convert probabilities to percentage
        category_percentage = [round(prob * 100, 2) for prob in category_probabilities]
        subcategory_percentage = [round(prob * 100, 2) for prob in subcategory_probabilities]
        
        # Return the results in the HTML response
        return render_template('result.html', 
                               category=category_result, 
                               subcategory=subcategory_result, 
                               advice=advice, 
                               translated_category=translated_category,
                               translated_subcategory=translated_subcategory,
                               translated_advice=translated_advice,
                               category_percentage=json.dumps(category_percentage),
                               subcategory_percentage=json.dumps(subcategory_percentage))

if __name__ == '__main__':
    app.run(debug=True)  # Start Flask app locally
