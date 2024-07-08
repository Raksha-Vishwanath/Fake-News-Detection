from flask import Flask, render_template, request
import pickle
import pandas as pd
from text_combiner import TextCombiner

app = Flask(__name__)

# Load serialized preprocessor and model
with open('preprocess.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('to_drop.pkl', 'rb') as to_drop_file:
    to_drop = pickle.load(to_drop_file)

# Load the trained model
with open('mlmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        # Get input data from form
        statement = request.form['statement']
        subject = request.form['subject']
        speaker = request.form['speaker']
        job_title = request.form['job_title']
        state_info = request.form['state_info']
        party_affiliation = request.form['party_affiliation']
        context = request.form['context']
        real_counts = float(request.form['real_counts'])  
        fake_counts = float(request.form['fake_counts'])  

        # Create DataFrame from input data
        input_df = pd.DataFrame([[statement, subject, speaker, job_title, state_info, party_affiliation, context, real_counts, fake_counts]],
                                columns=['Statement', 'Subject', 'Speaker', 'Speaker Job Title', 'State Info', 'Party Affiliation', 'Context', 'Real Counts', 'Fake Counts'])
        
        # Preprocess input data
        input_data = preprocessor.transform(input_df)

        # Convert to DataFrame for clarity and drop correlated columns
        input_data_dense=input_data.toarray()
        input_data_reduced = pd.DataFrame(input_data_dense)
        input_data_reduced = input_data_reduced.drop(columns=to_drop)
        input_data_reduced = input_data_reduced.values
        #input_data = pd.DataFrame(input_data, columns=preprocessor.transformers_[0][1]['vectorizer'].get_feature_names_out() + ['Real Counts', 'Fake Counts'])
        #input_data = input_data.drop(columns=to_drop)  

        # Use model to predict
        prediction = model.predict(input_data_reduced)

        # Return prediction
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
