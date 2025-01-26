# Music_classification
# Flask Web App for Musical Instrument Classification

This is a Flask-based web application that allows users to upload a `.wav` audio file, which is then processed to classify the musical instrument in the audio using a machine learning model. The app uses a model that classifies various musical instruments based on the audio features.

## Features
- **Upload .wav files**: Users can upload `.wav` files containing musical instrument sounds.
- **Classification**: The app processes the audio and classifies the instrument using a pre-trained machine learning model.
- **Display Results**: The classified instrument name and a corresponding image are displayed.


### Prerequisites
Make sure you have Python 3.x installed on your system. You'll also need **pip** for installing dependencies.
Install all the the libraries in requirement.txt for execution.

### Execution Process for Musical Instrument Classification  Web App 

  1. Start the Flask App
  Launch the App Locally: Once you’ve set up everything (dependencies, environment), you’ll start the Flask server using the command python app.py. This runs the app locally.
  2. User Interaction
  Upload .wav File: On the front-end of your web page, users can upload a .wav file containing audio from a musical instrument. This is done using an upload button on the interface.
  3. File Processing
  Backend Processing: When the file is uploaded, the Flask app receives the file and passes it to the backend where the machine learning model processes the audio.

Feature Extraction: The model extracts features from the .wav file such as frequency and time-domain information to classify the audio correctly.

  4. Model Classification
  Classify the Instrument: The extracted features are passed through a pre-trained machine learning model that classifies the audio into one of the musical instruments it has been trained to recognize.

Instrument Prediction: The model outputs the predicted instrument name based on the audio.

  5. Displaying the Results
  Show Instrument Name: The app displays the name of the classified instrument to the user.

Instrument Image: Along with the instrument name, a corresponding image of the musical instrument is displayed for the user to visualize the classification.

  6. Completion
  User Feedback: The user can upload another file if they wish to classify another musical instrument, or they can exit the web page.
  
  Logs/Debugging (optional): If there are any issues, the server logs or error messages will help debug.


### HIGHLIGHTS:
  1. Random Forest
  2. Supoort Vector Machine
  3. Voting classifier
  4. time-frequency domain audio feature extraction

### INSTRUMENTS:
  1. Cello
  2. Clarinet
  3. Hihat
  4. Flute
  5. Violin
  6. Doublebass
  7. BassDrum
  8. Tar
  9. Santur
  10. Setar
  11. Kamancheh
  12. Ney
  13. Saxophone




