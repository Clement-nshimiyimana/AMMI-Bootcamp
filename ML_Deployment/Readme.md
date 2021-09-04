This is an example of machine learning model deployement. The used packages are flask and flasgger which enable us to get user interface without using html and css.

Notice: For the interface to be well displayed, we have to add "apidocs" on Url(http://127.2.1.8000/apidocs/)
The interface allows user to input float values and also to upload file comprising different values, for the model to predict based on those inputs. you can use input.csv which is in Dataset as a file to be tested on.

The dataset used is Iris and the model is logistic regression with 5 split crossvalidation.

Run

cd ML_Deployment

python Deploy.py or ipython Deploy.py


Installation

pip install flask

pip install flasgger

pip install pickle