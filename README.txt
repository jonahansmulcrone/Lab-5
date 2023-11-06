(1). Learn the reference JAVA version of decision tree
Dr. Cao has provided a working java version of decision tree: Lab4_DTTrain.java and Lab4_DTPredict.java. Please download the training data TrainingData.txt and then compile the Lab4_DTTrain.java (javac Lab4_DTTrain.java), and then run it (java Lab4_DTTrain TrainingData.txt DT_model.model). You should be able to generate a decision tree model DT_model.model. After that, you can make predictions on the test dataset TestDataNoLabel.txt, by compiling Lab4_DTPredict.java (javac Lab4_DTPredict.java) and run it (java Lab4_DTPredict TestDataNoLabel.txt DT_model.model DT_prediction.txt). You should be able to find the prediction file DT_prediction.txt for your testing dataset.

(2). Implement Lab4.py (14 points)
There are two main functions (DTtrain and DTpredict - 7 points each), but you may need to add Tree node and other helper functions. Please refer to the JAVA code provided by Dr. Cao, and you are going to translate the JAVA to Python, but you will be in charge of the design for this program and you don't need to follow exactly the JAVA code. Run the Lab4_grading.py to check the grade of your program (The grading code didn't check the accuracy of your decision tree model).
In order to get full points, each group member must make contributions. You could upload a screenshot from github (check insights -> contributors)

(3). Discord comment (2 points)
Introduce the dataset for your project in discord and add comments on this lab! Each group member needs to post.

(4). Performance comparison (4 points)
We used weka's weka.classifiers.trees.J48 machine learning model to train and predict values in our dataset.

To train the model using weka
java -classpath .:/Users/stuartg/CS330/weka-3-8-6/weka.jar weka.classifiers.trees.J48 -t ARFFTrainingData.arff -c 1 -d weka_DT_model.model

To make predictions using the model
java -classpath .:/Users/stuartg/CS330/weka-3-8-6/weka.jar weka.classifiers.trees.J48 -T ARFFTestData.arff -l weka_DT_model.model -c 1 -p 1 > weka_predictions.txt

(5). Bonus points (up to 2 points)
Each group could earn bonus points for doing additional work. For example, different entropy for decision and add an option to choose it.