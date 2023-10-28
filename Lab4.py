"""
 Name: Jonah, Cole, Stuart
 Assignment: Lab 4 - Decision Tree
 Course: CS 330
 Semester: Fall 2023
 Instructor: Dr. Cao
 Date: the current date
 Sources consulted: any books, individuals, etc consulted

 Known Bugs: description of known bugs and other program imperfections

 Creativity: anything extra that you added to the lab

 Instructions: After a lot of practice in Python, in this lab, you are going to design the program for decision tree and implement it from scrath! Don't be panic, you still have some reference, actually you are going to translate the JAVA code to Python! The format should be similar to Lab 2!

"""
import sys
import argparse
import math
import os
import re

# You may need to define the Tree node and add extra helper functions here

def DTtrain(infile, model, percent):
    """
    This is the function for training a decision tree model
    """
    # implement your code here
    
    """
    the first part of this method handles reading the trainingdata.txt file 
    and creating the datamap, attvalues, and arr variables which are important for training the model
    """

    try: 
        """initialize map for storing data"""
        datamap = {}

        """open the trainingdata.txt file"""
        with open(infile, 'r') as file:
            line = file.readline()
            attline = line[1:]

            atts = attline.split('|')
            atts = [att.strip() for att in atts]

            numatts = len(atts)-1

            """Initialize a map of atttribute values"""
            attvalues = {}
            for att in atts:
                attvalues[att] = []

            """Read data into map"""

            index = 0

            for line in file:
                
                cleaned_line = []
                current_num = ''

                """Iterate through the elements in the line"""
                for element in line:
                    
                    if element.isdigit():
                        current_num += element
                        
                    elif current_num:

                        cleaned_line.append(current_num)
                        current_num = ''

                if current_num:
                    cleaned_line.append(current_num)

                dataclass = cleaned_line[0]

                arr = attvalues[atts[0]]

                if dataclass not in arr:
                    arr.append(dataclass)
                
                if dataclass not in datamap:
                    datamap[dataclass] = []

                a = datamap.get(dataclass)
                datapoint = []
        
                for i in range(1, numatts + 1):
                    datapoint.append(cleaned_line[i])
                    arr = attvalues.get(atts[i])

                    if cleaned_line[i] not in arr:
                        arr.append(cleaned_line[i])
                
                if index % 100 < percent:
                    a.append(datapoint)
            
            numclasses = len(datamap.keys())
            
    except IOError as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    """
    end of reading file
    """
        

def DTpredict(data, model, prediction):
    """
    This is the main function to make predictions on the test dataset. It will load saved model file,
    and also load testing data TestDataNoLabel.txt, and apply the trained model to make predictions.
    You should save your predictions in prediction file, each line would be a label, such as:
    1
    0
    0
    1
    ...
    """
    # implement your code here


    pass


def EvaDT(predictionLabel, realLabel, output):
    """
    This is the main function. You should compare line by line,
     and calculate how many predictions are correct, how many predictions are not correct. The output could be:

    In total, there are ??? predictions. ??? are correct, and ??? are not correct.

    """
    correct,incorrect, length = 0,0,0
    with open(predictionLabel,'r') as file1, open(realLabel, 'r') as file2:
        pred = [line for line in file1]
        real = [line for line in file2]
        length = len(pred)
        for i in range(length):
            if pred.pop(0) == real.pop(0):
                correct += 1
            else:
                incorrect += 1
    Rate = correct/length

    result = "In total, there are "+str(length)+" predictions. "+str(correct)+" are correct and "+ str(incorrect) + " are incorrect. The percentage is "+str(Rate)
    with open(output, "w") as fh:
        fh.write(result)

def main():
    options = parser.parse_args()
    mode = options.mode       # first get the mode
    print("mode is " + mode)
    if mode == "T":
        """
        The training mode
        """
        inputFile = options.input
        outModel = options.output
        if inputFile == '' or outModel == '':
            showHelper()
        DTtrain(inputFile, outModel, 100)
    elif mode == "P":
        """
        The prediction mode
        """
        inputFile = options.input
        modelPath = options.modelPath
        outPrediction = options.output
        if inputFile == '' or modelPath == '' or outPrediction == '':
            showHelper()
        DTpredict(inputFile, modelPath, outPrediction)
    elif mode == "E":
        """
        The evaluating mode
        """
        predictionLabel = options.input
        trueLabel = options.trueLabel
        outPerf = options.output
        if predictionLabel == '' or trueLabel == '' or outPerf == '':
            showHelper()
        EvaNB(predictionLabel,trueLabel, outPerf)
    pass

def showHelper():
    parser.print_help(sys.stderr)
    print("Please provide input augument. Here are examples:")
    print("python " + sys.argv[0] + " --mode T --input TrainingData.txt --output DTModel.txt")
    print("python " + sys.argv[0] + " --mode P --input TestDataNoLabel.txt --modelPath DTModel.txt --output TestDataLabelPrediction.txt")
    print("python " + sys.argv[0] + " --mode E --input TestDataLabelPrediction.txt --trueLabel LabelForTest.txt --output Performance.txt")
    sys.exit(0)


if __name__ == "__main__":
    #------------------------arguments------------------------------#
    #Shows help to the users                                        #
    #---------------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser._optionals.title = "Arguments"
    parser.add_argument('--mode', dest='mode',
    default = '',    # default empty!
    help = 'Mode: T for training, and P for making predictions, and E for evaluating the machine learning model')
    parser.add_argument('--input', dest='input',
    default = '',    # default empty!
    help = 'The input file. For T mode, this is the training data, for P mode, this is the test data without label, for E mode, this is the predicted labels')
    parser.add_argument('--output', dest='output',
    default = '',    # default empty!
    help = 'The output file. For T mode, this is the model path, for P mode, this is the prediction result, for E mode, this is the final result of evaluation')
    parser.add_argument('--modelPath', dest='modelPath',
    default = '',    # default empty!
    help = 'The path of the machine learning model ')
    parser.add_argument('--trueLabel', dest='trueLabel',
    default = '',    # default empty!
    help = 'The path of the correct label ')
    if len(sys.argv)<3:
        showHelper()
    main()