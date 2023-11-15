"""
 Name: Jonah, Cole, Stuart
 Assignment: Lab 4 - Decision Tree
 Course: CS 330
 Semester: Fall 2023
 Instructor: Dr. Cao
 Date: the current date
 Sources consulted: any books, individuals, etc. consulted

 Known Bugs: description of known bugs and other program imperfections

 Creativity: We added another entropy calculation called Gini impurity. To train a 
 model with this entropy calculation, you can use the argument --entropy with a value of option2

 Instructions: After a lot of practice in Python, in this lab, you are going to design the program for decision tree
 and implement it from scratch! Don't be panic, you still have some reference, actually you are going to translate the
 JAVA code to Python! The format should be similar to Lab 2!

"""
import sys
import argparse
import math


class TreeNode:
    def __init__(self, p):
        self.parent = p
        self.children = {}
        self.attribute = "none"
        self.return_val = None

class DTTrain:
    def __init__(self):
        self.datamap = {}  # Stores all data read in
        self.attvalues = {}
        self.atts = None  # A list of attributes read in. atts[0] is the classifier
        self.numAtts = 0  # The number of attributes used to predict atts[0]
        self.numClasses = 0  # The total number of classes to predict between
        self.root = None
        self.entropy_method = "default"

    def read_file(self, infile, percent, entropy_option):
        """
            the first part of this method handles reading the trainingdata.txt file
            and creating the datamap, attvalues, and arr variables which are important for training the model
        """
        try:
            """initialize map for storing data"""
            self.datamap = {}
            self.entropy_method = entropy_option

            """open the trainingdata.txt file"""
            with open(infile, 'r') as file:
                line = file.readline()
                attline = line[1:]
                self.atts = attline.split('|')
                self.atts = [att.strip() for att in self.atts]

                self.numAtts = len(self.atts) - 1

                """Initialize a map of attribute values"""
                self.attvalues = {}
                for att in self.atts:
                    self.attvalues[att] = []

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

                    arr = self.attvalues[self.atts[0]]

                    if dataclass not in arr:
                        arr.append(dataclass)

                    if dataclass not in self.datamap:
                        self.datamap[dataclass] = []

                    a = self.datamap.get(dataclass)
                    datapoint = []

                    for i in range(1, self.numAtts + 1):
                        datapoint.append(cleaned_line[i])
                        arr = self.attvalues.get(self.atts[i])

                        if cleaned_line[i] not in arr:
                            arr.append(cleaned_line[i])

                    if index % 100 < percent:
                        a.append(datapoint)

                self.numClasses = len(self.datamap.keys())

        except IOError as e:
            print("Error reading file: " + e)
            sys.exit(1)

        """
        end of reading file
        """

    def build_tree(self):
        self.root = self.build_tree_node(None, list(self.atts[1:]), self.datamap)

    def build_tree_node(self, parent, curr_free_atts, node_data):
        curr = TreeNode(parent)
        min_ent = 1
        min_att = None

        for i in range(self.numAtts):
            att = curr_free_atts[i]

            if att is not None:
                vals = self.attvalues[att]
                partition = [[0] * self.numClasses for _ in range(len(vals))]

                for j in range(self.numClasses):
                    outcome = self.attvalues[self.atts[0]][j]
                    l = node_data[outcome]

                    for data_point in l:
                        partition[vals.index(data_point[i])][j] += 1

                ent = self.partition_entropy(partition)

                if ent < min_ent:
                    min_ent = ent
                    min_att = att

        if min_att is None:
            max_count = 0
            max_class = "undefined"

            for j in range(self.numClasses):
                outcome = self.attvalues[self.atts[0]][j]

                if len(node_data[outcome]) >= max_count:
                    max_count = len(node_data[outcome])
                    max_class = outcome

            curr.return_val = max_class
            return curr

        curr.attribute = min_att
        att_index = curr_free_atts.index(min_att)
        curr_free_atts[att_index] = None

        for v in self.attvalues[min_att]:
            temp_map = {}

            for j in range(self.numClasses):
                outcome = self.attvalues[self.atts[0]][j]
                trim_list = [l for l in node_data[outcome] if l[att_index] == v]
                temp_map[outcome] = trim_list

            print(v + "---> ", end='')
            curr.children[v] = self.build_tree_node(curr, curr_free_atts, temp_map)

        curr_free_atts[att_index] = min_att
        return curr

    def partition_entropy(self, partition):
        total_ent = 0
        total = 0

        entropy_function = self.entropy if self.entropy_method == 'default' else self.entropy_2

        for i in range(len(partition)):
            n = sum(partition[i])
            total += sum(partition[i])
            total_ent += n * entropy_function(partition[i])

        return total_ent / total

    @staticmethod
    def entropy(class_counts):
        total = sum(class_counts)
        entropy_sum = 0

        for i in class_counts:
            if i > 0:
                entropy_sum -= (i / total) * DTTrain.log2(i / total)

        return entropy_sum
    
    @staticmethod
    def entropy_2(class_counts):
        total = sum(class_counts)
        if total == 0: 
            return 0
        impurity_sum = 0

        for count in class_counts:
            prob_of_lbl = count / total
            impurity_sum += prob_of_lbl * (1 - prob_of_lbl)

        return impurity_sum

    @staticmethod
    def log2(x):
        if x == 0:
            return 0
        return math.log(x) / math.log(2)

    def save_model(self, model_file):
        try:
            with open(model_file, 'w') as outfile:
                for i in range(self.numAtts):
                    outfile.write(self.atts[i + 1] + " ")
                outfile.write("\n")
                self.write_node(outfile, self.root)
        except IOError as e:
            print("Error writing to file:", e)
            sys.exit(1)

    def write_node(self, outfile, curr):
        if curr.return_val is not None:
            outfile.write("[" + curr.return_val + "] ")
            return

        outfile.write(curr.attribute + " ( ")
        for key, value in curr.children.items():
            outfile.write(key + " ")
            self.write_node(outfile, value)

        outfile.write(" ) ")


def DTtrain(infile, model, percent, entropy_option):
    """
    This is the function for training a decision tree model
    """
    t = DTTrain()
    t.read_file(infile, percent, entropy_option)
    t.build_tree()
    t.save_model(model)


class TreeNode:
    def __init__(self, attribute, children, return_val):
        self.attribute = attribute
        self.children = children
        self.return_val = return_val


class DTPredict:
    def __init__(self):
        self.root = None
        self.att_arr = None
        self.predictions = []

    def read_model(self, infile):

        try:
            with open(infile, 'r') as file:
                atts = file.readline().split()

                self.att_arr = []

                for i in range(len(atts)):
                    self.att_arr.append(atts[i])

                self.root = self.read_node(file)

        except IOError as e:
            print('Error reading file: ' + e)
            sys.exit(1)

    def read_node(self, file):

        line = file.readline().strip()
        
        if not line:
            return None 

        tree = line.split()
        n = tree[0][0]

        if n == "[":
            return TreeNode(None, None, tree[1][:-1])

        node = TreeNode(n, {}, None)

        i = 1
        while i < len(tree):
            val = tree[i]

            print("this is a val", val)
            
            if val == ")":
                return node

            node.children[val] = self.read_node(file)
            i += 1

        return node

    def predict_from_model(self, testfile):

        try:
            self.predictions = []
            with open(testfile, 'r') as file:
                for line in file:

                    data = line.strip().split()

                    if data:
                        data.pop(0)
                        print(self.root)
                        pred = self.trace_tree(self.root, data)
                        self.predictions.append(pred)

        except IOError as e:
            print(f"Error reading test file: {e}")
            sys.exit(1)

    def trace_tree(self, node, data):

        if node.return_val is not None:
            return node.return_val
        
        att = node.attribute
        val = data[self.att_arr.index(att)]
        t = node.children.get(val)

        return self.trace_tree(t, data)


    def save_predictions(self, outputfile):

        try:
            with open(outputfile, 'w') as file:
                for prediction in self.predictions:
                    file.write(f"{prediction}\n")

        except IOError as e:
            print(f"Error writing to file: {e}")

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
    
    p = DTPredict()
    p.read_model(model)
    p.predict_from_model(data)
    p.save_predictions(prediction)


def EvaDT(predictionLabel, realLabel, output):
    """
    This is the main function. You should compare line by line,
     and calculate how many predictions are correct, how many predictions are not correct. The output could be:

    In total, there are ??? predictions. ??? are correct, and ??? are not correct.

    """
    correct, incorrect, length = 0, 0, 0
    with open(predictionLabel, 'r') as file1, open(realLabel, 'r') as file2:
        pred = [line for line in file1]
        real = [line for line in file2]
        length = len(pred)
        for i in range(length):
            if pred.pop(0) == real.pop(0):
                correct += 1
            else:
                incorrect += 1
    Rate = correct / length

    result = "In total, there are " + str(length) + " predictions. " + str(correct) + " are correct and " + str(
        incorrect) + " are incorrect. The percentage is " + str(Rate)
    with open(output, "w") as fh:
        fh.write(result)


def main():
    options = parser.parse_args()
    inputFile = "TrainingData.txt"
    outModel = "DTModel.txt"
<<<<<<< HEAD
<<<<<<< HEAD
    mode = "T"  # first get the mode
    entropy_option = options.entropy
=======
    mode = "P"  # first get the mode
>>>>>>> 632a53f (DTPredict Implementation)
=======
    mode = "T"  # first get the mode
    entropy_option = options.entropy
>>>>>>> 77d148e (Add Second Entropy Option for Bonus Points)
    print("mode is " + mode)
    if mode == "T":
        """
        The training mode
        """
        # inputFile = options.input
        # outModel = options.output
        if inputFile == '' or outModel == '':
            showHelper()
        DTtrain(inputFile, outModel, 100, entropy_option)
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
        EvaDT(predictionLabel, trueLabel, outPerf)
    pass


def showHelper():
    parser.print_help(sys.stderr)
    print("Please provide input argument. Here are examples:")
    print("python " + sys.argv[0] + " --mode T --input TrainingData.txt --output DTModel.txt")
    print("python " + sys.argv[0] + " --mode P --input TestDataNoLabel.txt --modelPath DTModel.txt --output TestDataLabelPrediction.txt")
    print("python " + sys.argv[0] + " --mode E --input TestDataLabelPrediction.txt --trueLabel LabelForTest.txt --output Performance.txt")
    sys.exit(0)


if __name__ == "__main__":
    # ------------------------arguments------------------------------#
    # Shows help to the users                                        #
    # ---------------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser._optionals.title = "Arguments"
    parser.add_argument('--mode', dest='mode',
                        default='',  # default empty!
                        help='Mode: T for training, and P for making predictions, and E for evaluating the machine '
                             'learning model')
    parser.add_argument('--input', dest='input',
                        default='',  # default empty!
                        help='The input file. For T mode, this is the training data, for P mode, this is the test '
                             'data without label, for E mode, this is the predicted labels')
    parser.add_argument('--output', dest='output',
                        default='',  # default empty!
                        help='The output file. For T mode, this is the model path, for P mode, this is the prediction '
                             'result, for E mode, this is the final result of evaluation')
    parser.add_argument('--modelPath', dest='modelPath',
                        default='',  # default empty!
                        help='The path of the machine learning model ')
    parser.add_argument('--trueLabel', dest='trueLabel',
                        default='',  # default empty!
                        help='The path of the correct label ')
    parser.add_argument('--entropy', dest='entropy',
                        default='default',
                        help="There are two entropy methods. The 'default' and 'option2'")
    # if len(sys.argv) < 3:
    #     showHelper()
    main()
