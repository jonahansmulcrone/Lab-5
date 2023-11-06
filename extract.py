def extract_predictions(input_filepath, output_filepath):
    with open(input_filepath, 'r') as file:
        lines = file.readlines()

    predicted_labels = []

    for line in lines:
        if line.strip() and 'inst#' not in line and not line.startswith('==='): 
            parts = line.split()
            if len(parts) > 3:  
                predicted_label = parts[2].split(':')[1]
                predicted_labels.append(predicted_label)

    with open(output_filepath, 'w') as file:
        for label in predicted_labels:
            file.write(label + '\n')

input_filepath = './weka_predictions.txt'
output_filepath = './weka_predictions_formatted.txt'

extract_predictions(input_filepath, output_filepath)
