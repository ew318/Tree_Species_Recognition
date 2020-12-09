import math
import numpy as np
from sklearn.metrics import confusion_matrix


# Generate precision, recall, F1 score & confusion matrix given test predictions and test labels.
def get_metrics(model, test_data_gen):
    result = model.predict(test_data_gen)
    result_fix = []
    for item in result:
        result_fix.append([1 if y == max(item) else 0 for y in item])
    num_examples = len(test_data_gen.filenames)
    number_of_generator_calls = math.ceil(num_examples/(1*64))
    test_labels = []
    for k in range(0, int(number_of_generator_calls)):
        test_labels.extend(np.array(test_data_gen[k][1]))
    predictions = []
    labels = []
    for k in range(0,len(test_labels)):
        labels.append(list(test_labels[k]).index(1))
        predictions.append(result_fix[k].index(1))
    cm = confusion_matrix(labels, predictions)
    # TP / (TP + FN)
    recall = np.nan_to_num(np.diag(cm) / np.sum(cm, axis=1))
    # TP / (TP + FP)
    precision = np.nan_to_num(np.diag(cm) / np.sum(cm, axis=0))
    # 2 * precision * recall / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall)
    out_data = {
        'confusion_matrix': cm,
        'recall': recall,
        'avg_recall': np.mean(recall),
        'precision': precision,
        'avg_precision': np.mean(precision),
        'f1': f1
    }
    return out_data
