import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def label_prediction(predicted):
    if predicted == 'Positive':
        return 1
    else:
        return 0
    
def avg_confusion_matrix(matrix1, matrix2, matrix3):
    avg_matrix = []

    for i in range(4):
        avg_matrix.append((matrix1[i] + matrix2[i] + matrix3[i])/3)

    return avg_matrix
    
def get_confusion_matrix(f_name):
    model_eval = pd.read_csv(f_name, sep='\t')

    model_eval['predicted_label'] = model_eval['predicted'].apply(label_prediction) 
    
    
    true_labels = model_eval['target']

    predicted_labels = model_eval['predicted_label']

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    return [tn, fp, fn, tp]

def get_classification_report(f_name):
    model_eval = pd.read_csv(f_name, sep='\t')

    model_eval['predicted_label'] = model_eval['predicted'].apply(label_prediction) 
    
    
    true_labels = model_eval['target']

    predicted_labels = model_eval['predicted_label']

    # Generate classification report as a dictionary
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    
    return report


    
def find_model_accuracy(f_name):
    model_eval = pd.read_csv(f_name, sep='\t')

    model_eval['predicted_label'] = model_eval['predicted'].apply(label_prediction) # creates a new column that basically puts the predicted possitive and negatives in terms of 1s and 0s to compare with the correct labels

    model_correct_predictions = sum(model_eval['predicted_label'] == model_eval['target']) # true values are treated as 1 and false as 0 so this goes through all the columns and adds up all the comparisions 

    model_total_predictions = len(model_eval)

    model_accuracy = (model_correct_predictions/model_total_predictions) * 100

    return model_accuracy

def find_model_mcc(f_name):
    model_eval = pd.read_csv(f_name, sep='\t')

    model_eval['predicted_label'] = model_eval['predicted'].apply(label_prediction) 
    
    
    true_labels = model_eval['target']
    predicted_labels = model_eval['predicted_label']


    mcc = matthews_corrcoef(true_labels, predicted_labels)



    return mcc

def calculate_mcc_for_features(feature, model_eval, feature_table):

    # print(feature)
    # Subset data where the feature is present (1 in feature_table)
    feature_present_rows = feature_table[feature] == 1

    filtered_eval = model_eval[feature_present_rows]

    true_labels = filtered_eval['target']
    predicted_labels = filtered_eval['predicted_label']

    mcc = matthews_corrcoef(true_labels, predicted_labels)

    return mcc

def mean_mcc_for_features(model1_feature_mccs, model2_feature_mccs, model3_feature_mccs):
    mean_mccs = []

    for i in range(len(model1_feature_mccs)):
        mean_mcc = (model1_feature_mccs[i] + model2_feature_mccs[i] + model3_feature_mccs[i])/3
        mean_mccs.append(mean_mcc)

    return mean_mccs


    

def all_features_mcc(model_eval_fname, feature_table_fname):
    model_eval = pd.read_csv(model_eval_fname, sep='\t')

    model_eval['predicted_label'] = model_eval['predicted'].apply(label_prediction)

    feature_table = pd.read_csv(feature_table_fname)

    features = ["Simple", "Predicate", "Adjunct", "Argument Type", "Arg Altern", "Imperative", "Binding", "Question", "Comp Clause", "Auxiliary", "to-VP", "N, Adj", "S-Syntax", "Determiner", "Violations"]

    mccs = []

    for feature in features:
        mcc = calculate_mcc_for_features(feature, model_eval, feature_table)
        mccs.append(mcc)

    return mccs

# def avg_report(report1, report2, report3):
#     return (report1 + report2 + report3)/3
    
def check_if_sent_line_up(fname_tsv, fname_csv):
    same = True
    model1 = pd.read_csv(fname_tsv, sep='\t')
    model2 = pd.read_csv(fname_csv)
    
    if len(model1) != len(model2):
        print("Models do not have the same number of sentences")
        same = False
    else:
        for i in range(len(model1)):
            if model1['text'][i] != model2['Sentence'][i]:
                print("Sentences do not line up")
                print(i + 1)
                print(model1['text'][i])
                print(model2['Sentence'][i])
                same = False
            
        # for i in range(5):
        #     print(i + 1)
        #     print(model1['text'][i])
        #     print(model2['Sentence'][i])
    
    if same:
        return "All good"
    else:
        return "Not good"

def check_if_id_line_up(fname1_tsv, fname2_tsv):
    same = True
    model1 = pd.read_csv(fname1_tsv, sep='\t')
    model2 = pd.read_csv(fname2_tsv, sep='\t')

    if len(model1) != len(model2):
        print("Models do not have the same number of sentences")
        same = False
    else:
        for i in range(len(model1)):
            if model1['textid'][i] != model2['textid'][i]:
                print("Sentences do not line up")
                print(i + 1)
                print(model1['textid'][i])
                print(model2['textid'][i])
                same = False
            
        # for i in range(5):
        #     print(i + 1)
        #     print(model1['text'][i])
        #     print(model2['Sentence'][i])
    
    if same:
        return "All good"
    else:
        return "Not good"
    


        
def main():

    # print("Bert accuracys: ")

    # bert1_acc = find_model_accuracy("bert_trained_eval1.tsv")

    # print(bert1_acc)

    # bert2_acc = find_model_accuracy("bert_trained_eval2.tsv")

    # print(bert2_acc)

    # bert3_acc = find_model_accuracy("bert_trained_eval3.tsv")

    # print(bert3_acc)

    # print("bert avg: ")

    # bert_avg_acc = (bert1_acc + bert2_acc + bert3_acc)/3

    # print(bert_avg_acc)

    # print("GPT accuracys: ")

    # gpt1_acc = find_model_accuracy("gpt2model1Eval.tsv")

    # print(gpt1_acc)

    # gpt2_acc = find_model_accuracy("gpt2model2Eval.tsv")

    # print(gpt2_acc)

    # gpt3_acc = find_model_accuracy("gpt2model3Eval.tsv")

    # print(gpt3_acc)

    # print("GPT avg: ")

    # gpt_avg_acc = (gpt1_acc + gpt2_acc + gpt3_acc)/3

    # print(gpt_avg_acc)

    # print("Bert MCCs: ")

    bert1_mcc = find_model_mcc("bert_trained_eval1.tsv")

    # print(bert1_mcc)

    bert2_mcc = find_model_mcc("bert_trained_eval2.tsv")

    # print(bert2_mcc)

    bert3_mcc = find_model_mcc("bert_trained_eval3.tsv")

    # print(bert3_mcc)

    # print("bert avg: ")

    bert_avg_mcc = (bert1_mcc + bert2_mcc + bert3_mcc)/3

    # print(bert_avg_mcc)

    # print("GPT MCCs: ")

    gpt1_mcc = find_model_mcc("gpt2model1Eval.tsv")

    # print(gpt1_mcc)

    gpt2_mcc = find_model_mcc("gpt2model2Eval.tsv")

    # print(gpt2_mcc)

    gpt3_mcc = find_model_mcc("gpt2model3Eval.tsv")

    # print(gpt3_mcc)

    # print("GPT avg: ")

    gpt_avg_mcc = (gpt1_mcc + gpt2_mcc + gpt3_mcc)/3

    # print(gpt_avg_mcc)

    # print("Bert feature MCCs: ")

    bert1_feature_mccs = all_features_mcc("bert_trained_eval1.tsv", "sent_features.csv")

    # print(bert1_feature_mccs)

    bert2_feature_mccs = all_features_mcc("bert_trained_eval2.tsv", "sent_features.csv")

    # print(bert2_feature_mccs)

    bert3_feature_mccs = all_features_mcc("bert_trained_eval3.tsv", "sent_features.csv")

    # print(bert3_feature_mccs)

    # print("Bert avg feature MCCs: ")

    bert_avg_feature_mccs = mean_mcc_for_features(bert1_feature_mccs, bert2_feature_mccs, bert3_feature_mccs)

    # print(bert_avg_feature_mccs)

    # print("GPT feature MCCs: ")

    gpt1_feature_mccs = all_features_mcc("gpt2model1Eval.tsv", "sent_features.csv")

    # print(gpt1_feature_mccs)

    gpt2_feature_mccs = all_features_mcc("gpt2model2Eval.tsv", "sent_features.csv")

    # print(gpt2_feature_mccs)

    gpt3_feature_mccs = all_features_mcc("gpt2model3Eval.tsv", "sent_features.csv")

    # print(gpt3_feature_mccs)

    # print("GPT avg feature MCCs: ")

    gpt_avg_feature_mccs = mean_mcc_for_features(gpt1_feature_mccs, gpt2_feature_mccs, gpt3_feature_mccs)

    # print(gpt_avg_feature_mccs)

    bert1_confusion_matrix = get_confusion_matrix("bert_trained_eval1.tsv")

    bert2_confusion_matrix = get_confusion_matrix("bert_trained_eval2.tsv")

    bert3_confusion_matrix = get_confusion_matrix("bert_trained_eval3.tsv")

    bert_avg_confusion_matrix = avg_confusion_matrix(bert1_confusion_matrix, bert2_confusion_matrix, bert3_confusion_matrix)

    gpt1_confusion_matrix = get_confusion_matrix("gpt2model1Eval.tsv")

    gpt2_confusion_matrix = get_confusion_matrix("gpt2model2Eval.tsv")

    gpt3_confusion_matrix = get_confusion_matrix("gpt2model3Eval.tsv")

    gpt_avg_confusion_matrix = avg_confusion_matrix(gpt1_confusion_matrix, gpt2_confusion_matrix, gpt3_confusion_matrix)

    bert1_classification_report = get_classification_report("bert_trained_eval1.tsv")

    bert1_precision = [bert1_classification_report['0']['precision'], bert1_classification_report['1']['precision']]

    bert1_recall = [bert1_classification_report['0']['recall'], bert1_classification_report['1']['recall']]

    bert2_classification_report = get_classification_report("bert_trained_eval2.tsv")

    bert2_precision = [bert2_classification_report['0']['precision'], bert2_classification_report['1']['precision']]

    bert2_recall = [bert2_classification_report['0']['recall'], bert2_classification_report['1']['recall']]

    bert3_classification_report = get_classification_report("bert_trained_eval3.tsv")

    bert3_precision = [bert3_classification_report['0']['precision'], bert3_classification_report['1']['precision']]

    bert3_recall = [bert3_classification_report['0']['recall'], bert3_classification_report['1']['recall']]

    bert_avg_positive_precision = (bert1_precision[1] + bert2_precision[1] + bert3_precision[1])/3

    bert_avg_negative_precision = (bert1_precision[0] + bert2_precision[0] + bert3_precision[0])/3

    bert_avg_positive_recall = (bert1_recall[1] + bert2_recall[1] + bert3_recall[1])/3

    bert_avg_negative_recall = (bert1_recall[0] + bert2_recall[0] + bert3_recall[0])/3

    gpt1_classification_report = get_classification_report("gpt2model1Eval.tsv")

    gpt1_precision = [gpt1_classification_report['0']['precision'], gpt1_classification_report['1']['precision']]

    gpt1_recall = [gpt1_classification_report['0']['recall'], gpt1_classification_report['1']['recall']]

    gpt2_classification_report = get_classification_report("gpt2model2Eval.tsv")

    gpt2_precision = [gpt2_classification_report['0']['precision'], gpt2_classification_report['1']['precision']]

    gpt2_recall = [gpt2_classification_report['0']['recall'], gpt2_classification_report['1']['recall']]

    gpt3_classification_report = get_classification_report("gpt2model3Eval.tsv")

    gpt3_precision = [gpt3_classification_report['0']['precision'], gpt3_classification_report['1']['precision']]

    gpt3_recall = [gpt3_classification_report['0']['recall'], gpt3_classification_report['1']['recall']]

    gpt_avg_positive_precision = (gpt1_precision[1] + gpt2_precision[1] + gpt3_precision[1])/3

    gpt_avg_negative_precision = (gpt1_precision[0] + gpt2_precision[0] + gpt3_precision[0])/3

    gpt_avg_positive_recall = (gpt1_recall[1] + gpt2_recall[1] + gpt3_recall[1])/3

    gpt_avg_negative_recall = (gpt1_recall[0] + gpt2_recall[0] + gpt3_recall[0])/3

    bert1_f1 = [bert1_classification_report['0']['f1-score'], bert1_classification_report['1']['f1-score']]

    bert2_f1 = [bert2_classification_report['0']['f1-score'], bert2_classification_report['1']['f1-score']]

    bert3_f1 = [bert3_classification_report['0']['f1-score'], bert3_classification_report['1']['f1-score']]

    bert_avg_positive_f1 = (bert1_f1[1] + bert2_f1[1] + bert3_f1[1])/3

    bert_avg_negative_f1 = (bert1_f1[0] + bert2_f1[0] + bert3_f1[0])/3

    gpt1_f1 = [gpt1_classification_report['0']['f1-score'], gpt1_classification_report['1']['f1-score']]

    gpt2_f1 = [gpt2_classification_report['0']['f1-score'], gpt2_classification_report['1']['f1-score']]

    gpt3_f1 = [gpt3_classification_report['0']['f1-score'], gpt3_classification_report['1']['f1-score']]

    gpt_avg_positive_f1 = (gpt1_f1[1] + gpt2_f1[1] + gpt3_f1[1])/3

    gpt_avg_negative_f1 = (gpt1_f1[0] + gpt2_f1[0] + gpt3_f1[0])/3


    

    analyze_results = pd.DataFrame({ 
    'Model': ['Bert1', 'Bert2', 'Bert3', 'BertAvg', 'GPT1', 'GPT2', 'GPT3', 'GPTAvg'],
    'MCC': [bert1_mcc, bert2_mcc, bert3_mcc, bert_avg_mcc, gpt1_mcc, gpt2_mcc, gpt3_mcc, gpt_avg_mcc],
    'True Positive': [bert1_confusion_matrix[3], bert2_confusion_matrix[3], bert3_confusion_matrix[3], bert_avg_confusion_matrix[3], gpt1_confusion_matrix[3], gpt2_confusion_matrix[3], gpt3_confusion_matrix[3], gpt_avg_confusion_matrix[3]],
    'True Negative': [bert1_confusion_matrix[0], bert2_confusion_matrix[0], bert3_confusion_matrix[0], bert_avg_confusion_matrix[0], gpt1_confusion_matrix[0], gpt2_confusion_matrix[0], gpt3_confusion_matrix[0], gpt_avg_confusion_matrix[0]],
    'False Positive': [bert1_confusion_matrix[1], bert2_confusion_matrix[1], bert3_confusion_matrix[1], bert_avg_confusion_matrix[1], gpt1_confusion_matrix[1], gpt2_confusion_matrix[1], gpt3_confusion_matrix[1], gpt_avg_confusion_matrix[1]],
    'False Negative': [bert1_confusion_matrix[2], bert2_confusion_matrix[2], bert3_confusion_matrix[2], bert_avg_confusion_matrix[2], gpt1_confusion_matrix[2], gpt2_confusion_matrix[2], gpt3_confusion_matrix[2], gpt_avg_confusion_matrix[2]],
    'Positive Precision': [bert1_precision[1], bert2_precision[1], bert3_precision[1], bert_avg_positive_precision, gpt1_precision[1], gpt2_precision[1], gpt3_precision[1], gpt_avg_positive_precision],
    'Negative Precision': [bert1_precision[0], bert2_precision[0], bert3_precision[0], bert_avg_negative_precision, gpt1_precision[0], gpt2_precision[0], gpt3_precision[0], gpt_avg_negative_precision],
    'Positive Recall': [bert1_recall[1], bert2_recall[1], bert3_recall[1], bert_avg_positive_recall, gpt1_recall[1], gpt2_recall[1], gpt3_recall[1], gpt_avg_positive_recall],
    'Negative Recall': [bert1_recall[0], bert2_recall[0], bert3_recall[0], bert_avg_negative_recall, gpt1_recall[0], gpt2_recall[0], gpt3_recall[0], gpt_avg_negative_recall],
    'Positive F1': [bert1_f1[1], bert2_f1[1], bert3_f1[1], bert_avg_positive_f1, gpt1_f1[1], gpt2_f1[1], gpt3_f1[1], gpt_avg_positive_f1],
    'Negative F1': [bert1_f1[0], bert2_f1[0], bert3_f1[0], bert_avg_negative_f1, gpt1_f1[0], gpt2_f1[0], gpt3_f1[0], gpt_avg_negative_f1],
    'Simple MCC': [bert1_feature_mccs[0], bert2_feature_mccs[0], bert3_feature_mccs[0], bert_avg_feature_mccs[0], gpt1_feature_mccs[0], gpt2_feature_mccs[0], gpt3_feature_mccs[0], gpt_avg_feature_mccs[0]],
    'Predicate MCC': [bert1_feature_mccs[1], bert2_feature_mccs[1], bert3_feature_mccs[1], bert_avg_feature_mccs[1], gpt1_feature_mccs[1], gpt2_feature_mccs[1], gpt3_feature_mccs[1], gpt_avg_feature_mccs[1]],
    'Adjunct MCC': [bert1_feature_mccs[2], bert2_feature_mccs[2], bert3_feature_mccs[2], bert_avg_feature_mccs[2], gpt1_feature_mccs[2], gpt2_feature_mccs[2], gpt3_feature_mccs[2], gpt_avg_feature_mccs[2]],
    'Argument Type MCC': [bert1_feature_mccs[3], bert2_feature_mccs[3], bert3_feature_mccs[3], bert_avg_feature_mccs[3], gpt1_feature_mccs[3], gpt2_feature_mccs[3], gpt3_feature_mccs[3], gpt_avg_feature_mccs[3]],
    'Arg Altern MCC': [bert1_feature_mccs[4], bert2_feature_mccs[4], bert3_feature_mccs[4], bert_avg_feature_mccs[4], gpt1_feature_mccs[4], gpt2_feature_mccs[4], gpt3_feature_mccs[4], gpt_avg_feature_mccs[4]],
    'Imperative MCC': [bert1_feature_mccs[5], bert2_feature_mccs[5], bert3_feature_mccs[5], bert_avg_feature_mccs[5], gpt1_feature_mccs[5], gpt2_feature_mccs[5], gpt3_feature_mccs[5], gpt_avg_feature_mccs[5]],
    'Binding MCC': [bert1_feature_mccs[6], bert2_feature_mccs[6], bert3_feature_mccs[6], bert_avg_feature_mccs[6], gpt1_feature_mccs[6], gpt2_feature_mccs[6], gpt3_feature_mccs[6], gpt_avg_feature_mccs[6]],
    'Question MCC': [bert1_feature_mccs[7], bert2_feature_mccs[7], bert3_feature_mccs[7], bert_avg_feature_mccs[7], gpt1_feature_mccs[7], gpt2_feature_mccs[7], gpt3_feature_mccs[7], gpt_avg_feature_mccs[7]],
    'Comp Clause MCC': [bert1_feature_mccs[8], bert2_feature_mccs[8], bert3_feature_mccs[8], bert_avg_feature_mccs[8], gpt1_feature_mccs[8], gpt2_feature_mccs[8], gpt3_feature_mccs[8], gpt_avg_feature_mccs[8]],
    'Auxiliary MCC': [bert1_feature_mccs[9], bert2_feature_mccs[9], bert3_feature_mccs[9], bert_avg_feature_mccs[9], gpt1_feature_mccs[9], gpt2_feature_mccs[9], gpt3_feature_mccs[9], gpt_avg_feature_mccs[9]],
    'to-VP MCC': [bert1_feature_mccs[10], bert2_feature_mccs[10], bert3_feature_mccs[10], bert_avg_feature_mccs[10], gpt1_feature_mccs[10], gpt2_feature_mccs[10], gpt3_feature_mccs[10], gpt_avg_feature_mccs[10]],
    'N, Adj MCC': [bert1_feature_mccs[11], bert2_feature_mccs[11], bert3_feature_mccs[11], bert_avg_feature_mccs[11], gpt1_feature_mccs[11], gpt2_feature_mccs[11], gpt3_feature_mccs[11], gpt_avg_feature_mccs[11]],
    'S-Syntax MCC': [bert1_feature_mccs[12], bert2_feature_mccs[12], bert3_feature_mccs[12], bert_avg_feature_mccs[12], gpt1_feature_mccs[12], gpt2_feature_mccs[12], gpt3_feature_mccs[12], gpt_avg_feature_mccs[12]],
    'Determiner MCC': [bert1_feature_mccs[13], bert2_feature_mccs[13], bert3_feature_mccs[13], bert_avg_feature_mccs[13], gpt1_feature_mccs[13], gpt2_feature_mccs[13], gpt3_feature_mccs[13], gpt_avg_feature_mccs[13]],
    'Violations MCC': [bert1_feature_mccs[14], bert2_feature_mccs[14], bert3_feature_mccs[14], bert_avg_feature_mccs[14], gpt1_feature_mccs[14], gpt2_feature_mccs[14], gpt3_feature_mccs[14], gpt_avg_feature_mccs[14]],
    })

    # print(analyze_results)

    analyze_results.to_csv('midterm_analysis.csv', index=False)

    # print(check_if_sent_line_up("evaldatatest.tsv", "sent_features.csv"))

    # print(check_if_id_line_up("bert_trained_eval1.tsv", "gpt2model1Eval.tsv"))

    # print(check_if_id_line_up("bert_trained_eval1.tsv", "evaldatatest.tsv"))

    





    

if __name__ == "__main__":
    main()