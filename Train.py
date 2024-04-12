import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from time import time 






def train_model(args, model, lossfun, optimizer, data, labels):

    kf = KFold(n_splits = 10, shuffle = True)

    start_epoch = args.start_epoch
    end_epoch = args.end_epoch

    start = time()
    print("\n\n**************  Model Training  **************\n\n")

    accuracies = []
    weighted_precisions = []
    weighted_recalls = []
    weighted_f1_scores = []

    if (args.device == 'cuda:0'):
        torch.cuda.empty_cache()

    for train_index, test_index in kf.split(data):

        X_train, X_val = data[train_index], data[test_index]
        y_train, y_val = labels[train_index], labels[test_index]

        X_train = torch.tensor(X_train, dtype=torch.float32).to(args.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(args.device)

        y_train = torch.tensor(y_train, dtype=torch.long).to(args.device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(args.device)

        model.train()
        for epoch in range(end_epoch):
            outputs = model(X_train)
            loss = lossfun(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs, 1)

            precision = precision_score(y_val.cpu(), predicted.cpu(), average='weighted')
            recall = recall_score(y_val.cpu(), predicted.cpu(), average='weighted')
            f1 = f1_score(y_val.cpu(), predicted.cpu(), average='weighted')

            weighted_precisions.append(precision)
            weighted_recalls.append(recall)
            weighted_f1_scores.append(f1)

            acc = accuracy_score(y_val.cpu(), predicted.cpu())
            accuracies.append(acc)

        print(accuracies)
        
    avg_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)

    print(f"WAP : {100*np.mean(weighted_precisions)}, WAS : {100*np.mean(weighted_recalls)}, WAF1 : {100*np.mean(weighted_f1_scores)}")
    print("Avg ACC : ", 100*avg_accuracy)
    print("STD : ", 100*std_deviation)

    plt.boxplot(accuracies)
    plt.title('Accuracy Distribution')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.show()