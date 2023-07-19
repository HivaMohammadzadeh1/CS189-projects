import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io
import math
from sklearn import metrics
from save_csv import results_to_csv

if __name__ == "__main__":
    for data_name in ["mnist", "spam", "cifar10"]:
        data = np.load(f"../data/{data_name}-data.npz")
        print("\nloaded %s data!" % data_name)
        fields = "test_data", "training_data", "training_labels"
        for field in fields:
            print(field, data[field].shape)
    
    
    
### QUESTION 2: shuffle and partition each of the datasets in the assignment. 
#Shuffling prior to splitting crucially ensures that all classes are represented in your partitions. 

# MNIST
    print("\nMNIST Shuffling and Splitting:")
    data = np.load(f"../data/mnist-data.npz")
    training_data = data["training_data"]
    # print(training_data.shape)
    training_labels = data["training_labels"]
    # print(training_labels.shape)
    training_data = np.reshape(training_data, (60000, 784))
    training_labels = np.reshape(training_labels,(60000, 1))

    # Concatenating the labels with data in order to shuffle better
    concatenated_data = np.concatenate((training_data, training_labels), axis = 1)
    
    #Shuffling the data
    np.random.shuffle(concatenated_data)
    training_data = concatenated_data[:,:-1]
    training_labels = np.reshape(concatenated_data[:,-1], (60000, 1))
    # print(training_labels.shape)

    #Split using the amount we want in validation set
    amount_set_aside = 10000
    mnist_validation_data, mnist_validation_labels = training_data[:amount_set_aside,:], training_labels[:amount_set_aside,:]
    mnist_training_data, mnist_training_labels = training_data[amount_set_aside:,:], training_labels[amount_set_aside:,:]
    test_data = np.reshape(data["test_data"], (10000, 784))
    # print(test_data.shape)

    #Print the datasets' shapes
    print(f"Training data: {mnist_training_data.shape} \nTraining labels: {mnist_training_labels.shape}")
    print(f"Validation data: {mnist_validation_data.shape} \nValidation labels: {mnist_validation_labels.shape}")    



#SPAM
    print("\nSPAM Shuffling and Splitting:")
    data = np.load(f"../data/spam-data.npz")
    training_data = data["training_data"]
    # print(training_data.shape)
    training_labels = data["training_labels"]
    # print(training_labels.shape)
    # training_data = np.reshape(training_data, (60000, 784))
    training_labels = np.reshape(training_labels,(4172, 1))

    # Concatenating the labels with data in order to shuffle better
    concatenated_data = np.concatenate((training_data, training_labels), axis = 1)

    #shuffling the data
    np.random.shuffle(concatenated_data)
    training_data = concatenated_data[:,:-1]
    training_labels = np.reshape(concatenated_data[:,-1], (4172, 1))

    #Split using the amount we want in validation set
    #Convert 20% percent to amount 
    percent_set_aside = 0.20
    #print(training_data.shape)
    amount_set_aside = math.ceil(percent_set_aside * training_data.shape[0])
    spam_validation_data, spam_validation_labels = training_data[:amount_set_aside,:], training_labels[:amount_set_aside,:]
    spam_training_data, spam_training_labels = training_data[amount_set_aside:,:], training_labels[amount_set_aside:,:]

    #Print the datasets' shapes
    print(f"Training data: {spam_training_data.shape} \nTraining labels: {spam_training_labels.shape}")
    print(f"Validation data: {spam_validation_data.shape} \nValidation labels: {spam_validation_labels.shape}")



#CIFAR-10
    print("\nCIFAR-10 Shuffling and Splitting:")
    data = np.load(f"../data/cifar10-data.npz")
    training_data = data["training_data"]
    # print(training_data.shape)
    training_labels = data["training_labels"]
    # print(training_labels.shape)
    # training_data = np.reshape(training_data, (60000, 784))
    training_labels = np.reshape(training_labels,(50000, 1))

    # Concatenating the labels with data in order to shuffle better
    concatenated_data = np.concatenate((training_data, training_labels), axis = 1)

    #shuffling the data
    np.random.shuffle(concatenated_data)
    training_data = concatenated_data[:,:-1]
    training_labels = np.reshape(concatenated_data[:,-1], (50000, 1))

    #Split using the amount we want in validation set
    amount_set_aside = 5000
    cifar10_validation_data, cifar10_validation_labels = training_data[:amount_set_aside,:], training_labels[:amount_set_aside,:]
    cifar10_training_data, cifar10_training_labels = training_data[amount_set_aside:,:], training_labels[amount_set_aside:,:]

    #Print the datasets' shapes
    print(f"Training data: {cifar10_training_data.shape} \nTraining labels: {cifar10_training_labels.shape}")
    print(f"Validation data: {cifar10_validation_data.shape} \nValidation labels: {cifar10_validation_labels.shape}")




### QUESTION 3: SVM. Training the models and calculating validation accuracies

# Part a) MNIST Dataset: 
    print("\nMNIST Dataset accuracies:")
    mnist_model = svm.SVC()
    training_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    accuracies = {"training": [], "validation": []}

    for training_size in training_sizes:

        print(f"Training with {training_size} examples")

        training_labels = np.asarray(mnist_training_labels).reshape(-1)
        validation_labels = np.asarray(mnist_validation_labels).reshape(-1)
        
        # Preprocessing and normalizing
        # print(np.max(mnist_training_data[:training_size,:]))
        # 255 is the maximum 
        mnist_training_data = mnist_training_data / 255
        mnist_validation_data = mnist_validation_data / 255
       
        mnist_model.fit(mnist_training_data[:training_size, :], training_labels[:training_size])
        
        # Calculate training and validation accuracies 
        training_accuracy = metrics.accuracy_score(training_labels[:training_size], mnist_model.predict(mnist_training_data[:training_size,:]))
        validation_accuracy = metrics.accuracy_score(validation_labels[:training_size], mnist_model.predict(mnist_validation_data[:training_size,:]))

        accuracies["training"].append(training_accuracy)
        accuracies["validation"].append(validation_accuracy)

        print(f"Training accuracy: {training_accuracy} \nValidation Accuracy: {validation_accuracy}\n")

    #Graph the plots
    plt.figure(1)
    plt.plot(training_sizes, accuracies["training"], '.r-')
    plt.plot(training_sizes, accuracies["validation"], '.b-')
    plt.legend(['Training', 'Validation'], loc=4)
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.title("MNIST dataset")
    # plt.show()
    plt.savefig('mnist_accuracy.png')



# Part b) SPAM Dataset: 

    print("\nSPAM Dataset accuracies:")
    spam_model = svm.SVC()
    training_sizes = [100, 200, 500, 1000, 2000, spam_training_data.shape[0]]
    # print(spam_training_data_x.shape[0])
    accuracies = {"training": [], "validation": []}

    
    for training_size in training_sizes:

        print(f"Training with {training_size} examples")

        training_labels = np.asarray(spam_training_labels).reshape(-1)
        validation_labels = np.asarray(spam_validation_labels).reshape(-1)
        spam_model.fit(spam_training_data[:training_size,:], training_labels[:training_size])

        # Calculate training and validation accuracies 
        training_accuracy = metrics.accuracy_score(training_labels[:training_size], spam_model.predict(spam_training_data[:training_size,:]))
        validation_accuracy = metrics.accuracy_score(validation_labels[:training_size], spam_model.predict(spam_validation_data[:training_size,:]))

        accuracies["training"].append(training_accuracy)
        accuracies["validation"].append(validation_accuracy)

        print(f"Training accuracy: {training_accuracy} \nValidation Accuracy: {validation_accuracy}\n")

    #Graph the plots
    plt.figure(2)
    plt.plot(training_sizes, accuracies["training"], '.r-')
    plt.plot(training_sizes, accuracies["validation"], '.b-')
    plt.legend(['Training', 'Validation'], loc=4)
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.title("SPAM dataset")
    # plt.show()
    plt.savefig('spam_accuracy.png')



# Part c) CIFAR-10 Dataset: 

    print("\nCIFAR-10 Dataset accuracies:")
    cifar10_model = svm.SVC(kernel = 'rbf')
    training_sizes = [100, 200, 500, 1000, 2000, 5000]
    accuracies = {"training": [], "validation": []}

    for training_size in training_sizes:

        print(f"Training with {training_size} examples")

        training_labels = np.asarray(cifar10_training_labels).reshape(-1)
        validation_labels = np.asarray(cifar10_validation_labels).reshape(-1)

        # Preprocessing and normalization:
        cifar10_training_data = (cifar10_training_data - np.mean(cifar10_training_data))/np.std(cifar10_training_data)
        cifar10_validation_data = (cifar10_validation_data - np.mean(cifar10_validation_data))/np.std(cifar10_validation_data)

        cifar10_model.fit(cifar10_training_data[:training_size, :], training_labels[:training_size])

        # Calculate training and validation accuracies 
        training_accuracy = metrics.accuracy_score(training_labels[:training_size], cifar10_model.predict(cifar10_training_data[:training_size,:]))
        validation_accuracy= metrics.accuracy_score(validation_labels[:training_size], cifar10_model.predict(cifar10_validation_data[:training_size,:]))

        accuracies["training"].append(training_accuracy)
        accuracies["validation"].append(validation_accuracy)

        print(f"Training accuracy: {training_accuracy} \nValidation Accuracy: {validation_accuracy}\n")

    plt.figure(3)
    plt.plot(training_sizes, accuracies["training"], '.r-')
    plt.plot(training_sizes, accuracies["validation"], '.b-')
    plt.legend(['Training', 'Validation'], loc=4)
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 dataset")
    # plt.show()
    plt.savefig('cifar10_accuracy.png')




### Question 4: Hyperparameter Tuning
# The best C value for MNIST Dataset

    print("\nMNIST Dataset C value Calculation: ")
    # mnist_training_size = mnist_training_data.shape[0]
    mnist_training_size = 10000
    #Function to train the dataset on the given c_value and calculate the validation accuracy score for it
    #For Mnist and cifar since they need preprocessing 
    def calculate_validation_accuracy_MNIST_CIFAR (train_data, train_labels, val_data, val_labels, train_size, c_val):
        print(f"\nc_value: {c_val}")
        training_labels = train_labels.reshape(-1)
        validation_labels = val_labels.reshape(-1)
        model = svm.SVC( C = c_val)
        # Preprocessing and normalizing
        train_data = (train_data - np.mean(train_data))/np.std(train_data)
        val_data = (val_data - np.mean(val_data))/np.std(val_data)
       
        model.fit(train_data[:train_size], training_labels[:train_size])
        
        validation_accuracy = metrics.accuracy_score(validation_labels[:train_size], model.predict(val_data[:train_size,:]))
        print(f"validation_accuracy: {validation_accuracy}")
        return validation_accuracy

    all_validation_accuracies =[]

    # Geometric series generated by https://onlinenumbertools.com/generate-geometric-sequence using :
    # 1e-8 as the first element, 10 as the multiplication ratio, and 12 total numbers in the sequesnce

    C_values = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000, 10000]
    # C_values = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10]
    # C_values = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]

    # Calculate the accuracy for all the c_values
    for c_value in C_values: 
        validation_accuracy = calculate_validation_accuracy_MNIST_CIFAR(mnist_training_data, mnist_training_labels, mnist_validation_data, mnist_validation_labels, mnist_training_size, c_value)
        all_validation_accuracies.append(validation_accuracy)
        
    # Dictionary of all c values and their validation accuracies 
    dictionary = dict(zip(C_values, all_validation_accuracies))
    # Print the best C value that gives the highest validation accuracy
    print("\nBest c value for mnist is: " + str(C_values[all_validation_accuracies.index(max(all_validation_accuracies))])
                                                + "\nWith validation accuracy of: " + str(max(all_validation_accuracies)))

    plt.figure(4)
    plt.plot(C_values, all_validation_accuracies, label= "validation set")
    plt.xlabel("C value")
    plt.ylabel("Accuracy")
    plt.title("MNIST dataset")
    plt.xlim(max(C_values), -500)
    # plt.show()
    plt.savefig('MNIST_C_Value.png')




## Question 5: K-Fold Cross Validation
# The best C value for Spam dataset
# Using 5-fold cross-validation, with at least 8 c_values 

    #Function to train the dataset on the given c_value and calculate the validation accuracy score for it
    #For SPAM since it doesn't need preprocessing 
    def calculate_validation_accuracy_SPAM (train_data, train_labels, val_data, val_labels, train_size, c_val):
        print(f"\nc_value: {c_val}")
        training_labels = train_labels.reshape(-1)
        validation_labels = val_labels.reshape(-1)
        model = svm.SVC( C = c_val)

        model.fit(train_data[:train_size], training_labels[:train_size])
        
        validation_accuracy = metrics.accuracy_score(validation_labels[:train_size], model.predict(val_data[:train_size,:]))
        print(f"validation_accuracy: {validation_accuracy}")
        return validation_accuracy
    
    print("\nSPAM Dataset C value Calculation: ")
    spam_training_size = spam_training_data.shape[0]
    data = np.load(f"../data/spam-data.npz")
    #5-fold --> k=5
    k_value = 5
    spam_data = data["training_data"]
    spam_labels = data["training_labels"]
    # print(spam_labels.shape)
    spam_labels = np.reshape(spam_labels, (4172, 1))
    # print(spam_data.shape)
    fold = spam_data.shape[0]/k_value
    
    # Geometric series generated by https://onlinenumbertools.com/generate-geometric-sequence using :
    # 1e-8 as the first element, 10 as the multiplication ratio, and 12 total numbers in the sequesnce
    # C_values = [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
    # C_values = [0.0000001, 0.000001,0.00001,0.0001,0.001,0.01,0.1,1]]
    # C_values = [0.0001, 0.001,0.01,0.1,1,10,100,1000, 10000, 100000]
    C_values = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]



    #Function to train the dataset on the given c_value and calculate the 5-fold cross-validation accuracy score for it
    def calculate_k_folds_validation(c_value):
        all_validation_accuracies = []
        #"This process is repeated k times with each set chosen as the validation set once."
        for i in range(k_value):
            #"Model is trained on k − 1 sets and validated on the kth set."
            beginning_val = int(fold*i)
            ending_val = int(fold*(i+1))
            spam_validation_data = spam_data[beginning_val:ending_val]
            spam_validation_labels = spam_labels[beginning_val:ending_val]
            # print(spam_labels.shape)
            spam_training_data = np.vstack((spam_data[:beginning_val],spam_data[ending_val:]))
            spam_training_labels = np.vstack((spam_labels[:beginning_val],spam_labels[ending_val:]))
        
            #Call the function to calculate the validation accuracy 
            all_validation_accuracies.append(calculate_validation_accuracy_SPAM(spam_training_data, spam_training_labels, spam_validation_data, 
                                    spam_validation_labels, spam_training_size, c_value))
        return all_validation_accuracies

    # Calculate the k-folds accuracies for each C value
    cross_validation_accuracies = [calculate_k_folds_validation(c_value) for c_value in C_values]
    # "The cross-validation accuracy we report is the accuracy averaged over the k iterations." Average the validation accuracies 
    cross_validation_accuracy = [sum(i)/len(i) for i in cross_validation_accuracies]

    # dictionary of all c values and their validation accuracies 
    dictionary = dict(zip(C_values, cross_validation_accuracy))
    
    #Take the c value that gives the maximum average validation accuracy
    best_c_value = C_values[cross_validation_accuracy.index(max(cross_validation_accuracy))]
    print("Best c value for SPAM is: " + str(best_c_value) + "\nWith cross validation accuracy of: " + str(max(cross_validation_accuracy)))

    plt.figure(5)
    plt.plot(C_values, cross_validation_accuracy, label= "validation set")
    plt.xlabel("C value")
    plt.ylabel("Accuracy")
    plt.title("SPAM dataset")
    plt.xlim(max(C_values), -500)
    # plt.show()
    plt.savefig('SPAM_C_Value.png')




### QUESTION 6: Kaggle 

# MNIST Dataset:
    #Function to test the model
    def test(training_data, training_labels, testing_data, training_size, C_value):
        training_labels = training_labels.reshape(-1,)
        model = svm.SVC(C=C_value)
        
        model.fit(training_data[:training_size,:],training_labels[:training_size])
        return model.predict(testing_data)
    
    #List to hold the final predictions on the test data
    mnist_final_test_predictions= [] 
    
    print("\nTesting data for mnist")
    data = np.load(f"../data/mnist-data.npz")
    mnist_test_data = data["test_data"]
    # print(mnist_test_data.shape)
    mnist_test_data = np.reshape(mnist_test_data, (10000, 784))
    # print(mnist_test_data.shape)
    # Preprocessing and normalizing 
    mnist_training_data = (mnist_training_data - np.mean(mnist_training_data))/np.std(mnist_training_data)
    mnist_test_data = (mnist_test_data - np.mean(mnist_test_data))/np.std(mnist_test_data)
    # Calculate the final predictions (validation accuracies) for the test data 
    # Using the best C value for MNIST calculated in question 4 (0.01)
    mnist_test_result = test(mnist_training_data, mnist_training_labels, mnist_test_data, mnist_training_size, 10)
    # Save the result to a csv file
    results_to_csv(mnist_test_result)
    print("\nSuccessfully ran on test data for MNIST and saved to the csv file\n")
    
    
    
 # SPAM Dataset: 
    #List to hold the final predictions on the test data
    spam_final_test_predictions= [] 
    
    print("\nTesting data for spam")

    data = np.load(f"../data/spam-data.npz")
    spam_test_data = data["test_data"]
    # print(spam_test_data.shape)
    # spam_test_data = (spam_test_data - np.mean(spam_test_data))/np.std(spam_test_data)
    # Calculate the final predictions (validation accuracies) for the test data 
    # using the best C value for MNIST calculated in question 5 (0.01)

    spam_test_result = test(spam_training_data, spam_training_labels, spam_test_data, spam_training_size, 4096)
    # Save the result to a csv file
    results_to_csv(spam_test_result)
    print("\nSuccessfully ran on test data for SPAM and saved to the csv file\n")
    
    
     
        
## Repeating Question 4 for CIFAR-10: Hyperparameter Tuning
# The best C value for cifar-10 Dataset

    print("\nCIFAR-10 Dataset C value Calculation: ")
    # cifar_training_size = cifar10_training_data.shape[0]
    cifar_training_size = 10000

    #Function to train the dataset on the given c_value and calculate the validation accuracy score for it
    def calculate_validation_accuracy (train_data, train_labels, val_data, val_labels, train_size, c_val):
        print(f"\nc_value: {c_val}")
        training_labels = train_labels.reshape(-1)
        validation_labels = val_labels.reshape(-1)
        mnist_model = svm.SVC(C = c_val)
        mnist_model.fit(train_data[:train_size], training_labels[:train_size])
        
        validation_accuracy = metrics.accuracy_score(validation_labels[:train_size], mnist_model.predict(val_data[:train_size,:]))
        print(f"validation_accuracy: {validation_accuracy}")
        return validation_accuracy

    all_validation_accuracies =[]

    # Geometric series generated by https://onlinenumbertools.com/generate-geometric-sequence using :
    # 1e-8 as the first element, 10 as the multiplication ratio, and 12 total numbers in the sequesnce
    # C_values = [0.0000001, 0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
    # C_values = [0.01,0.1,1,10]
    # C_values = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
    C_values = [0.001,0.01,0.1,1,10,100,1000]   

    # Calculate the accuracy for all the c_values
    for c_value in C_values: 
        cifar10_training_data = (cifar10_training_data - np.mean(cifar10_training_data))/np.std(cifar10_training_data)
        cifar10_validation_data = (cifar10_validation_data - np.mean(cifar10_validation_data))/np.std(cifar10_validation_data)
        validation_accuracy = calculate_validation_accuracy(cifar10_training_data, cifar10_training_labels, cifar10_validation_data, cifar10_validation_labels, cifar_training_size, c_value)
        all_validation_accuracies.append(validation_accuracy)
        
    # Dictionary of all c values and their validation accuracies 
    dictionary = dict(zip(C_values, all_validation_accuracies))
    # Print the best C value that gives the highest validation accuracy
    print("\nBest c value for cifar is: " + str(C_values[all_validation_accuracies.index(max(all_validation_accuracies))]) 
                                                + "\nWith validation accuracy of: " + str(max(all_validation_accuracies)))

    plt.figure(5)
    plt.plot(C_values, all_validation_accuracies, label= "validation set")
    plt.xlabel("C value")
    plt.ylabel("Accuracy")
    plt.title("Cifar dataset")
    plt.xlim(max(C_values), -500)
    # plt.show()
    plt.savefig('cifar_10_C_Value.png')
        
        
        
        
# CIfAR-10 Dataset: 

    #List to hold the final predictions on the test data
    cifar10_final_test_predictions= [] 
    
    print("\nTesting data for cifar-10")

    data = np.load(f"../data/cifar10-data.npz")
    cifar10_test_data = data["test_data"]
    # print(cifar10_test_data.shape)
    
    ## cifar10_test_data = (cifar10_test_data - np.mean(cifar10_test_data))/np.std(cifar10_test_data)
    #Calculate the final predictions (validation accuracies) for the test data 
    # using the best C value for MNIST calculated in question 5 (0.01)
    cifar10_training_data = (cifar10_training_data - np.mean(cifar10_training_data))/np.std(cifar10_training_data)
    cifar10_test_data = (cifar10_test_data - np.mean(cifar10_test_data))/np.std(cifar10_test_data)

    # print("Hi")
    cifar10_test_result = test(cifar10_training_data, cifar10_training_labels, cifar10_test_data, cifar_training_size, 10)
    # Save the result to a csv file
    results_to_csv(cifar10_test_result)
    print("\nSuccessfully ran on test data for CIFAR-10 and saved to the csv file\n")
