# adaptiveDiversityBasedEnsembleLearning

# README #

# IMAGE CAPTIONING #

This project is the implementation of the algorithm published in 2017 paper called "Adaptive Diversity Based Ensemble Classifier"

Dataset:  Repeat Buyers Dataset(provided in the zip)

## Libraries Used ##
1. Python: Most popular language that suits for ML
2. NumPy:  Storing pixels as integers in an array, expanding dimensions.
3. Pandas: Used for data manipulation and storing and retrieving files.
4. Matplotlib: Plot graphs.
5. Skitlearn: for ML algorithms and models
6. math:for mathematical operations like sq root
7. xgboost: for xgboost model
8. mlxtend.classifier: for Stacking Classifier

All the files are included in the zip file
## Project File Structure ##
1. <b> train_final.csv</b> – Dataset contains the data of activity logs of different users.
2. <b> train_final.rar </b> zip file contains the dataset for the project
 
The below files have been created by us while making the project 
1. driver.py – It will start the project executiona and call other files functions when needed.
2. driver_auc.py– this will execute the project by selecting the base models according to auc evaluation when they are used to predict the test dataset
3. driver_f1.py – this will execute the project by selecting the base models based on their f1 scores
4. driver_hierarchical.py - this will execute the project by selecting the base models based on agglomerative clusering of root mean square deviations of pairwise diversities
5. driver_rmsd.py - this will execute the project by selecting the base models with highests root mean square deviations of pairwise diversities
6. driver_zol.py - this will execute the project by selecting the base models based on their zero one loss scores
7. base_model_selector.py - selecting base models
8. base_model_selector2.py - selecting base models for agglomerative clustering
9. manipulate_dataset.py - to extract the data from the dataset and putting in the execution storage for quick execution
10. meta_model_trainer.py - trains the meta model with the relabelled features on the training set
11. relabler.py - relabels the training set by their probabilities of generating positive result
12. soft-computing.pdf - 2017 paper in pdf form


## Step - wise execution ##
1. Execution starts from the execution of python files driver.py or driver_f1.py or any other file with driver in its name.
2. the driver file calls the functions in other files for execution, by first calling manipulate_dataset for removing NaN values and extracting the features from the dataset.
3. then the base models are selected by calling the function in base_model selector.py or base_model selector2.py
4. The function in relabler.py is called and the dataset is relabeled by maintaining a classifier map of all the models.
5. We have made a few changes to the meta_model_trainer, by training the meta model by an ensemble model of many classification models for better results to change a few things regarding execution.
6. the final model is evaluated and the f1 score is calculated to see how well the model fares(see screenshots in report)



