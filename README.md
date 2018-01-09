# Python/library versions
Python: 3.6.1 |Anaconda custom (x86_64)| (default, May 11 2017, 13:04:09) <br />
[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)] <br />
scipy: 0.19.1  <br />
numpy: 1.13.1 <br />
matplotlib: 2.0.2 <br />
pandas: 0.20.3 <br />
sklearn: 0.19.0 <br />

# Feature and Model Selection 
1. picked 100 random samples of each label/classification from the entire train data
2. Used TfidfVectorizer to tokenize and vectorize data using stop words to reduce number of features
3. Used TruncatedSVD for dimensionality reduction
4. Generated k-means clusters on the projected data
5. Added the k-means clusters as new features to the reduced data
6. Ran nested cross validation (with GridSearchCV) using 2 folds and averaged scores over 3 trials for each model
7. Picked the model with maximum average score as the best model that maximizes out of sample performance
8.  Tuned the best model from step 6 to obtain optimized parameters using GridSearchCV

# Improvements
1. More training samples
2. More models and parameters
3. More number of folds and trials for cross validation
4. Use other feature selection techniques and compare performace
5. Analyze the effect of scaling data before training
6. For TruncatedSVD, I set n_components param to int(no_features/2.5). Ideally, I could use the explianed_varaince
   measure to select the number of components to use without loosing much information
7. Increase max_iter for Perceptron model to improve the fit.
