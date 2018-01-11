import pickle
import json
import pandas as pd

# Read data
# --------------------------------------------------------------------------------------------------------------------
test_X_input = 'Data/test_X_languages_homework.json.txt'
model_pipeline = pickle.load(open('best_model.bin', 'rb'))

test_X = []
for line in open(test_X_input):
    strToJason = json.loads(line)
    test_X.append(strToJason['text'])

# use preprocessors from model_pipeline to pre-process input test data
# --------------------------------------------------------------------------------------------------------------------
data = model_pipeline.named_steps['vec'].transform(test_X)
data = model_pipeline.named_steps['svd'].transform(data)
labels = model_pipeline.named_steps['km'].predict(data)
data = pd.DataFrame(data)
data['clusters'] = labels

# Make predictions and write predictions to file
# --------------------------------------------------------------------------------------------------------------------
result = model_pipeline.named_steps['clf'].predict(data)

with open('Result/predictions.txt', "w+") as f:
    [f.write("{\"classification\":\"" + str(value)+'\"}\n') for value in result]
f.close()