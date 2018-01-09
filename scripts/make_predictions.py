import pickle
import re
import pandas as pd

# Read data
# --------------------------------------------------------------------------------------------------------------------
test_X = 'Data/test_X_languages_homework.json.txt'
model_pipeline = pickle.load(open('best_model.bin', 'rb'))

with open(test_X) as f:
    data = f.read()
test_X = []
[test_X.append(re.findall(':"(.*?)"}', value, re.S)[0]) for value in data.split('\n') if value]

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