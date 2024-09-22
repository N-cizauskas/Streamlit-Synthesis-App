# import packages

import streamlit as st

import pandas as pd

import random



import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression




# design page
if "randomise_clicked" not in st.session_state:
    st.session_state["randomise_clicked"] = False

if "randomised_data" not in st.session_state:
    st.session_state["randomised_data"] = []

if "synthesise" not in st.session_state:
    st.session_state["synthesise"] = False

def reset():
    for key in st.session_state.keys():
        del st.session_state[key]


# randomization of dataset
st.title("Let's Create Synthetic Data!")
st.button("Reset", on_click=reset, type="primary")

# randomization lists
def randomise_data():
    rand_sex=[]
    n=8
    for i in range(n):
        rand_sex.append(random.randint(0,1))
    rand_treatment=[]
    n=8
    for i in range(n):
        rand_treatment.append(random.randint(0,1))
    rand_outcome=[]
    n=8
    for i in range(n):
        rand_outcome.append(random.randint(0,1))
    rand_age=[]
    n=8
    for i in range(n):
        rand_age.append(random.randint(10,90))
    rand_race=[]
    n=8
    for i in range(n):
        rand_race.append(random.randint(1,4))

    rand_data = {"Sex": rand_sex,
     "Race": rand_race,
     "Age": rand_age ,
     "Treatment": rand_treatment,
     "Outcome": rand_outcome
     }

    return rand_data

def randomise():
    st.session_state["randomise_clicked"] = True
    st.session_state["randomised_data"] = randomise_data()

st.button("Randomise", on_click=randomise)


# button to randomize
if st.session_state["randomise_clicked"] == True:
    data = pd.DataFrame(st.session_state["randomised_data"])
else: data = pd.DataFrame(
    {"Sex": [0,1,0,1,0,1,0,1],
     "Race": [1, 2, 3, 4, 1, 2, 3, 4],
     "Age":[20, 65, 82, 31, 14, 41, 39, 18],
     "Treatment":[0,0,0,0,1,1,1,1],
     "Outcome":[1,1,0,0,1,1,0,0]
     }
)
    
if "synthesised_data" not in st.session_state:
    st.session_state["synthesised_data"] = data


st.dataframe(data)



# DEF CART FUNCTIONS :
class CARTDataSynthesizer:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    # gini Index for categorical splitting
    def gini_index(self, groups, dataset):
        n_instances = sum([len(group) for group in groups])
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            # score the group based on feature proportions
            unique_classes = [list(col) for col in np.array(dataset).T]  # col are the dataset variables
            for values in unique_classes:
                proportion = values.count(values[0]) / size
                score += proportion * proportion
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # split the dataset based on a feature and a value
    def test_split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if isinstance(value, (int, float)):  # for numerical features (age)
                if row[index] < value:
                    left.append(row)
                else:
                    right.append(row)
            else:  # for categorical features (everything else)
                if row[index] == value:
                    left.append(row)
                else:
                    right.append(row)
        return left, right

    # select the best split point for the dataset
    def get_best_split(self, dataset):
        best_index, best_value, best_score, best_groups = None, None, float('inf'), None
        for index in range(len(dataset[0])):  # by feature
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, dataset)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    # recursive split
    def split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= self.min_samples_split:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_best_split(left)
            self.split(node['left'], depth + 1)
        if len(right) <= self.min_samples_split:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_best_split(right)
            self.split(node['right'], depth + 1)

    # create terminal nodes (random sample from the group)
    def to_terminal(self, group):
        return random.choice(group)  # randomly select a row from the group

    # build the decision tree using the features
    def fit(self, dataset):
        self.tree = self.get_best_split(dataset)
        self.split(self.tree, 1)

    # traverse the tree to synthesize a new sample
    def traverse_tree(self, node):
        while isinstance(node, dict):  # while we haven't hit a terminal node
            if random.random() < 0.5:
                node = node['left']
            else:
                node = node['right']
        return node  # leaf node is a synthetic row

    # synthesize a new dataset by traversing the tree
    def synthesize(self, num_samples):
        new_data = []
        for _ in range(num_samples):
            new_row = self.traverse_tree(self.tree)
            new_data.append(new_row)
        return np.array(new_data)




# convert the df to a NumPy array 
dataset = data.values

# start CART synthesizer and fit the model
synthesizer = CARTDataSynthesizer(max_depth=3, min_samples_split=2)
synthesizer.fit(dataset)

# synthesize a new dataset 
synthetic_data = synthesizer.synthesize(num_samples=8)

# convert back to df 
synthetic_df = pd.DataFrame(synthetic_data, columns=data.columns)
print("Synthetic Data:")
print(synthetic_df)




# for lin/log regression:
# train a logistic regression model for binary variables
def train_logistic_model(data, target):
    X = data.drop(columns=[target])
    y = data[target]
    model = LogisticRegression()
    model.fit(X, y)
    return model

# train a linear regression model for continuous or categorical variables
def train_linear_model(data, target):
    X = data.drop(columns=[target])
    y = data[target]
    model = LinearRegression()
    model.fit(X, y)
    return model

# function to synthesize data based on the fitted model
def synthesize_data_logistic(model, X):
    probabilities = model.predict_proba(X)[:, 1]
    return np.random.binomial(1, probabilities)

def synthesize_data_linear(model, X):
    predictions = model.predict(X)
    # adding noise to linear predictions to make them more realistic
    noise = np.random.normal(0, 1, size=predictions.shape)
    return predictions + noise

# fill missing values with sampled data from original dataset
def fill_missing_values(X, reference_data):
    X_filled = X.copy()
    for col in X.columns:
        if X[col].isnull().any():
            X_filled[col].fillna(reference_data[col].sample(n=1).values[0], inplace=True)
    return X_filled

# fit models to each column based on type 
def create_synthetic_data(data, num_samples=5):
    synthetic_data = pd.DataFrame(columns=data.columns)
    
    # list variables by type
    binary_columns = ["Sex", "Treatment", "Outcome"]
    continuous_columns = ["Age", "Race"]
    
    models = {}
    
    # train models 
    for col in binary_columns:
        models[col] = train_logistic_model(data, col)
    for col in continuous_columns:
        models[col] = train_linear_model(data, col)
    
    # start generating synthetic data
    for _ in range(num_samples):
        new_sample = {}

        # predict each column based on the trained model and other variables
        for col in binary_columns:
            # use the same columns for prediction as used for fitting
            if new_sample:
                X = pd.DataFrame([new_sample], columns=data.drop(columns=[col]).columns)
                X = fill_missing_values(X, data)  # fill any missing values with original data samples
            else:
                X = data.drop(columns=[col]).sample(n=1)
            new_sample[col] = int(synthesize_data_logistic(models[col], X)[0])

        for col in continuous_columns:
            if new_sample:
                X = pd.DataFrame([new_sample], columns=data.drop(columns=[col]).columns)
                X = fill_missing_values(X, data)  # fill any missing values with original data samples
            else:
                X = data.drop(columns=[col]).sample(n=1)
            new_sample[col] = synthesize_data_linear(models[col], X)[0]
            # round the predictions 
            new_sample[col] = int(np.round(synthesize_data_linear(models[col], X)[0]))
        
        # append the new sample row to the synthetic data 
        synthetic_data = pd.concat([synthetic_data, pd.DataFrame([new_sample])], ignore_index=True)
    
    return synthetic_data

# create the synthesize function with three methods as options
def synthesise(data, method):
    if method == "CART":
        dataset = data.values
        synthesizer = CARTDataSynthesizer(max_depth=3, min_samples_split=2)
        synthesizer.fit(dataset)
        synth_data = pd.DataFrame(synthetic_data, columns=data.columns)
        return synth_data
    if method == "Random Sampling":
        n = len(data)
        return pd.DataFrame(
    {"Sex": random.choices(data["Sex"], k=n),
     "Race": random.choices(data["Race"], k=n),
     "Age":random.choices(data["Age"], k=n),
     "Treatment":random.choices(data["Treatment"], k=n),
     "Outcome":random.choices(data["Outcome"], k=n)
     })
    if method == "Linear/Logistic Regression":
       synth_data = create_synthetic_data(data, num_samples=8)
       return synth_data
    
def synthesise_clicked(data, method): # synthesize button
    st.session_state["synthesise_clicked"] = True
    st.session_state["synthesised_data"] = synthesise(data, method)


method = st.radio('Select Synthesis Method:', ['CART','Random Sampling', 'Linear/Logistic Regression'])
new_data = data

st.button("Synthesise", on_click=synthesise_clicked, args=[data, method])
        
new_data = st.session_state["synthesised_data"]

st.dataframe(new_data)





# visualize the results and compare
if st.button('Analyse'):



    variables = ["Age"]
    df1 = data["Age"]
    df2 = new_data["Age"]


    # colors for the two data sources
    colors = ['lightblue', 'lightgreen']

    # create boxplot
    plt.figure(figsize=(12, 6))

    plt.boxplot(df1, 
                positions=[2], widths=0.6,
                patch_artist=True, boxprops=dict(facecolor=colors[0]), 
                medianprops=dict(color='blue'))
    plt.boxplot(df2, 
                positions=[2 + 0.6], widths=0.6,
                patch_artist=True, boxprops=dict(facecolor=colors[1]), 
                medianprops=dict(color='green'))

    plt.gca().set_xticks([])

    # title and labels
    plt.title('Boxplot Comparison of Age in Original and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Age')

    # legend
    plt.legend([plt.Line2D([0], [0], color='lightblue', lw=4),
                plt.Line2D([0], [0], color='lightgreen', lw=4)],
            ['Original Data', 'Synthetic Data'])

    # show the plot
    plt.tight_layout()
    plt.savefig("box.png")
    plt.show()




    # display the image
    st.title('Dataset Comparisons')

    image = Image.open('box.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')




    # binary variables

    # normalize the value counts to get proportions
    proportions1 = data['Sex'].value_counts(normalize=True)
    proportions2 = new_data['Sex'].value_counts(normalize=True)

    # df for easier plotting
    comparison_df = pd.DataFrame({'Original Data': proportions1, 'Synthetic Data': proportions2})



    # plot stacked bar plot 
    ax = comparison_df.T.plot(
        kind='bar', 
        stacked=True, 
        width=1, 
        color=['pink', 'mediumpurple'], 
        edgecolor='black',  
        figsize=(10,6)
    )


    # set x-axis labels to horizontal
    plt.xticks(rotation=0)


    ax.legend(['Female', 'Male'], loc='upper right')

    # title and labels
    plt.title('Stacked Barplot Comparison of Sex in Original and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Porportion of Participants')


    # show the plot
    plt.tight_layout()
    plt.savefig("bar_chart.png")
    plt.show()



    # display the image

    image = Image.open('bar_chart.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')


    # normalize the value counts to get proportions
    proportions1 = data['Treatment'].value_counts(normalize=True)
    proportions2 = new_data['Treatment'].value_counts(normalize=True)

    # dffor easier plotting
    comparison_df = pd.DataFrame({'Original Data': proportions1, 'Synthetic Data': proportions2})



    # plot stacked bar plot 
    ax = comparison_df.T.plot(
        kind='bar', 
        stacked=True, 
        width=1, 
        color=['pink', 'mediumpurple'], 
        edgecolor='black',  
        figsize=(10,6)
    )


    # set x-axis labels to horizontal
    plt.xticks(rotation=0)


    ax.legend(['Untreated', 'Treated'], loc='upper right')

    # title and labels
    plt.title('Stacked Barplot Comparison of Treatment in Original and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Porportion of Participants')


    # show the plot
    plt.tight_layout()
    plt.savefig("bar_chart_treat.png")
    plt.show()




    # display the image

    image = Image.open('bar_chart_treat.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')







    # normalize the value counts to get proportions
    proportions1 = data['Outcome'].value_counts(normalize=True)
    proportions2 = new_data['Outcome'].value_counts(normalize=True)

    # df for easier plotting
    comparison_df = pd.DataFrame({'Original Data': proportions1, 'Synthetic Data': proportions2})



    # plot stacked bar plot
    ax = comparison_df.T.plot(
        kind='bar', 
        stacked=True, 
        width=1, 
        color=['pink', 'mediumpurple'], 
        edgecolor='black',  
        figsize=(10,6)
    )


    # set x-axis labels to horizontal
    plt.xticks(rotation=0)


    ax.legend(['Not Present', 'Present'], loc='upper right')

    # title and labels
    plt.title('Stacked Barplot Comparison of Outcome in Original and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Porportion of Participants')


    # show the plot
    plt.tight_layout()
    plt.savefig("bar_chart_out.png")
    plt.show()




    # display the image

    image = Image.open('bar_chart_out.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')










    # categorical column - race
    # get the proportions of each category in both datasets
    categories = [1, 2, 3, 4]
    proportions1 = data['Race'].value_counts(normalize=True).reindex(categories, fill_value=0)
    proportions2 = new_data['Race'].value_counts(normalize=True).reindex(categories, fill_value=0)

    # df for easier plotting
    comparison_df = pd.DataFrame({'Original Data': proportions1, 'Synthetic Data': proportions2})


    # reindex df for all four categories

    comparison_df = comparison_df.reindex(categories).fillna(0)

   

    # plot stacked bar plot
    ax = comparison_df.T.plot(
        kind='bar', 
        stacked=True, 
        width=1, 
        color=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon'],  # colors for each category
        edgecolor='black', 
        figsize=(10,6)
    )

    # set x-axis labels to horizontal
    plt.xticks(rotation=0)

    # update the legend to display the category names
    categories_names = ["Asian/Asian British", "Black/Black British", "Mixed Race", "White"]
    ax.legend(categories_names, loc='upper right')


    # title and labels
    plt.title('Stacked Barplot Comparison of Race in Original and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Porportion of Participants')


    # show the plot
    plt.tight_layout()
    plt.savefig("bar_chart_race.png")
    plt.show()


    # display the image

    image = Image.open('bar_chart_race.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')
