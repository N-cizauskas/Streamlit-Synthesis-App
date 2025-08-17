# import packages

import streamlit as st

import pandas as pd

import random



import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression

from datetime import datetime

import gspread

from oauth2client.service_account import ServiceAccountCredentials

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Home", "Virtual Synthesizer", "What are Synthetic Controls?", "Synthetic Control Quality", "Bayesian VS Synthetic", "About Me", "Zines", "Leave Feedback"])

with tab1:
    st.title("Welcome to the World of Synthetic Controls!") # home page title
    st.write("Please navigate using the tabs at the top.")
    st.image("Home.gif")
with tab2:
    # design page
    if "randomize_clicked" not in st.session_state:
        st.session_state["randomize_clicked"] = False

    if "randomized_data" not in st.session_state:
        st.session_state["randomized_data"] = []

    if "synthesize" not in st.session_state:
        st.session_state["synthesize"] = False

    def reset():
        for key in st.session_state.keys():
            del st.session_state[key]


    # randomization of dataset
    st.title("Let's Create Synthetic Data!") # main title

    st.header('Try my virtual data synthesizer below:')

    st.subheader('1) Randomizer:')
    st.write('The first section creates the data. In the virtual synthesizer, this data is randomized. This will form our "original data". Click the randomize button to begin!')






    st.button("Reset", on_click=reset, type="primary")

    # randomization lists
    def randomize_data():
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

    def randomize():
        st.session_state["randomize_clicked"] = True
        st.session_state["randomized_data"] = randomize_data()

    st.button("Randomize", on_click=randomize)


    # button to randomize
    if st.session_state["randomize_clicked"] == True:
        data = pd.DataFrame(st.session_state["randomized_data"])
    else: data = pd.DataFrame(
        {"Sex": [0,1,0,1,0,1,0,1],
        "Race": [1, 2, 3, 4, 1, 2, 3, 4],
        "Age":[20, 65, 82, 31, 14, 41, 39, 18],
        "Treatment":[0,0,0,0,1,1,1,1],
        "Outcome":[1,1,0,0,1,1,0,0]
        }
    )
        
    if "synthesized_data" not in st.session_state:
        st.session_state["synthesized_data"] = data


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

                # predict the continuous value
                predicted_value = synthesize_data_linear(models[col], X)[0]
        
                # round
                predicted_value = int(np.round(predicted_value))
        
                # truncation 
                if col == "Race":
                    new_sample[col] = np.clip(predicted_value, 1, 4)  # Ensure Race is between 1 and 4
                elif col == "Age":
                    new_sample[col] = np.clip(predicted_value, 10, 90)  # Ensure Age is between 10 and 90
                else:
                    new_sample[col] = predicted_value  
            
                # ensure no NAs after truncation
                new_sample[col] = new_sample[col] if pd.notna(new_sample[col]) else fill_missing_values(pd.DataFrame([new_sample]), data)[col].values[0]


                # append new synthetic data 
            synthetic_data = pd.concat([synthetic_data, pd.DataFrame([new_sample])], ignore_index=True)

        return synthetic_data

    # create the synthesize function with three methods as options
    def synthesize(data, method):
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
        
    def synthesize_clicked(data, method): # synthesize button
        st.session_state["synthesize_clicked"] = True
        st.session_state["synthesized_data"] = synthesize(data, method)




    st.subheader('2) Synthesizer:')
    st.write('The second section synthesizes new data based on the original data created in the first section. There are three options for synthesis method: CART (Classification and Regression Trees), random sampling, and linear/logisitc regression.  Select your method of synthesis and then click the button to synthesize new data!')

    st.markdown("- CART: a decision tree that can create a new dataset by recursively partitioning the data based on feature splits and then assigning synthetic values to new points using the statistical properties of the data in each terminal leaf node")
    st.markdown("- Random sampling: selects data points at random from each column in the original data until a new dataset of the desired size is formed")
    st.markdown("- Linear/logistic regression: uses models trained on the original dataset to generate a new dataset by predicting continuous and categorical variables (via linear regression) and binary variables (via logistic regression) based on relationships learned from the original data")




    method = st.radio('Select Synthesis Method:', ['CART','Random Sampling', 'Linear/Logistic Regression'])
    new_data = data

    


    st.button("Synthesize", on_click=synthesize_clicked, args=[data, method])
            
    new_data = st.session_state["synthesized_data"]

    st.dataframe(new_data)



    st.subheader('3) Analysis:')
    st.write('The third section allows you to visually compare the original dataset to the new, synthetic dataset you have created.  Continuous variables are compared using a box plot and cateogical/binary variables are compared using stacked bar plots. Click the analysis button to generate graphs!')


    # visualize the results and compare
    if st.button('Analyze'):



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
    

        image = Image.open('box.png')
        st.image(image)



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
        st.image(image)


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
        st.image(image)







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
        st.image(image)










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
        st.image(image)
with tab3:
    st.title("What are Synthetic Controls?")
    st.header("Synthetic Data")
    st.subheader("Types of Synthetic Data")
    st.write('Synthetic data widely refers to data that is artificially generated, and not measured through real-world events. Synthetic controls fall under the umbrella of synthetic data, along with similar (and overlapping) methods, such as digital twins. The diagram below shows some of the other definitions under the synthetic data umbrella.')
    st.image("names.png")
    st.subheader("Synthetic Controls")
    st.write("Synthetic data can used to substitute the control arm of a clinical randomized control trial (RCT). Synthetic control arms are generated based on external (to the study) patient data with similar attributes to the experimental group. They are designed based on historical RCTs, observational study data, or external data.")
    st.header("Why are Synthetic Controls Useful?")
    st.subheader("Participant Recruitment")
    st.write("Synthetic control arms are useful for any RCT that would otherwise be restricted by participant recruitment.  This includes rare diseases and pediatric trials. Synthetic controls can reduce the number of participants needed to recruit by removing the need to allocate to the control arm.")
    st.subheader("Cost and Speed")
    st.write("Synthetic controls can reduce the cost and speed it takes to perform a RCT by reducing the number of participants needed to recruit. This is useful in specific circumstances that require a very fast, cheap trial – such as pandemics.")
    st.subheader("Ethics")
    st.write("Synthetic controls can also be used in cases where it is unethical to assign participants to the control group, such as situations where no current quality standard of care exists or the standard of care is expected to be significantly less effective based on proof of concept trials.")
    st.header("Types of Synthetic Controls")
    st.subheader("Data Sources")
    st.write("Three different data sources can be used to create synthetic controls:")
    st.markdown("- Historical RCTs with similar populations, demographics, and measured outcome")
    st.markdown("- Observational or single-arm studies ")
    st.markdown("- External data (e.g. electronic health records, routinely collected data, survey results)")
    st.subheader("Synthesis Methods")
    st.write("Synthetic controls can be generated from a variety of methods, including random sampling, linear and logistic regression, and machine learning techniques.")
    st.subheader("Why Not Just use a Historical Control Arm?")
    st.write("Some studies do use historical control arms in the place of a new or synthetic one. One downside of this are that you need to recollect participant consent to use their data in a new trial.  In addition to this, there are other benefits of using synthetic controls over historical ones. You can amplify characteristics of interest (i.e. age, sex) in your synthetic controls. You can also combine multiple data sources to create one synthetic control arm, resulting in a decrease of overall bias.")
with tab4:
    st.title("Does the Data Source Used Affect the Quality of a Synthetic Control?") # main title
    st.header("What We Know So Far")
    st.subheader("Data Source Quality")
    st.write("Previous studies have stated that there is a quality tradeoff in the choice of data source used to create a synthetic control, with RCTs being the highest quality, observational studies being less good, and external data having the least quality results. This premise is compatible with the hierarhcy of evidence, where RCTs are the gold standard.")
    st.image("hierarchy.png")
    st.subheader("How Much Quality is Lost in the Tradeoff?")
    st.write("We don't know.")
    st.subheader("What Does Quality Refer to?")
    st.write("We don't know.")
    st.header('My Work')
    st.subheader('How to Measure Quality')
    st.write("I propose two metrics to measure the quality of a synthetic control:")
    st.markdown(" - Respone rate maintenance")
    st.markdown(" - Closeness to original data")
    st.write("The response rate maintenance ensures the rate of response to treatment is the kept the same in the control group throughout the synthesis process.  This is calculated by dividing the number of responders to a treatment in the control group by the total size of the control group.")
    st.write("The closenss to original data compares how well the synthetic data reflects the original data. This is measured by using the standard mean difference, and each variable is calculated individually.")
    st.subheader("Testing the Quality")
    st.write("To test the quality differences between synthetic controls created from different data sources, three distinct synthetic control arms were generated from two case studies:")
    st.markdown(" - COVID-19 and the BNT162b vaccine")
    st.markdown(" - Crohn's disease and Ustekinumab")
    st.write("These were selected based on availability of matched studies featuring similar populations in similar years.")
    st.write("Summary statistics were exported from the original data sources and used to simulate individual-level patient dataframes. These were used as the original data sources. Synthetic datasets were then created from the simulated dataframes. The diagram below shows the flow of data during the study.")
    st.image("data_flow.png")
    st.write("Three data types were tested, using three synthesis methods, in four different sample sizes. This resulted in 36 unique scenarios. Each scenario was simulated 10,000 times, resulting in a total of 360,000 simulations.")
    st.image("study_design.png")
    st.write("Realistic sample size refers to using the exact sample sizes from the original studies.")
    st.header("Results")
    st.write("The following graphs show the results for the CART synthesis method only. There were not significant differences between the synthesis methods.")
    st.subheader("COVID-19")
    st.write("The response rate maintenance analysis showed the largest range flucuations in observational data.")
    st.image("cov_crr.jpg")
    st.write("The closeness to original data analysis showed the largest SMD range for all data types in the smallest sample size scenario (scenario 1) and a larger RCT range compared to the other data types in the realistic sample size scenario.")
    st.image("cov_smd.jpg")
    st.subheader("Crohn's Disease")
    st.write("The response rate maintenance analysis showed the largest difference between original and synthetic data in the smallest sample size scenario.")
    st.image("cro_crr.jpg")
    st.write("Similarly, the closeness to original data analysis again showed the largest SMD range for all data types in the smallest sample size scenario.")
    st.image("cro_smd.jpg")
    st.header("Conclusions")
    st.subheader("What Does it Mean?")
    st.write("The quality of synthetic controls may change depending on:")
    st.markdown(" - sample size")
    st.markdown(" - Data type")
    st.subheader("What Should I do About it?")
    st.write("There is no evidence to suggest the assumed hierarchy of quality, with RCTs being the best and external data being the worst. If you are creating your own synthetic controls for a clinical study, you may want to consider sources outside of the traditional RCTs. You should also consider checking the response rate maintenance and closeness to the original data for any synthetic controls you create.  If you are reading other synthetic control studies, keep an eye out for their reported sample sizes and data sources.  The quality of a synthetic control may also differ between disease types, as it did between COVID-19 and Crohn's disease; this is important to consider when reviewing literature.")

    
    
    
   

with tab5:
    st.title("Which is Better: Bayesian Dynamic Borrowing or Synthetic Control Methods?") # main title
    st.header("What are They?")
    st.write("Bayesian dynamic borrowing (BDB) and synthetic control methods (SCM) are both used in clinical trial design to reduce dependence on large randomized control arms.")
    st.subheader("Bayesian Dynamic Borrowing")
    st.write("BDB creates a prior distribution informed by historical data, which carries information from the sample sizes and outcomes of previous trials with similar populations and the same standard of care/placebo. This prior can be used to estimate the maximum number of patients needed in a control arm to achieve the required power and type 1 error threshold for the study. In dynamic borrowing, the influence of the prior is adjusted based on the similarity between the historical data and observed trial data. If the control group begins to show results incongruent with expectations while using a commensurate or robust MAP (maximum a posteriori) prior, the prior can be downweighted.")
    st.write("Benefits:")
    st.markdown(" - can account for variability between historical studies")
    st.markdown( " - can adjust influence during the trial based on preliminary results")
    
    
    st.write("Drawbacks:")
    st.markdown(" - still need to recruit a small number of real control arm participants")
    st.write("The product of BDB is a personalized trial design that gives a specific sample size aim for the control group.")
    
    st.subheader("Synthetic Control Methods")
    st.write("SCM generate a control arm based on previous study data. These control arms have similar distributions, means, and treatment effects to the studies they are based on. Popular approaches include propensity score matching on baseline covariates or linear regression weighting, although decision trees like CART (classification and regression tree) can also be used.  The generated data can be used to augment an existing control arm, increasing the overall sample size; this is referred to as a hybrid synthetic control. The generated data can also be used directly as a full synthetic control arm. In this case, a randomised control trial would recruit only for the treatment arm and the findings would be compared to the synthetic control arm directly.")
    st.write("Benefits:")
    st.markdown(" - one-and-done approach for simplicity")
    st.markdown(" - no need for control arm recruitment whatsoever")
    st.markdown(" - can incorpate covariate influence (i.e. sex, race)")
    st.markdown(" - can increase distribution of certain characteristics (e.g. lowering the distribution of age in controls produced for pediatric studies)")
    
    st.write("Drawbacks")
    st.markdown(" - will not be downweighted if controls are not representative")
    st.markdown(" - less established method for regulatory approval")
    st.write("The product of SCM is the control group itself.")

    st.header("Comparison")
    st.subheader("Previous Comparisons")
    st.write("There are none.")
    
    st.header("My Work")
    st.subheader("Metrics of Comparison")
    st.write("Response rate, power, and type 1 error are used to compare these methods.")
    st.subheader("Study Design")
    st.write("Pediatric atopic dermatitis was chosen as a case study for comparing BDB and SCM. Pediatric trials are often difficult to recruit for, leading to reliance on one of these methods.")
    st.write("The same six historical RCTs were selected for use in both methods. Placebo group sample sizes and response rates were included from each study.")
    st.write("A MAP prior was created for the BDB method.  The CART method was used for the SCM method. To calculate the power and type 1 error in the SCM method, the synthesis process was repeated 10,000 times. The sample sizse for the synthetic controls was set to equal the mean of the historical study sample sizes.")

    st.header("Results")
    st.subheader("Response Rate")
    st.write("The response rate of both BDB and SCM were based on the historical studies.  The forest plots below show the differences alongside the historical study rates. ")
    st.image("forest.png")
    st.write("The MAP prior had a mean of 0.2 with a large credible interval from 0.02 to 0.71, and the synthetic control had a mean of 0.25 with a confidence interval from 0.16 to 0.35.  Note that the credible interval and confidence interval are not a direct comparison due to being Bayesian and frequentist respectively: a credible interval means there is a 95% probability that the true parameter lies within that range, and a confidence interval means that in 100 repeated samples from the population, the true population parameter would be within that range 95 times. The methodologies incorporate the historical information differently: BDB is looking at the cumulative information from the intervals of previous studies, while SCM is creating a new interval based on the historical information.  ")
    st.subheader("Power and Type 1 Error")
    st.write("BDB produced a robust power of 0.580 and a robust type 1 error rate of 0.026. SCM produced a power of 0.676 and a type 1 error rate of 0.027.")

    st.header("Conclusion")
    st.subheader("What Happened?")
    st.write("The type 1 error rate was similar between the two methods. SCM showed an increased power over BDB.  The response rates had a similar mean in both studies, but the credible interval in BDB was wider than the confidence interval in SCM. ")
    st.subheader("What Does it Mean?")
    st.write("While the increased power in SCM may lead some to choose that method, the real answer to the question 'which is better?' is more nuanced. BDB and SCM have different use cases in the real world.  BDB is used to determine a maximum sample size needed for a control group, and SCM is used to generate the control group itself. Practically, determining which method is “better” will depend on the specific needs of the study, such as how difficult recruitment is. In cases where recruitment for the control group is near impossible, SCM is a much better choice. In cases where recruitment as a whole is a challenge but allocation to the control group is not an issue, BDB may have more regulatory appeal. Overall, the results provided evidence that both methods are viable.  ")



with tab6:
    new_title = '<p style="font-size: 35px;"><strong style="font-weight: 900;">About Me!</strong></p>'

    st.markdown(new_title, unsafe_allow_html=True)
    st.write('This is me! Please come ask me questions about my work if you see me around!')
    image = Image.open('me.jpg')
    # Resize the image
    new_size = (300, 300)  # Width, Height
    image = image.resize(new_size)
    st.image(image)
    st.write('(she/her)')
    st.subheader('Current Position')
    st.write('Nicole Cizauskas')
    st.write('Newcastle University, PGR in Biostatistics')
    st.write('Biostatistics Research Group')
    st.subheader('Contact Me:')
    st.write('Email: n.cizauskas@newcastle.ac.uk')
    st.write('Github: https://github.com/N-cizauskas')

with tab7:
    st.title("Check Out My Zines!")
    st.write("These zines were created to help advertise my talk at ISCB 2025.")
    st.image("zine1.png")
    st.image("zine2.png")
with tab8:

    # Define the scope for accessing Google Sheets and Google Drive
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # Load credentials from Streamlit secrets
    credentials_info = st.secrets["google_sheets"]

    # Authorize the client using the credentials from secrets
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
    client = gspread.authorize(credentials)

    # Open your Google Sheet
    sheet = client.open("Feedback").sheet1

    # Streamlit app to collect feedback
    st.header('Feedback Form')
    st.write('I would love to hear your thoughts on this app or any of my work!')

    feedback = st.text_area("Enter your comments or feedback here:")

    if st.button("Submit"):
        if feedback:
        # Append the feedback and a timestamp to the Google Sheet
            sheet.append_row([feedback, str(datetime.now())])
            st.write("Response submitted. Thank you for your feedback!")
            
        else:
            st.write("Please enter your feedback before submitting.")

    
   









