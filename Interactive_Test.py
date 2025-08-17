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
    st.write("Previous studies have stated that there is a quality tradeoff in the choice of data source used to create a synthetic control, with RCTs being the highest quality, observational studies being less good, and external data having the least quality results. This premise is compatible with the hierarhcy of clinical data, where RCTs are the gold standard.")
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

    
    
    
    st.header('References')
    st.markdown("""
1. Bouttell J, Craig P, Lewsey J, Robinson M, Popham F. Synthetic control methodology as a tool for evaluating population-level health interventions. J Epidemiol Community Health. 2018 Aug 1;72(8):673–8. 
2. Thorlund K, Dron L, Park JJH, Mills EJ. Synthetic and External Controls in Clinical Trials – A Primer for Researchers. Clin Epidemiol. 2020 May 8;12:457–67. 
3. Ali Awan A. What is Synthetic Data? [Internet]. 2023 [cited 2024 Feb 28]. Available from: https://www.datacamp.com/blog/what-is-synthetic-data
4. Lyman JP, Doucette A, Zheng-Lin B, Cabanski CR, Maloy MA, Bayless NL, et al. Feasibility and utility of synthetic control arms derived from real-world data to support clinical development. J Clin Oncol. 2022 Feb;40(4_suppl):528–528. 
5. Goldsack J. Synthetic control arms can save time and money in clinical trials [Internet]. STAT. 2019 [cited 2023 Oct 2]. Available from: https://www.statnews.com/2019/02/05/synthetic-control-arms-clinical-trials/
6. Sibbald B, Roland M. Understanding controlled trials: Why are randomised controlled trials important? BMJ. 1998 Jan 17;316(7126):201. 
7. Akobeng AK. Understanding randomised controlled trials. Arch Dis Child. 2005 Aug 1;90(8):840–4.
8. Hariton E, Locascio JJ. Randomised controlled trials—the gold standard for effectiveness research. BJOG Int J Obstet Gynaecol. 2018 Dec;125(13):1716. 
9. University College London. MRC Clinical Trials Unit at UCL. 2018 [cited 2024 Feb 19]. What is an observational study? Available from: https://www.mrcctu.ucl.ac.uk/patients-public/about-clinical-trials/what-is-an-observational-study/
10. Song JW, Chung KC. Observational Studies: Cohort and Case-Control Studies. Plast Reconstr Surg. 2010 Dec;126(6):2234–42
11. Aaser M, McElhaney D. Harnessing the power of external data. McKinsey Digit. 2021;
12. Burger HU, Gerlinger C, Harbron C, Koch A, Posch M, Rochon J, et al. The use of external controls: To what extent can it currently be recommended? Pharm Stat. 2021;20(6):1002–16. 
13. Chevret S, Timsit JF, Biard L. Challenges of using external data in clinical trials- an illustration in patients with COVID-19. BMC Med Res Methodol. 2022 Dec 15;22(1):321. 
14. Burcu M, Dreyer NA, Franklin JM, Blum MD, Critchlow CW, Perfetto EM, et al. Real-world evidence to support regulatory decision-making for medicines: Considerations for external control arms. Pharmacoepidemiol Drug Saf. 2020;29(10):1228–35. 
15. Lyman JP, Doucette A, Zheng-Lin B, Cabanski CR, Maloy MA, Bayless NL, et al. Feasibility and utility of synthetic control arms derived from real-world data to support clinical development. J Clin Oncol. 2022 Feb;40(4_suppl):528–528. 
16. Commissioner O of the. FDA. FDA; 2020 [cited 2023 Oct 2]. Statement from FDA Commissioner Scott Gottlieb, M.D., on FDA’s new strategic framework to advance use of real-world evidence to support development of drugs and biologics. Available from: https://www.fda.gov/news-events/press-announcements/statement-fda-commissioner-scott-gottlieb-md-fdas-new-strategic-framework-advance-use-real-world
17. Berry DA, Elashoff M, Blotner S, Davi R, Beineke P, Chandler M, et al. Creating a synthetic control arm from previous clinical trials: Application to establishing early end points as indicators of overall survival in acute myeloid leukemia (AML). J Clin Oncol. 2017 May 20;35(15_suppl):7021–7021. 
18. Blondeau K, Schneider A, Ngwa I. A synthetic control arm from observational data to estimate the background incidence rate of an adverse event in patients with Alzheimer’s disease matched to a clinical trial population. Alzheimers Dement. 2020;16(S10):e043657. 
19. Ko YA, Chen Z, Liu C, Hu Y, Quyyumi AA, Waller LA, et al. Developing a synthetic control group using electronic health records: Application to a single-arm lifestyle intervention study. Prev Med Rep. 2021 Dec 1;24:101572.
20. GOV.UK. GOV.UK. 2021 [cited 2024 Mar 8]. Demographic data for coronavirus (COVID-19) testing (England): 28 May to 26 August. Available from: https://www.gov.uk/government/publications/demographic-data-for-coronavirus-testing-england-28-may-to-26-august/demographic-data-for-coronavirus-covid-19-testing-england-28-may-to-26-august
21. Yang ZR, Jiang YW, Li FX, Liu D, Lin TF, Zhao ZY, et al. Efficacy of SARS-CoV-2 vaccines and the dose–response relationship with three major antibodies: a systematic review and meta-analysis of randomised controlled trials. Lancet Microbe. 2023 Apr 1;4(4):e236–46. 
22. Bernal JL, Andrews N, Gower C, Stowe J, Robertson C, Tessier E, et al. Early effectiveness of COVID-19 vaccination with BNT162b2 mRNA vaccine and ChAdOx1 adenovirus vector vaccine on symptomatic disease, hospitalisations and mortality in older adults in England [Internet]. medRxiv; 2021 [cited 2024 Feb 26]. p. 2021.03.01.21252652. Available from: https://www.medrxiv.org/content/10.1101/2021.03.01.21252652v1              
23. NHS. COVID-19 Vaccinations Archive [Internet]. 2021 [cited 2024 Feb 28]. Available from: https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-vaccinations/covid-19-vaccinations-archive/
24. GOV.UK. Cases in England | Coronavirus in the UK [Internet]. 2021 [cited 2024 Feb 28]. Available from: https://coronavirus.data.gov.uk/details/cases?areaType=nation&areaName=England
25. Nowok B, Raab GM, Dibben C. synthpop: Bespoke Creation of Synthetic Data in R. J Stat Softw. 2016 Oct 28;74:1–26.
""")

with tab5:
    st.title("Which is Better: Bayesian Dynamic Borrowing or Synthetic Control Methods?") # main title

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

    
   









