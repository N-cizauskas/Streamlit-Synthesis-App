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

tab1, tab2, tab3, tab4 = st.tabs(["Virtual Synthesizer", "About my Project", "About Me", "Leave Feedback"])


with tab1:
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
                new_sample[col] = synthesize_data_linear(models[col], X)[0]
                # round the predictions 
                new_sample[col] = int(np.round(synthesize_data_linear(models[col], X)[0]))
            
            # append the new sample row to the synthetic data 
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

with tab2:
    #new_title = '<p style="font-size: 35px;"><strong style="font-weight: 900;">Analyzing Differences in Data Quality of Synthetic Data Generated from Randomized Control Trials, Observational Studies, and External Data</strong></p>'

    #st.markdown(new_title, unsafe_allow_html=True)
    st.title("Analyzing Differences in Data Quality of Synthetic Data Generated from Randomized Control Trials, Observational Studies, and External Data") # main title

    st.header('What is Synthetic Data?')
    st.subheader('Synthetic Data')
    st.write('Synthetic data widely refers to data that is artifically generated, and not measured through real-world events.')
    st.subheader('Synthetic Data for Clinical Trials')
    st.write('Synthetic data can be implemented in clinical trials through the use of synthetic control arms. Synthetic control arms are control groups generated based on real-world patient data with similar attributes to the experimental group (1). They are typically designed based on previous clinical trial data, observational study data, or external data (2). ')
    st.subheader('Synthetic Data VS Simulated Data')
    st.write('Colloquially, synthetic data is algorithmic and data-driven in its generation and simulated data requires specific real-world characteristics to frame the data such as probabilities of key variables (3).')


    st.header('Why are Synthetic Controls Useful for Clinical Trials?')
    st.subheader('Rare Diseases')
    st.write('Synthetic control arms are particularly useful in randomized control trials (RCTs) that would otherwise be restricted by participant recruitment, cost, or ethics (4,5) - such as rare disease clinical trials. These synthetic control arms can cut the number of participants required to run an RCT by providing data to represent the control or placebo group. This also solves the problem  of ethics in cases where giving participants the control or placebo could be detrimental to their health.')
    st.subheader('Cost Reduction')
    st.write('Synthetic control arms can also be used to reduce the cost of otherwise expensive to run clinical trials by reducing the amount of real participants needed.')



    st.header('What Types of Synthetic Data are There?')
    st.subheader('Methods')
    st.write('There are many ways of generating new data. One of the most popular methods for generating synthetic data is CART (classification and regression trees). In the virtual synthesizer, random sampling and linear/logistic regression were also available. This is because CART is a nonparametric model and linear/logistic regression modelling is a parametric model. Random sampling provides a "control" method to compare to.  A reminder of the definitions of each method can be found below:')
    st.markdown("- CART: a decision tree that can create a new dataset by recursively partitioning the data based on feature splits and then assigning synthetic values to new points using the statistical properties of the data in each terminal leaf node")
    st.markdown("- Random sampling: selects data points at random from each column in the original data until a new dataset of the desired size is formed")
    st.markdown("- Linear/logistic regression: uses models trained on the original dataset to generate a new dataset by predicting continuous and categorical variables (via linear regression) and binary variables (via logistic regression) based on relationships learned from the original data")
    st.subheader('Data Types')
    st.write('There are three main types of data that can be used to create synthetic controls: RCT data, observational study data, and external data.')
    st.write('RCTs are the gold standard of clinical research (6-8). By randomly sorting participants into two or more groups and assigning one for each treatment and one placebo or standard of care, a study reduces the influence of biases and confounding factors. RCTs are typically required for treatment and drug approval (9). Well-powered RCTs require a larger number of participants than single-arm studies due to the random allocation of participants into two independent groups. RCTs can be time-consuming and costly to run compared to other trial methodologies (10).')
    st.write('A clinical observational study is any study where the researcher does not intervene in the result and simply collects data. These studies are often used to provide evidence of an association between a variable and disease of interest (11). There are multiple types of observational studies, including cross-sectional studies and case-control studies (12).')  
    st.write('External data is any data that was collected from sources other than the relevant party. It can include consumer purchasing habits, digital activity, weather forecasts, and any other publicly available information (13). When used in clinical trials, external data often refers to data collected by hospitals or other healthcare institutions, large surveys, census data, electronic health records, or registries (12,13). These sources of information are also referred to as Real-World Data (RWD) (14,15). External data can have many levels, from population-level statistics to patient-level information. ')


    st.header('How Does Using Different Data Types Impact the Quality of Synthetic Data in Clinical Trials?')
    st.subheader('The Answer')
    st.write("We don't know.")
    st.subheader('My Project')
    st.write('The aim of my project is to determine the difference in quality, if any, between synthetic data created from different types of clinical data. These three data types may not produce equal quality of synthetic controls, with RCTs hypothesized to produce a much higher quality (2). However, no research has measured the difference in quality of the synthetic controls produced by these methods. ')



    st.header('How Can Synthetic Data Quality be Measured and Compared?')
    st.subheader('Standard Mean Difference')
    st.write('The standardized mean difference (SMD) measures how closely the synthetic data matches the original data. A SMD of 0 means that there is no difference between datasets and the range of a SMD can be from -1 to 1. Small SMD values are between 0.2-0.5, medium SMD values are between 0.5-0.8, and large SMD values are greater than 0.8.')
    st.subheader('Treatment Effect Maintenance')
    st.write('Treatment effect maintenance includes measuring whether a treatment effect analysis using the synthetic data would produce the same results as one using the original data. For example, if the treatment and outcome are both binary, a chi-squared test can be used to determine the treatment effect for each dataset.  If the treatment effect is maintained, we would expect to see a similar significance level in both chi-squared tests.')



    st.header('Why is this Important for Researchers?')
    st.subheader('The Past')
    st.write('Successful studies have been approved using synthetic controls generated from all three types of data in the past (16-19). However, there is currently no way of understanding the difference in data quality between these studies. ')
    st.subheader('The Future')
    st.write('It is important to know whether the difference in quality is statistically significant, and by how much, so that future models can balance the need for hard-to-get RCT data with an appropriate estimation of how much better it is in producing quality synthetic data than observational data or external data.')



    st.header('Want to Know More About my Current Study?')
    st.write('Of Course You Do!')
    st.subheader('Data Sources')
    st.write('Data was simulated using the programming language R to reflect the distributions and ranges expected from external data, observational studies, and RCTs. Simulated datasets took the place of “real” data, and be compared to synthetic data as if it were such.  COVID-19 data is used as a case study to illustrate the differences between data type, chosen based on data availability.')
    st.write('The outcome and intervention set looked at was testing positive for COVID-19 and the efficacy of the BNT162b2 mRNA vaccine. Data comes from February-August 2021 in the United Kingdom. In RCTs and observational studies, vaccine efficacy was looked at after 10 days following the intervention. Demographic confounders, such as sex, race, and age were incorporated into the analyses based on statistics of COVID-19 data in the UK (20). The trials and data used in the simulations were early after the initial rollout of the vaccine in 2021. RCT data was based on probabilities from a systematic review and meta-analysis (21). Observational study data was based on probabilities from an early vaccination efficacy study on older adults in England (22). External data probabilities were devised from statistics of vaccination rates from the National Health Service in England (23) and accounts of COVID-19 test results from England (24).  ')
    
    st.subheader('Synthpop')
    st.write('Synthpop is an R packaged designed to create synthetic data from input datasets (25). The process of using synthpop in this study has three steps: ')
    st.markdown('- Preparing the data for synthesis ')
    st.markdown('- Synthesizing new data')
    st.markdown('- Comparing the synthetic data')
    st.write('In the first step, three datasets (external, observational, and RCT) are cleaned so that they have similarly named matching variables to each other. All datasets have an outcome variable, an exposure/treatment variable, and certain demographic variables. ')
    st.write('Next, the data is synthesized using synthpop. CART modelling is used as the default, but other models are compared within the sensitivity analyses. A new, synthetic dataset is formed for each data type at the end of this step. ')
    st.write('The final step compares each synthetic dataset to synthetic datasets of other types as well as the original data. Distributions of each variable are compared, and a boxplot is created to visualize the standard mean difference of the simulated datasets to the synthetic datasets. ')
  
    st.subheader('Selection of Characteristics')
    st.write('The outcome will differ depending on the disease or treatment of interest in each study, and therefore it is important to keep the labelling of the treatment/exposure and outcome generic within the data. Demographic characteristics are selected based on availability; all studies must have the same characteristics to be compared. Sex, race, and age were selected based on this metric. The categories included in race differ based on the trial and were therefore simplified to the four main groups presented in the external data: Asian/Asian British, Black/Black British, Mixed race, and White. ')
    st.subheader('Sample Size Selection')
    st.write('Having a robust sample size is a crucial aspect of generating accurate synthetic data (12). To mitigate the influence of sample size on the quality of synthetic data produced in this study, simulated studies will be generated three times using three different sample sizes. A generic sample size of 20,000 observations will be created for all data types first. A smaller sample size of 100 observations will also be created for all data types. A third version of the simulations of each data type will be created with a “realistic” sample size based off the original data source. It is expected that RCT data type simulations will have significantly less observations in this version than observational studies or external data. The purpose of doing this is to measure whether there is a change in the quality of synthetic data generated from generic or realistic sample size simulations. Two versions of the generic sample size (100 and 20,000) are used to detect any differences in synthetic data quality change between data types at different sampling amounts. ')
    st.write('These sample sizes form the four scenarios used within the simulation and synthesis: ')
    st.markdown('- Scenario 1: n = 100 ')
    st.markdown('- Scenario 2: n = 20,000 ')
    st.markdown('- Scenario 3: realistic sample sizes (RCT: n = 18,575; Observational: n = 156, 930; External: n = 20,248,632')
    st.markdown('- Scenario 4: n = 100 in simulated datasets, n = 20,000 in synthetic datasets ')
    st.write('Simulations will be run 10,000 times for each sample size scenario. ')

    st.subheader('Results so Far')
    st.write('CART method:')
    st.write('The SMD of all variables did not show any significance difference between the simulated data and the synthetic data. This was true across all data types and all scenarios. A boxplot of the results can be seen in Figure 1.')
    image = Image.open('CART_results.png')
    st.image(image)
    st.write('Figure 1: These boxplots show the standard mean difference between the simulated and synthetic data for each measured variable. The boxplots are separated by scenario, with each scenario corresponding to a specified sample size. Each variable has three boxes for different data types, with white being RCT, pink being observational, and grey being external data.')


    st.write('Random sampling method:')
    st.write('The SMD of all variables did not show any significant difference between the simulated data and the synthetic data. This was true across all data types and all scenarios. A boxplot of the results can be seen in Figure 2.')
    image = Image.open('RS_results.png')
    st.image(image)
    st.write('Figure 2: These boxplots show the standard mean difference between the simulated and synthetic data for each measured variable. The boxplots are separated by scenario, with each scenario corresponding to a specified sample size. Each variable has three boxes for different data types, with white being RCT, pink being observational, and grey being external data.')
   



    st.write('Linear/logistic regression method:')
    st.write('The SMD of all variables did not show any significance difference between the simulated data and the synthetic data. This was true across all data types and all scenarios. A boxplot of the results can be seen in Figure 3.')
    image = Image.open('LL_results.png')
    st.image(image)
    st.write('Figure 3: These boxplots show the standard mean difference between the simulated and synthetic data for each measured variable. The boxplots are separated by scenario, with each scenario corresponding to a specified sample size. Each variable has three boxes for different data types, with white being RCT, pink being observational, and grey being external data.')
    
    

    st.write('More results coming soon!')



    st.subheader('What it Means')
    st.write('Preliminary results do not show significant differences in SMD between data types. SMD is used to measure the data replicability of the synthetic data; in this way, all three data types are showing high quality synthetic data in every scenario and for all tested synthesis methods.')
    st.write('This is a case study of a single disease, and these results are not representative of all clinical trials.')
    st.write('There are several limitations of the study that should be mentioned. The main weakness of the synthesis methods is the computational power and time required to run the simulation study and synthetic data generation at an appropriate number of simulations (N = 10,000 for each scenario) Due to having four different sample size scenarios and three methods tested, the total number of simulations run was 120,000. The choice of COVID-19 as an outcome in the simulation study was not representative of other disease outcomes. While originally picked due to the wide availability of all three data types, it is now apparent that the sample sizes for all three data types are inflated compared to other diseases. This primarily affects the treatment effect estimations and the realistic sample size scenario.')
    st.subheader('What is Left to do')
    st.write('So far, this is a case study on a specific treatment and outcome. To confirm these findings, more research will need to be done on other diseases and treatments.')

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


with tab3:
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

with tab4:

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
    st.write('I would love to hear your thoughts on this app, my poster, or any of my work!')

    feedback = st.text_area("Enter your comments or feedback here:")

    if st.button("Submit"):
        if feedback:
        # Append the feedback and a timestamp to the Google Sheet
            sheet.append_row([feedback, str(datetime.now())])
            st.write("Response submitted. Thank you for your feedback!")
            
        else:
            st.write("Please enter your feedback before submitting.")

    
   









