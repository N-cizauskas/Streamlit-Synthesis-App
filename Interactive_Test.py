# import packages

import streamlit as st

import pandas as pd

import random

from synthpop import Synthpop

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

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




#st.data_editor('Edit data', data)

def synthesise(data, method):
    if method == "CART":
        spop = Synthpop()
        dtypes = None
        spop.fit(data, dtypes)
        num_rows = len(data)
        synth_data = spop.generate(k=num_rows)
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
        data = data[["Sex", "Race", "Treatment", "Outcome"]]
        spop = Synthpop(syn_method='logreg') 
        dtypes = None
        spop.fit(data, dtypes)
        num_rows = len(data)
        synth_data_log = spop.generate(k=num_rows)

        data = data["Age"]
        spop = Synthpop(syn_method='lm') 
        dtypes = None
        spop.fit(data, dtypes)
        num_rows = len(data)
        synth_data_lin = spop.generate(k=num_rows)

        return pd.merge(synth_data_log, synth_data_lin, left_index = True, right_index = True)
    
def synthesise_clicked(data, method):
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


    # Colors for the two data sources
    colors = ['lightblue', 'lightgreen']

    # Create the boxplot
    plt.figure(figsize=(12, 6))

    # Organize data for grouped boxplots with no space between boxes

    plt.boxplot(df1, 
                positions=[2], widths=0.6,
                patch_artist=True, boxprops=dict(facecolor=colors[0]), 
                medianprops=dict(color='blue'))
    plt.boxplot(df2, 
                positions=[2 + 0.6], widths=0.6,
                patch_artist=True, boxprops=dict(facecolor=colors[1]), 
                medianprops=dict(color='green'))

    # Labels for the x-axis corresponding to the variables
    plt.gca().set_xticks([])

    # Add title and labels
    plt.title('Boxplot Comparison of Age in Simulated and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Age')

    # Add a legend
    plt.legend([plt.Line2D([0], [0], color='lightblue', lw=4),
                plt.Line2D([0], [0], color='lightgreen', lw=4)],
            ['Simulated Data', 'Synthetic Data'])

    # Show the plot
    plt.tight_layout()
    plt.savefig("box.png")
    plt.show()




    # Load and display the image
    st.title('Dataset Comparisons')

    image = Image.open('box.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')








    # binary variables

    # Normalize the value counts to get proportions
    proportions1 = data['Sex'].value_counts(normalize=True)
    proportions2 = new_data['Sex'].value_counts(normalize=True)

    # Create a DataFrame for easier plotting
    comparison_df = pd.DataFrame({'Simulated Data': proportions1, 'Synthetic Data': proportions2})



    # Plot stacked bar plot with touching bars and outlines
    ax = comparison_df.T.plot(
        kind='bar', 
        stacked=True, 
        width=1, 
        color=['pink', 'mediumpurple'], 
        edgecolor='black',  # Outline color
        figsize=(10,6)
    )


    # Set x-axis labels to horizontal
    plt.xticks(rotation=0)


    ax.legend(['Female', 'Male'], loc='upper right')

    # Add title and labels
    plt.title('Stacked Barplot Comparison of Sex in Simulated and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Porportion of Participants')


    # Show the plot
    plt.tight_layout()
    plt.savefig("bar_chart.png")
    plt.show()




    # Load and display the image

    image = Image.open('bar_chart.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')







    # Normalize the value counts to get proportions
    proportions1 = data['Treatment'].value_counts(normalize=True)
    proportions2 = new_data['Treatment'].value_counts(normalize=True)

    # Create a DataFrame for easier plotting
    comparison_df = pd.DataFrame({'Simulated Data': proportions1, 'Synthetic Data': proportions2})



    # Plot stacked bar plot with touching bars and outlines
    ax = comparison_df.T.plot(
        kind='bar', 
        stacked=True, 
        width=1, 
        color=['pink', 'mediumpurple'], 
        edgecolor='black',  # Outline color
        figsize=(10,6)
    )


    # Set x-axis labels to horizontal
    plt.xticks(rotation=0)


    ax.legend(['Untreated', 'Treated'], loc='upper right')

    # Add title and labels
    plt.title('Stacked Barplot Comparison of Treatment in Simulated and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Porportion of Participants')


    # Show the plot
    plt.tight_layout()
    plt.savefig("bar_chart_treat.png")
    plt.show()




    # Load and display the image

    image = Image.open('bar_chart_treat.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')







    # Normalize the value counts to get proportions
    proportions1 = data['Outcome'].value_counts(normalize=True)
    proportions2 = new_data['Outcome'].value_counts(normalize=True)

    # Create a DataFrame for easier plotting
    comparison_df = pd.DataFrame({'Simulated Data': proportions1, 'Synthetic Data': proportions2})



    # Plot stacked bar plot with touching bars and outlines
    ax = comparison_df.T.plot(
        kind='bar', 
        stacked=True, 
        width=1, 
        color=['pink', 'mediumpurple'], 
        edgecolor='black',  # Outline color
        figsize=(10,6)
    )


    # Set x-axis labels to horizontal
    plt.xticks(rotation=0)


    ax.legend(['Not Present', 'Present'], loc='upper right')

    # Add title and labels
    plt.title('Stacked Barplot Comparison of Outcome in Simulated and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Porportion of Participants')


    # Show the plot
    plt.tight_layout()
    plt.savefig("bar_chart_out.png")
    plt.show()




    # Load and display the image

    image = Image.open('bar_chart_out.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')










    # categorical column - race
    # Get the proportions of each category in both datasets
    categories = ['1', '2', '3', '4']
    proportions1 = data['Race'].value_counts(normalize=True).reindex(categories, fill_value=0)
    proportions2 = new_data['Race'].value_counts(normalize=True).reindex(categories, fill_value=0)

    # Create a DataFrame for easier plotting
    comparison_df = pd.DataFrame({'Dataset 1': proportions1, 'Dataset 2': proportions2})


    # Reindex the DataFrame to ensure all four categories are included, even if some values are missing

    comparison_df = comparison_df.reindex(categories).fillna(0)

    st.dataframe(comparison_df)

    # Plot stacked bar plot
    ax = comparison_df.T.plot(
        kind='bar', 
        stacked=True, 
        width=1, 
        color=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon'],  # Colors for each category
        edgecolor='black',  # Outline color
        figsize=(10,6)
    )

    # Set x-axis labels to horizontal
    plt.xticks(rotation=0)

    # Update the legend to display the category names
    categories_names = ["Asian/Asian British", "Black/Black British", "Mixed Race", "White"]
    ax.legend(categories_names, loc='upper right')


    # Add title and labels
    plt.title('Stacked Barplot Comparison of Race in Simulated and Synthetic Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Porportion of Participants')


    # Show the plot
    plt.tight_layout()
    plt.savefig("bar_chart_race.png")
    plt.show()




    # Load and display the image

    image = Image.open('bar_chart_race.png')
    st.image(image, caption='Comparison of Original and Synthetic Data')