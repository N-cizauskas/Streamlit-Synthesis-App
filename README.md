# Comparing-Simulated-and-Synthetic-Data-Types


## Description

This contains the python code for a streamlit app.  The app displays a virtual synthesizer, where synthetic data can be created from random data using one of three methods: CART, random sampling, or linear/logistic regression.  The differences between the original data and synthetic data can be analyzed using boxplots and stacked bar plots.  The app also displays a tab with a description of my research project and a tab with my contact details. 

This app was created with the ICTMC 2024 conference (Edinburgh) in mind.


## Installation

The script "Interactive_Test.py" can be opened in any python-friendly compiler. 

To access the website, see [synthesis.streamlit.app](https://synthesis.streamlit.app/).

## Libraries

The Python packages used include the following (also in "requirements.txt")


- streamlit
- pandas
- random
- matplotlib.pyplot
- PIL
- numpy
- sklearn.linear_model 



## Usage

This code is to display my research through a streamlit website.

## Citations

### As referenced in the references section of the website:

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
