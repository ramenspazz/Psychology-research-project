# Comparison of Social and Emotional Isolation and Loneliness of Males: The Male Loneliness Epidemic
This repository contains all materials used for a survey based research project on the Male Loneliness Epidemic. We attempted to use opensource tools whenever possible.
Tools used:
1) PSPP as an alternative to SPSS, though results were verified with SPSS on University computers to satisfy academic requirements
2) python libraries pandas, matplotlib, scipy, numpy, and statsmodels for secondary validation of results and to test methods used by other software
3) DesignCraft to design and build the poster we used for presentation
4) WordCraft to write the abstract and report

All analysis can be preformed in PSPP, SPSS, or any tool of choice. The results we used came directly from our faculty recommended methods, and all other methods should be considered exploratory or secondary validation.

We would like to make a special thanks to our faculty mentor Dr. Diane J. Pfahler for her immense help in this project, without which we would not have gotten to this point.
We love you Dr. Pfahler!

# Installing python dependencies:
> First, create the python environment by running `python -m venv data_analysis`,
> then activate the environment with `source data_analysis/bin/activate` on 
> Linux/macOS or `data_analysis` on Windows.

> Next, run `pip install -r requirements.txt`. Do not commit the venv folder to version control.

# run the python analysis:
> with the venv data_analysis activated, run `python3 survey_subscale_analysis.py --csv survey_scores_final.csv --comparison male-female`. Results will print to a terminal window and a graph window will open.

# run the SPSS tests:
> open the SPSS file named `survey_abstract_spss19.sps`. Edit the code `/FILE=<location>` to match the operating system type.
> Then, go to run and select `All`. The results will be displayed in a seperate window.
