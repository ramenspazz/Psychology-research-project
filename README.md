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
