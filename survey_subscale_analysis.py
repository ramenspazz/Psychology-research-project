#!/usr/bin/env python3
"""Analyze SSS and four individual ESLI subscales (no composites).
Run: python3 survey_subscale_analysis.py --csv survey_scores.csv --comparison male-female --manova
Dependencies: python3 -m pip install numpy pandas 'scipy>=1.11' statsmodels matplotlib
Gender codes: 1=male, 2=female, 3=other (categorical labels).
Accepts entry-script lowercase headers or the display names below.
Does not modify the input CSV. Default: exploratory male versus female/other comparison.
Use --comparison male-female for the original comparison.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from statsmodels.multivariate.manova import MANOVA

SUBSCALES = [
    'ESLI_social_loneliness',
    'ESLI_emotional_loneliness',
    'ESLI_social_isolation',
    'ESLI_emotional_isolation',
]
OUTCOMES = ['SSS'] + SUBSCALES
GENDERS = {1: 'Male', 2: 'Female', 3: 'Other'}
MAX_SCORES = dict(zip(['SSS'] + SUBSCALES, [63, 24, 24, 21, 21]))


def load_data(path):
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    # Normalize capitalization, then use the requested names and order.
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={'sss_total': 'sss'})
    if df.columns.duplicated().any():
        raise ValueError('Duplicate column names after normalizing headers.')
    names = {name.lower(): name for name in ['SSS'] + SUBSCALES}
    df = df.rename(columns=names)
    required = ['gender', 'SSS'] + SUBSCALES
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df = df[required].copy()
    df[required] = df[required].apply(pd.to_numeric, errors='raise')
    if not df['gender'].isin(GENDERS).all():
        raise ValueError('Gender must be 1=male, 2=female, or 3=other; no missing codes.')
    df['gender'] = df['gender'].astype(int)
    for outcome, maximum in MAX_SCORES.items():
        values = df[outcome].dropna()
        if not np.isfinite(values).all():
            raise ValueError(f'{outcome} contains infinite scores.')
        if not values.between(0, maximum).all() or not (values % 1 == 0).all():
            raise ValueError(f'{outcome} must contain whole-number totals from 0 to {maximum}.')
    return df[['gender'] + OUTCOMES]


def welch_tests(df, comparison="male-vs-rest"):
    codes = [2] if comparison == "male-female" else [2, 3]
    label = "Female" if comparison == "male-female" else "Female/other"
    rows = []
    for outcome in OUTCOMES:
        men = df.loc[df['gender'] == 1, outcome].dropna().to_numpy(dtype=float)
        women = df.loc[df['gender'].isin(codes), outcome].dropna().to_numpy(dtype=float)
        if len(men) < 2 or len(women) < 2:
            raise ValueError(f'Too few observations for {outcome}: need at least two per comparison group.')
        if np.var(men, ddof=1) == 0 and np.var(women, ddof=1) == 0:
            raise ValueError(f'No within-group variation for {outcome}.')
        test = stats.ttest_ind(men, women, equal_var=False, alternative='two-sided')
        ci = test.confidence_interval(confidence_level=0.95)
        rows.append({
            'outcome': outcome,
            'comp_group': label,
            'n_male': len(men), 'n_comparison': len(women),
            'μ_male': men.mean(), 'mean_comparison': women.mean(),
            'std_male': np.sqrt(men.var(ddof=1)), 'std_comparison': np.sqrt(women.var(ddof=1)),
            'diff_male-comp': men.mean() - women.mean(),
            'ci95_l-u': ci.low, 'ci95_h-u': ci.high,
            't': test.statistic, 'df': test.df, 'p_raw': test.pvalue,
        })
    results = pd.DataFrame(rows)
    if not np.isfinite(results['p_raw']).all():
        raise ValueError('At least one test produced an invalid p-value.')
    reject, adjusted, _, _ = multipletests(results['p_raw'].to_numpy(), alpha=0.05, method='holm')
    results['p_holm'] = adjusted
    results['reject_H0_holm'] = reject
    return results




def pearson_tests(df):
    """Pearson r for SSS against each ESLI subscale, with pairwise complete cases."""
    rows = []
    for subscale in SUBSCALES:
        pairs = df[['SSS', subscale]].dropna()
        if len(pairs) < 3 or pairs['SSS'].nunique() < 2 or pairs[subscale].nunique() < 2:
            rows.append(dict(subscale=subscale, n=len(pairs), r=np.nan, p_raw=np.nan))
            continue
        result = stats.pearsonr(pairs['SSS'].to_numpy(float), pairs[subscale].to_numpy(float))
        rows.append(dict(subscale=subscale, n=len(pairs), r=result.statistic, p_raw=result.pvalue))
    output = pd.DataFrame(rows)
    valid = np.isfinite(output['p_raw'])
    output['p_holm'] = np.nan
    if valid.any():
        # Keep all four intended tests in the family, including untestable pairs.
        p_values = output['p_raw'].where(valid, 1.0).to_numpy()
        corrected = multipletests(p_values, method='holm')[1]
        output.loc[valid, 'p_holm'] = corrected[valid.to_numpy()]
    return output


def variance_tests(df, codes):
    rows = []
    for outcome in OUTCOMES:
        male = df.loc[df.gender.eq(1), outcome].dropna().to_numpy(float)
        comparison = df.loc[df.gender.isin(codes), outcome].dropna().to_numpy(float)
        test = stats.levene(male, comparison, center='median')
        var_a, var_b = male.var(ddof=1), comparison.var(ddof=1)
        rows.append(dict(outcome=outcome, n_male=len(male), n_comparison=len(comparison),
                         sd_male=np.sqrt(var_a), sd_comparison=np.sqrt(var_b),
                         variance_ratio=var_a / var_b if var_b else np.nan,
                         W=test.statistic, df1=1, df2=len(male)+len(comparison)-2,
                         p_raw=test.pvalue))
    results = pd.DataFrame(rows)
    valid = np.isfinite(results['p_raw'])
    results['p_holm'] = np.nan
    if valid.any():
        # An undefined test still counts in the planned five-outcome family.
        adjusted = multipletests(results['p_raw'].where(valid, 1), method='holm')[1]
        results.loc[valid, 'p_holm'] = adjusted[valid.to_numpy()]
    return results


def pillai_test(df, comparison="male-vs-rest"):
    codes = [1, 2] if comparison == "male-female" else [1, 2, 3]
    selected = df.loc[df['gender'].isin(codes), ['gender'] + OUTCOMES].copy()
    data = selected.dropna(subset=OUTCOMES).copy()
    data['group'] = np.where(data['gender'] == 1, 'Male', 'Comparison')
    counts = data['group'].value_counts()
    if len(counts) != 2 or (counts < 2).any():
        raise ValueError('Need at least two complete participants in each group.')
    if len(data) - 2 < len(OUTCOMES):
        raise ValueError('Insufficient residual degrees of freedom for five outcomes.')
    residuals = data[OUTCOMES] - data.groupby('group')[OUTCOMES].transform('mean')
    if np.linalg.matrix_rank(residuals.to_numpy()) < len(OUTCOMES):
        raise ValueError('Within-group covariance is singular; outcomes are redundant or constant.')
    model = MANOVA.from_formula(' + '.join(OUTCOMES) + ' ~ C(group)', data=data)
    # Explicitly test the group coefficient, not the intercept.
    contrast = np.zeros((1, len(model.exog_names)))
    contrast[0, [i for i, name in enumerate(model.exog_names) if name != 'Intercept']] = 1
    output = model.mv_test([('group', contrast)])
    row = output.results['group']['stat'].loc["Pillai's trace"].astype(float)
    if not np.isfinite(row.to_numpy()).all() or not 0 <= row['Pr > F'] <= 1:
        raise ValueError('Nonfinite or invalid MANOVA result; inspect covariance and scores.')
    return row, counts, len(selected) - len(data)


def report_pillai(df, comparison):
    row, counts, dropped = pillai_test(df, comparison)
    label = 'female' if comparison == 'male-female' else 'female/other'
    print(f'\nEXPLORATORY MANOVA: male versus {label}; five outcomes jointly.')
    print(f"Complete cases: male={counts['Male']}, {label}={counts['Comparison']}; excluded={dropped}.")
    print('Assumes independent observations, multivariate-normal errors, and equal group covariance matrices.')
    print('Pillai trace is not a variance test or an unequal-covariance Welch test.')
    print(row.to_frame('Pillai result').to_string())
    p = row['Pr > F']
    p_text = 'p < .001' if p < .001 else f'p = {p:.3f}'
    conclusion = 'found evidence of' if p < .05 else 'did not find statistically significant evidence of'
    print(f"\nAn exploratory MANOVA comparing male participants with {label} participants "
          f"{conclusion} a difference in the joint mean profile of SSS and the four individual ESLI subscales, "
          f"Pillai's V = {row['Value']:.3f}, "
          f"F({row['Num DF']:.0f}, {row['Den DF']:.0f}) = {row['F Value']:.2f}, {p_text}.")
    print('This omnibus p-value is unadjusted for the other analyses/groupings explored.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='survey_scores.csv')
    parser.add_argument('--no-plot', action='store_true', help='Print statistics without opening plots.')
    parser.add_argument('--comparison', choices=['male-vs-rest', 'male-female'],
                        default='male-vs-rest', help='Default is exploratory male versus female/other.')
    parser.add_argument('--manova', action='store_true',
                        help='Also report exploratory MANOVA using Pillai trace.')
    args = parser.parse_args()
    codes = [2] if args.comparison == 'male-female' else [2, 3]
    label = 'Female' if args.comparison == 'male-female' else 'Female/other'
    df = load_data(args.csv)
    print('SSS total and four individual ESLI subscales (no composites).')
    print('Gender categories:')
    counts = df['gender'].value_counts().reindex([1, 2, 3], fill_value=0)
    counts.index = [f'{code} = {GENDERS[code]}' for code in counts.index]
    print(counts.to_string())
    # Show every group descriptively, including Other.
    grouped = df.assign(gender=df['gender'].map(GENDERS)).groupby('gender', sort=False)
    print('\nDescriptive statistics (all observed gender groups):')
    print(grouped[OUTCOMES].agg(['count', 'mean', 'std', 'var', 'median', 'min', 'max']).round(2).to_string())
    print('\nMissing outcome scores:')
    print(grouped[OUTCOMES].agg(lambda x: x.isna().sum()).to_string())
    selected = df.loc[df['gender'].isin([1] + codes)].copy()
    selected['group'] = np.where(selected['gender'] == 1, 'Male', label)
    print('\nDescriptive statistics for the selected comparison:')
    print(selected.groupby('group', sort=False)[OUTCOMES].agg(
        ['count', 'mean', 'std', 'var', 'median']).round(2).to_string())
    print(f'\nWelch tests: male (1) versus {label} {codes}.')
    if args.comparison == 'male-vs-rest':
        print('EXPLORATORY: regrouping after examining the original male-female results.')
    results = welch_tests(df, args.comparison)
    print(results.round(4).to_string(index=False))
    print('\nDifference = male mean minus comparison mean. CIs are unadjusted; p_holm adjusts across five outcomes in this run, not across alternative groupings.')
    print('\nMedian-centered Levene tests (Brown–Forsythe):')
    print(variance_tests(df, codes).round(4).to_string(index=False))
    print('Variance p-values are Holm-adjusted separately from the Welch p-values.')
    print('\nPearson correlations: SSS versus each individual ESLI subscale (all genders):')
    print(pearson_tests(df).round(5).to_string(index=False))
    print('Correlations use pairwise complete cases; p_holm corrects the four Pearson tests separately.')
    if args.manova:
        try:
            report_pillai(df, args.comparison)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f'\nMANOVA unavailable: {exc}')
    if args.no_plot:
        return
    fig = plot_scores(df, codes, label, args.comparison == 'male-vs-rest')
    plt.show()


def plot_scores(df, codes, label, exploratory=True):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    colors = ['#3978a8', '#d18b40']
    for ax, outcome in zip(axes, OUTCOMES):
        samples = [df.loc[df['gender'].isin(group), outcome].dropna().to_numpy(dtype=float)
                   for group in ([1], codes)]
        boxes = ax.boxplot(samples, positions=[0, 1], widths=.48, patch_artist=True,
                           showfliers=False, whis=1.5,
                           medianprops={'color': 'black', 'linewidth': 2})
        for box, color in zip(boxes['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(.22)
        for position, (values, color) in enumerate(zip(samples, colors)):
            x = position + rng.uniform(-.12, .12, size=len(values))
            ax.scatter(x, values, color=color, alpha=.65, s=23, zorder=3)
            if len(values):
                ax.scatter(position, values.mean(), color='crimson', marker='D',
                           edgecolor='white', s=60, zorder=4)
        ax.set_xticks([0, 1], [f'Male\n(n={len(samples[0])})',
                                  f'{label}\n(n={len(samples[1])})'])
        ax.set_title({'SSS': 'SSS total',
                      'ESLI_social_loneliness': 'Social Loneliness',
                      'ESLI_emotional_loneliness': 'Emotional Loneliness',
                      'ESLI_social_isolation': 'Social Isolation',
                      'ESLI_emotional_isolation': 'Emotional Isolation'}[outcome])
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[-1].axis('off')
    axes[-1].text(.04, .9,
        'Reading the plots\n\n'
        'Box: 25th to 75th percentile\n'
        'Black line: median\n'
        'Red diamond: mean\n'
        'Dots: individual participants\n'
        'Whiskers: most extreme observed scores\n'
        'within 1.5 IQR of the box edges\n\n'
        'Horizontal spread separates overlap.\n',
        # 'Sample variance is printed in the tables.',
        transform=axes[-1].transAxes, va='top', fontsize=16)
    fig.suptitle(('Exploratory: ' if exploratory else '') + f'Male versus {label}')
    fig.tight_layout(rect=[0, 0, 1, .96])
    return fig



if __name__ == '__main__':
    main()
