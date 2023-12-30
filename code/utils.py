import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

def calculate_ppl(seq_losses):
    '''
    seq_losses should be a tensor containing the per-element loss for each 
    element in the sequence (as returned by nn.CrossEntropyLoss(reduction='none')).
    Returns the perplexity of the sequence, normalized by sequence length.
    '''
    return float(torch.exp(sum(seq_losses)/len(seq_losses)))

def stat_test_differences(ppl_dict):
    twa, tna, nwa, nna = (ppl_dict['test_group_with_attribute'], 
                          ppl_dict['test_group_no_attribute'], 
                          ppl_dict['norm_group_with_attribute'], 
                          ppl_dict['norm_group_no_attribute'])
    test_diffs = np.array(twa)-np.array(tna)
    norm_diffs = np.array(nwa)-np.array(nna)
    test_shapiro = stats.shapiro(test_diffs)
    norm_shapiro = stats.shapiro(norm_diffs)
    if (test_shapiro.statistic < 0.5 and test_shapiro.pvalue < 0.05) or (norm_shapiro.statistic < 0.5 and norm_shapiro.pvalue < 0.05):
        print('Warning: data do not seem to be normally distributed;')
        print(f'Shapiro test result for norm group differences: {norm_shapiro}')
        print(f'Shapiro test result for test group differences: {test_shapiro}')
        print('t-test result may be unreliable.')
    return stats.ttest_ind(norm_diffs, test_diffs, equal_var=False)

def make_ppl_df(key, value):
    df = pd.DataFrame(value, columns=['Model perplexity'])
    words = key.split('_')
    df['Group'] = ' '.join(words[:2]).capitalize()
    df['Sentence version'] = ' '.join(words[2:]).capitalize()+' word'
    return df

def result_to_df(ppl_dict):
    dfs = [make_ppl_df(key, value) for key, value in ppl_dict.items()]
    df = pd.concat(dfs) 
    return df

def plot_groups(df, max_z=3, savepath=False, theme='dark', filling=False, colors=('b', '.35')):
    no_outliers = df[(np.abs(stats.zscore(df['Model perplexity'])) < max_z)]
    sns.set_theme(style=theme)
    # Draw a nested violinplot and split the violins for easier comparison
    plot = sns.violinplot(data=no_outliers, x="Group", y="Model perplexity", hue="Sentence version",
                   split=True, inner="quart", fill=filling,
                   palette={"With attribute word": colors[0], "No attribute word": colors[1]})
    if savepath:
        plot.get_figure().savefig(savepath)
        
    return plot
    






















