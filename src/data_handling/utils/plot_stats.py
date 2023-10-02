
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint as pp


# Bivariate Analysis (Continous - Categorical)
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
     '''
     takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sampled Z-Test
     '''
     import numpy as np
     from scipy.stats import norm
     ovr_sigma = np.sqrt(sigma1**2/N1 + sigma2**2/N2)
     z = (X1 - X2)/ovr_sigma
     pval = 2*(1 - norm.cdf(abs(z)))
     return pval

def TwoSampT(X1, X2, sd1, sd2, n1, n2):
     '''
     takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sample T-Test
     '''
     import numpy as np
     from scipy.stats import t as t_dist
     ovr_sd = np.sqrt(sd1**2/n1 + sd2**2/n2)
     t = (X1 - X2)/ovr_sd
     df = n1+n2-2
     pval = 2*(1 - t_dist.cdf(abs(t),df))
     return pval

def Bivariate_cont_cat(data, cont, cat, category):
     
     import numpy as np
     
     #creating 2 samples
     x1 = data[cont][data[cat]==category][:]
     x2 = data[cont][~(data[cat]==category)][:]
     
     #calculating descriptives
     n1, n2 = x1.shape[0], x2.shape[0]
     m1, m2 = x1.mean(), x2.mean()
     std1, std2 = x1.std(), x2.std()
     
     #calculating p-values
     t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
     z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

     #table
     table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc=np.mean)

     #plotting
     plt.figure(figsize = (15,6), dpi=140)
     
     #barplot
     plt.subplot(1,2,1)
     sns.barplot(['not {}'.format(category), str(category)], [m2, m1])
     plt.ylabel('mean {}'.format(cont))
     plt.xlabel(cat)
     plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                 z_p_val,
                                                                 table))

     # boxplot
     plt.subplot(1,2,2)
     sns.boxplot(x=cat, y=cont, data=data)
     plt.title('categorical boxplot')


# Bivariate Analysis (Categorical - Categorical)
def BVA_categorical_plot(data, tar, cat):
     '''
     take data and two categorical variables,
     calculates the chi2 significance between the two variables 
     and prints the result with countplot & CrossTab
     '''
     #isolating the variables
     data = data[[cat,tar]][:]

     #forming a crosstab
     table = pd.crosstab(data[tar],data[cat],)
     f_obs = np.array([table.iloc[0][:].values,
                         table.iloc[1][:].values])

     #performing chi2 test
     from scipy.stats import chi2_contingency
     chi, p, dof, expected = chi2_contingency(f_obs)
     
     #checking whether results are significant
     if p<0.05:
          sig = True
     else:
          sig = False

     #plotting grouped plot
     sns.countplot(x=cat, hue=tar, data=data)
     plt.title("p-value = {}\n difference significant? = {}\n".format(round(p,8),sig))

     #plotting percent stacked bar plot
     #sns.catplot(ax, kind='stacked')
     ax1 = data.groupby(cat)[tar].value_counts(normalize=True).unstack()
     ax1.plot(kind='bar', stacked='True',title=str(ax1))
     int_level = data[cat].value_counts()


# ChatGPT function

def stacked_bar_chart_with_ttest(data, x, y):
    
    import pingouin as pg
    
    # Create stacked bar chart
    sns.set(style="whitegrid")
    sns.countplot(data=data, x=x, hue=y)
    plt.show()

    # Create stacked bar chart (percentages)

    # Calculate the percentage of each category
    grouped_data = data.groupby([x, y]).size().unstack()
    percent_data = grouped_data.apply(lambda row: row / row.sum(), axis=1)

    # Create stacked bar chart with percentages
    # sns.set(style="whitegrid")
    ax = percent_data.plot(kind="bar", stacked=True)

    # Set y-axis as percentage
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
    plt.show()

    # Perform t-test
    pp("T-Test Result:")
    pp(pg.chi2_independence(data, x=x, y=y))
    

# ChatGPT using plotly

def px_stacked_bar_chart_with_ttest(data, x, y):
    
    import pingouin as pg
    import plotly.express as px
    
    # copy data
    data_cp = data.copy()

    # convert x and y to categorical
    data_cp[x] = data_cp[x].astype('category')
    data_cp[y] = data_cp[y].astype('category')

    # Calculate the percentage of each category
    data_stack=data_cp.groupby([x, y]).size().reset_index()
    
    data_stack['Percentage']=data_cp.groupby([x, y]).size()  \
     .groupby(level=0).apply(lambda x:100 * x/float(x.sum())).values
    
    data_stack.columns= [x, y, 'Counts', 'Percentage']
    
    # data_stack['Percentage'] = data_stack['Percentage'].map('{:,.2f}%'.format)

    # Create bar chart
    fig = px.bar(data_stack, x = x, y = 'Counts', color = y, barmode = 'group')
    
    fig.show()

    # Create stacked bar chart with percentages
    fig2=px.bar(data_stack, x = x, y = 'Percentage', color = y, barmode ='relative')
    
    fig2.show()

    # Perform t-test
    pp("T-Test Result:")
    pp(pg.chi2_independence(data = data_cp, x=x, y=y))
