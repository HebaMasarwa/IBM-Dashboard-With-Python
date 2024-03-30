## Imports
import numpy as np
import pandas as pd
import squarify 
from PIL import Image

# bokeh
from bokeh.io import show, save
from bokeh.models import ColumnDataSource, FactorRange, LabelSet, Div
from bokeh.plotting import figure 
from bokeh.palettes import GnBu
from bokeh.layouts import layout

# to upload the logo image and make plots in matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

## opening the data

try:
    df = pd.read_csv('IBM data.csv')
except:
    print('Error while loading the file')

## color pallete used

palette_name = 'GnBu'
palette = GnBu[9] 

##########################################
#
#   preparing graph functions for the dashboard
#
##########################################

# making a function to display images in the dashboard
# this is due to some graphs being made using matplot and you can only
# display matplot figures using images in a bokeh dashboard
# also this is to display IBM logo

def display_image_bokeh(file_path,width=400,height=400):
    img = Image.open(file_path).convert('RGBA')
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Flip the image data vertically
    img_array_flipped = np.flipud(img_array)
    
    # Convert the flipped image data to a format suitable for Bokeh 
    img_data_flipped = img_array_flipped.view(dtype=np.uint32).reshape(img_array_flipped.shape[:2])
    
    # Create a Bokeh figure
    p = figure(width=width, height=height)
    
    # Display the flipped image
    p.image_rgba(image=[img_data_flipped], x=0, y=0, dw=10, dh=10)  
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.axis.visible = False
    return p

######################################################
#
#   Figure 1: Average Age by Department and Job Role
#
######################################################

# matplot graph 

# preparing the data for this section: 
avg_age_by_depart_job = df.groupby(['Department', 'JobRole','Gender'
                                    ])['Age'].mean().round(2).reset_index(name='AvgAge')

# arranging the data to be ascending according to departmenet and descending by avgage
sorted_avg_age_by_depart_job = avg_age_by_depart_job.sort_values(by=['Department', 'AvgAge'
                                                                     ], ascending=[True, False])

# sepearting the genders to diff data frames
females_only_df = sorted_avg_age_by_depart_job.loc[sorted_avg_age_by_depart_job['Gender'] == 'Female']
males_only_df = sorted_avg_age_by_depart_job.loc[sorted_avg_age_by_depart_job['Gender'] == 'Male']

# Merge the DataFrames on 'Department' and 'JobRole'
combined_df = pd.merge(females_only_df[['Department', 'JobRole', 'AvgAge']],
                       males_only_df[['Department', 'JobRole', 'AvgAge']],
                       on=['Department', 'JobRole'], 
                       suffixes=('_Female', '_Male'))

# Transforming the data into a list of lists
avg_age_by_depart_job_list = [['Department', 'Job Role', 'Female', 'Male']]  

for _, row in combined_df.iterrows():
    avg_age_by_depart_job_list.append([row['Department'], row['JobRole'], row['AvgAge_Female'], row['AvgAge_Male']])

def avg_age_by_depart_job_figure(data):
    # data is a list of lists with the structure similar to
    
    # Extracting counts and normalizing
    counts_female = np.array([row[2] for row in data[1:]])
    normalized_counts_female = (counts_female - counts_female.min()) / (counts_female.max() - counts_female.min())

    counts_male = np.array([row[3] for row in data[1:]])
    normalized_counts_male = (counts_male - counts_male.min()) / (counts_male.max() - counts_male.min())

    # Setting the colormap
    colormap_female = plt.cm.GnBu(normalized_counts_female)
    colormap_male = plt.cm.GnBu(normalized_counts_male)
    title = 'Average Age by Department and Job Role'
    # Preparing the graph
    fig, ax = plt.subplots(figsize=(3,4))
    ax.set_title('Average Age by Department and Job Role', pad=130)
    ax.axis('off')

    cell_colors = []
    for i, row in enumerate(data):
        if i == 0:  # Header row
            cell_colors.append(['#f0f0f0' for _ in row])
        else:
            row_colors = ['w', 'w'] + [colormap_female[i-1], colormap_male[i-1]]
            cell_colors.append(row_colors)

    table = ax.table(cellText=data, loc='center', cellLoc='center', cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(3, 3)
    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300) # saving the figure to display in the dashboard

#################################################################################
#
#   Figure 2: Gender Percentage in the Company
# 
#################################################################################

# matplot figure

# preparing the data for this section: 
# Count occurrences of each gender
gender_counts = df['Gender'].value_counts()

# Defining a Dataframe for the data
gender_counts = pd.DataFrame({'Gender': gender_counts.index, 
                              'Count': gender_counts.values})

# Calculating Percentage
gender_counts['Percentage'] = (gender_counts['Count'] / gender_counts['Count'].sum() * 100).round(2)

# formatted percentage strings with '%' sign
gender_counts['FormattedPercentage'] = gender_counts['Percentage'].apply(lambda x: f"{x:0.2f}%")

def gender_piechart(data):
    fig, ax = plt.subplots(figsize=(4, 3))
    title='Gender in The Company'
    ax.set_title(title)

    ax.pie(data['Percentage'] , 
        labels=data['Gender'] , 
        autopct='%1.1f%%',
        colors= [palette[0],palette[-3]])
    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)


#################################################################################
#
#   Figure 3: Gender and Martial States
# 
#################################################################################

# matplot figure

# preparing the data for this section: 
martial_count = df.groupby(['Gender', 'MaritalStatus']).size().reset_index(name='count')

# Sorting the DataFrame by 'count' in descending order
martial_count_sorted = martial_count.sort_values(by='count', ascending=False)

def martial_count_figure(data):
    # Preparing the graph 
    sizes = data['count']
    labels = data.apply(lambda x: f"{x['Gender']}\n{x['MaritalStatus']}\n({x['count']})", axis=1)

    # Normalizing sizes and generating colors based on count
    norm = plt.Normalize(vmin=min(sizes), vmax=max(sizes))
    cmap = cm.ScalarMappable(norm=norm, cmap=palette_name)  
    colors = [cmap.to_rgba(size) for size in sizes]
    title='Gender and Martial States'
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title)
    ax.axis('off') 

    squarify.plot(sizes=sizes, 
                label=labels, 
                alpha=0.8, 
                color=colors)

    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)

#################################################################################
#
#   Figure 4: Total Employees by Education Field
# 
#################################################################################

# matplot figure

# preparing the data for this section: 
tot_emp_by_edu = df.groupby(['EducationField'])['EmployeeCount'].count().reset_index(name='TotalEmployees')
sorted_tot_emp_by_edu = tot_emp_by_edu.sort_values(by='TotalEmployees', ascending=False)

# Transforming the data into a list of lists
# Uncomment for headers: 
#sorted_tot_emp_by_edu_list = [['Education Field','Total Employees']]  
tot_emp_by_edu_list = []  

# Iterating over DataFrame rows
for col, row in sorted_tot_emp_by_edu.iterrows():
    tot_emp_by_edu_list.append([row['EducationField'], row['TotalEmployees']])

def tot_emp_by_edu_figure(data):
    # Extracting the counts and normalizing
    counts = np.array([row[1] for row in data])
    normalized_counts = (counts - counts.min()) / (counts.max() - counts.min())

    # Getting the colormap
    colormap = plt.cm.GnBu(normalized_counts)

    # Creating the figure
    fig, ax = plt.subplots(figsize=(4, 3))
    title='Total Employees by Education Field'
    ax.set_title(title)
    ax.axis('off') # Hiding the axes for a cleaner look

    # Creating the table with colored cells based on counts
    cell_colors = [['w', colormap[i]] for i in range(len(data))]

    table = ax.table(cellText=data, 
                    loc='center', 
                    cellLoc='center', 
                    cellColours=cell_colors)

    table.set_fontsize(10)
    table.scale(1, 1.5)  

    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)

#################################################################################
#
#   Figure 5: Total Employees, Average Age, Avg. Years at Company, Avg. Total Working Years
# 
#################################################################################

# preparing the data for this section: 
total_emp = df['EmployeeCount'].count()
avg_age = df['Age'].mean().round(2)
avg_years = df['YearsAtCompany'].mean().round(3)
avg_working = df['TotalWorkingYears'].mean().round(2)

def textPlot(ax, data, title):
    ax.axis('off')  # Hiding the axes for a cleaner look
    ax.set_title(title, fontsize=14, loc='center')
    
    # Displaying the data as text in the middle of the subplot
    ax.text(0.5, 0.5, str(data), ha='center', fontsize=14, transform=ax.transAxes)

def textSubPlot():
    fig, axs = plt.subplots(2, 2, figsize=(6,3))  # Create a 2x2 grid of subplots
    title='textPlot'
    # Use the textPlot function to display text in each subplot
    textPlot(axs[0, 0], total_emp, 'Total Employees')
    textPlot(axs[0, 1], avg_age, 'Average Age')
    textPlot(axs[1, 0], avg_years, 'Avg. Years at Company')
    textPlot(axs[1, 1], avg_working, 'Avg. Total Working Years')

    plt.tight_layout()  # Adjust the layout
    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)

#################################################################################
#
#   Figure 6: Avg. Monthly Income by Depart. and Job Role
# 
#################################################################################

# bokeh figure
# preparing the data for this section

avg_income_by_depart_job = df.groupby(['Department', 'JobRole'])[
    'MonthlyIncome'].mean().round(2).reset_index(name='AvgIncome').round()

# sorting the data 
sorted_avg_income_by_depart_job = avg_income_by_depart_job.sort_values(
    by=['Department', 'AvgIncome'],ascending=[True, False])

# Extracting unique departments and job roles for plotting
# Departments serve as the primary category, and job roles are nested within each department.
Department = sorted_avg_income_by_depart_job['Department'].unique()
JobRole = sorted_avg_income_by_depart_job['JobRole'].unique()

# data deoart jobrole title column name for counts 
def nestedHbar(data,Department,JobRole,colName,title):
    
    y = [(depart, job) for depart in Department for job in JobRole 
        if job in data[data['Department'] == depart]['JobRole'].values]

    # Create a ColumnDataSource
    source = ColumnDataSource(data=dict(y=y, counts=data[colName]))

    p = figure(y_range=FactorRange(*y), height=400, width=400,
            title=title,
            toolbar_location=None, tools="")
    p.hbar(y='y', right='counts', height=0.9, source=source,color=palette[0])
    # Add labels to each bar to display the exact income value.
    labels = LabelSet(x='counts', y='y', text='counts', level='glyph',
                    x_offset=-30, y_offset=0, source=source, text_font_size='8pt', text_color='white')

    p.add_layout(labels)
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None
    
    return p

#################################################################################
#
#   Figure 7: Gender and Job Role
# 
#################################################################################

# matplot figure
# preparing the data for this section: 
emp_count_by_role_gender = df.groupby(['JobRole', 'Gender'])['Gender'].count().unstack(fill_value=0)
sorted_emp_count_by_role_gender = emp_count_by_role_gender.sort_values(by=['Female', 'Male'], ascending=False)

# Transforming the data into a list of lists
emp_count_by_role_gender_list = [['Job Role', 'Female', 'Male']]  # Starting with headers

# Iterating over DataFrame rows
for job_role, row in sorted_emp_count_by_role_gender.iterrows():
    emp_count_by_role_gender_list.append([job_role, row['Female'], row['Male']])
    

def empCountByRoleGender(data):
    # Extracting the counts and normalizing
    counts_female = np.array([row[1] for row in data[1:]])
    normalized_counts_female = (counts_female - counts_female.min()) / (counts_female.max() - counts_female.min())

    counts_male = np.array([row[2] for row in data[1:]])
    normalized_counts_male = (counts_male - counts_male.min()) / (counts_male.max() - counts_male.min())

    colormap_female = plt.cm.GnBu(normalized_counts_female)
    colormap_male = plt.cm.GnBu(normalized_counts_male)

    fig, ax = plt.subplots()
    title='Gender and Job Role'
    ax.set_title(title,pad=20)
    ax.axis('off') 

    cell_colors = []
    for i, row in enumerate(data):
        if i == 0:  # Header row
            cell_colors.append(['#f0f0f0' for _ in row])  # Header color
        else:
            row_colors = ['w']  # Default color for the first column
            row_colors.append(colormap_female[i-1])  # Female column color
            row_colors.append(colormap_male[i-1])  # Male column color
            cell_colors.append(row_colors)

    table = ax.table(cellText=data, 
                    loc='center', 
                    cellLoc='center', 
                    cellColours=cell_colors)

    table.auto_set_font_size(False)
    table.set_fontsize(10)  
    table.scale(1.5, 2)  
    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)
    
#################################################################################
#
#   Figure 8: Total Companies Worked For
# 
#################################################################################

# preparing the data
tot_compaies_gender_count = df.groupby(['NumCompaniesWorked','Gender'])['Gender'].count().unstack(fill_value=0)

def totCompaiesGenderCount_bokeh(data):
    x = data.index
    y_male = data['Male']
    y_female = data['Female']
    
    p = figure(title="Total Companies Worked For", 
               x_axis_label='Num. of Companies Worked', 
               y_axis_label='Employees Gender Count',
               height =400,
               width=400)  
    p.line(x, y_male, legend_label="Male", color=palette[0], line_width=2)
    p.line(x, y_female, legend_label="Female", color=palette[-4], line_width=2) 
    
    return p  
#################################################################################
#
#   Figure 9: Total Empolyees by Department and Job Role
# 
#################################################################################

# uses the "nestedHbar(data,Department,JobRole,colName,title)" function

# preparing the data for this figure
emp_count_by_dept_and_role = df.groupby(['Department','JobRole'])['JobRole'].count().reset_index(name='JobCount')
sorted_emp_count_by_dept_and_role = emp_count_by_dept_and_role.sort_values(by=['Department', 'JobCount'], 
                                                                           ascending=[True, False])

Department2 = sorted_emp_count_by_dept_and_role['Department'].unique()
JobRole2 = sorted_emp_count_by_dept_and_role['JobRole'].unique() 

#######################################
#
#   Assembling the Dashboard
#
#######################################

# Header
title = Div(text='<h1 style="text-align: center">IBM Dashboard</h1>')


fig1 = avg_age_by_depart_job_figure(avg_age_by_depart_job_list)
img1 = display_image_bokeh('Average Age by Department and Job Role.png')

fig2 = gender_piechart(gender_counts) 
img2= display_image_bokeh('Gender in The Company.png')

fig3 = empCountByRoleGender(emp_count_by_role_gender_list) 
img3 = display_image_bokeh('Gender and Job Role.png')

fig4 = martial_count_figure(martial_count_sorted) 
img4= display_image_bokeh('Gender and Martial States.png')

fig5 = display_image_bokeh('IBM_logo.png',400,200 )

fig6 = textSubPlot() 
img6 = display_image_bokeh('textPlot.png',400,200)

fig7 = totCompaiesGenderCount_bokeh(tot_compaies_gender_count)
 
fig8 = tot_emp_by_edu_figure(tot_emp_by_edu_list) 
img8 = display_image_bokeh('Total Employees by Education Field.png')

fig9 = nestedHbar(sorted_avg_income_by_depart_job,Department,JobRole,
                                'AvgIncome',
                                "Avg. Monthly Income by Depart. and Job Role")
fig10=nestedHbar(sorted_emp_count_by_dept_and_role,Department2,JobRole2,
                                'JobCount',
                                'Total Empolyees by Department and Job Role')

# creating the layout

row1 = [img1,img2,img3]
middle_column  = [fig5,img6]
row2 = [img4,middle_column,fig7]
row3 =[img8,fig9,fig10]
final_layout = layout([title,row1, row2, row3])



show(final_layout)
save(final_layout)