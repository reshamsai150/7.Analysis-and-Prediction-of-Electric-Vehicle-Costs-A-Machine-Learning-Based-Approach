import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import base64

base_dir= os.path.dirname(os.path.abspath(__file__))
ev_path= os.path.join(base_dir,"..","media","ev_final.xlsx")

df= pd.read_excel(ev_path)

def gen_dashboard():
    fig, axes= plt.subplots(2,2,figsize=(15,15))

# plot for city distribution

    city_distribution = df['city'].value_counts()
    city_distribution = np.log1p(city_distribution)
    axes[0,0].bar(city_distribution.index, city_distribution.values,color='skyblue')
    axes[0,0].set_title('EV Charging Stations Distribution Across Different Cities in India')
    axes[0,0].tick_params(axis ='x',rotation=90)

# pie chart for types of Stations

    station_type = df['station_type'].value_counts()
    axes[0,1].pie(station_type, labels = station_type.index,autopct='%1.1f%%',
              startangle =90, colors = sns.color_palette('Set3'))
    axes[0,1].set_title('Types of EV Charging Stations')

# Distribution based on Staffed and unstaffed 

    staff_distribution = df['staff'].value_counts()
    axes[1,0].bar(staff_distribution.index, staff_distribution.values,color='lightcoral')
    axes[1,0].set_title('Distribution based on Staffed and unstaffed')

# power Capacity

    df['capacity'] = df['capacity'].replace({'kW':''},regex=True)
    df['capacity'] = pd.to_numeric(df['capacity'],errors='coerce')
    sns.histplot(df['capacity'],bins =20, kde=True,color='mediumseagreen',ax = axes[1,1])
    axes[1,1].set_title('Distribution of Charging Stations Based on Power Capacity (kW)')
    axes[1,1].set_xlabel('Power Capacity (kW)')

    plt.tight_layout()

    output_path  = os.path.join(base_dir,"..","static","dashboard.png")
    plt.savefig(output_path)
    plt.close(fig)

    return "dashboard.png"
