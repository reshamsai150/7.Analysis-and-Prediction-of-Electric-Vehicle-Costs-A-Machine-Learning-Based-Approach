import os
import pandas as pd
import folium
from folium.plugins import MarkerCluster

def ev_map(selected_city=None):
    base_dir= os.path.dirname(os.path.abspath(__file__))
    ev_path= os.path.join(base_dir,"..","media","ev_final.xlsx")
    ev_path=os.path.abspath(ev_path)

    df= pd.read_excel(ev_path)

    df['capacity']= df['capacity'].fillna('0 KW')
    #df['city']=df['city'].replace('Delhi', 'New Delhi')
    df['staff'] = df['staff'].replace({
        'Staffed': 'staffed',
        'UnStaffed': 'unstaffed',
        'Unstaffed': 'unstaffed'
    })

    if selected_city and selected_city.lower()!= "all":
        df= df[df['city'] ==selected_city]

    df= df.dropna(subset=['latitude','longitude'])

    if df.empty:
        map_base =[20.5937 , 78.9629]
        charging_map = folium.Map(location=map_base,zoom_start=5)
        map_path =os.path.join(os.path.dirname(__file__),"..","templates","ev_maps.html")
        map_path= os.path.abspath(map_path)
        charging_map.save(map_path)
        return "ev_maps.html"

    map_base = [df['latitude'].mean() , df['longitude'].mean()]
    charging_map = folium.Map(location=map_base,zoom_start=5)

    marker_cluster = MarkerCluster().add_to(charging_map)

    for _ ,row in df.iterrows():
        contact = row.get('contact_numbers','N/A')
        if pd.isna(contact):
            contact='N/A'
        elif isinstance(contact,list):
            contact =','.join(map(str,contact))
        else:
            contact =str(contact)
        
        popup_text = f"<b>{row['name']}</b><br>Vendor: {row['vendor_name']}<br>Contact :{contact}"
        folium.Marker([row['latitude'],row['longitude']],popup =popup_text).add_to(marker_cluster)

    #file = f"ev_maps.html" if not city else f"ev_maps_{city}.html"
    map_path = os.path.join(os.path.dirname(__file__),"..","templates","ev_maps.html")
    map_path= os.path.abspath(map_path)
    charging_map.save(map_path)

    return "ev_maps.html"

if __name__ =="__main__":
    ev_map()


