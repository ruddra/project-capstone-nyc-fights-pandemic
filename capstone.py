import os
import webbrowser
import requests
from bs4 import BeautifulSoup
import pandas as pd
import geocoder
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import sklearn
from sklearn.cluster import KMeans
import folium
import numpy as np
import matplotlib.pyplot as plt

def save_map(m):
    filepath = 'map.html'
    m.save(filepath)

def get_geo_location(address):
    geolocator = Nominatim(user_agent="ny_explorer")
    location = geolocator.geocode(address)
    if location:
        latitude = location.latitude
        longitude = location.longitude
        return [latitude,longitude]
    return [None, None]


def get_new_york_data():
    NY_DATASET = "https://cocl.us/new_york_dataset"
    resp=requests.get(NY_DATASET).json()
    features=resp['features']
    column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude']
    new_york_data = pd.DataFrame(columns=column_names)
    
    for data in features:
        borough = data['properties']['borough'] 
        neighborhood_name = data['properties']['name']
        
        neighborhood_latlon = data['geometry']['coordinates']
        neighborhood_lat = neighborhood_latlon[1]
        neighborhood_lon = neighborhood_latlon[0]
    
        new_york_data = new_york_data.append({'Borough': borough,
                                          'Neighborhood': neighborhood_name,
                                          'Latitude': neighborhood_lat,
                                          'Longitude': neighborhood_lon}, ignore_index=True)
    
    return new_york_data


def get_population_per_neighbourhood(read_from_csv=False):
    if not read_from_csv:
        WIKI_LINK = "https://en.wikipedia.org/wiki/Neighborhoods_in_New_York_City"
        ROOT_WIKI_LINK = "https://en.wikipedia.org"
        page = requests.get(WIKI_LINK)
        soup = BeautifulSoup(page.text, 'html.parser')
        population_list = []
        for table_row in soup.select("table.wikitable tr"):
            cells = table_row.findAll('td')
            if len(cells) > 0:
                borough = cells[0].text.strip().replace('\xa0', ' ').split(' ')[0]
                population = int(cells[3].text.strip().replace(',', ''))
                for item in cells[4].findAll('a'):
                    neighborhood = item.text
                    neighbourhood_page = requests.get(ROOT_WIKI_LINK+item['href'])
                    soup = BeautifulSoup(neighbourhood_page.text, 'html.parser')
                    table = soup.select("table.infobox tr")
                    should_record = False
                    for row in table:
                        head = row.find('th')
                        body = row.find('td')
                        if head and 'population' in head.text.lower():
                            should_record = True
                            continue
                        if should_record:
                            try:
                                population_list.append([borough, neighborhood, int(body.text.replace(',', ''))])
                            except:
                                pass
                            should_record = False
        df = pd.DataFrame(population_list, columns=[ "Borough", "Neighborhood", "Population"])
        df.to_csv('population.csv')
    else:
        df = pd.read_csv('population.csv')
    df = df.sort_values(by=['Borough'])
    df = df.drop_duplicates(subset='Neighborhood', keep='last')
    return df


def get_hospital_data(lat, lng):
    radius=1000
    LIMIT=100
    VERSION = '20200328'
    # FS_CLIENT_ID = "0LVR2DN1KYTA0PDB1AIBAMAKIVLKWZ0GBEUH3WLZFBDN5OST"
    FS_CLIENT_ID = "A5S2CJNU43XNBJEADGVEDLOR024ZP5BC5KZY2E1F0WT0DZEI"
    # FS_CLIENT_SECRET = "2FYWN25KKTQATH14AM0W4BNIWYNYUWR2JUWKMOGQTY3WTMW0"
    FS_CLIENT_SECRET = "GIPWZSDNB1GYTVSRWTFV2E2JZBHBDYCORNL3MVRVDUOWQADI"
    FS_HOSPITAL_KEY = "4bf58dd8d48988d196941735"
    url = 'https://api.foursquare.com/v2/venues/search?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}&categoryId={}'.format(
            FS_CLIENT_ID, 
            FS_CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT,
            FS_HOSPITAL_KEY)
    response = requests.get(url)
    if not response.status_code == 200:
        return None
    results = response.json()
    venue_data=results["response"]["venues"]
    venue_details=[]
    for row in venue_data:
        try:
            venue_id=row['id']
            venue_name=row['name']
            lat=row["location"]["lat"]
            lng=row["location"]["lng"]
            venue_details.append([venue_id,venue_name,lat, lng])
        except KeyError:
            pass
        
    column_names=['ID','Name','Latitude', 'Longitude']
    df = pd.DataFrame(venue_details,columns=column_names)
    return df

def get_hospital_per_neighborhood(row):
    data = get_hospital_data(row["Latitude"], row["Longitude"])
    if data is not None:
        return len(data.index)
    return 0

def plot_kmeans(dataset):
    obs = dataset.copy()
    silhouette_score_values = list()
    number_of_clusters = range(3,30)
    for i in number_of_clusters:
        classifier = KMeans(i,init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=10)
        classifier.fit(obs)
        labels = classifier.predict(obs)
        silhouette_score_values.append(sklearn.metrics.silhouette_score(obs,labels ,metric='euclidean', random_state=0))
    
    plt.plot(number_of_clusters, silhouette_score_values)
    plt.title("Silhouette score values vs Numbers of Clusters ")
    plt.show()
    
    optimum_number_of_components=number_of_clusters[silhouette_score_values.index(max(silhouette_score_values))]
    print("Optimal number of components is:")
    print(optimum_number_of_components)

def hospital_vs_population(row):
    return row["Hospitals"] / row ["Population"]
# ny_df = get_new_york_data()
# ny_borough_df = get_population_per_neighbourhood(True)
# ny_df.set_index('Neighborhood')
# ny_borough_df.set_index('Neighborhood')
# df = pd.merge(ny_df, ny_borough_df)
# df['Hospitals'] = df.apply(lambda row:get_hospital_per_neighborhood(row),axis=1)
# # df = pd.read_csv('hospital.csv')
# print(df[['Hospitals', "Neighborhood", "Population"]].head(20))
# df.to_csv('hospitals.csv')
df = pd.read_csv('hospitals.csv')
df["Hospital vs Population"] = df.apply(lambda row:hospital_vs_population(row),axis=1)
plot_kmeans(df[["Latitude", "Longitude", "Hospital vs Population"]])

# set number of clusters
kclusters = 3
# run k-means clustering
df_clusters = df[["Latitude", "Longitude", "Hospitals", "Population", "Hospital vs Population"]]
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(df_clusters)
# check cluster labels generated for each row in the dataframe
print(kmeans.labels_[0:24])

df.insert(0, 'Cluster Labels', kmeans.labels_)
print(df.head(30))

# create map

def render_map_clusters(df, df_clusters, kclusters=3):
    map_clusters = folium.Map(location=get_geo_location("New York"), zoom_start=11)
    colours = ['red', 'black', 'blue']
    x = np.arange(kclusters)
    ys = [i + x + (i*x)**2 for i in range(kclusters)]
    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    rainbow = [colors.rgb2hex(i) for i in colors_array]
    markers_colors = []
    for lat, lon, poi, cluster, hos_v_pop in zip(df['Latitude'], df['Longitude'], df['Borough'], df['Cluster Labels'], df["Hospital vs Population"]):
        label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
        folium.CircleMarker(
            [lat, lon],
            radius=hos_v_pop*5+15,
            popup=label,
            color=colours[cluster],
            fill=True,
            fill_color=colours[cluster],
            fill_opacity=0.7).add_to(map_clusters)
        
    save_map(map_clusters)

y_kmeans = kmeans.predict(df_clusters)
plt.scatter(df_clusters[["Population"]], df_clusters[["Hospitals"]], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Population')
plt.ylabel('Hospitals')
plt.show()

y_kmeans = kmeans.predict(df_clusters)
plt.scatter(df_clusters[["Population"]], df_clusters[["Hospitals"]], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 2], c='black', s=200, alpha=0.5);
plt.xlabel('Population')
plt.ylabel('Hospitals')
plt.show()


render_map_clusters(df, df_clusters)

print(df[(df['Cluster Labels'] == 0)].head())
print(df[(df['Cluster Labels'] == 2)].head())
print(df[(df['Cluster Labels'] == 1)].head())