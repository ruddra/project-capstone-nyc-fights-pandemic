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
from selenium import webdriver
import time
from fuzzywuzzy import fuzz


def save_map(m):
    filepath = 'map.html'
    m.save(filepath)


def get_geo_location(address):
    geolocator = Nominatim(user_agent="ny_explorer")
    location = geolocator.geocode(address)
    if location:
        latitude = location.latitude
        longitude = location.longitude
        return [latitude, longitude]
    return [None, None]


def get_new_york_data():
    NY_DATASET = "https://cocl.us/new_york_dataset"
    resp = requests.get(NY_DATASET).json()
    features = resp['features']
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
                borough = cells[0].text.strip().replace(
                    '\xa0', ' ').split(' ')[0]
                population = int(cells[3].text.strip().replace(',', ''))
                for item in cells[4].findAll('a'):
                    neighborhood = item.text
                    neighbourhood_page = requests.get(
                        ROOT_WIKI_LINK+item['href'])
                    soup = BeautifulSoup(
                        neighbourhood_page.text, 'html.parser')
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
                                population_list.append(
                                    [borough, neighborhood, int(body.text.replace(',', ''))])
                            except:
                                pass
                            should_record = False
        df = pd.DataFrame(population_list, columns=[
                          "Borough", "Neighborhood", "Population"])
        df.to_csv('population.csv')
    else:
        df = pd.read_csv('population.csv')
    df = df.sort_values(by=['Borough'])
    df = df.drop_duplicates(subset='Neighborhood', keep='last')
    return df


def get_hospital_data(lat, lng, borough, neighborhood):
    radius = 1000
    LIMIT = 100
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
        print("ERROR", response)
        return None
    results = response.json()
    venue_data = results["response"]["venues"]
    venue_details = []
    for row in venue_data:
        try:
            venue_id = row['id']
            venue_name = row['name']
            lat = row["location"]["lat"]
            lng = row["location"]["lng"]
            venue_details.append(
                [venue_id, venue_name, lat, lng, borough, neighborhood])
        except KeyError:
            pass

    column_names = ['ID', 'Name', 'Latitude',
                    'Longitude', "Borough", "Neighborhood"]
    df = pd.DataFrame(venue_details, columns=column_names)
    return df


def get_hospital_per_neighborhood(row):
    data = get_hospital_data(
        row["Latitude"], row["Longitude"], row["Borough"], row["Neighborhood"])
    if data is not None:
        return len(data.index)
    return 0


def plot_kmeans(dataset):
    obs = dataset.copy()
    silhouette_score_values = list()
    number_of_clusters = range(3, 30)
    for i in number_of_clusters:
        classifier = KMeans(i, init='k-means++', n_init=10,
                            max_iter=300, tol=0.0001, random_state=10)
        classifier.fit(obs)
        labels = classifier.predict(obs)
        silhouette_score_values.append(sklearn.metrics.silhouette_score(
            obs, labels, metric='euclidean', random_state=0))

    plt.plot(number_of_clusters, silhouette_score_values)
    plt.title("Silhouette score values vs Numbers of Clusters ")
    plt.show()

    optimum_number_of_components = number_of_clusters[silhouette_score_values.index(
        max(silhouette_score_values))]
    print("Optimal number of components is:")
    print(optimum_number_of_components)


def hospital_vs_population(row):
    return row["Hospitals"] / row["Population"]


# create map


def show_bar_chart(df, field="Population", title='Number of Neighborhood for each Borough in New York City', x_label="Borough", y_label="No.of Neighborhood"):
    plt.figure(figsize=(9, 5), dpi=100)
    plt.title(title)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    df.groupby('Borough')[field].sum().plot(kind='bar')
    plt.legend()
    plt.show()


def render_map_clusters(df, df_clusters, kclusters=3):
    map_clusters = folium.Map(
        location=get_geo_location("New York"), zoom_start=11)
    colours = ['red', 'black', 'blue']
    x = np.arange(kclusters)
    ys = [i + x + (i*x)**2 for i in range(kclusters)]
    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    rainbow = [colors.rgb2hex(i) for i in colors_array]
    markers_colors = []
    for lat, lon, poi, cluster, bed_per_people in zip(df['Latitude'], df['Longitude'], df['Borough'], df['Cluster Labels'], df_clusters[:, 1]):
        label = folium.Popup(
            str(poi) + ' Cluster ' + str(cluster),
            parse_html=True
        )
        folium.CircleMarker(
            [lat, lon],
            radius=bed_per_people,
            popup=label,
            color=colours[cluster],
            fill=True,
            fill_color=colours[cluster],
            fill_opacity=0.7).add_to(map_clusters)

    save_map(map_clusters)


def get_chunks(lst, n, chunk_id=0):
    nlist = []
    for i in range(0, len(lst), n):
        nlist.append(lst[i:i + n])
    return nlist[chunk_id]


def get_bed_per_hospital():
    ROOT_URL = "https://profiles.health.ny.gov/hospital/printview/{}"
    NYM_NYC = [
        103016, 106804, 102908, 103035, 102934, 1256608, 105117, 103009, 102974, 103006, 103041, 105086, 103056, 103086, 102973,
        102970, 102950, 103074, 103008, 103007, 102985, 103012, 106809, 102937, 103068, 102944, 102995, 106803, 102916, 105109,
        102914, 102960, 103038, 106810, 106811, 102961, 102940, 102933, 103078, 254693, 103065, 103021, 103080, 103033, 102919,
        105116, 106825, 103084, 103087, 102989, 102929, 106817, 106819, 103073, 103085, 103025
    ]
    NYM_LI = [
        102999, 103062, 102928, 103002, 102980, 103077, 103049, 103011, 102918, 102965, 102994, 102966, 103069, 1189331, 102926,
        103088, 103045, 103000, 103070, 105137, 103082, 102954, 103072
    ]
    BRONX = [
        102908, 106804, 105117, 102973, 102950, 106809, 102937, 103068, 102944, 103078, 103087
    ]
    QUEENS = [
        102974, 103006, 102912, 103074, 103008, 105109, 102933, 103033, 103084
    ]

    HOSPITALS = list(set(NYM_LI + NYM_NYC + BRONX + QUEENS))
    print('Total hospitals', len(HOSPITALS))

    hospital_data = []

    for val in HOSPITALS:
        print("Processing hospital", val)
        url = ROOT_URL.format(val)
        browser = webdriver.Safari()
        try:
            browser.get(url)
            time.sleep(10)
            html = browser.page_source
            soup = BeautifulSoup(html, 'html.parser')
            hospital_name = soup.find('h2').text
            table = soup.select("table", id="number-of-beds")[0]
            rows = table.findAll('tr')
            hospital_name = soup.find('h2').text.strip()
            icu_beds = 0
            for row in rows:
                tds = row.findAll('td')
                should_record = False
                for td in tds:
                    if "intensive care beds" == td.text.lower():
                        should_record = True
                        continue
                    if should_record:
                        icu_beds = td.text

            bed_number = rows[-1].findAll('td')[-1].text
            print(hospital_name, bed_number, icu_beds)
            hospital_data.append([hospital_name, bed_number, icu_beds])
        except Exception as e:
            print(e)
        browser.quit()
    df = pd.DataFrame(
        hospital_data, columns=[
            "Hospital Name", "Bed Number", "ICU Bed Number"
        ]
    )
    df = df.drop_duplicates(subset='Hospital Name', keep='last')
    df.to_csv('hospital_beds_.csv')


def get_hospital_per_neighborhood_borough(df):
    column_names = ['ID', 'Name', 'Latitude',
                    'Longitude', "Borough", "Neighborhood"]
    data = []
    for i, row in df.iterrows():
        h_df = get_hospital_data(
            row["Latitude"], row["Longitude"], row["Borough"], row["Neighborhood"])
        if h_df is not None:
            for x, hrow in h_df.iterrows():
                data.append([hrow[column] for column in column_names])

    n_df = pd.DataFrame(data, columns=column_names)
    n_df.to_csv('hospital_per_boro_nei.csv')


df = pd.read_csv("hospitals.csv")
h_df = pd.read_csv("hospital_beds.csv")
h_pbn_df = pd.read_csv("hospital_per_boro_nei.csv")
# get_hospital_per_neighborhood_borough(df)
# get_bed_per_hospital()


def combine_hospital_beds_with_boro_neighborhood(
    hospital_df,
    hospital_boro_nei_df
):
    data = []
    column_names = [
        "Hospital Name", "Bed Number", "ICU Bed Number"
    ]
    boro_neig_column_names = [
        "Borough", "Neighborhood"
    ]
    for i, row in hospital_df.iterrows():
        data_per_hospital = None
        max_ratio = 0
        for x, hrow in hospital_boro_nei_df.iterrows():
            ratio = fuzz.token_sort_ratio(row["Hospital Name"], hrow["Name"])
            if ratio > max_ratio:
                max_ratio = ratio
                data_per_hospital = [
                    row[column] for column in column_names] + \
                    [hrow[column] for column in boro_neig_column_names]
        if data_per_hospital:
            data.append(data_per_hospital)

    df = pd.DataFrame(data, columns=column_names+boro_neig_column_names)
    df = df.drop_duplicates(
        subset=["Borough", "Neighborhood", "Hospital Name"], keep="last"
    )
    df.to_csv('cleaned_hospital_data.csv')
    return df


# print(h_df.head())
# print(h_pbn_df.head())
# combine_hospital_beds_with_boro_neighborhood(h_df, h_pbn_df)

c_df = pd.read_csv('cleaned_hospital_data.csv')

print(c_df.head(20))

c_df = c_df.groupby(
    ["Neighborhood", "Borough"]).agg({'Bed Number': "sum", "ICU Bed Number": "sum"})


print(c_df.head(20))


ny_df = get_new_york_data()
ny_borough_df = get_population_per_neighbourhood(True)
ny_df.set_index('Neighborhood')
ny_borough_df.set_index('Neighborhood')
ny_p_df = pd.merge(ny_df, ny_borough_df)
# df['Hospitals'] = df.apply(lambda row:get_hospital_per_neighborhood(row),axis=1)


def get_bed_per_person(row):
    return row["Bed Number"] * 100 / row["Population"]


def get_icu_bed_per_person(row):
    return row["ICU Bed Number"] * 100 / row["Population"]


ny_p_df.set_index('Neighborhood')

df = pd.merge(c_df, ny_p_df, how="inner", on=["Borough", "Neighborhood"])
df.to_csv('final_data.csv')

df["Bed per Person"] = df.apply(
    lambda row: get_bed_per_person(row), axis=1)

df["ICU Bed per Person"] = df.apply(
    lambda row: get_icu_bed_per_person(row), axis=1)

plot_kmeans(df[["Latitude", "Longitude",
                "Bed per Person"]])

# set number of clusters
kclusters = 3
# run k-means clustering
df_clusters = df[["Latitude", "Longitude",
                  "Population", "Bed per Person", "ICU Bed per Person"]]
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(df_clusters)
# check cluster labels generated for each row in the dataframe
print(kmeans.labels_[0:24])

df.insert(0, 'Cluster Labels', kmeans.labels_)
print(df.head(30))
render_map_clusters(df, df_clusters)


show_bar_chart(ny_p_df)
# show_bar_chart(df, "Hospitals", title="Hospitals per Borough", y_label="Hospital")
# show_bar_chart(df, "Hospital vs Population", title="Hospital vs Population Borough", y_label="Hospital vs Population")

y_kmeans = kmeans.predict(df_clusters)
plt.scatter(df_clusters[["Population"]], df_clusters[[
            "Bed per Person"]], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Population')
plt.ylabel('Bed per Person')
plt.show()

y_kmeans = kmeans.predict(df_clusters)
plt.scatter(df_clusters[["Population"]], df_clusters[[
            "Bed per Person"]], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 2], c='black', s=200, alpha=0.5)
plt.xlabel('Population')
plt.ylabel('Bed per Person')
plt.show()


render_map_clusters(df, df_clusters)

print(df[(df['Cluster Labels'] == 0)].head())
print(df[(df['Cluster Labels'] == 2)].head())
print(df[(df['Cluster Labels'] == 1)].head())
