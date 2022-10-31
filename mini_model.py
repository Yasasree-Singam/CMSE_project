import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
import hiplot as hip
from PIL import Image



st.title('Penguins dataset', anchor=None)
st.write("Can we name the species of the penguin just from the given data? Is the dataset biased? or do we have all the data necessary to detect a particular type of penguin just by looking at it?")
st.markdown('**Name**: Yasasree Singam')
st.markdown('**libraries**: seaborn,pandas,altair,matplotlib,plotly,missingno,hiplot')
# im = Image.open('C:/Users/singa/OneDrive/Desktop/CMSE_830/project2/penguins.jpg')
# st.image(im, caption='Species of penguin as per the dataset')

#st.markdown('![AltText|50](https://th.bing.com/th/id/R.e6cb5ffcb56d1994f9d5c3f33ac9211f?rik=9orasCzReXcM%2fg&riu=http%3a%2f%2fd3i3l3kraiqpym.cloudfront.net%2fwp-content%2fuploads%2f2016%2f04%2f26094914%2fAd%c3%a9lie-Chinstrap-and-gentoo-penguin-species.jpg&ehk=0Cy9AWjOykJ1QTbhtzopFANK6pLNtIPkiNVK6s92%2bGk%3d&risl=&pid=ImgRaw&r=0)')

st.markdown('![Species of penguins as per dataset](https://th.bing.com/th/id/OIP.Wdcx6lmxYpqK_U7Bsh4UJAHaDA?w=290&h=142&c=7&r=0&o=5&dpr=1.5&pid=1.7)')

st.write("The penguin dataset is a collection of images of penguin colonies in Antarctica coming from the larger penguin watch project, which was setup with the purpose of monitoring their changes in population. The images are taken by fixed cameras in over 40 different locations, which have been capturing an image per hour for several years. In order to track the colony sizes, the number of penguins in each of the images in the dataset is required.")
df = sns.load_dataset("penguins")
st.subheader('Penguins Dataframe')
st.dataframe(df.head(10))
dr = df.dropna()
dn = df.dropna()
col =dr.columns.to_list()
col2= dr.columns.to_list()

st.subheader('Attribute Information:')
str_text= '\n - This dataset has Seven columns, Three categorical variables, and Four Numeric variables.\n - The count of the numeric variables is the same for each column because the count is the number of not-empty values and two rows of all the columns have NaNs(344-2 =342).\n - The sex column has 14 NaNs.\n - The dataset consists of 7 columns.\n      species: penguin species (Chinstrap, Adélie, or Gentoo).\n      bill_length_mm: bill length (mm).\n      bill_depth_mm: bill depth (mm).\n      flipper_length_mm: flipper length (mm).\n      body_mass_g: body mass (g).\n      island: island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)\n      sex: penguin sex'
st.text(str_text)

st.subheader("Statistical information of the penguins dataset")
stats= df.describe()
st.dataframe(stats)

#tab1, tab2, tab3, tab4, tab5 = st.tabs(["Alatir plot","Heatmap","Parallel coordinate plot","Facet plot","Hi plot"])

tab1,tab2,tab3 = st.tabs(['Univariate','bivariate','multivariate'])

with tab1:
    st.header('Histogram',anchor = None)
    axis11 = st.selectbox(label = "select a column name for labeling on x-axis", options=col,key=11)

    plt.figure(figsize=(6,5))
    img_1 = px.histogram(dn, x = axis11, color = 'species', text_auto= True)
    st.plotly_chart(img_1)

    st.header("Distribution plot")
    axis12 = st.selectbox(label = "select a column name for labeling on x-axis", options = col, key = 21 )
    plt.figure(figsize= (6,5))
    img_2 = px.histogram(dn, x= axis12, color='island', marginal="box",hover_data=dr.columns, text_auto= True)
    st.plotly_chart(img_2)






with tab2:
    st.header('Altair Plot', anchor=None)
    axis1 = st.selectbox(label = "select a column name for labeling on x-axis", options=col,key=1)
    axis2 = st.selectbox(label = "select a column name for labeling on y-axis", options=col2,key=2)

    df1 = alt.Chart(df).mark_point(size = 50).encode(
    x=axis1,y=axis2,
        color=alt.Color('species', scale = alt.Scale(scheme='dark2'),legend=alt.Legend(title="Species by color",orient="left"))
    ).configure_axis(grid = False).configure_view(strokeWidth=0).properties(width = 700)

    st.altair_chart(df1)
    #plt.figure(figsize=(6, 5))



    
    st.header('Facet Plot', anchor=None)
    axis3 = st.selectbox(label = "select a column name for labeling on x-axis", options=col,key=3)
    axis4 = st.selectbox(label = "select a column name for labeling on y-axis", options=col2,key=4)

    dom = ['Biscoe', 'Dream', 'Torgersen']
    rng = ['red', 'green', 'black']
    pengs_model = alt.Chart(dr).mark_point().encode(x=alt.X(axis3,axis=alt.Axis(title='flipper_length_mm', grid=False)),y=alt.Y(axis4,axis=alt.Axis(title='body_mass_g', grid=False)),
                                        color=alt.Color('island', legend=alt.Legend(title="island by color",orient='left'),scale=alt.Scale(domain=dom, range=rng)),tooltip=['island','flipper_length_mm','bill_length_mm','body_mass_g','sex'], 
                                    facet=alt.Facet('species', columns=3),).properties(title='filpper_length vs body_mass of various species',width=300,height=250).configure_view(strokeWidth=0).interactive()
    st.altair_chart(pengs_model)


with tab3:

    st.header("Heatmap")
    fig, ax = plt.subplots()
    msno.heatmap(df,ax=ax)
    #sns.heatmap(df.corr(), ax=ax)

    st.write('What category of missingness do you think it is? MAR,MNAR or MCAR')
    st.write(fig)
    st.write("A value near 1 means if one variable appears then the other variable is very likely to be present.")
    st.write("You are right!!! It's **MAR**,Missing At Random because missingness of one value is dependent on the other.")

    st.header("pair plot")
    pair =sns.pairplot(dn, hue ='species',
    x_vars=["bill_length_mm", "bill_depth_mm", "flipper_length_mm"],
    y_vars=["bill_length_mm", "bill_depth_mm","flipper_length_mm"])
    st.pyplot(pair)

    
    
    
    
    st.header('Parallel coordinate plot', anchor=None)

    dr.replace({'species':{'Adelie':0,'Chinstrap':2,'Gentoo':4}},inplace=True)
    fig = px.parallel_coordinates(dr,color = "species",
                                dimensions=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"],
                                color_continuous_scale=px.colors.diverging.Tealrose,
                                color_continuous_midpoint=2)
    st.plotly_chart(fig, )


    st.header('Hi Plot', anchor=None)
    xp = hip.Experiment.from_dataframe(dn)
    xp.to_streamlit(ret="selected_uids", key="hip").display()
