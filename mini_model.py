import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
import hiplot as hip
from PIL import Image
import io
import plotly.graph_objects as go
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import scikitplot as skplt


st.title('Penguins dataset', anchor=None)
st.write("Can we name the species of the penguin just from the given data? Is the dataset biased? or do we have all the data necessary to detect a particular type of penguin just by looking at it?")
st.markdown('**Name**: Yasasree Singam')
st.markdown('**libraries**: seaborn,pandas,altair,matplotlib,plotly,missingno,hiplot')
st.markdown('**Github link**: https://github.com/Yasasree-Singam/CMSE_project')
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

tab1,tab2,tab3, tab4, tab5 = st.tabs(['Univariate','bivariate','multivariate','Machine learning model','user_input'])

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

with tab4:
    st.header("classification model")
    df = sns.load_dataset("penguins")
    df_sex = df.dropna(subset=['sex']).copy()
    buffer = io.StringIO()
    df_sex.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("Now we can see that there are no NaN values. For our understanding, let's just delete the bill length in 2nd column which has a value of 40.3 and species in 4th column which is Adelie")
    # df_sex.loc[2, 'bill_length_mm'] = np.nan
    # df_sex.loc[4, 'species'] = np.nan
    X = df_sex.copy()  # copy of the dataframe
    y = X.pop('sex')      # pops out the sex column from the dataframe
    lb = LabelBinarizer()   # converts the string data into int(0, 1....)
    y = lb.fit_transform(y)
    y = y.ravel()
    categorical_columns = []
    numerical_columns = []

    # Seperating the categorical columns and numeric columns with its dtype
    for col in X.columns:
        if X[col].dtype == 'O':
            categorical_columns.append(col)
        else:
            numerical_columns.append(col)
    X_cat = X[categorical_columns]
    X_num = X[numerical_columns]

# For categorical data we are imputing nans using the strategy - most frequent
# Converting string values into numeric values which is done by OneHotEncoder by assigning unique values to a number

    # categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')),('onehot',OneHotEncoder())])
    si_cat = SimpleImputer(strategy = 'most_frequent')
    imputed_X_cat = si_cat.fit_transform(X_cat)
    onehot_cat = OneHotEncoder()
    dummy_coded_X_cat = onehot_cat.fit_transform(imputed_X_cat)

# for numeric data we are imputing the nans using the strategy - mean(default)
# scaling the data using StandardScaler

    # numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer()),('z_scaler',StandardScaler())])
    si_num = SimpleImputer()
    imputed_X_num = si_num.fit_transform(X_num)
    z_scaler = StandardScaler()
    X_num_std = z_scaler.fit_transform(imputed_X_num)

# horizontal stacking of the categorical data and numeric data into a single dataframe and converting it into array 

    # preprocessor = ColumnTransformer(transformers = [('cat', categorical_transformer, categorical_columns), ('num',  numeric_transformer, numerical_columns)] )
    X_stupid = np.hstack((X_num_std,dummy_coded_X_cat.toarray()))
#     names = ["Nearest Neighbors","RBF SVM","Decision Tree", "Random Forest","Neural Net", "Logistic Regression"] 
    
    st.write("""
# Explore different classifier and datasets
Which one is the best?
""")
    classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ("KNN","SVM","Decision Tree", "Random Forest","Neural Net", "Logistic Regression")
)
#     classifiers = [
#     KNeighborsClassifier(3),
#     svm.SVC(gamma=0.001),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     LogisticRegression(n_jobs = -1)
# ]
    st.write('Shape of dataset:', X_stupid.shape)
    st.write('number of classes:', len(np.unique(y)))
    # names = {"Nearest Neighbors": KNeighborsClassifier(3),"RBF SVM": svm.SVC(gamma=0.001),"Decision Tree":DecisionTreeClassifier(max_depth=5), "Random Forest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),"Neural Net":MLPClassifier(alpha=1, max_iter=1000),"Logistic Regression":LogisticRegression(n_jobs = -1)}
    
    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
        elif clf_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K
        elif clf_name == "Decision Tree":
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
        elif clf_name == "Neural Net":
            alpha = st.sidebar.slider('alpha', 1, 5)
            params['alpha'] =alpha
            max_iter = st.sidebar.slider('max_iter', 1, 100)
            params['max_iter'] = max_iter
        elif clf_name == "Logistic Regression":
            n_jobs = st.sidebar.slider('alpha', -1, 1)
            params['n_jobs'] =n_jobs
        else:
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            params['n_estimators'] = n_estimators
        return params

    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'])
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        elif clf_name == "Decision Tree":
            clf = DecisionTreeClassifier(max_depth=params['max_depth'])
        elif clf_name == "Neural Net":
            clf = MLPClassifier(alpha= params['alpha'], max_iter= params['max_iter'])
        
        elif clf_name == "Logistic Regression":
            clf = LogisticRegression(n_jobs = params['n_jobs'])
        else:
            clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], random_state=1234)
        return clf

    clf = get_classifier(classifier_name, params)

    X_train, X_test, y_train, y_test = train_test_split(X_stupid, y, test_size=0.2, random_state=1234)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    
    st.header("Confusion Matrix")

    fig=plt.figure(figsize=(1,1))
    sns.heatmap(cm,annot=True)
    st.pyplot(plt)

    st.write(f'Males predicted as males= {cm[0,0]}')
    st.write(f'Males predicted as Females = {cm[0,1]}')
    st.write(f'Females predicted as males = {cm[1,0]}')
    st.write(f'Females predicted as females= {cm[1,1]}')
    st.write('For 0')

    st.write(f'True positive = {cm[0,0]}')
    st.write(f'Flase Negative = {cm[0,1]}')
    st.write(f'False positive = {cm[1,0]}')
    st.write(f'True Negative= {cm[1,1]}')

    st.write('For 1' )
    st.write(f'True positive = {cm[1,1]}')
    st.write(f'Flase Negative = {cm[1,0]}')
    st.write(f'False positive = {cm[0,1]}')
    st.write(f'True Negative = {cm[0,0]}')

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cr = pd.DataFrame(report_dict)
    st.header("Classification report")
    st.write(cr)


with tab5:
    st.header('User_inputs',anchor = None)
    species= st.radio("What species do you want to check for?",
    ('Adelie', 'Gentoo', 'Chinstrap'))

    if species == 'Adelie':
        st.write('You selected Adelie.')
    elif species == 'Gentoo':
        st.write('You selected Gento0.')
    elif species == 'Chinstrap':
        st.write('You selected Chinstrap')
    else:
        st.write('You didnt select anything')

    island= st.radio("What island do you want to check for?",
    ('Torgersen', 'Biscoe', 'Dream'))

    if island == 'Torgersen':
        st.write('You selected Torgersen.')
    elif island == 'Biscoe':
        st.write('You selected Biscoe.')
    elif island == 'Dream':
        st.write('You selected Dream.')
    else:
        st.write('You didnt select anything')
    
    bill_length_mm = st.slider('bill_length_mm',30,60)
    st.write('Bill length is ', bill_length_mm)

    bill_depth_mm = st.slider('bill_depth_mm', 13, 22)
    st.write('Bill depth is ', bill_depth_mm)

    flipper_length_mm = st.slider('flipper_length_mm',172,231)
    st.write('flipper length is ', flipper_length_mm)

    body_mass_g = st.slider('body_mass_g',2700,6300)
    st.write('Body mass is ', body_mass_g)

    features={"species":species,'island':island,"bill_length_mm":bill_length_mm, "bill_depth_mm":bill_depth_mm,'flipper_length_mm':flipper_length_mm,'body_mass_g': body_mass_g}
    smap = pd.DataFrame(features, index =[0])

    categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')),('onehot',OneHotEncoder())])

    numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer()),('z_scaler',StandardScaler())])

    preprocessor = ColumnTransformer(transformers = [('cat', categorical_transformer, categorical_columns), ('num',  numeric_transformer, numerical_columns)] )

    pipe = Pipeline(steps = [('preprocessor', preprocessor),('classifier', LogisticRegression(n_jobs = -1))] )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
    pipe.fit(X_train, y_train)
    st.write(smap)

    # clf = LogisticRegression(n_jobs = -1)
    y_pred = pipe.predict(smap)

    if y_pred[0] == 0:
        st.write('prediction = Male')
    else:
        st.write('prediction = Female')
    
