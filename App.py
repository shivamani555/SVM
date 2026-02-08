import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
# Logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
# Session State 
if "clean_saved" not in st.session_state:
    st.session_state["clean_saved"] = False
# folder setup
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR=os.path.join(BASE_DIR,"data","raw")
CLEAN_DIR=os.path.join(BASE_DIR,"data","clean")
os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEAN_DIR,exist_ok=True)
log("Application started.")
log(f"RAW_DIR = {RAW_DIR}")
log(f"CLEAN_DIR = {CLEAN_DIR}")
# page configuration
st.set_page_config("End-to-End SVM Platform",layout="wide")
st.title("End-to-End SVM Platform")
# Sidebar: Model Settings
st.sidebar.header("SVM Settings")
kernel=st.sidebar.selectbox("Kernel",options=["linear","poly","rbf","sigmoid"])
C=st.sidebar.number_input("C(Regularization)",0.01,10.0,1.0)
gamma=st.sidebar.selectbox("Gamma",options=["scale","auto"])
log(f"Svm settings ---> kernel: {kernel}, C: {C}, gamma: {gamma}")
# Step-1: Data Ingestion
st.header("Step 1: Data Ingestion")
log("Step 1 started: Data Ingestion")
option=st.radio("Choose data source",["Download Dataset","Upload CSV"])
df=None
raw_path=None
if option=="Download Dataset":
    if st.button("Download Iris Dataset"):
       log("Downloading Iris dataset")
       url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
       response=requests.get(url)
       raw_path=os.path.join(RAW_DIR,"iris.csv")
       with open(raw_path,"wb") as f:
            f.write(response.content)
       df=pd.read_csv(raw_path)
       st.success("Dataset downloaded successfully!")
       log(f"Iris dataset saved at {raw_path}")
if option=="Upload CSV":
    uploaded_file=st.file_uploader("Upload CSV file",type=["csv"])
    if uploaded_file:
        raw_path=os.path.join(RAW_DIR,uploaded_file.name)
        with open(raw_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        df=pd.read_csv(raw_path)
        st.success("File uploaded successfully!")
        log(f"File uploaded and saved at {raw_path}")
# Step-2: EDA
if df is not None:
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    log("Step 2 started: EDA")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing Values", df.isnull().sum())
    fig,ax=plt.subplots()
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm",ax=ax) 
    st.pyplot(fig)
    log("EDA completed.")

#Step 3 : Data Cleaning
if df is not None:
    st.header("Step 3 : Data Cleaning")

    strategy = st.selectbox(
        "Missing Values Strategy",
        ["Mean","Median","Drop Rows"]
    )

    df_clean = df.copy()

    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()

    else:
        for col in df_clean.select_dtypes(include = np.number):
            if strategy == "Mean":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    st.session_state.df_clean = df_clean
    st.success("Data Cleaning Completeted")

else:
    st.info("Please Complete the step 1 (Data Ingestion) First..")


# Step 4: Save Cleaned Data
st.header("Step 4: Save Cleaned Data")

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

if st.button("Save Cleaned Data"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data to save")
    else:
        fname = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(CLEAN_DIR, fname)
        st.session_state.df_clean.to_csv(path, index=False)
        st.success("Cleaned data saved successfully")
        log(f"Cleaned data saved at {path}")

# Step 5 : Load cleaned dataset
st.header("Step 5 : Load Cleaned Dataset")

clean_files = os.listdir(CLEAN_DIR)

if not clean_files:
    st.warning("No cleaned datasets found.")
    log("No cleaned datasets found.")
    df_model = None
else:
    selected = st.selectbox("Select Cleaned dataset", clean_files)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))

    st.success(f"Loaded dataset: {selected}")
    log(f"Loaded cleaned dataset: {selected}")

    st.dataframe(df_model.head())

# Step 6: Train SVM
if df_model is not None:
    st.header("Step 6: Train SVM")
    log("Step 6 started: Training SVM")

    target = st.selectbox("Select Target Variable", df_model.columns)
    Y = df_model[target]

    # Validate target for classification
    if Y.nunique() < 2 or Y.nunique() > 20:
        st.error("Invalid target variable for classification. Please select a categorical target with 2–20 unique values.")
        st.stop()

    # Encode target if categorical
    if Y.dtype == 'object' or Y.dtype.name == 'category':
        Y = LabelEncoder().fit_transform(Y)
        log("Target variable encoded")

    # Select numeric features only
    X = df_model.drop(columns=[target])
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        st.error("No numeric features available for training.")
        st.stop()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Model
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, Y_train)

    # Evaluate
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)

    st.success(f"Accuracy: {acc:.2f}")
    log(f"SVM trained with accuracy: {acc:.2f}")

    cm = confusion_matrix(Y_test, Y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)
