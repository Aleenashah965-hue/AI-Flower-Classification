# ============================================================
# AI Flower Classification System
# DecodeLabs Internship Project
# ============================================================
"""
A Streamlit application that demonstrates an end-to-end
supervised learning workflow on the classic Iris dataset:
dataset exploration, visualization, model training, and
interactive prediction.
"""
 
import io
 
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
 
# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fc;
    }
    h1, h2, h3 {
        color: #1f4e79;
    }
    .card {
        background: white;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    [data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    [data-testid="stSidebar"] * {
        color: #f0f2f6 !important;
    }
    div[data-testid="stMetric"] {
        background: white;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetric"] * {
        color: #1f4e79 !important;
    }
    div[data-testid="stMetricLabel"] * {
        color: #5a6b7d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
SPECIES_STYLE = {
    "setosa": {"emoji": "🌼", "color": "#2e7d32", "banner": st.success},
    "versicolor": {"emoji": "🌺", "color": "#1565c0", "banner": st.info},
    "virginica": {"emoji": "🌷", "color": "#c62828", "banner": st.error},
}
 
FEATURE_LABELS = {
    "sepal length (cm)": "Sepal Length (cm)",
    "sepal width (cm)": "Sepal Width (cm)",
    "petal length (cm)": "Petal Length (cm)",
    "petal width (cm)": "Petal Width (cm)",
}
 
 
# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the Iris dataset into a labeled DataFrame."""
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=iris.feature_names)
    frame["Species"] = iris.target_names[iris.target]
    return frame, iris
 
 
df, iris = load_data()
 
 
# ------------------------------------------------------------
# Sidebar / Navigation
# ------------------------------------------------------------
def render_sidebar() -> str:
    st.sidebar.title("🌸 AI Flower Classifier")
 
    selected_page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dataset",
            "📈 Visualization",
            "🤖 Train Model",
            "🌸 Prediction",
            "ℹ️ About",
        ],
    )
 
    st.sidebar.markdown("---")
    st.sidebar.success("DecodeLabs Internship")
    st.sidebar.caption("Built with Streamlit • scikit-learn • Plotly")
 
    return selected_page
 
 
# ------------------------------------------------------------
# Home Page
# ------------------------------------------------------------
def page_home():
    st.title("🌸 AI Flower Classification System")
 
    st.write(
        """
        Welcome to the **Iris Flower Classification** project.
 
        This application predicts the species of an Iris flower using a
        Decision Tree classifier, and walks through the full ML workflow
        from raw data to live prediction.
 
        **Features:**
        - ✅ Dataset preview & summary statistics
        - ✅ Interactive visualizations
        - ✅ Model training with live metrics
        - ✅ Classification report & confusion matrix
        - ✅ Real-time prediction on custom measurements
        """
    )
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(iris.feature_names))
    with col3:
        st.metric("Classes", len(iris.target_names))
 
    st.markdown("---")
 
    left, right = st.columns(2)
    with left:
        st.subheader("Dataset Features")
        st.dataframe(
            pd.DataFrame({"Feature Name": iris.feature_names}),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.subheader("Flower Classes")
        st.dataframe(
            pd.DataFrame({"Species": iris.target_names}),
            use_container_width=True,
            hide_index=True,
        )
 
    st.info("Use the sidebar to explore the dataset, train the model, and make predictions.")
 
 
# ------------------------------------------------------------
# Dataset Page
# ------------------------------------------------------------
def page_dataset():
    st.title("📊 Iris Dataset")
 
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)
 
    st.markdown("---")
 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: **{df.shape[0]}**")
        st.write(f"Columns: **{df.shape[1]}**")
 
        st.subheader("Column Names")
        st.write(df.columns.tolist())
 
    with col2:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().rename("Missing Count"))
 
        st.subheader("Data Types")
        st.dataframe(df.dtypes.astype(str).rename("Data Type"))
 
    st.markdown("---")
 
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
 
 
# ------------------------------------------------------------
# Visualization Page
# ------------------------------------------------------------
def page_visualization():
    st.title("📈 Data Visualization")
 
    chart = st.selectbox(
        "Choose Visualization",
        ["Histogram", "Scatter Plot", "Box Plot", "Pie Chart"],
    )
 
    numeric_columns = iris.feature_names
 
    if chart == "Histogram":
        feature = st.selectbox("Select Feature", numeric_columns)
        fig = px.histogram(
            df, x=feature, color="Species",
            title=f"Distribution of {FEATURE_LABELS.get(feature, feature)}",
        )
        st.plotly_chart(fig, use_container_width=True)
 
    elif chart == "Scatter Plot":
        x_axis = st.selectbox("X-Axis", numeric_columns)
        y_axis = st.selectbox("Y-Axis", numeric_columns, index=2)
        fig = px.scatter(
            df, x=x_axis, y=y_axis, color="Species",
            title="Feature Relationship",
        )
        st.plotly_chart(fig, use_container_width=True)
 
    elif chart == "Box Plot":
        feature = st.selectbox("Select Feature", numeric_columns, key="box")
        fig = px.box(
            df, x="Species", y=feature, color="Species",
            title=f"Spread of {FEATURE_LABELS.get(feature, feature)} by Species",
        )
        st.plotly_chart(fig, use_container_width=True)
 
    elif chart == "Pie Chart":
        count_df = df["Species"].value_counts().reset_index()
        count_df.columns = ["Species", "Count"]
        fig = px.pie(count_df, names="Species", values="Count", title="Species Distribution")
        st.plotly_chart(fig, use_container_width=True)
 
 
# ------------------------------------------------------------
# Train Model Page (training + dashboard combined)
# ------------------------------------------------------------
def page_train_model():
    st.title("🤖 Train Decision Tree Model")
    st.write("Train a Decision Tree classifier on the Iris dataset and review its performance.")
 
    if st.button("🚀 Train Model", use_container_width=True):
        X = df[iris.feature_names]
        y = df["Species"]
 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
 
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
 
        # Persist everything needed for later reruns in session_state,
        # so the dashboard below doesn't depend on the button still
        # being "pressed" on this particular script run.
        st.session_state["model"] = model
        st.session_state["accuracy"] = accuracy
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["prediction"] = prediction
        st.session_state["train_size"] = len(X_train)
        st.session_state["test_size"] = len(X_test)
 
        st.success("✅ Model trained successfully!")
        st.balloons()
 
    if "accuracy" not in st.session_state:
        return  # Nothing trained yet — stop here.
 
    accuracy = st.session_state["accuracy"]
    y_test = st.session_state["y_test"]
    prediction = st.session_state["prediction"]
 
    st.markdown("---")
    st.subheader("📋 Classification Report")
    report_df = pd.DataFrame(
        classification_report(y_test, prediction, output_dict=True)
    ).transpose()
    st.dataframe(report_df, use_container_width=True)
 
    st.markdown("---")
    st.subheader("🔲 Confusion Matrix")
    cm = confusion_matrix(y_test, prediction, labels=iris.target_names)
    cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)
    st.dataframe(cm_df, use_container_width=True)
 
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", st.session_state["train_size"])
    with col2:
        st.metric("Testing Samples", st.session_state["test_size"])
    with col3:
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
 
    # ---------------- Dashboard ----------------
    st.markdown("---")
    st.header("📊 AI Dashboard")
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")
    with col2:
        st.metric("🌸 Classes", len(iris.target_names))
    with col3:
        st.metric("📄 Total Samples", len(df))
 
    st.markdown("---")
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=list(iris.target_names),
        y=list(iris.target_names),
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="Confusion Matrix Heatmap",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("---")
    prediction_df = pd.DataFrame({"Actual": y_test.values, "Predicted": prediction})
    csv_buffer = io.StringIO()
    prediction_df.to_csv(csv_buffer, index=False)
 
    st.download_button(
        label="📥 Download Prediction CSV",
        data=csv_buffer.getvalue(),
        file_name="prediction_results.csv",
        mime="text/csv",
    )
 
    st.success("Model training and evaluation complete ✅")
 
 
# ------------------------------------------------------------
# Prediction Page
# ------------------------------------------------------------
def page_prediction():
    st.title("🌸 Iris Flower Prediction")
 
    if "model" not in st.session_state:
        st.warning("⚠️ Please train the model first from the '🤖 Train Model' page.")
        return
 
    st.subheader("Enter Flower Measurements")
 
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5, 0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 0.2, 0.1)
 
    st.markdown("---")
 
    if not st.button("🌼 Predict Flower", use_container_width=True):
        return
 
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=iris.feature_names,
    )
    flower = st.session_state["model"].predict(sample)[0]
    style = SPECIES_STYLE.get(flower, SPECIES_STYLE["versicolor"])
 
    st.markdown("## 🌸 Prediction Result")
    style["banner"](f"{style['emoji']} Predicted Flower: **{flower.upper()}**")
 
    st.markdown(
        f"""
        <div class="card" style="border-left: 6px solid {style['color']};">
            <h3 style="margin:0;">{style['emoji']} {flower.capitalize()}</h3>
            <p style="color:#555;margin-top:6px;">
                Predicted from the measurements you entered, using the trained
                Decision Tree model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    st.markdown("---")
    st.subheader("Input Values")
    result = pd.DataFrame(
        {
            "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width],
        }
    )
    st.dataframe(result, use_container_width=True, hide_index=True)
 
 
# ------------------------------------------------------------
# About Page
# ------------------------------------------------------------
def page_about():
    st.title("ℹ️ About This Project")
 
    st.write(
        """
        **AI Flower Classification System** is a machine learning demo built
        around the classic Iris dataset, showcasing a complete workflow from
        data exploration to live inference.
 
        **Tech stack**
        - Streamlit — application framework & UI
        - scikit-learn — Decision Tree classifier, train/test split, metrics
        - Plotly — interactive charts and heatmaps
        - Pandas — data handling
 
        **Workflow**
        1. Explore the dataset and its summary statistics
        2. Visualize feature distributions and relationships
        3. Train a Decision Tree model and review accuracy, the
           classification report, and the confusion matrix
        4. Enter custom measurements to get a live prediction
        """
    )
 
    st.markdown("---")
    st.caption("DecodeLabs Internship Project")
 
 
# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
PAGES = {
    "🏠 Home": page_home,
    "📊 Dataset": page_dataset,
    "📈 Visualization": page_visualization,
    "🤖 Train Model": page_train_model,
    "🌸 Prediction": page_prediction,
    "ℹ️ About": page_about,
}
 
 
def main():
    page = render_sidebar()
    PAGES[page]()
 
 
if __name__ == "__main__":
    main()