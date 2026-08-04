import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Malnutrition AI System",
    page_icon="🧑‍⚕️",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "malnutrition_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "state_encoder.pkl"
)

IMAGE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "image_model.keras"
)

MAP_PATH = os.path.join(
    BASE_DIR,
    "visualizations",
    "malnutrition_map.html"
)

IMAGE_RESULTS_PATH = os.path.join(
    BASE_DIR,
    "output",
    "image_evaluation_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🧑‍⚕️ Geo-Specific Malnutrition AI System"
)

st.markdown(
    "### AI-based EHR and neonatal image analysis system"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a module:",
    [
        "🏠 Home",
        "📊 EHR Analysis",
        "📷 Image Analysis",
        "🗺️ District Map",
        "📈 Model Performance"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.header("Welcome 👋")

    st.write(
        """
        This system combines health-record analysis and
        neonatal image analysis to identify malnutrition risk.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "📊 **EHR Analysis**\n\n"
            "Upload health data and analyze malnutrition risk."
        )

    with col2:

        st.info(
            "📷 **Image Analysis**\n\n"
            "Upload a neonatal image for nutrition prediction."
        )

    with col3:

        st.info(
            "🗺️ **District Map**\n\n"
            "View district-wise malnutrition risk."
        )

    st.divider()

    st.subheader("🔍 System Workflow")

    st.write(
        """
        **EHR Dataset**
        ↓
        **Column Mapping**
        ↓
        **State Encoding**
        ↓
        **AI Risk Prediction**
        ↓
        **District-wise Results**
        ↓
        **District Map**
        """
    )

    st.write(
        """
        **Neonatal Image**
        ↓
        **Image Preprocessing**
        ↓
        **Image AI Model**
        ↓
        **Nutrition Class Prediction**
        """
    )


# ============================================================
# EHR ANALYSIS
# ============================================================

elif page == "📊 EHR Analysis":

    st.header("📊 EHR Dataset Analysis")

    st.write(
        "Upload a CSV dataset to begin the analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload EHR Dataset",
        type=["csv"],
        key="ehr_upload"
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.success(
                "✅ Dataset uploaded successfully!"
            )

        except Exception as e:

            st.error(
                "❌ Could not read the CSV file."
            )

            st.code(
                str(e)
            )

            st.stop()


        # ----------------------------------------------------
        # DATASET PREVIEW
        # ----------------------------------------------------

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.write(
            f"**Rows:** {df.shape[0]} | "
            f"**Columns:** {df.shape[1]}"
        )


        # ----------------------------------------------------
        # COLUMN MAPPING
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🧠 Map Dataset Columns"
        )

        columns = list(
            df.columns
        )

        district = st.selectbox(
            "District Column",
            columns,
            key="district_column"
        )

        state = st.selectbox(
            "State Column",
            columns,
            key="state_column"
        )

        diarrhoea = st.selectbox(
            "Diarrhoea Column",
            columns,
            key="diarrhoea_column"
        )

        diet = st.selectbox(
            "Adequate Diet Column",
            columns,
            key="diet_column"
        )

        anaemia = st.selectbox(
            "Anaemia Column",
            columns,
            key="anaemia_column"
        )


        # ----------------------------------------------------
        # MAPPING BUTTON
        # ----------------------------------------------------

        if st.button(
            "Continue with Mapping",
            type="primary"
        ):

            mapped_df = pd.DataFrame()

            mapped_df[
                "District Names"
            ] = df[district]

            mapped_df[
                "State/UT"
            ] = (
                df[state]
                .astype(str)
                .str.strip()
            )

            mapped_df[
                "Diarrhoea"
            ] = pd.to_numeric(
                df[diarrhoea],
                errors="coerce"
            )

            mapped_df[
                "Adequate_Diet"
            ] = pd.to_numeric(
                df[diet],
                errors="coerce"
            )

            mapped_df[
                "Anaemia"
            ] = pd.to_numeric(
                df[anaemia],
                errors="coerce"
            )


            # ------------------------------------------------
            # REMOVE MISSING VALUES
            # ------------------------------------------------

            mapped_df = mapped_df.dropna(
                subset=[
                    "District Names",
                    "State/UT",
                    "Diarrhoea",
                    "Adequate_Diet",
                    "Anaemia"
                ]
            )

            mapped_df = mapped_df.reset_index(
                drop=True
            )


            st.success(
                "✅ Columns mapped successfully!"
            )


            # ------------------------------------------------
            # SHOW MAPPED DATA
            # ------------------------------------------------

            st.subheader(
                "📋 Mapped Dataset"
            )

            st.dataframe(
                mapped_df.head(),
                use_container_width=True
            )

            st.write(
                f"**Valid rows:** {len(mapped_df)}"
            )


            # ------------------------------------------------
            # SAVE MAPPED DATASET
            # ------------------------------------------------

            mapped_path = os.path.join(
                OUTPUT_DIR,
                "mapped_dataset.csv"
            )

            mapped_df.to_csv(
                mapped_path,
                index=False
            )


            # =================================================
            # LOAD EHR MODEL
            # =================================================

            st.divider()

            st.subheader(
                "🤖 EHR Model"
            )

            if not os.path.exists(
                MODEL_PATH
            ):

                st.error(
                    "❌ EHR model not found."
                )

                st.code(
                    MODEL_PATH
                )

                st.stop()


            if not os.path.exists(
                ENCODER_PATH
            ):

                st.error(
                    "❌ State encoder not found."
                )

                st.code(
                    ENCODER_PATH
                )

                st.stop()


            # ------------------------------------------------
            # LOAD MODEL
            # ------------------------------------------------

            try:

                model = joblib.load(
                    MODEL_PATH
                )

                st.success(
                    "✅ EHR model loaded successfully!"
                )

            except Exception as e:

                st.error(
                    "❌ Could not load EHR model."
                )

                st.code(
                    str(e)
                )

                st.stop()


            # ------------------------------------------------
            # LOAD ENCODER
            # ------------------------------------------------

            try:

                encoder = joblib.load(
                    ENCODER_PATH
                )

                st.success(
                    "✅ State encoder loaded successfully!"
                )

            except Exception as e:

                st.error(
                    "❌ Could not load state encoder."
                )

                st.code(
                    str(e)
                )

                st.stop()


            # =================================================
            # STATE ENCODING
            # =================================================

            st.subheader(
                "🔢 State/UT Encoding"
            )

            try:

                state_values = (
                    mapped_df["State/UT"]
                    .astype(str)
                    .str.strip()
                )

                known_states = set(
                    encoder.classes_
                )

                uploaded_states = set(
                    state_values.unique()
                )

                unknown_states = (
                    uploaded_states
                    - known_states
                )

                if unknown_states:

                    st.error(
                        "❌ The uploaded dataset contains "
                        "State/UT values that were not present "
                        "during model training."
                    )

                    st.write(
                        "Unknown State/UT values:"
                    )

                    st.write(
                        sorted(
                            unknown_states
                        )
                    )

                    st.info(
                        "Use the same state names as the "
                        "training dataset."
                    )

                    st.stop()


                mapped_df[
                    "State/UT"
                ] = encoder.transform(
                    state_values
                )

                st.success(
                    "✅ State/UT encoded successfully!"
                )

            except Exception as e:

                st.error(
                    "❌ State/UT encoding failed."
                )

                st.code(
                    str(e)
                )

                st.stop()


            # =================================================
            # MODEL INPUT
            # =================================================

            st.subheader(
                "🔎 Model Input"
            )

            X = mapped_df[
                [
                    "State/UT",
                    "Diarrhoea",
                    "Adequate_Diet",
                    "Anaemia"
                ]
            ].copy()

            st.dataframe(
                X.head(),
                use_container_width=True
            )


            # =================================================
            # PREDICTION
            # =================================================

            st.subheader(
                "🎯 Running Prediction"
            )

            try:

                predictions = model.predict(
                    X
                )

                mapped_df[
                    "Predicted Risk"
                ] = predictions

                st.success(
                    "✅ Prediction completed successfully!"
                )

            except Exception as e:

                st.error(
                    "❌ Prediction failed."
                )

                st.code(
                    str(e)
                )

                st.stop()


            # =================================================
            # SAVE PREDICTIONS
            # =================================================

            prediction_path = os.path.join(
                OUTPUT_DIR,
                "district_predictions.csv"
            )

            mapped_df.to_csv(
                prediction_path,
                index=False
            )

            st.success(
                "💾 Predictions saved successfully!"
            )


            # =================================================
            # RESULTS
            # =================================================

            st.divider()

            st.subheader(
                "🎯 Malnutrition Risk Prediction"
            )

            result_df = mapped_df[
                [
                    "District Names",
                    "State/UT",
                    "Predicted Risk"
                ]
            ]

            st.dataframe(
                result_df,
                use_container_width=True
            )


            # =================================================
            # RISK SUMMARY
            # =================================================

            st.subheader(
                "📊 Risk Summary"
            )

            risk_counts = (
                mapped_df[
                    "Predicted Risk"
                ]
                .value_counts()
            )

            high_count = int(
                risk_counts.get(
                    "High",
                    0
                )
            )

            medium_count = int(
                risk_counts.get(
                    "Medium",
                    0
                )
            )

            low_count = int(
                risk_counts.get(
                    "Low",
                    0
                )
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "🔴 High Risk",
                    high_count
                )

            with col2:

                st.metric(
                    "🟡 Medium Risk",
                    medium_count
                )

            with col3:

                st.metric(
                    "🟢 Low Risk",
                    low_count
                )


            # =================================================
            # DOWNLOAD
            # =================================================

            st.subheader(
                "💾 Download Results"
            )

            csv_data = result_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_data,
                file_name="district_predictions.csv",
                mime="text/csv"
            )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "📷 Image Analysis":

    st.header(
        "📷 Neonatal Image Analysis"
    )

    st.write(
        "Upload a neonatal image to predict the nutrition class."
    )

    uploaded_image = st.file_uploader(
        "Upload Neonatal Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="image_upload"
    )

    if uploaded_image is not None:

        try:

            from PIL import Image
            import tensorflow as tf

            image = Image.open(
                uploaded_image
            ).convert(
                "RGB"
            )

            st.subheader(
                "🖼️ Uploaded Image"
            )

            st.image(
                image,
                caption="Uploaded neonatal image",
                width=400
            )


            if not os.path.exists(
                IMAGE_MODEL_PATH
            ):

                st.error(
                    "❌ Image model not found."
                )

                st.code(
                    IMAGE_MODEL_PATH
                )

                st.stop()


            # ------------------------------------------------
            # LOAD IMAGE MODEL
            # ------------------------------------------------

            image_model = (
                tf.keras.models.load_model(
                    IMAGE_MODEL_PATH
                )
            )

            st.success(
                "✅ Image model loaded successfully!"
            )


            # ------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------

            img = image.resize(
                (224, 224)
            )

            img_array = np.array(
                img,
                dtype=np.float32
            )

            img_array = (
                img_array / 255.0
            )

            img_array = np.expand_dims(
                img_array,
                axis=0
            )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = image_model.predict(
                img_array,
                verbose=0
            )

            predicted_index = int(
                np.argmax(
                    prediction[0]
                )
            )

            confidence = (
                float(
                    np.max(
                        prediction[0]
                    )
                ) * 100
            )


            # IMPORTANT:
            # These class names match the order
            # used by your current working app.

            class_names = [
                "Normal",
                "Mild",
                "Moderate",
                "Severe"
            ]


            if predicted_index < len(
                class_names
            ):

                predicted_class = (
                    class_names[
                        predicted_index
                    ]
                )

            else:

                predicted_class = "Unknown"


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🎯 Nutrition Prediction"
            )

            st.success(
                f"Prediction: **{predicted_class}**"
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )


        except Exception as e:

            st.error(
                "❌ Image prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# DISTRICT MAP
# ============================================================

elif page == "🗺️ District Map":

    st.header(
        "🗺️ District-wise Malnutrition Risk Map"
    )

    st.write(
        "Interactive map showing district-wise "
        "malnutrition risk."
    )


    # --------------------------------------------------------
    # LOAD LATEST PREDICTIONS
    # --------------------------------------------------------

    prediction_path = os.path.join(
        OUTPUT_DIR,
        "district_predictions.csv"
    )


    if os.path.exists(
        prediction_path
    ):

        try:

            latest_predictions = pd.read_csv(
                prediction_path
            )

            st.success(
                "✅ Latest EHR prediction data found!"
            )

            st.write(
                f"**Districts available:** "
                f"{len(latest_predictions)}"
            )


            if "Predicted Risk" in (
                latest_predictions.columns
            ):

                st.subheader(
                    "📊 Current Prediction Summary"
                )

                risk_counts = (
                    latest_predictions[
                        "Predicted Risk"
                    ]
                    .value_counts()
                )

                high_count = int(
                    risk_counts.get(
                        "High",
                        0
                    )
                )

                medium_count = int(
                    risk_counts.get(
                        "Medium",
                        0
                    )
                )

                low_count = int(
                    risk_counts.get(
                        "Low",
                        0
                    )
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "🔴 High Risk",
                        high_count
                    )

                with col2:

                    st.metric(
                        "🟡 Medium Risk",
                        medium_count
                    )

                with col3:

                    st.metric(
                        "🟢 Low Risk",
                        low_count
                    )


        except Exception as e:

            st.warning(
                "⚠️ Could not read latest prediction data."
            )

            st.code(
                str(e)
            )


    # --------------------------------------------------------
    # LOAD MAP
    # --------------------------------------------------------

    if os.path.exists(
        MAP_PATH
    ):

        st.success(
            "✅ District map loaded successfully!"
        )

        try:

            with open(
                MAP_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                map_html = f.read()


            components.html(
                map_html,
                height=700,
                scrolling=True
            )


        except Exception as e:

            st.error(
                "❌ Could not display the map."
            )

            st.code(
                str(e)
            )


    else:

        st.error(
            "❌ District map file not found."
        )

        st.code(
            MAP_PATH
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.header(
        "📈 Model Performance"
    )

    st.write(
        """
        Performance summary of the EHR-based malnutrition
        risk model and neonatal image classification model.
        """
    )


    # ========================================================
    # EHR MODEL
    # ========================================================

    st.divider()

    st.subheader(
        "📊 EHR Model"
    )

    ehr_accuracy = 75.35

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "EHR Model Accuracy",
            f"{ehr_accuracy:.2f}%"
        )

    with col2:

        if os.path.exists(
            MODEL_PATH
        ):

            st.success(
                "✅ EHR model available"
            )

        else:

            st.error(
                "❌ EHR model not found"
            )


    st.write(
        """
        The EHR model uses health-related features including
        State/UT, diarrhoea, adequate diet, and anaemia to
        predict district-level malnutrition risk.
        """
    )


    # ========================================================
    # IMAGE MODEL
    # ========================================================

    st.divider()

    st.subheader(
        "📷 Neonatal Image Model"
    )


    image_accuracy = 77.57

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Image Accuracy",
            f"{image_accuracy:.2f}%"
        )

    with col2:

        st.metric(
            "Images Evaluated",
            "214"
        )

    with col3:

        if os.path.exists(
            IMAGE_MODEL_PATH
        ):

            st.success(
                "✅ Image model available"
            )

        else:

            st.error(
                "❌ Image model not found"
            )


    # ========================================================
    # CLASS PERFORMANCE
    # ========================================================

    st.subheader(
        "📋 Image Classification Performance"
    )


    performance_data = pd.DataFrame(
        {
            "Nutrition Class": [
                "Mild",
                "Moderate",
                "Normal",
                "Severe"
            ],
            "Precision": [
                0.82,
                0.71,
                0.79,
                0.75
            ],
            "Recall": [
                0.86,
                0.82,
                0.69,
                0.63
            ],
            "F1-Score": [
                0.84,
                0.76,
                0.74,
                0.69
            ]
        }
    )


    display_data = performance_data.copy()

    display_data[
        "Precision"
    ] = (
        display_data[
            "Precision"
        ] * 100
    ).round(2).astype(str) + "%"


    display_data[
        "Recall"
    ] = (
        display_data[
            "Recall"
        ] * 100
    ).round(2).astype(str) + "%"


    display_data[
        "F1-Score"
    ] = (
        display_data[
            "F1-Score"
        ] * 100
    ).round(2).astype(str) + "%"


    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )


    confusion_data = pd.DataFrame(
        {
            "Mild": [
                70,
                3,
                9,
                3
            ],
            "Moderate": [
                4,
                41,
                2,
                11
            ],
            "Normal": [
                6,
                2,
                31,
                0
            ],
            "Severe": [
                1,
                4,
                3,
                24
            ]
        },
        index=[
            "Mild",
            "Moderate",
            "Normal",
            "Severe"
        ]
    )


    st.dataframe(
        confusion_data,
        use_container_width=True
    )


    st.caption(
        "Rows represent actual classes and columns represent predicted classes."
    )


    # ========================================================
    # RESULTS FILE
    # ========================================================

    if os.path.exists(
        IMAGE_RESULTS_PATH
    ):

        st.success(
            "✅ Detailed image evaluation results are available."
        )

        try:

            results_df = pd.read_csv(
                IMAGE_RESULTS_PATH
            )

            st.download_button(
                label="⬇️ Download Image Evaluation Results",
                data=results_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="image_evaluation_results.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.warning(
                "Could not load detailed evaluation results."
            )

            st.code(
                str(e)
            )


    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "📌 Overall System Performance"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            f"""
            **EHR Model**

            Accuracy: **{ehr_accuracy:.2f}%**

            Predicts district-level malnutrition
            risk as High, Medium, or Low.
            """
        )


    with col2:

        st.info(
            f"""
            **Image Model**

            Accuracy: **{image_accuracy:.2f}%**

            Classifies neonatal images into
            Mild, Moderate, Normal, or Severe.
            """
        )


    st.success(
        """
        ✅ Both AI modules have been evaluated and
        integrated into the Streamlit application.
        """
    )