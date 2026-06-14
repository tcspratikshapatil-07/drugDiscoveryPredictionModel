import streamlit as st
from cancer_drug_predictor import CancerDrugDiscoveryPredictor

def main():
    st.set_page_config(page_title="Cancer Drug Discovery", layout="wide")

    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    section = st.sidebar.radio(
        "Select a section to display:",
        [
            "Data Loading & Model Training",
            "Feature Importance",
            "Drug Prediction",
            "Virtual Screening",
            "Model Comparison",
            "Summary Statistics",
            "Download Results"
        ]
    )

    # Initialize
    predictor = CancerDrugDiscoveryPredictor()
    smiles_df, _ = predictor.download_and_load_data()
    data = predictor.generate_cancer_specific_data(smiles_df, n_compounds=1000)
    results = predictor.train_cancer_models(data)
    predictor.feature_columns = results['feature_columns']
    feature_importance = predictor.analyze_cancer_features(data)

    # Show selected section
    if section == "Data Loading & Model Training":
        st.header("📥 Data Loading & Model Training")
        st.success(f"Generated dataset with {len(data)} compounds.")
        st.write({
            "Bioactivity MSE": round(results['bioactivity_mse'], 4),
            "Toxicity Accuracy": round(results['toxicity_accuracy'], 4),
            "Selectivity MSE": round(results['selectivity_mse'], 4)
        })
        
    elif section == "Feature Importance":
        st.header("📈 Top 10 Important Features for Cancer Drugs")
        st.dataframe(feature_importance.head(10))
    elif section == "Drug Prediction":
        st.header("Predictions of Cancer Drugs")
        examples = {
            "Imatinib-like": 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5',
            "Dasatinib-like": 'CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)N=CN3C4CCNCC4',
            "Erlotinib-like": 'CCNC(=O)C1=CC(=C(C=C1OC2=CC=NC3=C2C=C(C=C3)OC)F)OC'
        }
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Select a Known SMILES String")
            selected_label = st.selectbox("Choose an example compound:", list(examples.keys()))
            selected_smiles = examples[selected_label]
        with col2:
            st.subheader("Or Enter a Custom SMILES String")
            custom_smiles = st.text_input("Enter a SMILES string only:", "")
        smiles_to_predict = custom_smiles.strip() if custom_smiles else selected_smiles
        if smiles_to_predict:
            prediction = predictor.predict_cancer_drug_properties(smiles_to_predict)
            if prediction:
                st.subheader("Prediction Result")
                st.code(smiles_to_predict, language="none")
                st.write({
                    "Potency (μM)": round(prediction['predicted_cancer_potency'], 2),
                    "Toxicity": prediction['toxicity_prediction'],
                    "Toxicity Probability": round(prediction['toxicity_probability'], 3),
                    "Selectivity": round(prediction['cancer_selectivity'], 2),
                    "Therapeutic Index": round(prediction['therapeutic_index'], 2),
                    "Drug-likeness Score": prediction['cancer_drug_likeness_score']
            })
        else:
            st.warning("⚠️ Unable to process the entered SMILES. Please check the format.")
    elif section == "Virtual Screening":
        st.header("Virtual Screening Results")
        screening_library = data['SMILES'].sample(300).tolist()
        hits = predictor.cancer_virtual_screening(screening_library)
        st.success(f"Found {len(hits)} promising cancer drug candidates.")

        num_to_display = st.slider("Select number of top hits to be displayed", min_value=5, max_value=len(hits), value=10, step=5)

        table_data = []
        for hit in hits[:num_to_display]:
            table_data.append({
                "SMILES": hit['smiles'],
                "Potency (μM)": round(hit['predicted_cancer_potency'], 2),
                "Selectivity": round(hit['cancer_selectivity'], 2),
                "Therapeutic Index": round(hit['therapeutic_index'], 2),
                "Toxicity Probability": round(hit['toxicity_probability'], 3)
            })
        st.dataframe(table_data, use_container_width=True)
    elif section == "Model Comparison":
        st.header("📊 Model Comparison")
        st.info("This compares traditional ML models and deep learning for potency, toxicity, and selectivity.")

        if predictor.feature_columns is None:
            st.warning("Run model training first to set feature columns.")
        else:
            with st.spinner("Running model comparisons..."):
                result_df = predictor.compare_models(data)
                st.success("Comparison completed!")

             # Display table
                st.dataframe(result_df)

                # Plot metrics
                task_to_plot = st.selectbox("Select Task", ['Bioactivity', 'Toxicity', 'Selectivity'])
                metric = 'R2' if task_to_plot != 'Toxicity' else 'Accuracy'

                chart_df = result_df[result_df['Task'] == task_to_plot]
                st.bar_chart(data=chart_df.set_index('Model')[metric])
    elif section == "Summary Statistics":
        st.header("📊 Summary Statistics")
        st.write({
            "Total Compounds": len(data),
            "Avg. Potency (μM)": round(data['overall_cancer_potency'].mean(), 2),
            "Good Selectivity (>1.5)": int((data['selectivity_index'] > 1.5).sum()),
            "Non-toxic Compounds": int((data['is_toxic'] == 0).sum()),
            "High Therapeutic Index (>3)": int((data['therapeutic_index'] > 3).sum())
        })

    elif section == "Download Results":
        st.header("⬇️ Download Outputs")
        st.download_button("📥 Download Results CSV", data.to_csv(index=False), file_name="cancer_drug_discovery_results.csv")
        st.download_button("📊 Download Feature Importance", feature_importance.to_csv(index=False), file_name="cancer_drug_feature_importance.csv")

if __name__ == "__main__":
    main()
