import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
import pickle
import os
from cancer_drug_predictor import CancerDrugDiscoveryPredictor

# Configure Streamlit page
st.set_page_config(
    page_title="Cancer Drug Discovery Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e88e5;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .warning-card {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .error-card {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def save_predictor(predictor, filename="cancer_drug_model.pkl"):
    """Save trained predictor to file"""
    with open(filename, 'wb') as f:
        pickle.dump(predictor, f)

def load_predictor(filename="cancer_drug_model.pkl"):
    """Load trained predictor from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def draw_molecule(smiles):
    """Draw molecule structure from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            rdDepictor.Compute2DCoords(mol)
            img = Draw.MolToImage(mol, size=(300, 300))
            return img
        return None
    except:
        return None

def create_radar_chart(data_dict, title):
    """Create radar chart for drug properties"""
    categories = list(data_dict.keys())
    values = list(data_dict.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )),
        showlegend=True,
        title=title
    )
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🧬 Cancer Drug Discovery Platform</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Train Model", "Drug Prediction", "Virtual Screening", "Feature Analysis", "Batch Processing"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Train Model":
        show_training()
    elif page == "Drug Prediction":
        show_prediction()
    elif page == "Virtual Screening":
        show_screening()
    elif page == "Feature Analysis":
        show_feature_analysis()
    elif page == "Batch Processing":
        show_batch_processing()

def show_home():
    """Home page with overview and introduction"""
    st.markdown('<h2 class="sub-header">Welcome to the Cancer Drug Discovery Platform</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Platform Overview
        
        This platform leverages machine learning to accelerate cancer drug discovery by:
        
        - **Predicting Cancer Potency**: Estimate IC50 values across multiple cancer cell lines
        - **Toxicity Assessment**: Predict potential toxic effects of compounds
        - **Selectivity Analysis**: Evaluate cancer vs. normal cell selectivity
        - **Drug-likeness Scoring**: Assess compound suitability as cancer therapeutics
        - **Virtual Screening**: Screen large compound libraries efficiently
        
        ### 🔬 Supported Cancer Cell Lines
        - MCF-7 (Breast Cancer)
        - HeLa (Cervical Cancer)
        - A549 (Lung Cancer)
        - HCT-116 (Colorectal Cancer)
        - PC-3 (Prostate Cancer)
        - HepG2 (Liver Cancer)
        - U87 (Brain Cancer)
        - MDA-MB-231 (Triple-negative Breast Cancer)
        
        ### 🚀 Getting Started
        1. **Train Model**: Generate training data and train ML models
        2. **Drug Prediction**: Enter SMILES to predict drug properties
        3. **Virtual Screening**: Screen compound libraries
        4. **Feature Analysis**: Understand important molecular features
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Model Performance
        """)
        
        # Example metrics (these would be real after training)
        if st.session_state.model_trained:
            predictor = st.session_state.predictor
            st.success("✅ Models Trained Successfully")
        else:
            st.info("ℹ️ Models not yet trained")
            
        st.markdown("""
        ### 🔗 Key Features
        - Real-time SMILES validation
        - Interactive molecular visualization
        - Comprehensive drug property analysis
        - Export results to CSV
        - Batch processing capabilities
        """)

def show_training():
    """Training page for model creation"""
    st.markdown('<h2 class="sub-header">🔧 Model Training</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Training Configuration")
        
        n_compounds = st.slider(
            "Number of compounds for training:",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100
        )
        
        use_real_data = st.checkbox(
            "Attempt to download real SMILES data",
            value=True,
            help="Try to download real molecular data from Kaggle"
        )
        
        if st.button("🚀 Start Training", type="primary"):
            train_models(n_compounds, use_real_data)
    
    with col2:
        st.markdown("### Training Status")
        
        if st.session_state.model_trained:
            st.success("✅ Models Successfully Trained!")
            
            # Show model performance metrics
            if hasattr(st.session_state.predictor, 'training_results'):
                results = st.session_state.predictor.training_results
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric(
                        "Bioactivity MSE",
                        f"{results.get('bioactivity_mse', 0):.4f}"
                    )
                
                with metrics_col2:
                    st.metric(
                        "Toxicity Accuracy",
                        f"{results.get('toxicity_accuracy', 0):.3f}"
                    )
                
                with metrics_col3:
                    st.metric(
                        "Selectivity R²",
                        f"{results.get('selectivity_r2', 0):.3f}"
                    )
                
                # Training progress visualization
                st.markdown("### Training Progress")
                if 'training_history' in results:
                    history = results['training_history']
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Bioactivity Loss', 'Toxicity Loss', 
                                      'Selectivity Loss', 'Overall Score'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    epochs = list(range(1, len(history['bioactivity_loss']) + 1))
                    
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['bioactivity_loss'], 
                                 name="Bioactivity Loss"),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['toxicity_loss'], 
                                 name="Toxicity Loss"),
                        row=1, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['selectivity_loss'], 
                                 name="Selectivity Loss"),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['overall_score'], 
                                 name="Overall Score"),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No trained models available")
            st.markdown("""
            **Training Steps:**
            1. Configure training parameters
            2. Click "Start Training" 
            3. Wait for model training to complete
            4. Review performance metrics
            """)
def train_models(n_compounds, use_real_data):
    """Train the cancer drug discovery models"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing predictor...")
        progress_bar.progress(10)
        
        # Initialize predictor
        predictor = CancerDrugDiscoveryPredictor()
        
        status_text.text("Generating training data...")
        progress_bar.progress(30)
        
        # Generate training data
        training_data = predictor.generate_training_data(
            n_compounds=n_compounds,
            use_real_smiles=use_real_data
        )
        
        status_text.text("Training models...")
        progress_bar.progress(60)
        
        # Train models
        results = predictor.train_models(training_data)
        
        status_text.text("Evaluating performance...")
        progress_bar.progress(90)
        
        # Store results
        predictor.training_results = results
        st.session_state.predictor = predictor
        st.session_state.training_data = training_data
        st.session_state.model_trained = True
        
        # Save model
        save_predictor(predictor)
        
        progress_bar.progress(100)
        status_text.text("Training completed successfully!")
        
        st.success(f"✅ Successfully trained models on {len(training_data)} compounds!")
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def show_prediction():
    """Single drug prediction page"""
    st.markdown('<h2 class="sub-header">💊 Drug Prediction</h2>', 
                unsafe_allow_html=True)
    
    # Check if model is trained
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' section.")
        return
    
    predictor = st.session_state.predictor
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Enter Compound Information")
        
        # SMILES input
        smiles_input = st.text_input(
            "SMILES String:",
            value="CCO",
            help="Enter a valid SMILES representation of the molecule"
        )
        
        # Validate SMILES
        if smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol is not None:
                st.success("✅ Valid SMILES")
            else:
                st.error("❌ Invalid SMILES")
                return
        
        # Optional compound name
        compound_name = st.text_input(
            "Compound Name (optional):",
            value="",
            help="Enter a name for this compound"
        )
        
        if st.button("🔍 Predict Properties", type="primary"):
            predict_single_compound(predictor, smiles_input, compound_name)
    
    with col2:
        st.markdown("### Molecule Visualization")
        
        if smiles_input:
            mol_img = draw_molecule(smiles_input)
            if mol_img:
                st.image(mol_img, caption=f"Structure: {smiles_input}")
            else:
                st.error("Cannot generate molecule structure")

def predict_single_compound(predictor, smiles, compound_name=""):
    """Predict properties for a single compound"""
    
    try:
        with st.spinner("Predicting compound properties..."):
            results = predictor.predict_single_compound(smiles)
        
        if results is None:
            st.error("❌ Prediction failed. Please check the SMILES string.")
            return
        
        # Display results
        st.markdown("### 🎯 Prediction Results")
        
        name = compound_name if compound_name else "Unknown Compound"
        st.markdown(f"**Compound:** {name}")
        st.markdown(f"**SMILES:** `{smiles}`")
        
        # Overall drug score
        overall_score = results.get('drug_score', 0)
        score_color = "success" if overall_score > 0.6 else "warning" if overall_score > 0.3 else "error"
        
        st.markdown(f"""
        <div class="metric-card {score_color}-card">
            <h3>Overall Drug Score: {overall_score:.3f}</h3>
            <p>Composite score based on bioactivity, selectivity, and safety</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different result categories
        tab1, tab2, tab3, tab4 = st.tabs(["🧬 Bioactivity", "⚠️ Toxicity", "🎯 Selectivity", "📊 Properties"])
        
        with tab1:
            st.markdown("#### Cancer Cell Line IC50 Predictions (μM)")
            
            bioactivity_data = results.get('bioactivity_predictions', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            cell_lines = ['MCF-7', 'HeLa', 'A549', 'HCT-116', 'PC-3', 'HepG2', 'U87', 'MDA-MB-231']
            
            for i, cell_line in enumerate(cell_lines):
                ic50 = bioactivity_data.get(cell_line, 0)
                potency = "High" if ic50 < 1 else "Medium" if ic50 < 10 else "Low"
                
                with [col1, col2, col3, col4][i % 4]:
                    st.metric(
                        cell_line,
                        f"{ic50:.2f} μM",
                        delta=f"{potency} potency"
                    )
            
            # Bioactivity radar chart
            bio_dict = {k: max(0.1, 20 - v) for k, v in bioactivity_data.items()}  # Convert IC50 to activity score
            fig_bio = create_radar_chart(bio_dict, "Bioactivity Profile")
            st.plotly_chart(fig_bio, use_container_width=True)
        
        with tab2:
            st.markdown("#### Toxicity Predictions")
            
            toxicity_data = results.get('toxicity_predictions', {})
            
            tox_col1, tox_col2, tox_col3 = st.columns(3)
            
            with tox_col1:
                hepatotoxicity = toxicity_data.get('hepatotoxicity', 0)
                hep_risk = "High" if hepatotoxicity > 0.7 else "Medium" if hepatotoxicity > 0.3 else "Low"
                st.metric("Hepatotoxicity Risk", f"{hepatotoxicity:.3f}", delta=hep_risk)
            
            with tox_col2:
                cardiotoxicity = toxicity_data.get('cardiotoxicity', 0)
                card_risk = "High" if cardiotoxicity > 0.7 else "Medium" if cardiotoxicity > 0.3 else "Low"
                st.metric("Cardiotoxicity Risk", f"{cardiotoxicity:.3f}", delta=card_risk)
            
            with tox_col3:
                nephrotoxicity = toxicity_data.get('nephrotoxicity', 0)
                neph_risk = "High" if nephrotoxicity > 0.7 else "Medium" if nephrotoxicity > 0.3 else "Low"
                st.metric("Nephrotoxicity Risk", f"{nephrotoxicity:.3f}", delta=neph_risk)
            
            # Overall toxicity assessment
            avg_toxicity = np.mean(list(toxicity_data.values()))
            if avg_toxicity < 0.3:
                st.success("✅ Low overall toxicity risk")
            elif avg_toxicity < 0.7:
                st.warning("⚠️ Moderate toxicity risk - requires careful evaluation")
            else:
                st.error("❌ High toxicity risk - not recommended for development")
        
        with tab3:
            st.markdown("#### Selectivity Analysis")
            
            selectivity_data = results.get('selectivity_predictions', {})
            
            sel_col1, sel_col2 = st.columns(2)
            
            with sel_col1:
                cancer_selectivity = selectivity_data.get('cancer_selectivity', 0)
                st.metric(
                    "Cancer vs Normal Selectivity",
                    f"{cancer_selectivity:.2f}x",
                    help="Fold selectivity for cancer cells over normal cells"
                )
            
            with sel_col2:
                therapeutic_window = selectivity_data.get('therapeutic_window', 0)
                st.metric(
                    "Therapeutic Window",
                    f"{therapeutic_window:.2f}",
                    help="Ratio of toxic dose to effective dose"
                )
            
            # Selectivity interpretation
            if cancer_selectivity > 10:
                st.success("✅ Excellent selectivity for cancer cells")
            elif cancer_selectivity > 3:
                st.info("ℹ️ Good selectivity for cancer cells")
            else:
                st.warning("⚠️ Limited selectivity - may affect normal cells")
        
        with tab4:
            st.markdown("#### Molecular Properties")
            
            properties = results.get('molecular_properties', {})
            
            prop_col1, prop_col2, prop_col3 = st.columns(3)
            
            with prop_col1:
                st.metric("Molecular Weight", f"{properties.get('molecular_weight', 0):.1f} Da")
                st.metric("LogP", f"{properties.get('logp', 0):.2f}")
            
            with prop_col2:
                st.metric("HBD Count", f"{properties.get('hbd_count', 0)}")
                st.metric("HBA Count", f"{properties.get('hba_count', 0)}")
            
            with prop_col3:
                st.metric("TPSA", f"{properties.get('tpsa', 0):.1f} Ų")
                st.metric("Rotatable Bonds", f"{properties.get('rotatable_bonds', 0)}")
            
            # Lipinski's Rule of Five assessment
            st.markdown("#### Drug-likeness Assessment")
            
            mw = properties.get('molecular_weight', 0)
            logp = properties.get('logp', 0)
            hbd = properties.get('hbd_count', 0)
            hba = properties.get('hba_count', 0)
            
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1
            
            if violations == 0:
                st.success("✅ Passes Lipinski's Rule of Five")
            elif violations == 1:
                st.info("ℹ️ One Rule of Five violation")
            else:
                st.warning(f"⚠️ {violations} Rule of Five violations")
        
        # Export results
        if st.button("📥 Export Results"):
            export_results(results, smiles, compound_name)
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

def show_screening():
    """Virtual screening page for compound libraries"""
    st.markdown('<h2 class="sub-header">🔬 Virtual Screening</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' section.")
        return
    
    predictor = st.session_state.predictor
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Screening Configuration")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Upload SMILES file", "Generate random library", "Enter SMILES list"]
        )
        
        compounds_to_screen = []
        
        if input_method == "Upload SMILES file":
            uploaded_file = st.file_uploader(
                "Upload CSV/TXT file with SMILES",
                type=['csv', 'txt'],
                help="File should contain SMILES strings, optionally with compound names"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        if 'SMILES' in df.columns:
                            compounds_to_screen = df['SMILES'].tolist()
                        elif 'smiles' in df.columns:
                            compounds_to_screen = df['smiles'].tolist()
                        else:
                            compounds_to_screen = df.iloc[:, 0].tolist()
                    else:
                        content = uploaded_file.read().decode('utf-8')
                        compounds_to_screen = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    st.success(f"✅ Loaded {len(compounds_to_screen)} compounds")
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        elif input_method == "Generate random library":
            n_random = st.slider(
                "Number of random compounds:",
                min_value=10,
                max_value=500,
                value=100,
                step=10
            )
            
            if st.button("🎲 Generate Random Library"):
                compounds_to_screen = predictor.generate_random_smiles(n_random)
                st.success(f"✅ Generated {len(compounds_to_screen)} random compounds")
        
        elif input_method == "Enter SMILES list":
            smiles_text = st.text_area(
                "Enter SMILES (one per line):",
                height=200,
                placeholder="CCO\nCCN\nC1CCCCC1"
            )
            
            if smiles_text:
                compounds_to_screen = [line.strip() for line in smiles_text.split('\n') if line.strip()]
                st.info(f"ℹ️ {len(compounds_to_screen)} compounds entered")
        
        # Screening filters
        st.markdown("### Screening Filters")
        
        min_drug_score = st.slider(
            "Minimum drug score:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        max_toxicity = st.slider(
            "Maximum toxicity score:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        min_selectivity = st.slider(
            "Minimum cancer selectivity:",
            min_value=1.0,
            max_value=20.0,
            value=3.0,
            step=0.5
        )
        
        if st.button("🚀 Start Screening", type="primary") and compounds_to_screen:
            screen_compounds(
                predictor, 
                compounds_to_screen, 
                min_drug_score, 
                max_toxicity, 
                min_selectivity
            )
    
    with col2:
        st.markdown("### Screening Results")
        
        if 'screening_results' in st.session_state:
            results = st.session_state.screening_results
            
            st.markdown(f"**Total Screened:** {results['total_screened']}")
            st.markdown(f"**Hits Found:** {results['hits_found']}")
            st.markdown(f"**Hit Rate:** {results['hit_rate']:.1%}")
            
            if results['hits_found'] > 0:
                hits_df = results['hits_dataframe']
                
                # Display top hits
                st.markdown("#### Top Hits")
                st.dataframe(
                    hits_df.head(10),
                    use_container_width=True
                )
                
                # Download results
                csv = hits_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Hits",
                    data=csv,
                    file_name="screening_hits.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("ℹ️ No screening results available")

def screen_compounds(predictor, compounds, min_drug_score, max_toxicity, min_selectivity):
    """Screen a list of compounds"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        hits = []
        total_screened = 0
        
        for i, smiles in enumerate(compounds):
            status_text.text(f"Screening compound {i+1}/{len(compounds)}: {smiles[:50]}...")
            progress_bar.progress((i+1)/len(compounds))
            
            try:
                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Predict properties
                results = predictor.predict_single_compound(smiles)
                if results is None:
                    continue
                
                total_screened += 1
                
                # Apply filters
                drug_score = results.get('drug_score', 0)
                avg_toxicity = np.mean(list(results.get('toxicity_predictions', {}).values()))
                cancer_selectivity = results.get('selectivity_predictions', {}).get('cancer_selectivity', 0)
                
                if (drug_score >= min_drug_score and 
                    avg_toxicity <= max_toxicity and 
                    cancer_selectivity >= min_selectivity):
                    
                    hit = {
                        'SMILES': smiles,
                        'Drug_Score': drug_score,
                        'Toxicity_Score': avg_toxicity,
                        'Cancer_Selectivity': cancer_selectivity,
                        'MCF7_IC50': results.get('bioactivity_predictions', {}).get('MCF-7', 0),
                        'HeLa_IC50': results.get('bioactivity_predictions', {}).get('HeLa', 0),
                        'A549_IC50': results.get('bioactivity_predictions', {}).get('A549', 0)
                    }
                    hits.append(hit)
            
            except Exception as e:
                continue
        
        # Create results dataframe
        if hits:
            hits_df = pd.DataFrame(hits)
            hits_df = hits_df.sort_values('Drug_Score', ascending=False)
        else:
            hits_df = pd.DataFrame()
        
        # Store results
        screening_results = {
            'total_screened': total_screened,
            'hits_found': len(hits),
            'hit_rate': len(hits) / total_screened if total_screened > 0 else 0,
            'hits_dataframe': hits_df
        }
        
        st.session_state.screening_results = screening_results
        
        progress_bar.progress(1.0)
        status_text.text("Screening completed!")
        
        st.success(f"✅ Screening completed! Found {len(hits)} hits out of {total_screened} compounds")
        
    except Exception as e:
        st.error(f"❌ Screening failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def show_feature_analysis():
    """Feature importance analysis page"""
    st.markdown('<h2 class="sub-header">📊 Feature Analysis</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' section.")
        return
    
    predictor = st.session_state.predictor
    
    st.markdown("### Feature Importance Analysis")
    
    # Get feature importance from trained models
    try:
        feature_importance = predictor.get_feature_importance()
        
        tab1, tab2, tab3 = st.tabs(["🧬 Bioactivity Features", "⚠️ Toxicity Features", "🎯 Selectivity Features"])
        
        with tab1:
            st.markdown("#### Most Important Features for Bioactivity Prediction")
            
            bio_features = feature_importance.get('bioactivity', {})
            if bio_features:
                bio_df = pd.DataFrame(
                    list(bio_features.items()), 
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    bio_df.head(20), 
                    x='Importance', 
                    y='Feature',
                    title="Top 20 Bioactivity Features",
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### Most Important Features for Toxicity Prediction")
            
            tox_features = feature_importance.get('toxicity', {})
            if tox_features:
                tox_df = pd.DataFrame(
                    list(tox_features.items()), 
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    tox_df.head(20), 
                    x='Importance', 
                    y='Feature',
                    title="Top 20 Toxicity Features",
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### Most Important Features for Selectivity Prediction")
            
            sel_features = feature_importance.get('selectivity', {})
            if sel_features:
                sel_df = pd.DataFrame(
                    list(sel_features.items()), 
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    sel_df.head(20), 
                    x='Importance', 
                    y='Feature',
                    title="Top 20 Selectivity Features",
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Feature analysis failed: {str(e)}")

def show_batch_processing():
    """Batch processing page"""
    st.markdown('<h2 class="sub-header">⚡ Batch Processing</h2>', 
                unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' section.")
        return
    
    predictor = st.session_state.predictor
    
    st.markdown("### Upload Compound Library for Batch Processing")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with SMILES and compound information",
        type=['csv'],
        help="CSV should contain at least a 'SMILES' column"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} compounds")
            
            # Show preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head())
            
            # Select SMILES column
            smiles_column = st.selectbox(
                "Select SMILES column:",
                options=df.columns.tolist()
            )
            
            if st.button("🚀 Process All Compounds", type="primary"):
                process_batch(predictor, df, smiles_column)
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def process_batch(predictor, df, smiles_column):
    """Process compounds in batch"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        results = []
        
        for i, row in df.iterrows():
            smiles = row[smiles_column]
            status_text.text(f"Processing compound {i+1}/{len(df)}: {smiles[:50]}...")
            progress_bar.progress((i+1)/len(df))
            
            try:
                prediction = predictor.predict_single_compound(smiles)
                if prediction:
                    result = {
                        'SMILES': smiles,
                        'Drug_Score': prediction.get('drug_score', 0),
                        'Avg_Toxicity': np.mean(list(prediction.get('toxicity_predictions', {}).values())),
                        'Cancer_Selectivity': prediction.get('selectivity_predictions', {}).get('cancer_selectivity', 0)
                    }
                    
                    # Add original columns
                    for col in df.columns:
                        if col != smiles_column:
                            result[col] = row[col]
                    
                    results.append(result)
            except:
                continue
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        progress_bar.progress(1.0)
        status_text.text("Batch processing completed!")
        
        st.success(f"✅ Processed {len(results)} compounds successfully")
        
        # Display results
        st.markdown("#### Processing Results")
        st.dataframe(results_df)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="batch_processing_results.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"❌ Batch processing failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def export_results(results, smiles, compound_name):
    """Export prediction results to CSV"""
    
    try:
        # Flatten results for CSV export
        export_data = {
            'Compound_Name': compound_name,
            'SMILES': smiles,
            'Drug_Score': results.get('drug_score', 0)
        }
        
        # Add bioactivity predictions
        bioactivity = results.get('bioactivity_predictions', {})
        for cell_line, ic50 in bioactivity.items():
            export_data[f'{cell_line}_IC50'] = ic50
        
        # Add toxicity predictions
        toxicity = results.get('toxicity_predictions', {})
        for tox_type, score in toxicity.items():
            export_data[f'{tox_type}_score'] = score
        
        # Add selectivity predictions
        selectivity = results.get('selectivity_predictions', {})
        for sel_type, value in selectivity.items():
            export_data[f'{sel_type}'] = value
        
        # Add molecular properties
        properties = results.get('molecular_properties', {})
        for prop_name, value in properties.items():
            export_data[prop_name] = value
        
        # Create DataFrame and CSV
        export_df = pd.DataFrame([export_data])
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name=f"prediction_{compound_name or 'compound'}.csv",
            mime="text/csv"
        )
        
        st.success("✅ Results exported successfully!")
    
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

if __name__ == "__main__":
    main()