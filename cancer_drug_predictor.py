import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, classification_report
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import kagglehub
import os
warnings.filterwarnings('ignore')

class CancerDrugDiscoveryPredictor:
    #nitialize the class with default models and variables.
    def __init__(self):
        self.molecular_scaler = StandardScaler()
        self.bioactivity_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.toxicity_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
        self.cancer_selectivity_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.label_encoder = LabelEncoder()
        self.cancer_cell_lines = ['MCF-7', 'HeLa', 'A549', 'HCT-116', 'PC-3', 'HepG2', 'U87', 'MDA-MB-231']
        self.feature_columns = None
        self.is_trained = False

    #Load dataset from a local CSV file.
    def download_and_load_data(self, filepath = r"C:\Users\HP\OneDrive\Desktop\webdevplopment\Credit_Fraud_Detection\DrugDiscovery\jak2_data.csv"
):
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded data from: {filepath}")
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            if 'SMILES' not in df.columns or 'pIC50' not in df.columns:
                raise ValueError("CSV file must contain 'SMILES' and 'pIC50' columns")
            return df, os.path.dirname(filepath)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
        return None, None
   
    #Extract chemical features (descriptors) from SMILES strings.
    def calculate_molecular_descriptors(self, smiles_list):
        descriptors = []
        valid_smiles = []
        
        print(f"Calculating descriptors for {len(smiles_list)} compounds...")
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(smiles_list)} (Valid: {len(descriptors)})")
            
            # Skip empty or NaN SMILES
            if pd.isna(smiles) or not isinstance(smiles, str) or len(smiles.strip()) == 0:
                continue
                
            # Clean the SMILES string
            smiles_clean = smiles.strip()
            
            try:
                mol = Chem.MolFromSmiles(smiles_clean)
                if mol is not None:
                    # Sanitize the molecule
                    Chem.SanitizeMol(mol)
                    
                    desc_dict = {}
                    # Basic molecular properties
                    desc_dict['molecular_weight'] = Descriptors.MolWt(mol)
                    desc_dict['logp'] = Descriptors.MolLogP(mol)
                    desc_dict['num_hbd'] = Descriptors.NumHDonors(mol)
                    desc_dict['num_hba'] = Descriptors.NumHAcceptors(mol)
                    desc_dict['tpsa'] = Descriptors.TPSA(mol)
                    desc_dict['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                    desc_dict['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
                    
                    # Handle FractionCsp3 compatibility
                    try:
                        desc_dict['fraction_csp3'] = Descriptors.FractionCsp3(mol)
                    except AttributeError:
                        try:
                            desc_dict['fraction_csp3'] = rdMolDescriptors.CalcFractionCsp3(mol)
                        except:
                            desc_dict['fraction_csp3'] = 0.0
                    
                    desc_dict['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
                    desc_dict['num_rings'] = Descriptors.RingCount(mol)
                    desc_dict['num_heavy_atoms'] = Descriptors.HeavyAtomCount(mol)
                    desc_dict['molar_refractivity'] = Descriptors.MolMR(mol)
                    
                    # Cancer-relevant descriptors
                    try:
                        desc_dict['slogp'] = Descriptors.SlogP(mol)
                    except:
                        desc_dict['slogp'] = desc_dict['logp']  # Fallback to MolLogP
                    
                    desc_dict['num_saturated_rings'] = Descriptors.NumSaturatedRings(mol)
                    desc_dict['num_aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
                    
                    try:
                        desc_dict['balaban_j'] = Descriptors.BalabanJ(mol)
                    except:
                        desc_dict['balaban_j'] = 0.0  # Default value if calculation fails
                    
                    # Lipinski's Rule of Five compliance
                    desc_dict['lipinski_violations'] = sum([
                        desc_dict['molecular_weight'] > 500,
                        desc_dict['logp'] > 5,
                        desc_dict['num_hbd'] > 5,
                        desc_dict['num_hba'] > 10
                    ])
                    
                    # Cancer drug-likeness features
                    desc_dict['is_lipinski_compliant'] = desc_dict['lipinski_violations'] == 0
                    
                    # Avoid division by zero
                    if desc_dict['num_heavy_atoms'] > 0:
                        desc_dict['molecular_complexity'] = (desc_dict['num_rings'] + 
                                                           desc_dict['num_rotatable_bonds'] + 
                                                           desc_dict['num_heteroatoms']) / desc_dict['num_heavy_atoms']
                    else:
                        desc_dict['molecular_complexity'] = 0.0
                    
                    # Validate all descriptors are finite numbers
                    if all(np.isfinite(v) if isinstance(v, (int, float)) else True for v in desc_dict.values()):
                        descriptors.append(desc_dict)
                        valid_smiles.append(smiles_clean)
                    
            except Exception as e:
                if i < 10:  # Print first few errors for debugging
                    print(f"Error processing SMILES '{smiles}': {e}")
                continue
                    
        print(f"Successfully calculated descriptors for {len(descriptors)} compounds out of {len(smiles_list)}")
        
        if len(descriptors) == 0:
            print("No valid descriptors calculated. Checking SMILES format...")
            # Print sample SMILES for debugging
            sample_smiles = [s for s in smiles_list[:10] if pd.notna(s)]
            print(f"Sample SMILES: {sample_smiles}")
            
        return pd.DataFrame(descriptors), valid_smiles
    
    def generate_cancer_specific_data(self, smiles_df=None, n_compounds=1500):
        """Generate cancer-specific drug discovery data"""
        np.random.seed(42)
        
        if smiles_df is not None and 'SMILES' in smiles_df.columns:
            # Use real SMILES data
            print("Using real SMILES data for cancer drug discovery...")
            
            # Filter out invalid SMILES first
            valid_smiles_df = smiles_df[smiles_df['SMILES'].notna() & (smiles_df['SMILES'] != '')]
            print(f"Found {len(valid_smiles_df)} valid SMILES entries out of {len(smiles_df)}")
            
            if len(valid_smiles_df) == 0:
                print("No valid SMILES found in dataset, using synthetic data...")
                smiles_list = self._generate_synthetic_smiles(n_compounds)
            else:
                # Sample from valid SMILES
                sample_size = min(n_compounds, len(valid_smiles_df))
                smiles_list = valid_smiles_df['SMILES'].sample(sample_size).tolist()
        else:
            # Generate synthetic cancer-relevant SMILES
            print("Generating synthetic cancer-specific SMILES data...")
            smiles_list = self._generate_synthetic_smiles(n_compounds)
        
        # Calculate molecular descriptors
        molecular_features, valid_smiles = self.calculate_molecular_descriptors(smiles_list)
        
        if len(molecular_features) == 0:
            print("No valid descriptors from real data, falling back to synthetic data...")
            smiles_list = self._generate_synthetic_smiles(n_compounds)
            molecular_features, valid_smiles = self.calculate_molecular_descriptors(smiles_list)
            
            if len(molecular_features) == 0:
                raise ValueError("Unable to generate valid molecular descriptors")
        
        n_valid = len(molecular_features)
        print(f"Successfully processed {n_valid} compounds for cancer drug discovery")
        
        # Generate cancer-specific bioactivity data
        # IC50 values for different cancer cell lines (lower = more potent)
        cell_line_activities = {}
        for cell_line in self.cancer_cell_lines:
            # Generate correlated activities based on molecular properties
            base_activity = np.random.lognormal(mean=1.5, sigma=1.2, size=n_valid)
            
            # Modify based on molecular properties for realistic correlations
            mw_factor = np.where(molecular_features['molecular_weight'] > 600, 1.5, 1.0)
            logp_factor = np.where(molecular_features['logp'] > 6, 1.3, 1.0)
            
            cell_line_activities[f'{cell_line}_IC50'] = base_activity * mw_factor * logp_factor
        
        # Generate toxicity data (more realistic for cancer drugs)
        toxicity_prob = 1 / (1 + np.exp(-(molecular_features['molecular_weight'] - 400) / 100))
        toxicity = np.random.binomial(1, toxicity_prob)
        
        # Generate selectivity indices (cancer vs normal cells)
        selectivity_index = np.random.lognormal(mean=0.5, sigma=0.8, size=n_valid)
        
        # Generate drug resistance potential
        resistance_score = np.random.uniform(0, 1, n_valid)
        
        # Create comprehensive cancer drug dataset
        data = molecular_features.copy()
        data['SMILES'] = valid_smiles
        
        # Add cell line activities
        for cell_line, activities in cell_line_activities.items():
            data[cell_line] = activities
        
        data['is_toxic'] = toxicity
        data['selectivity_index'] = selectivity_index
        data['resistance_score'] = resistance_score
        
        # Calculate overall cancer potency (geometric mean of IC50s)
        ic50_columns = [col for col in data.columns if col.endswith('_IC50')]
        data['overall_cancer_potency'] = np.exp(np.mean(np.log(data[ic50_columns]), axis=1))
        
        # Add therapeutic index (selectivity/toxicity ratio)
        data['therapeutic_index'] = data['selectivity_index'] / (data['is_toxic'] + 0.1)
        
        return data
    
    def _generate_synthetic_smiles(self, n_compounds):
        """Generate synthetic cancer-relevant SMILES"""
        cancer_drug_templates = [
            'CCc1ccccc1',  # Simple aromatic
            'CC(=O)Nc1ccccc1',  # Acetanilide-like
            'Nc1ccc(O)cc1',  # Aminophenol-like
            'CCOc1ccccc1',  # Ethoxybenzene-like
            'c1ccc2[nH]c3ccccc3c2c1',  # Carbazole-like
            'CC(C)Nc1ccccc1',  # Isopropylaniline-like
            'CCN(CC)c1ccccc1',  # Diethylaniline-like
            'COc1ccccc1O',  # Methoxyphenol-like
            'Cc1ccc(N)cc1',  # Toluidine-like
            'c1ccc(Cl)cc1',  # Chlorobenzene-like
            'CC(=O)c1ccccc1',  # Acetophenone-like
            'Nc1cccc(N)c1',  # Diaminobenzene-like
            'CCCc1ccccc1',  # Propylbenzene-like
            'COc1ccc(N)cc1',  # Anisidine-like
            'c1ccc2ccccc2c1',  # Naphthalene-like
        ]
        
        # Generate more diverse compounds by combining templates
        extended_smiles = []
        for _ in range(n_compounds):
            template = np.random.choice(cancer_drug_templates)
            extended_smiles.append(template)
        
        return extended_smiles
    
    def train_cancer_models(self, data):
        """Train cancer-specific prediction models"""
        # Prepare features
        feature_columns = [col for col in data.columns if col not in 
                          ['SMILES', 'is_toxic', 'selectivity_index', 'resistance_score',
                           'overall_cancer_potency', 'therapeutic_index'] and 
                          not col.endswith('_IC50')]
        
        X = data[feature_columns]
        
        # Train bioactivity model (overall cancer potency)
        print("Training cancer bioactivity model...")
        X_scaled = self.molecular_scaler.fit_transform(X)
        bio_mse, _, _, _ = self._train_regression_model(
            X_scaled, data['overall_cancer_potency'], self.bioactivity_model
        )
        print(f"Cancer Bioactivity Model MSE: {bio_mse:.4f}")
        
        # Train toxicity model
        print("Training toxicity prediction model...")
        tox_acc, _, _, _ = self._train_classification_model(
            X_scaled, data['is_toxic'], self.toxicity_model
        )
        print(f"Toxicity Model Accuracy: {tox_acc:.4f}")
        
        # Train cancer selectivity model
        print("Training cancer selectivity model...")
        sel_mse, _, _, _ = self._train_regression_model(
            X_scaled, data['selectivity_index'], self.cancer_selectivity_model
        )
        print(f"Cancer Selectivity Model MSE: {sel_mse:.4f}")
        
        self.feature_columns = feature_columns
        self.is_trained = True
        
        return {
            'bioactivity_mse': bio_mse,
            'toxicity_accuracy': tox_acc,
            'selectivity_mse': sel_mse,
            'feature_columns': feature_columns
        }
    
    def _train_regression_model(self, X, y, model):
        """Helper function to train regression models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse, X_test, y_test, predictions
    
    def _train_classification_model(self, X, y, model):
        """Helper function to train classification models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy, X_test, y_test, predictions
    
    def predict_cancer_drug_properties(self, smiles):
        """Predict cancer drug properties for a given SMILES string"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
            
        molecular_desc, valid_smiles = self.calculate_molecular_descriptors([smiles])
        
        if len(molecular_desc) == 0:
            return None
        
        # Select only the feature columns used in training
        feature_desc = molecular_desc[self.feature_columns]
        X_scaled = self.molecular_scaler.transform(feature_desc)
        
        # Predict properties
        bioactivity_pred = self.bioactivity_model.predict(X_scaled)[0]
        toxicity_pred = self.toxicity_model.predict(X_scaled)[0]
        toxicity_prob = self.toxicity_model.predict_proba(X_scaled)[0][1]
        selectivity_pred = self.cancer_selectivity_model.predict(X_scaled)[0]
        
        # Calculate cancer drug-likeness score
        desc = molecular_desc.iloc[0]
        cancer_drug_likeness = self.calculate_cancer_drug_likeness(desc)
        
        # Calculate therapeutic index
        therapeutic_index = selectivity_pred / (toxicity_prob + 0.1)
        
        return {
            'smiles': smiles,
            'predicted_cancer_potency': bioactivity_pred,
            'toxicity_prediction': 'Toxic' if toxicity_pred == 1 else 'Non-toxic',
            'toxicity_probability': toxicity_prob,
            'cancer_selectivity': selectivity_pred,
            'therapeutic_index': therapeutic_index,
            'cancer_drug_likeness_score': cancer_drug_likeness,
            'molecular_weight': desc['molecular_weight'],
            'logp': desc['logp'],
            'lipinski_violations': desc['lipinski_violations'],
            'tpsa': desc['tpsa']
        }
    
    def calculate_cancer_drug_likeness(self, descriptors):
        """Calculate cancer drug-likeness score"""
        score = 100
        
        # Standard drug-likeness penalties
        if descriptors['molecular_weight'] > 600:  # Cancer drugs can be larger
            score -= 15
        if descriptors['logp'] > 6:
            score -= 20
        if descriptors['num_hbd'] > 6:
            score -= 15
        if descriptors['num_hba'] > 12:
            score -= 15
        if descriptors['tpsa'] > 160:
            score -= 10
        if descriptors['num_rotatable_bonds'] > 12:
            score -= 10
        
        # Cancer-specific bonuses
        if 200 < descriptors['molecular_weight'] < 600:
            score += 10
        if 2 < descriptors['logp'] < 5:
            score += 10
        if descriptors['num_aromatic_rings'] >= 2:
            score += 5
        
        return max(0, score)
    
    def cancer_virtual_screening(self, compound_library, 
                                threshold_potency=5.0,
                                max_toxicity_prob=0.4,
                                min_selectivity=1.0,
                                min_therapeutic_index=2.0):
        """Perform virtual screening focused on cancer drugs"""
        results = []
        
        print(f"Screening {len(compound_library)} compounds for cancer activity...")
        
        for i, smiles in enumerate(compound_library):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(compound_library)}")
                
            prediction = self.predict_cancer_drug_properties(smiles)
            if prediction:
                # Apply cancer-specific filters
                if (prediction['predicted_cancer_potency'] < threshold_potency and
                    prediction['toxicity_probability'] < max_toxicity_prob and
                    prediction['cancer_selectivity'] >= min_selectivity and
                    prediction['therapeutic_index'] >= min_therapeutic_index):
                    results.append(prediction)
        
        # Sort by therapeutic index (higher is better)
        return sorted(results, key=lambda x: x['therapeutic_index'], reverse=True)
    
    def analyze_cancer_features(self, data):
        """Analyze important features for cancer drug discovery"""
        if not self.is_trained:
            raise ValueError("Models must be trained before analyzing features")
            
        feature_columns = self.feature_columns
        
        # Feature importance for bioactivity
        bio_importance = pd.DataFrame({
            'feature': feature_columns,
            'bioactivity_importance': self.bioactivity_model.feature_importances_
        })
        
        # Feature importance for selectivity  
        sel_importance = pd.DataFrame({
            'feature': feature_columns,
            'selectivity_importance': self.cancer_selectivity_model.feature_importances_
        })
        
        # Combine importance scores
        combined_importance = pd.merge(bio_importance, sel_importance, on='feature')
        combined_importance['combined_score'] = (
            combined_importance['bioactivity_importance'] + 
            combined_importance['selectivity_importance']
        ) / 2
        
        return combined_importance.sort_values('combined_score', ascending=False)
    def compare_models(self, data, epochs=30, batch_size=32):
        if self.feature_columns is None:
            raise ValueError("Run train_cancer_models first or set feature_columns.")
        X = data[self.feature_columns]
        X_scaled = self.molecular_scaler.fit_transform(X)

            # Targets
        y_bio = data['overall_cancer_potency']
        y_tox = data['is_toxic']
        y_sel = data['selectivity_index']

            # Train/test splits
        X_train_bio, X_test_bio, y_train_bio, y_test_bio = train_test_split(X_scaled, y_bio, test_size=0.2, random_state=42)
        X_train_tox, X_test_tox, y_train_tox, y_test_tox = train_test_split(X_scaled, y_tox, test_size=0.2, random_state=42)
        X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_scaled, y_sel, test_size=0.2, random_state=42)



        regression_models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR()
        }

        classification_models = {
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'SVC': SVC(probability=True),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
        }

        results = []

            # Bioactivity regression models
        for name, model in regression_models.items():
            model.fit(X_train_bio, y_train_bio)
            y_pred = model.predict(X_test_bio)
            mse = mean_squared_error(y_test_bio, y_pred)
            r2 = r2_score(y_test_bio, y_pred)
            results.append({'Task': 'Bioactivity', 'Model': name, 'MSE': mse, 'R2': r2, 'Accuracy': None, 'F1': None})

            # Deep learning model for Bioactivity
        model_bio = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_bio.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model_bio.compile(optimizer=Adam(0.001), loss='mse')
        model_bio.fit(X_train_bio, y_train_bio, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model_bio.predict(X_test_bio).flatten()
        mse = mean_squared_error(y_test_bio, y_pred)
        r2 = r2_score(y_test_bio, y_pred)
        results.append({'Task': 'Bioactivity', 'Model': 'DeepNN', 'MSE': mse, 'R2': r2, 'Accuracy': None, 'F1': None})

        # Toxicity classification models
        for name, model in classification_models.items():
            model.fit(X_train_tox, y_train_tox)
            y_pred = model.predict(X_test_tox)
            acc = accuracy_score(y_test_tox, y_pred)
            f1 = f1_score(y_test_tox, y_pred)
            results.append({'Task': 'Toxicity', 'Model': name, 'MSE': None, 'R2': None, 'Accuracy': acc, 'F1': f1})

            # Deep learning model for Toxicity
        model_tox = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_tox.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model_tox.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model_tox.fit(X_train_tox, y_train_tox, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model_tox.predict(X_test_tox).flatten()
        y_pred_label = (y_pred > 0.5).astype(int)
        acc = accuracy_score(y_test_tox, y_pred_label)
        f1 = f1_score(y_test_tox, y_pred_label)
        results.append({'Task': 'Toxicity', 'Model': 'DeepNN', 'MSE': None, 'R2': None, 'Accuracy': acc, 'F1': f1})

            # Selectivity regression models
        for name, model in regression_models.items():
            model.fit(X_train_sel, y_train_sel)
            y_pred = model.predict(X_test_sel)
            mse = mean_squared_error(y_test_sel, y_pred)
            r2 = r2_score(y_test_sel, y_pred)
            results.append({'Task': 'Selectivity', 'Model': name, 'MSE': mse, 'R2': r2, 'Accuracy': None, 'F1': None})

            # Deep learning model for Selectivity
        model_sel = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_sel.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model_sel.compile(optimizer=Adam(0.001), loss='mse')
        model_sel.fit(X_train_sel, y_train_sel, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model_sel.predict(X_test_sel).flatten()
        mse = mean_squared_error(y_test_sel, y_pred)
        r2 = r2_score(y_test_sel, y_pred)
        results.append({'Task': 'Selectivity', 'Model': 'DeepNN', 'MSE': mse, 'R2': r2, 'Accuracy': None, 'F1': None})
        return pd.DataFrame(results)