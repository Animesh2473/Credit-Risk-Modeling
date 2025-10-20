
import joblib
import pandas as pd
import numpy as np

class FairCreditRiskPredictor:
    """
    Fair Credit Risk Model Predictor
    Ensures fair predictions across gender groups
    """

    def __init__(self, package_path='complete_model_package_20251020_113604.pkl'):
        """Load the complete model package"""
        self.package = joblib.load(package_path)
        self.mitigator = self.package['mitigator']
        self.scaler = self.package['scaler']
        self.gender_encoder = self.package['gender_encoder']
        self.feature_columns = self.package['feature_columns']
        self.scaled_columns = self.package['scaled_columns']
        self.metadata = self.package['metadata']

    def preprocess(self, df, gender_col='gender'):
        """
        Preprocess input data

        Parameters:
        -----------
        df : pandas DataFrame
            Input data with all required features
        gender_col : str
            Name of the gender column

        Returns:
        --------
        X : pandas DataFrame
            Preprocessed features
        gender : array
            Encoded gender values
        """
        # Make a copy
        df_copy = df.copy()

        # Extract and encode gender
        if gender_col in df_copy.columns:
            gender = self.gender_encoder.transform(df_copy[gender_col])
            df_copy = df_copy.drop(columns=[gender_col])
        else:
            raise ValueError(f"Gender column '{gender_col}' not found in input data")

        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df_copy.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Select and order features
        X = df_copy[self.feature_columns].copy()

        # Scale numerical features
        scaled_cols_present = [col for col in self.scaled_columns if col in X.columns]
        if scaled_cols_present:
            X[scaled_cols_present] = self.scaler.transform(X[scaled_cols_present])

        return X, gender

    def predict(self, df, gender_col='gender', return_proba=False):
        """
        Make fair predictions

        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        gender_col : str
            Name of gender column
        return_proba : bool
            If True, return probabilities

        Returns:
        --------
        predictions : array
            Binary predictions (0 or 1)
        probabilities : array (optional)
            Prediction probabilities
        """
        X, gender = self.preprocess(df, gender_col)

        if return_proba:
            try:
                proba = self.mitigator.predict_proba(X)[:, 1]
                predictions = self.mitigator.predict(X)
                return predictions, proba
            except:
                predictions = self.mitigator.predict(X)
                return predictions, None
        else:
            return self.mitigator.predict(X)

    def get_model_info(self):
        """Return model metadata"""
        return self.metadata

# Example usage:
if __name__ == "__main__":
    # Load model
    predictor = FairCreditRiskPredictor()

    # Display model info
    print("Model Information:")
    print(predictor.get_model_info())

    # Example prediction (replace with your actual data)
    # sample_data = pd.DataFrame({
    #     'feature1': [value1],
    #     'feature2': [value2],
    #     'gender': ['M']
    # })

    # Make prediction
    # prediction = predictor.predict(sample_data)
    # print(f"Prediction: {prediction}")
