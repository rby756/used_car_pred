import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from pathlib import Path
from src.carPricePrediction.entity.config_entity import DataTransformationConfig
from src.carPricePrediction import logger
import os


class TargetEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.encoder = TargetEncoder(cols=self.cols)
    
    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols], y)
        return self
    
    def transform(self, X, y=None):
        X[self.cols] = self.encoder.transform(X[self.cols])
        return X



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize the DataTransformation class with a given configuration.
        """
        self.config = config


    def _drop_irrelevant_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop irrelevant columns from the data.

        Parameters:
        data (pd.DataFrame): The input data.

        Returns:
        pd.DataFrame: The data without the dropped columns.
        """

        data=data.drop(columns=['car_name','registration_year'])
        print(data.columns)

        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing target values and handle other missing values as needed.

        Parameters:
        data (pd.DataFrame): The raw input data.

        Returns:
        pd.DataFrame: The data with missing target values removed.
        """

        print("self.config.target_column : ", self.config.target_column)
        data = data.dropna(subset=[self.config.target_column])

        # Additional handling for missing values in features can be added here if needed
        return data

    def _separate_features_and_target(self, data: pd.DataFrame) -> tuple:
        """
        Separate features and target variable from the data.

        Parameters:
        data (pd.DataFrame): The input data.

        Returns:
        tuple: Features (X) and target (y).
        """
        X = data.drop(columns=self.config.target_column)
        y = data[self.config.target_column]
        return X, y

    def _select_columns_by_type(self, X: pd.DataFrame) -> tuple:
        """
        Select numerical and categorical columns from the feature data.

        Parameters:
        X (pd.DataFrame): The feature data.

        Returns:
        tuple: Lists of numerical and categorical column names.
        """
        num_cols = ['manufacturing_year', 'seats', 'kms_driven',
       'mileage(kmpl)', 'engine(cc)', 'torque(Nm)']

        cat_cols = ['insurance_validity', 'fuel_type', 'ownsership', 'transmission']

        return num_cols, cat_cols

    def _create_transformer(self, num_cols: list, cat_cols: list) -> ColumnTransformer:
        """
        Create a column transformer for preprocessing.

        Parameters:
        num_cols (list): List of numerical column names.
        cat_cols (list): List of categorical column names.

        Returns:
        ColumnTransformer: The column transformer.
        """
        num_pipeline = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=3)),  # Using KNNImputer
            ('scaler', StandardScaler())
        ])


        cat_pipeline = ColumnTransformer(
        transformers=[
            # Label Encoding for insurance_validity
            ('insurance_validity', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['insurance_validity']),

            # One-Hot Encoding for fuel_type and transmission
            ('fuel_type', OneHotEncoder(handle_unknown='ignore'), ['fuel_type']),
            ('transmission', OneHotEncoder(handle_unknown='ignore'), ['transmission']),

            # Ordinal Encoding for ownership
            ('ownsership', OrdinalEncoder(categories=[['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner','Fifth Owner']]), ['ownsership'])
        ],
        
        remainder='passthrough'  # Pass the numeric features through without transformation
)

        transformer = ColumnTransformer(transformers=[
            ('num_pipeline', num_pipeline, num_cols),
            ('cat_pipeline', cat_pipeline, cat_cols),
        ], remainder='drop', n_jobs=-1)

        return transformer

    def _save_transformer(self, transformer: ColumnTransformer) -> None:
        """
        Save the fitted transformer to a file.

        Parameters:
        transformer (ColumnTransformer): The fitted column transformer.
        """
        joblib.dump(transformer, os.path.join(self.config.root_dir, self.config.preprocessor_name))


    def _apply_target_encoding(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:

        """
        Apply target encoding to specified categorical columns.

        Parameters:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target variable.

        Returns:
        pd.DataFrame: The feature data with target-encoded columns.
        """
        
        target_encoder = TargetEncodingTransformer(cols=['short_carname'])
        target_encoder.fit(X, y)
        X = target_encoder.transform(X)
        return X

    def preprocess_data(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Perform the complete preprocessing pipeline on the data.

        This includes:
        - Dropping irrelevant columns
        - Handling missing values (dropping rows with missing target)
        - Separating features and target
        - Applying target encoding
        - Selecting numerical and categorical columns
        - Creating and saving the transformer
        - Transforming the feature data

        Parameters:
        data (pd.DataFrame): The raw input data.
        fit (bool): Whether to fit the transformers or just transform using pre-fitted transformers.

        Returns:
        tuple: Transformed feature data (X_transformed) and target variable (y).
        """
        data = self._drop_irrelevant_columns(data)
        data = self._handle_missing_values(data)
        X, y = self._separate_features_and_target(data)
        X = self._apply_target_encoding(X, y if fit else None)
        num_cols, cat_cols = self._select_columns_by_type(X)
        transformer = self._create_transformer(num_cols, cat_cols)
        if fit:
            self._save_transformer(transformer)
            X_transformed = transformer.fit_transform(X)
        else:
            transformer = joblib.load(os.path.join(self.config.root_dir, self.config.preprocessor_name))
            X_transformed = transformer.transform(X)
        return X_transformed, y

    def _save_transformer(self, transformer: ColumnTransformer) -> None:
        joblib.dump(transformer, os.path.join(self.config.root_dir, self.config.preprocessor_name))

    def _save_target_encoder(self, encoder: TargetEncodingTransformer) -> None:
        joblib.dump(encoder, os.path.join(self.config.root_dir, 'target_encoder.pkl'))

    def _load_target_encoder(self) -> TargetEncodingTransformer:
        return joblib.load(os.path.join(self.config.root_dir, 'target_encoder.pkl'))

    def train_test_splitting(self) -> None:
        """
        Load data, preprocess it, and split into training and test sets.
        """
        try:
            data = pd.read_csv(self.config.data_path)
            X, y = self.preprocess_data(data)

            print("type of data before splitting",type(X))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            print("type of training  data before splitting",type(X_train))

            # X_train = X_train.to_frame().reset_index(drop=True)
            # X_test = X_test.to_frame().reset_index(drop=True)

            # y_train_df = y_train.to_frame().reset_index(drop=True)
            # y_test_df = y_test.to_frame().reset_index(drop=True)

            X_train = pd.DataFrame(X_train).reset_index(drop=True)
            X_test = pd.DataFrame(X_test).reset_index(drop=True)

            y_train_df = pd.DataFrame(y_train).reset_index(drop=True)
            y_test_df = pd.DataFrame(y_test).reset_index(drop=True)

            train_processed = pd.concat([X_train, y_train], axis=1)
            test_processed = pd.concat([X_test, y_test], axis=1)

            train_processed.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test_processed.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

            logger.info("Data split into training and test sets")
            logger.info(f"Shape of preprocessed training data: {train_processed.shape}")
            logger.info(f"Shape of preprocessed test data: {test_processed.shape}")

        except Exception as e:
            logger.error("An error occurred during train-test splitting", exc_info=True)
            raise


    