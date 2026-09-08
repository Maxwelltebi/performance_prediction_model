import pandas as pd
import joblib
import random
from sklearn.preprocessing import OrdinalEncoder

# --- 1. Define global variables used in preprocessing ---
nominal_cols = ['gender', 'school_type', 'internet_access', 'extra_activities', 'study_method']
ordinal_cols = ['parent_education', 'travel_time']

parent_education_order = ['no formal', 'diploma', 'high school', 'graduate', 'post graduate', 'phd']
travel_time_order = ['<15 min', '15-30 min', '30-60 min', '>60 min']

# Re-instantiate OrdinalEncoders with categories they were trained on.
# In a full production pipeline, these fitted encoders would also be saved/loaded.
encoder_parent_education = OrdinalEncoder(categories=[parent_education_order])
encoder_parent_education.fit(pd.DataFrame({'parent_education': parent_education_order})) # Fit to make transform work

encoder_travel_time = OrdinalEncoder(categories=[travel_time_order])
encoder_travel_time.fit(pd.DataFrame({'travel_time': travel_time_order})) # Fit to make transform work

# --- 2. Load the trained model and scaler ---
try:
    model_rf = joblib.load('best_random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or Scaler files not found. Make sure 'best_random_forest_model.joblib' and 'scaler.joblib' are in the current directory.")
    # In a real application, you might raise an exception or exit here.
    exit()

# --- 3. Define production_train_columns ---
# This list represents the exact order and names of features the model expects after preprocessing.
production_train_columns = ['age', 'study_hours', 'attendance_percentage', 'math_score', 'science_score',
                            'english_score', 'parent_education_encoded', 'travel_time_encoded',
                            'gender_male', 'gender_other', 'school_type_public', 'internet_access_yes',
                            'extra_activities_yes', 'study_method_group study', 'study_method_mixed',
                            'study_method_notes', 'study_method_online videos', 'study_method_textbook']

# --- 4. Define the prediction function ---
def predict_student_grade(raw_input_data):
    """
    Preprocesses raw student input data and predicts the final grade.

    Args:
        raw_input_data (dict): A dictionary containing student features,
                                e.g., {'age': 18, 'gender': 'male', ...}
                                Excludes 'overall_score' and 'final_grade'.

    Returns:
        str: The predicted final grade.
    """
    # Create DataFrame from raw input
    input_df_raw = pd.DataFrame([raw_input_data])

    # Generate random student_id internally
    if 'student_id' not in input_df_raw.columns:
        input_df_raw['student_id'] = random.randint(100000, 999999)

    # Make a copy for processing
    input_df_processed = input_df_raw.copy()

    # Ordinal Encoding
    input_df_processed['parent_education_encoded'] = encoder_parent_education.transform(input_df_processed[['parent_education']])
    input_df_processed['travel_time_encoded'] = encoder_travel_time.transform(input_df_processed[['travel_time']])

    # One-Hot Encoding
    input_df_processed = pd.get_dummies(input_df_processed, columns=nominal_cols, drop_first=True)

    # Drop original categorical columns and other irrelevant columns
    columns_to_drop = ordinal_cols + nominal_cols + ['student_id', 'overall_score', 'final_grade']
    input_df_processed = input_df_processed.drop(columns=columns_to_drop, errors='ignore')

    # Convert boolean columns to int
    boolean_cols_input = input_df_processed.select_dtypes(include='bool').columns
    if not boolean_cols_input.empty:
        input_df_processed[boolean_cols_input] = input_df_processed[boolean_cols_input].astype(int)

    # Align columns with production_train_columns
    input_df_processed = input_df_processed.reindex(columns=production_train_columns, fill_value=0)

    # Scale the input data
    input_scaled = scaler.transform(input_df_processed)

    # Make prediction
    predicted_grade = model_rf.predict(input_scaled)

    return predicted_grade[0]

# --- 5. Example Usage ---
# Example input data, similar to what a user might provide via an API or form.
# Note: 'student_id', 'overall_score', 'final_grade' are not needed from the user.
example_user_input = {
    'age': 18,
    'gender': 'male',
    'school_type': 'private',
    'parent_education': 'phd',
    'study_hours': 7.0,
    'attendance_percentage': 90.0,
    'internet_access': 'yes',
    'travel_time': '<15 min',
    'extra_activities': 'yes',
    'study_method': 'notes',
    'math_score': 85.0,
    'science_score': 92.0,
    'english_score': 88.0
}

# Make a prediction using the function
final_prediction = predict_student_grade(example_user_input)
print(f"The predicted final grade for the example input is: {final_prediction}")