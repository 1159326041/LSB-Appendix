import pandas as pd
import numpy as np

# Load the dataset
file_path = r".\Uncleaned_employees_final_dataset.csv"
data = pd.read_csv(file_path)

# Filter the dataset to keep only the relevant columns
relevant_columns = ['education', 'gender', 'no_of_trainings', 'age', 
                    'previous_year_rating', 'length_of_service', 
                    'KPIs_met_more_than_80', 'awards_won', 'avg_training_score']

filtered_data = data[relevant_columns].dropna()

# Mapping for categorical variables
education_mapping = {'Bachelors': 0, 'Masters & above': 1, 'Below Secondary': -1}
gender_mapping = {'m': 0, 'f': 1}

# Apply the mappings
filtered_data['education'] = filtered_data['education'].map(education_mapping)
filtered_data['gender'] = filtered_data['gender'].map(gender_mapping)

# Create the matrix and vector
matrix_columns = ['education', 'gender', 'no_of_trainings', 'age', 
                  'length_of_service', 'KPIs_met_more_than_80', 
                  'awards_won', 'avg_training_score']
vector_column = 'previous_year_rating'

matrix = filtered_data[matrix_columns].to_numpy()
vector = filtered_data[vector_column].to_numpy()



# Save the matrix and vector to .npy files
matrix_file_path = 'employee_matrix.npy'
vector_file_path = 'employee_vector.npy'

np.save(matrix_file_path, matrix)
np.save(vector_file_path, vector)

print(f"Matrix saved to {matrix_file_path}")
print(f"Vector saved to {vector_file_path}")