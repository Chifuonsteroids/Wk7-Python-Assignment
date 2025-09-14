# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Task 1: Load and Explore the Dataset
print("="*50)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("="*50)

try:
    # Load the Iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Explore data structure
    print("\nData types:")
    print(df.dtypes)
    
    print("\nDataset information:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing values:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    # Clean dataset (though Iris dataset typically has no missing values)
    if missing_values.sum() > 0:
        df = df.dropna()
        print(f"\nDropped {missing_values.sum()} missing values")
    else:
        print("\nNo missing values found - dataset is clean!")
        
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\n" + "="*50)
print("TASK 2: BASIC DATA ANALYSIS")
print("="*50)

# Basic statistics
print("Basic statistics for numerical columns:")
print(df.describe())

# Group by species and compute means
print("\nMean values by species:")
species_means = df.groupby('species').mean()
print(species_means)

# Additional analysis
print("\nAdditional analysis:")
print(f"Number of samples per species:")
print(df['species'].value_counts())

# Find patterns
print("\nInteresting findings:")
max_sepal_length = df.loc[df['sepal length (cm)'].idxmax()]
print(f"Sample with maximum sepal length: {max_sepal_length['species']} "
      f"({max_sepal_length['sepal length (cm)']} cm)")

min_sepal_length = df.loc[df['sepal length (cm)'].idxmin()]
print(f"Sample with minimum sepal length: {min_sepal_length['species']} "
      f"({min_sepal_length['sepal length (cm)']} cm)")

# Task 3: Data Visualization
print("\n" + "="*50)
print("TASK 3: DATA VISUALIZATION")
print("="*50)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis - Visualizations', fontsize=16, fontweight='bold')

# 1. Line chart - Trends by species (using index as pseudo-time)
plt.subplot(2, 2, 1)
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.plot(species_data.index[:30], species_data['sepal length (cm)'][:30], 
             label=species, marker='o', linestyle='-', markersize=4)
plt.title('Sepal Length Trends (First 30 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Bar chart - Comparison of mean values across species
plt.subplot(2, 2, 2)
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x_pos = np.arange(len(features))
width = 0.25

for i, species in enumerate(df['species'].unique()):
    species_data = df[df['species'] == species]
    means = species_data[features].mean()
    plt.bar(x_pos + i*width, means, width, label=species, alpha=0.8)

plt.title('Mean Feature Values by Species')
plt.xlabel('Features')
plt.ylabel('Mean Value (cm)')
plt.xticks(x_pos + width, features, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 3. Histogram - Distribution of sepal length
plt.subplot(2, 2, 3)
plt.hist(df['sepal length (cm)'], bins=15, alpha=0.7, edgecolor='black', color='skyblue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 4. Scatter plot - Relationship between sepal length and petal length
plt.subplot(2, 2, 4)
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.scatter(species_data['sepal length (cm)'], 
                species_data['petal length (cm)'], 
                label=species, alpha=0.7, s=50, c=colors[species])

plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional visualizations
print("\nAdditional visualizations:")

# Box plot to show distribution by species
plt.figure(figsize=(12, 6))
df.boxplot(column=['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)'], 
           by='species', figsize=(12, 8))
plt.suptitle('Feature Distributions by Species')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

# Findings and observations
print("\n" + "="*50)
print("FINDINGS AND OBSERVATIONS")
print("="*50)

print("1. Dataset Overview:")
print(f"   - Total samples: {len(df)}")
print(f"   - Features: {len(df.columns) - 1} numerical features + 1 categorical target")
print(f"   - Species distribution: {df['species'].value_counts().to_dict()}")

print("\n2. Key Patterns:")
print("   - Setosa species has the smallest petal measurements")
print("   - Virginica species has the largest measurements overall")
print("   - Strong positive correlation between petal length and petal width")
print("   - Versicolor appears to be intermediate between setosa and virginica")

print("\n3. Visualization Insights:")
print("   - Line chart shows natural variation within each species")
print("   - Bar chart clearly shows size differences between species")
print("   - Histogram reveals normal distribution of sepal length")
print("   - Scatter plot shows clear separation between species clusters")

print("\n4. Data Quality:")
print("   - No missing values detected")
print("   - All data types are appropriate")
print("   - Dataset is well-structured and ready for analysis")