import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Generate realistic synthetic data for customer engagement
np.random.seed(42) # Ensures the data is the same every time
num_customers = 500
data = {
    'NPS Score': np.random.randint(1, 11, size=num_customers),
    'Time on Site (min)': np.clip(np.random.normal(60, 20, num_customers), 5, 120),
    'Purchase Frequency': np.clip(np.random.randint(1, 15, num_customers), 1, 15),
    'Avg. Order Value ($)': np.clip(np.random.normal(75, 25, num_customers), 20, 300),
    'Support Tickets': np.clip(np.random.randint(0, 5, num_customers), 0, 10)
}
df = pd.DataFrame(data)

# 2. Calculate the correlation matrix from the data
correlation_matrix = df.corr()

# 3. Create the heatmap using Seaborn
# Set a professional style and the exact figure size for 512x512 output
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 8)) # 8 inches x 8 inches

# Create the heatmap
heatmap = sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="vlag"
)

# Add a title and adjust labels
plt.title('Customer Engagement Correlation', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Manually adjust subplot parameters to prevent labels from being cut off,
# which is a more rigid alternative to tight_layout()
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.9)


# 4. Save the chart with the required DPI
# 8 inches * 64 dpi (dots per inch) = 512 pixels.
# We do NOT use bbox_inches='tight' or tight_layout() as they alter the final size.
plt.savefig('chart.png', dpi=64)
