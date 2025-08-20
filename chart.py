import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. Data Generation ---
# Generate realistic synthetic data for customer engagement patterns.
# We set a random seed for reproducibility.
np.random.seed(42)

# Number of customers (data points)
num_customers = 500

# Create correlated data
# Base metric: Customer satisfaction (NPS Score)
nps_score = np.random.randint(1, 11, size=num_customers)

# Time on Site (positively correlated with NPS)
time_on_site = nps_score * 10 + np.random.normal(0, 20, num_customers)
time_on_site = np.clip(time_on_site, 5, 120) # Clamp values to a realistic range (minutes)

# Purchase Frequency (positively correlated with NPS and Time on Site)
purchase_frequency = (nps_score * 0.5) + (time_on_site / 20) + np.random.normal(0, 1, num_customers)
purchase_frequency = np.clip(purchase_frequency, 1, 15) # Clamp to 1-15 purchases/quarter

# Average Order Value (weakly correlated with frequency)
avg_order_value = 50 + purchase_frequency * 5 + np.random.normal(0, 25, num_customers)
avg_order_value = np.clip(avg_order_value, 20, 300) # Clamp to $20-$300

# Support Tickets (negatively correlated with NPS)
support_tickets = 5 - (nps_score / 2) + np.random.normal(0, 1, num_customers)
support_tickets = np.clip(support_tickets, 0, 10).astype(int) # Clamp to 0-10 tickets

# Create a pandas DataFrame
data = {
    'NPS Score': nps_score,
    'Time on Site (min)': time_on_site,
    'Purchase Frequency': purchase_frequency,
    'Avg. Order Value ($)': avg_order_value,
    'Support Tickets': support_tickets
}
df = pd.DataFrame(data)

# --- 2. Create Correlation Matrix ---
# Calculate the correlation between the different metrics
correlation_matrix = df.corr()

# --- 3. Style and Create the Heatmap ---
# Apply a professional Seaborn style
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.8)

# Set the figure size to produce a 512x512 pixel image (8 inches * 64 dpi)
plt.figure(figsize=(8, 8))

# Create the heatmap with appropriate parameters
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,          # Display the correlation values on the map
    fmt=".2f",           # Format the values to two decimal places
    cmap='vlag',         # Use a divergent colormap (blue for positive, red for negative)
    linewidths=.5,       # Add lines between cells
    cbar_kws={'label': 'Correlation Coefficient'} # Add a label to the color bar
)

# Style the chart with a title and proper labels
plt.title('Customer Engagement Metrics Correlation Matrix', fontsize=18, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Ensure the layout is tight to prevent labels from being cut off
plt.tight_layout()

# --- 4. Export the Chart ---
# Save the chart as a PNG with exactly 512x512 pixel dimensions
plt.savefig('chart.png', dpi=64, bbox_inches='tight')

print("Successfully generated and saved chart.png (512x512 pixels).")
