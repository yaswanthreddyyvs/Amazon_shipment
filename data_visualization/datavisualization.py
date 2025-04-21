import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df=pd.read_csv("C:/Users/yvsya/OneDrive/Desktop/amazon_product_shipment_dataset.csv")
df


print(df.info())
print(df.describe())

df['Mode_of_Shipment'].fillna(df['Mode_of_Shipment'].mode().iloc[0],inplace=True)
df['Cost_of_the_Product'].fillna(df['Cost_of_the_Product'].mean(),inplace=True)
df['Prior_purchases'].fillna(df['Prior_purchases'].mean(),inplace=True)
df['Product_importance'].fillna(df['Product_importance'].mode().iloc[0],inplace=True)
df['Gender'].fillna(df['Gender'].mode().iloc[0],inplace=True)
df['Discount_offered'].fillna(df['Discount_offered'].mean(),inplace=True)
print(df.info())

#1. Histograms
plt.figure(figsize=(12, 5))
sb.histplot(df['Cost_of_the_Product'],bins=30,kde=True, color='skyblue')
plt.title('Cost of the Product')
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.show()

#2. Pie chart for shipment mode
shipment_counts = df['Mode_of_Shipment'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(shipment_counts, labels=shipment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Shipment Mode Distribution')
plt.axis('equal')
plt.show()

#3. Bar plot of product importance
sb.countplot(data=df, x='Product_importance', palette='Set2')
plt.title('Product Importance Levels')
plt.xlabel('Importance Level')
plt.ylabel('Count')
plt.show()

#4. Countplot of delivery status by mode of shipment
sb.countplot(data=df, x='Mode_of_Shipment', hue='Reached.on.Time_Y.N', palette='Set1')
plt.title('Delivery Status by Shipment Mode')
plt.xlabel('Mode of Shipment')
plt.ylabel('Count')
plt.legend(title='Reached On Time (1=Yes)')
plt.show()

#5. Boxplot: weight vs delivery status
sb.boxplot(data=df, x='Reached.on.Time_Y.N', y='Weight_in_gms')
plt.title('Weight of Shipment vs Delivery Status')
plt.xlabel('Reached on Time (1=Yes)')
plt.ylabel('Weight (grams)')
plt.show()

#6. Histogram for discount
plt.figure(figsize=(8, 4))
sb.histplot(df['Discount_offered'],bins=30,kde=True, color='orange')
plt.title('Distribution of Discounts Offered')
plt.xlabel('Discount Offered')
plt.ylabel('Frequency')
plt.show()

#7. Correlation heatmap
plt.figure(figsize=(10, 6))
sb.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
