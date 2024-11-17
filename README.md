# Title: Whole_sale_customer

# Overview
The project is focused on analyzing wholesale customer purchasing behavior. It uses clustering and Self-Organizing Maps (SOMs) to analyze customer spending data. The data appears to be sourced from the UCI Machine Learning Repository.
# Data source [Download](https://github.com/Pritex32/whole_sale_customer/blob/main/Wholesale%20customers%20data.csv)
The dataset used for this project is from UCI machine learning repository (Wholesale customers - UCI Machine Learning Repository)
Its features includes Annual spending on various products (e.g., milk, grocery, frozen foods).The data set is multivariate with 440 rows and 8 columns.
It includes the annual spending in monetary units (m.u.) on diverse product categories.


## Dataset Description
### Purpose: Analyze annual spending in monetary units (m.u.) across different product categories.
### Columns Explained:
### Region: Geographic location of customers (1 = Lisbon, 2 = Oporto, 3 = Other regions).
### Channel: Customer type (1 = Hotel/Restaurant/Cafe (HoReCa), 2 = Retail).
### Fresh: Spending on fresh products (e.g., vegetables, fruits, seafood).
### Milk: Spending on dairy products.
### Grocery: Spending on general groceries.
### Frozen: Spending on frozen foods.
### Detergents_Paper: Spending on detergents and paper products.
### Delicassen: Spending on delicatessen items.
## Techniques Used
- Clustering: Likely involves grouping customers based on their spending patterns.
- Self-Organizing Maps (SOMs): A neural network technique to visualize and cluster high-dimensional data.
- Pandas
- Matplotlib
-	Seaborn
-	Sklearn
-	Self-organizing map
-	Numpy
-	Jupyter notebook

# Exploratory Data Analysis (EDA)
## 1.	Determining the High spender and low spenders among channels 
It is discovered that the both customer base (hotel and retail customers) have very low spending on item Delicassen. Retailers/individuals are the customers with high purchasing power as they spend more while Hotel/restaurant purchases only frozen foods in a very high quantity. The retails customers buy grocery in a very large quantity and verylittle of frozen foods and other products moderately.
## 2.Identifying the purchasing behavior of customers among regions
All regions purchases fresh food and grocery in high quantity but very little of Delicassen items. Products like frozen foods, detergent and delicassen are in very low quantity across all regions.
## Clusters interpretation
I went ahead map out the values of customers in the white boxes called clusters with this code ‘mappings=som.win_map (x_scaled)’.
Findings:
1. Both clusters spends very low on Detergent papers.
2. There are 3 regions but only Oporto and other regions spends more.
## Recommendations:
1. I suggest the marketing team to identify why customers don't purchase other products may be through questionnaire and suggestion box.
2. The marketers should focus on advertising these low purchase products such as detergent paper, delicassen especially to hotels/restaurants in oporto and lisbon regions.
3. To communicate product benefits to customers through emails.
4. Include bonuses and incentives on those products.






