import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy_financial as npf

TOTAL_NUMBER_OF_SCHOOLS = 25022 # Total number of schools in the UK

ONE_CS_PER_CUSTOMERS = 50 # One customer support specialist per 20 customers
ONE_MANAGER_PER_EMPLOYEES = 10 # One manager per 10 employees

def calculate_npv(cash_flows, discount_rate):
    return np.sum(cash_flows / (1 + discount_rate) ** np.arange(len(cash_flows)))

def calculate_irr(cash_flows):
    return npf.irr(cash_flows)

def calculate_uk_corporation_tax(profit, year):
    # UK Corporation Tax rates (as of 2023)
    if year < 2023:
        return profit * 0.19
    else:
        if profit <= 50000:
            return profit * 0.19
        elif profit > 250000:
            return profit * 0.25
        else:
            base_tax = 50000 * 0.19
            marginal_profit = profit - 50000
            marginal_rate = (profit - 50000) / (250000 - 50000) * (0.25 - 0.19) + 0.19
            return base_tax + marginal_profit * marginal_rate

def run_monte_carlo(initial_investment, revenues, costs, iterations=1000):
    npv_results = []
    irr_results = []
    
    for _ in range(iterations):
        revenue_mult = norm.rvs(1, 0.1)  # Revenue multiplier with 10% standard deviation
        cost_mult = norm.rvs(1, 0.05)    # Cost multiplier with 5% standard deviation
        
        adjusted_revenues = revenues * revenue_mult
        adjusted_costs = costs * cost_mult
        adjusted_ebitda = adjusted_revenues - adjusted_costs
        
        taxes = [calculate_uk_corporation_tax(ebitda, year) for year, ebitda in enumerate(adjusted_ebitda, start=2023)]
        cash_flows = adjusted_ebitda - taxes
        cash_flows = np.insert(cash_flows, 0, -initial_investment)
        
        npv = calculate_npv(cash_flows, 0.1)  # Assuming 10% discount rate
        npv_results.append(npv)
        
        if np.all(cash_flows[1:] >= 0) and np.any(cash_flows[1:] > 0):
            irr = calculate_irr(cash_flows)
            irr_results.append(irr)
    
    return npv_results, irr_results

def calculate_hr_resources(customer_base, years):
    roles = [
        {"title": "CEO", "base_salary": 65000, "start_year": 1, "end_year": 5, "seniority": "Executive"},
        {"title": "CTO", "base_salary": 65000, "start_year": 3, "end_year": 5, "seniority": "Executive"},
        {"title": "CFO", "base_salary": 60000, "start_year": 2, "end_year": 5, "seniority": "Executive"},
        {"title": "Sales Manager", "base_salary": 46000, "start_year": 1, "end_year": 5, "seniority": "Senior"},
        {"title": "Marketing Manager", "base_salary": 46000, "start_year": 2, "end_year": 5, "seniority": "Senior"},
        {"title": "Customer Success Manager", "base_salary": 46000, "start_year": 2, "end_year": 5, "seniority": "Senior"},
        {"title": "Software Engineer", "base_salary": 46000, "start_year": 1, "end_year": 5, "seniority": "Mid-level"},
        {"title": "UI/UX Designer", "base_salary": 38000, "start_year": 2, "end_year": 5, "seniority": "Mid-level"},
        {"title": "Sales Representative", "base_salary": 40000, "start_year": 1, "end_year": 5, "seniority": "Junior"},
        {"title": "Customer Support Specialist", "base_salary": 35000, "start_year": 1, "end_year": 5, "seniority": "Junior"},
    ]
    
    hr_resources = []
    for year in range(1, years + 1):
        customer_count = customer_base[year - 1]
        role_counts = {}
        for role in roles:
            if role["start_year"] <= year <= role["end_year"]:
                if role["title"] in ["CEO", "CTO", "CFO"]:
                    count = 1
                elif role["title"] == "Customer Support Specialist":
                    count = max(1, int(customer_count / ONE_CS_PER_CUSTOMERS))  # 1 specialist per 50 customers
                else:
                    base_count = max(1, int(customer_count / 200))
                    count = base_count + (base_count // ONE_MANAGER_PER_EMPLOYEES)  # Add a manager for every 10 employees
                
                role_counts[role["title"]] = count
                
                for i in range(count):
                    seniority = role["seniority"]
                    if i > 0 and (i + 1) % 6 == 0:  # Every 6th person (including the first) is a manager
                        seniority = "Manager"
                        salary = role["base_salary"] * 1.3  # 30% more for managers
                    else:
                        salary = role["base_salary"]
                    
                    salary *= (1.03 ** (year - 1))  # 3% annual raise
                    labor_charges = salary * 0.15  # Assuming 15% labor charges
                    benefits = salary * 0.20  # Assuming 20% benefits
                    total = salary + labor_charges + benefits
                    hr_resources.append({
                        "Role": role["title"],
                        "Start Year": year,
                        "End Year": min(year, role["end_year"]),
                        "Seniority": seniority,
                        "Salary": salary,
                        "Labor Charges": labor_charges,
                        "Benefits": benefits,
                        "Total": total,
                        "Necessary Resources": role_counts[role["title"]]
                    })
        
        # Add the number of necessary resources for each role
        # for item in hr_resources:
        #     if item["Start Year"] == year:
        #         item["Necessary Resources"] = role_counts[item["Role"]]
    
    
    return pd.DataFrame(hr_resources)

def validate_cost_structure(sales_marketing, customer_success, admin, operations, engineering):
    total = sales_marketing + customer_success + admin + operations + engineering
    return abs(total - 100) < 0.01  # Allow for small floating-point errors

def calculate_additional_costs(years, employee_counts, customer_counts):
    costs = []
    for year, employees, customers in zip(range(1, years + 1), employee_counts, customer_counts):
        accounting_cost = 12000 * (1 + 0.1 * (year - 1))  # Base 12000/year, 10% increase each year
        legal_cost = 10000 * (1 + 0.05 * (year - 1))  # Base 10000/year, 5% increase each year
        rent_cost = 200 * 12 * employees  # £200/person/month
        office_materials = 500 * employees  # £500/person/year
        software_licenses = 1000 * employees  # £1000/person/year for various software
        marketing_cost = 5000 * (1 + 0.2 * (year - 1)) + 50 * customers  # Base 5000/year, 20% increase each year, plus £50 per customer
        misc_cost = 10000 * (1 + 0.05 * (year - 1))  # Base 10000/year, 5% increase each year

        total_cost = accounting_cost + legal_cost + rent_cost + office_materials + software_licenses + marketing_cost + misc_cost
        
        costs.append({
            'Year': year,
            'Accounting': accounting_cost,
            'Legal Advisory': legal_cost,
            'Rent (Co-working)': rent_cost,
            'Office Materials': office_materials,
            'Software Licenses': software_licenses,
            'Marketing': marketing_cost,
            'Miscellaneous': misc_cost,
            'Total Additional Costs': total_cost
        })
    
    return pd.DataFrame(costs)

def calculate_cagr(start_value, end_value, num_years):
    return (end_value / start_value) ** (1 / num_years) - 1

# Main Streamlit app code
st.title('UK SaaS Financial Projection for Schools')

st.sidebar.header('Input Parameters')
initial_investment = st.sidebar.number_input('Initial Investment (£)', min_value=0, value=500000, step=10000)
initial_customers = st.sidebar.number_input('Initial number of customers', min_value=1, value=10)
monthly_growth_rate = st.sidebar.slider('Monthly growth rate (%)', 0.0, 10.0, 8.5) / 100
churn_rate = st.sidebar.slider('Monthly churn rate (%)', 0.0, 5.0, 1.0) / 100
annual_subscription_price = st.sidebar.number_input('Annual subscription price (£)', min_value=0, value=6000)

# Cost structure (as % of revenue)
st.sidebar.subheader('Cost Structure (% of Revenue)')
sales_marketing_pct = st.sidebar.slider('Sales & Marketing (%)', 0, 100, 30)
customer_success_pct = st.sidebar.slider('Customer Success (%)', 0, 100, 15)
admin_pct = st.sidebar.slider('Administration (%)', 0, 100, 10)
operations_pct = st.sidebar.slider('Operations (%)', 0, 100, 20)
engineering_pct = st.sidebar.slider('Engineering (%)', 0, 100, 25)

# Validate cost structure
is_valid_cost_structure = validate_cost_structure(
    sales_marketing_pct, customer_success_pct, admin_pct, operations_pct, engineering_pct
)

if not is_valid_cost_structure:
    st.sidebar.error('Error: Cost Structure percentages must add up to exactly 100%.')
    st.stop()  # This will halt the execution of the app if the cost structure is invalid

# Convert percentages to decimals
sales_marketing_pct /= 100
customer_success_pct /= 100
admin_pct /= 100
operations_pct /= 100
engineering_pct /= 100

# Calculate projections
months = 60  # 5 years
customer_base = [initial_customers]
for _ in range(1, months):
    new_customers = customer_base[-1] * monthly_growth_rate
    churned_customers = customer_base[-1] * churn_rate
    customer_base.append(customer_base[-1] + new_customers - churned_customers)

monthly_revenues = [c * (annual_subscription_price / 12) for c in customer_base]
annual_revenues = [sum(monthly_revenues[i:i+12]) for i in range(0, months, 12)]

# Calculate costs
sales_marketing_costs = [r * sales_marketing_pct for r in annual_revenues]
customer_success_costs = [r * customer_success_pct for r in annual_revenues]
admin_costs = [r * admin_pct for r in annual_revenues]
operations_costs = [r * operations_pct for r in annual_revenues]
engineering_costs = [r * engineering_pct for r in annual_revenues]

total_costs = [sum(costs) for costs in zip(sales_marketing_costs, customer_success_costs, admin_costs, operations_costs, engineering_costs)]

# HR Resources Table
st.subheader('HR Resources Projection')
hr_df = calculate_hr_resources([customer_base[i] for i in range(11, months, 12)], 5)

# Make the salary column editable
edited_hr_df = st.data_editor(
    hr_df,
    column_config={
        "Salary": st.column_config.NumberColumn(
            "Salary",
            help="Edit the salary for each role",
            min_value=0,
            max_value=1000000,
            step=1000,
            format="£%d"
        )
    },
    hide_index=True,
    num_rows="dynamic",
)

# Recalculate labor charges, benefits, and total based on edited salaries
edited_hr_df['Labor Charges'] = edited_hr_df['Salary'] * 0.15
edited_hr_df['Benefits'] = edited_hr_df['Salary'] * 0.20
edited_hr_df['Total'] = edited_hr_df['Salary'] + edited_hr_df['Labor Charges'] + edited_hr_df['Benefits']

# Display the updated HR costs
edited_hr_df.style.format({
    'Salary': '£{:,.0f}',
    'Labor Charges': '£{:,.0f}',
    'Benefits': '£{:,.0f}',
    'Total': '£{:,.0f}'
})

# st.dataframe(edited_hr_df)

# Calculate total HR costs per year
hr_costs_per_year = edited_hr_df.groupby('Start Year')['Total'].sum().reset_index()
hr_costs_per_year.columns = ['Year', 'Total HR Costs']

st.subheader('Total HR Costs per Year')
st.dataframe(hr_costs_per_year.style.format({
    'Total HR Costs': '£{:,.0f}'
}))

st.subheader('Total HR Costs per Year, per Role')

# Prepare data for the chart
hr_costs_by_role = edited_hr_df.groupby(['Start Year', 'Role'])['Total'].sum().unstack()

# Create the horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 8))

hr_costs_by_role.plot(kind='barh', stacked=True, ax=ax)

ax.set_xlabel('Total Cost (£)')
ax.set_ylabel('Year')
ax.set_title('Total HR Costs per Year, per Role')
ax.legend(title='Role', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add value labels on the bars
for i, year in enumerate(hr_costs_by_role.index):
    total = hr_costs_by_role.loc[year].sum()
    ax.text(total, i, f'£{total:,.0f}', va='center')

plt.tight_layout()
st.pyplot(fig)

# Add a table with the exact figures
st.subheader('HR Costs Breakdown by Role and Year')
st.dataframe(hr_costs_by_role.style.format('£{:,.0f}'))

# Calculate and display the percentage breakdown
st.subheader('HR Costs Percentage Breakdown by Role and Year')
hr_costs_percentage = hr_costs_by_role.apply(lambda x: x / x.sum() * 100, axis=1)
st.dataframe(hr_costs_percentage.style.format('{:.2f}%'))

st.subheader('Total HR Costs per Year, by Business Area')

# Define role to area mapping
role_to_area = {
    'CEO': 'Administration',
    'CTO': 'Engineering',
    'CFO': 'Administration',
    'Sales Manager': 'S&M',
    'Marketing Manager': 'S&M',
    'Customer Success Manager': 'Operations',
    'Software Engineer': 'Engineering',
    'UI/UX Designer': 'Engineering',
    'Sales Representative': 'S&M',
    'Customer Support Specialist': 'Operations'
}

# Add the 'Area' column to the DataFrame
edited_hr_df['Area'] = edited_hr_df['Role'].map(role_to_area)

# Prepare data for the chart
hr_costs_by_area = edited_hr_df.groupby(['Start Year', 'Area'])['Total'].sum().unstack()

# Create the horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 8))

hr_costs_by_area.plot(kind='barh', stacked=True, ax=ax)

ax.set_xlabel('Total Cost (£)')
ax.set_ylabel('Year')
ax.set_title('Total HR Costs per Year, by Business Area')
ax.legend(title='Business Area', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add value labels on the bars
for i, year in enumerate(hr_costs_by_area.index):
    total = hr_costs_by_area.loc[year].sum()
    ax.text(total, i, f'£{total:,.0f}', va='center')

plt.tight_layout()
st.pyplot(fig)

# Add a table with the exact figures
st.subheader('HR Costs Breakdown by Business Area and Year')
st.dataframe(hr_costs_by_area.style.format('£{:,.0f}'))

# Calculate and display the percentage breakdown
st.subheader('HR Costs Percentage Breakdown by Business Area and Year')
hr_costs_percentage_area = hr_costs_by_area.apply(lambda x: x / x.sum() * 100, axis=1)
st.dataframe(hr_costs_percentage_area.style.format('{:.2f}%'))



# Calculate additional operational costs
employee_counts = edited_hr_df.groupby('Start Year')['Necessary Resources'].sum().tolist()
customer_counts = [customer_base[i] for i in range(11, months, 12)]
additional_costs_df = calculate_additional_costs(5, employee_counts, customer_counts)

st.subheader('Additional Operational Costs')
st.dataframe(additional_costs_df.style.format({col: '£{:,.0f}' for col in additional_costs_df.columns if col != 'Year'}))


# Prepare DataFrame for display
df = pd.DataFrame({
    'Year': range(1, 6),
    'Customers': [customer_base[i] for i in range(11, months, 12)],
    'Revenue': annual_revenues,
    'Total Costs': total_costs,
    'HR Costs': hr_costs_per_year['Total HR Costs'],
})



# Calculate EBITDA, Tax, and Net Profit
df['HR Costs'] = hr_costs_per_year['Total HR Costs']
df['Additional Costs'] = additional_costs_df['Total Additional Costs']
df['Total Costs'] = df['Total Costs'] + df['HR Costs'] + df['Additional Costs']
df['EBITDA'] = df['Revenue'] - df['Total Costs']
df['Tax'] = df['EBITDA'].apply(lambda x: calculate_uk_corporation_tax(x, 2023))
df['Net Profit'] = df['EBITDA'] - df['Tax']

st.subheader('5-Year Financial Projection')
st.dataframe(df.style.format({
    'Customers': '{:.0f}',
    'Revenue': '£{:,.0f}',
    'Total Costs': '£{:,.0f}',
    'HR Costs': '£{:,.0f}',
    'Additional Costs': '£{:,.0f}',
    'EBITDA': '£{:,.0f}',
    'Tax': '£{:,.0f}',
    'Net Profit': '£{:,.0f}'
}))

# Calculate NPV and IRR
cash_flows = [-initial_investment] + df['Net Profit'].tolist()
npv = calculate_npv(cash_flows, 0.1)  # Assuming 10% discount rate
irr = calculate_irr(cash_flows)
marketshare = df['Customers'].iloc[-1] / TOTAL_NUMBER_OF_SCHOOLS

st.subheader('Financial Indicators')

# Calculate monthly growth rate (already defined in input parameters)
monthly_growth = monthly_growth_rate * 100  # Convert to percentage

# Calculate CAGR for customers and revenue
customer_cagr = calculate_cagr(df['Customers'].iloc[0], df['Customers'].iloc[-1], 5)
revenue_cagr = calculate_cagr(df['Revenue'].iloc[0], df['Revenue'].iloc[-1], 5)

col1, col2 = st.columns(2)
col1.metric('Net Present Value (NPV)', f'£{npv:,.0f}')
col2.metric('Internal Rate of Return (IRR)', f'{irr:.2%}')

col1, col2 = st.columns(2)
col1.metric('Monthly Growth Rate', f'{monthly_growth:.2f}%')
col2.metric('Customer CAGR (5 years)', f'{customer_cagr:.2%}')

col1, col2 = st.columns(2)
col1.metric('Revenue CAGR (5 years)', f'{revenue_cagr:.2%}')
col2.metric('Number of Customers (5 years)', int(df['Customers'].iloc[-1]))

col1.metric('Market Share (5 years)', f'{marketshare:.2%}')

# Growth rates table
growth_rates = pd.DataFrame({
    'Metric': ['Monthly Growth Rate', 'Customer CAGR (5 years)', 'Revenue CAGR (5 years)'],
    'Rate': [monthly_growth, customer_cagr * 100, revenue_cagr * 100]
})

st.subheader('Growth Rates Summary')
st.dataframe(growth_rates.style.format({
    'Rate': '{:.2f}%'
}))

# Monte Carlo simulation
st.subheader('Monte Carlo Simulation')
npv_results, irr_results = run_monte_carlo(initial_investment, np.array(annual_revenues), np.array(df['Total Costs']))

col1, col2 = st.columns(2)
col1.metric('NPV (Mean)', f'£{np.mean(npv_results):,.0f}')
col2.metric('IRR (Mean)', f'{np.mean(irr_results):.2%}')

# Visualizations
st.subheader('Business Evolution and Scalability')

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(range(1, 61), customer_base, color='blue', label='Customers')
ax1.set_xlabel('Months')
ax1.set_ylabel('Number of Customers', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(range(1, 61), monthly_revenues, color='green', label='Monthly Revenue')
ax2.set_ylabel('Monthly Revenue (£)', color='green')
ax2.tick_params(axis='y', labelcolor='green')

fig.legend(loc='upper left', bbox_to_anchor=(0.1, 1), ncol=2)
plt.title('Business Evolution: Customer Base and Monthly Revenue')
st.pyplot(fig)

# NPV and IRR Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(npv_results, bins=30, edgecolor='black')
ax1.set_title('NPV Distribution')
ax1.set_xlabel('NPV (£)')
ax1.set_ylabel('Frequency')

ax2.hist(irr_results, bins=30, edgecolor='black')
ax2.set_title('IRR Distribution')
ax2.set_xlabel('IRR')
ax2.set_ylabel('Frequency')

st.pyplot(fig)

# Profit vs Loss Chart
st.subheader('Profit vs Loss Over 5 Years')

fig, ax = plt.subplots(figsize=(12, 6))

years = df['Year']
revenue = df['Revenue']
total_costs = df['Total Costs'] + df['HR Costs']
profit_loss = df['Net Profit']

ax.bar(years, revenue, label='Revenue', alpha=0.8, color='g')
ax.bar(years, -total_costs, label='Costs', alpha=0.8, color='r')
ax.bar(years, profit_loss, label='Net Profit/Loss', alpha=0.8, color='b')

ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

for i, v in enumerate(profit_loss):
    ax.text(years[i], v, f'£{v:,.0f}', ha='center', va='bottom' if v > 0 else 'top')

ax.set_xlabel('Year')
ax.set_ylabel('Amount (£)')
ax.set_title('Profit vs Loss Over 5 Years')
ax.legend()

plt.tight_layout()
st.pyplot(fig)

# Add a table showing the yearly breakdown
st.subheader('Yearly Financial Breakdown')
yearly_breakdown = pd.DataFrame({
    'Year': years,
    'Revenue': revenue,
    'Total Costs': total_costs,
    'Net Profit/Loss': profit_loss
})
st.dataframe(yearly_breakdown.style.format({
    'Revenue': '£{:,.0f}',
    'Total Costs': '£{:,.0f}',
    'Net Profit/Loss': '£{:,.0f}'
}))

# After the existing visualizations, add the following code:

st.subheader('Annual Budget Breakdown by Department')

# Prepare data for the chart
years = df['Year']
sm_costs = [r * sales_marketing_pct for r in df['Revenue']]
engineering_costs = [r * engineering_pct for r in df['Revenue']]
operations_costs = [r * operations_pct for r in df['Revenue']]
admin_costs = [r * admin_pct for r in df['Revenue']]

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(years, sm_costs, label='Sales & Marketing', alpha=0.8)
ax.bar(years, engineering_costs, bottom=sm_costs, label='Engineering', alpha=0.8)
ax.bar(years, operations_costs, bottom=[i+j for i,j in zip(sm_costs, engineering_costs)], label='Operations', alpha=0.8)
ax.bar(years, admin_costs, bottom=[i+j+k for i,j,k in zip(sm_costs, engineering_costs, operations_costs)], label='Administration', alpha=0.8)

ax.set_xlabel('Year')
ax.set_ylabel('Budget (£)')
ax.set_title('Annual Budget Breakdown by Department')
ax.legend(loc='upper left')

# Add value labels on the bars
for i, year in enumerate(years):
    total = sm_costs[i] + engineering_costs[i] + operations_costs[i] + admin_costs[i]
    ax.text(year, total, f'£{total:,.0f}', ha='center', va='bottom')

plt.tight_layout()
st.pyplot(fig)

# Add a table with the exact figures
st.subheader('Annual Budget Breakdown (Detailed)')
budget_breakdown = pd.DataFrame({
    'Year': years,
    'Sales & Marketing': sm_costs,
    'Engineering': engineering_costs,
    'Operations': operations_costs,
    'Administration': admin_costs,
    'Total Budget': [sum(x) for x in zip(sm_costs, engineering_costs, operations_costs, admin_costs)]
})

st.dataframe(budget_breakdown.style.format({col: '£{:,.0f}' for col in budget_breakdown.columns if col != 'Year'}))
