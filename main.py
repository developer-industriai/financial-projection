import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy_financial as npf

class SaaSFinancialModel:
    TOTAL_NUMBER_OF_SCHOOLS = 25022 # Total number of schools in the UK
    ONE_CS_PER_CUSTOMERS = 50 # One customer support specialist per 20 customers
    ONE_SALES_REP_PER_CUSTOMERS = 50 # One customer support specialist per 20 customers
    ONE_MANAGER_PER_EMPLOYEES = 10 # One manager per 10 employees
    ONE_ENGINEER_PER_CUSTOMER = 100 # One Engineer per 100 customers
    NUMBER_OF_SHARES = 100

    def __init__(self):
        self.df = None
        self.hr_df = None
        self.services_costs_df = None
        self.npv_results = None
        self.irr_results = None
        self.months = 60  # 5 years
     
    @staticmethod   
    def calculate_npv(cash_flows, discount_rate):
        return np.sum(cash_flows / (1 + discount_rate) ** np.arange(len(cash_flows)))
     
    @staticmethod
    def calculate_irr(cash_flows):
        return npf.irr(cash_flows)
    
    # @staticmethod
    def calculate_uk_corporation_tax(self, profit, year):
        # UK Corporation Tax rates (as of 2023)
        if year < 2023:
            return profit * 0.19
        else:
            if profit <= 0:
                return 0
            if profit <= 50000:
                return profit * 0.19
            elif profit > 250000:
                return profit * 0.25
            else:
                base_tax = 50000 * 0.19
                marginal_profit = profit - 50000
                marginal_rate = (profit - 50000) / (250000 - 50000) * (0.25 - 0.19) + 0.19
                return base_tax + marginal_profit * marginal_rate
    
    # @staticmethod
    def run_monte_carlo(self, initial_investment, revenues, costs, discount_rate, iterations=1000):
        npv_results = []
        irr_results = []
        
        for _ in range(iterations):
            revenue_mult = norm.rvs(1, 0.1)  # Revenue multiplier with 10% standard deviation
            cost_mult = norm.rvs(1, 0.05)    # Cost multiplier with 5% standard deviation
            
            adjusted_revenues = revenues * revenue_mult
            adjusted_costs = costs * cost_mult
            adjusted_ebitda = adjusted_revenues - adjusted_costs
            
            taxes = [self.calculate_uk_corporation_tax(ebitda, year) for year, ebitda in enumerate(adjusted_ebitda, start=2023)]
            cash_flows = adjusted_ebitda - taxes
            cash_flows = np.insert(cash_flows, 0, -initial_investment)
            
            npv = self.calculate_npv(cash_flows, discount_rate)  # Assuming 10% discount rate
            npv_results.append(npv)
            
            if np.all(cash_flows[1:] >= 0) and np.any(cash_flows[1:] > 0):
                irr = self.calculate_irr(cash_flows)
                irr_results.append(irr)
        
        return npv_results, irr_results

    # @staticmethod
    def calculate_hr_resources(self, customer_base, years):
        self.roles = [
            {"title": "CEO", "base_salary": 65000, "start_year": 1, "end_year": 5, "seniority": "Executive"},
            {"title": "COO", "base_salary": 48000, "start_year": 3, "end_year": 5, "seniority": "Executive"},
            {"title": "CFO", "base_salary": 46000, "start_year": 1, "end_year": 5, "seniority": "Executive"},
            # {"title": "Sales Manager", "base_salary": 30000, "start_year": 2, "end_year": 5, "seniority": "Senior"},
            # {"title": "Marketing Manager", "base_salary": 30000, "start_year": 2, "end_year": 5, "seniority": "Senior"},
            {"title": "Marketing Analyst", "base_salary": 26000, "start_year": 2, "end_year": 5, "seniority": "Senior"},
            {"title": "Software Engineer", "base_salary": 46000, "start_year": 1, "end_year": 5, "seniority": "Mid-level"},
            {"title": "UI/UX Designer", "base_salary": 26000, "start_year": 2, "end_year": 5, "seniority": "Mid-level"},
            {"title": "Sales Representative", "base_salary": 26000, "start_year": 1, "end_year": 5, "seniority": "Junior"},
            {"title": "Customer Support Specialist", "base_salary": 22000, "start_year": 1, "end_year": 5, "seniority": "Junior"},
        ]
        
        self.hr_resources = []
        for year in range(1, years + 1):
            customer_count = customer_base[year - 1]
            role_counts = {}
            for role in self.roles:
                if role["start_year"] <= year <= role["end_year"]:
                    if role["title"] in ["CEO", "COO", "CFO"]:
                        count = 1
                        # Set allocation percentage for C-level executives
                        if year == 1:
                            allocation = 0.3
                        elif year == 2:
                            allocation = 0.5
                        else:
                            allocation = 1.0
                            
                    elif role["title"] == "Customer Support Specialist":
                        count = max(1, int(customer_count / self.ONE_CS_PER_CUSTOMERS))
                        allocation = 1.0
                        
                    elif role["title"] == "Sales Representative":
                        count = max(1, int(customer_count / self.ONE_SALES_REP_PER_CUSTOMERS))
                        allocation = 1.0
                        
                    elif role["title"] == "Software Engineer":
                        count = max(1, int(customer_count / self.ONE_ENGINEER_PER_CUSTOMER))
                        allocation = 1.0
                        
                    else:
                        base_count = max(1, int(customer_count / 200))
                        count = base_count + (base_count // self.ONE_MANAGER_PER_EMPLOYEES)
                        allocation = 1.0
                    
                    role_counts[role["title"]] = count
                    
                    for i in range(count):
                        seniority = role["seniority"]
                        if i > 0 and (i + 1) % 6 == 0:
                            seniority = "Manager"
                            salary = role["base_salary"] * 1.3
                        else:
                            salary = role["base_salary"]
                        
                        salary *= (1.03 ** (year - 1))  # 3% annual raise
                        labor_charges = salary * 0.15
                        benefits = salary * 0.05
                        total = (salary + labor_charges + benefits) * allocation
                        self.hr_resources.append({
                            "Role": role["title"],
                            "Start Year": year,
                            "End Year": min(year, role["end_year"]),
                            "Seniority": seniority,
                            "Salary": salary,
                            "Labor Charges": labor_charges,
                            "Benefits": benefits,
                            "AllocationPercentage": allocation * 100,
                            "Total": total,
                            "Necessary Resources": role_counts[role["title"]]
                        })
        
        return pd.DataFrame(self.hr_resources)
    
    @staticmethod
    def validate_cost_structure(sales_marketing, customer_success, admin, operations, engineering):
        total = sales_marketing + customer_success + admin + operations + engineering
        return abs(total - 100) < 0.01  # Allow for small floating-point errors
    
    @staticmethod
    def calculate_services_costs(years, employee_counts, customer_counts):
        costs = []
        for year, employees, customers in zip(range(1, years + 1), employee_counts, customer_counts):
            accounting_cost = 2000 * (1 + 0.1 * (year - 1))  # Base 2000/year, 10% increase each year
            legal_cost = 10000 * (1 + 0.05 * (year - 1))  # Base 10000/year, 5% increase each year
            rent_cost = 200 * 12 * employees * 0.5  # £200/person/month * 50% ocupancy on hybrid model
            office_materials = 20 * employees  # £20/person/year
            software_licenses = 50 * employees  # £1000/person/year for various software
            marketing_cost = 5000 * (1 + 0.2 * (year - 1)) + 50 * customers  # Base 5000/year, 20% increase each year, plus £50 per customer
            misc_cost = 2000 * (1 + 0.05 * (year - 1))  # Base 10000/year, 5% increase each year
            equipment_cost = 1000 * employees  # £1000/employee/year for equipment
            
            total_cost = accounting_cost + legal_cost + rent_cost + office_materials + software_licenses + marketing_cost + misc_cost + equipment_cost
            
            costs.append({
                'Year': year,
                'Accounting': accounting_cost,
                'Legal Advisory': legal_cost,
                'Rent (Co-working)': rent_cost,
                'Office Materials': office_materials,
                'Software Licenses': software_licenses,
                'Advertisement/Media': marketing_cost,
                'Miscellaneous': misc_cost,
                'Equipment': equipment_cost,
                'Total Services Costs': total_cost
            })
        
        return pd.DataFrame(costs)
    
    @staticmethod
    def calculate_cagr(start_value, end_value, num_years):
        return (end_value / start_value) ** (1 / num_years) - 1

    # RENDERING FUNCTIONS
    def render_sidebar_parameters(self):
        st.title('UK Start-up Financial Projection (SaaS, B2B)')

        st.sidebar.header('Input Parameters')
        self.market_size = st.sidebar.number_input('Market Size  (number of customers)', min_value=0, value=self.TOTAL_NUMBER_OF_SCHOOLS, step=10000)
        self.initial_investment = st.sidebar.number_input('Initial Investment (£)', min_value=0, value=500000, step=10000)
        self.initial_customers = st.sidebar.number_input('Initial number of customers', min_value=1, value=10)
        self.monthly_growth_rate = st.sidebar.slider('Monthly growth rate (%)', 0.0, 10.0, 9.2) / 100
        self.churn_rate = st.sidebar.slider('Monthly churn rate (%)', 0.0, 5.0, 1.0) / 100
        self.annual_subscription_price = st.sidebar.number_input('Annual subscription price (£)', min_value=0, value=10000)
        self.cogs_per_customer = st.sidebar.number_input('Cost of Goods Sold (COGS) per customer (£)', min_value=0, value=500)
        self.discount_rate = st.sidebar.slider('Discount rate (%)', 0.0, 20.0, 10.0) / 100
       
        # Cost structure (as % of revenue)
        # st.sidebar.subheader('Cost Structure (% of Revenue)')
        # self.sales_marketing_pct = st.sidebar.slider('Sales & Marketing (%)', 0, 100, 30)
        # self.customer_success_pct = st.sidebar.slider('Customer Success (%)', 0, 100, 15)
        # self.admin_pct = st.sidebar.slider('Administration (%)', 0, 100, 10)
        # self.operations_pct = st.sidebar.slider('Operations (%)', 0, 100, 20)
        # self.engineering_pct = st.sidebar.slider('Engineering (%)', 0, 100, 25)

        # # Validate cost structure
        # is_valid_cost_structure = self.validate_cost_structure(
        #     self.sales_marketing_pct, self.customer_success_pct, self.admin_pct, self.operations_pct, self.engineering_pct
        # )

        # if not is_valid_cost_structure:
        #     st.sidebar.error('Error: Cost Structure percentages must add up to exactly 100%.')
        #     st.stop()  # This will halt the execution of the app if the cost structure is invalid

    def calculate_projections(self):
        # Convert percentages to decimals
        # self.sales_marketing_pct /= 100
        # self.customer_success_pct /= 100
        # self.admin_pct /= 100
        # self.operations_pct /= 100
        # self.engineering_pct /= 100

        # Calculate projections
        self.customer_base = [self.initial_customers]
        for _ in range(1, self.months):
            new_customers = self.customer_base[-1] * self.monthly_growth_rate
            churned_customers = self.customer_base[-1] * self.churn_rate
            self.customer_base.append(self.customer_base[-1] + new_customers - churned_customers)

        self.monthly_revenues = [c * (self.annual_subscription_price / 12) for c in self.customer_base]
        self.annual_revenues = [sum(self.monthly_revenues[i:i+12]) for i in range(0, self.months, 12)]

        # Calculate costs
        # self.sales_marketing_costs = [r * self.sales_marketing_pct for r in self.annual_revenues]
        # self.customer_success_costs = [r * self.customer_success_pct for r in self.annual_revenues]
        # self.admin_costs = [r * self.admin_pct for r in self.annual_revenues]
        # self.operations_costs = [r * self.operations_pct for r in self.annual_revenues]
        # self.engineering_costs = [r * self.engineering_pct for r in self.annual_revenues]
        
        self.cogs = [r * self.cogs_per_customer for r in self.customer_base]

        # self.total_costs = [sum(costs) for costs in zip(self.sales_marketing_costs, self.customer_success_costs, self.admin_costs, self.operations_costs, self.engineering_costs, self.cogs)]

    def render_hr_resources(self):
        # HR Resources Table
        st.subheader('HR Resources Projection')
        self.hr_df = self.calculate_hr_resources([int(self.customer_base[i]) for i in range(11, self.months, 12)], 5)

        # Make the salary column editable
        self.edited_hr_df = st.data_editor(
            self.hr_df,
            column_config={
                "Salary": st.column_config.NumberColumn(
                    "Salary",
                    help="Edit the salary for each role",
                    min_value=0,
                    max_value=1000000,
                    step=1000,
                    format="£%d"
                ),
                'Labor Charges': st.column_config.NumberColumn(
                    "Labor Charges",
                    help="Automatically calculated as 15% of the salary",
                    format="£%d"
                ),
                'Benefits': st.column_config.NumberColumn(
                    "Benefits",
                    help="Automatically calculated as 20% of the salary",
                    format="£%d"
                ),
                'AllocationPercentage': st.column_config.NumberColumn(
                    "Allocation %",
                    help="Percentage of time allocated to the role",
                    min_value=0,
                    max_value=100,
                    step=1,
                    format="%d%%"
                ),
                'Total': st.column_config.NumberColumn(
                    "Total",
                    help="Automatically calculated as the sum of Salary, Labor Charges, and Benefits, adjusted for allocation",
                    format="£%d"
                )
            },
            hide_index=True,
            num_rows="dynamic",
        )

        # Calculate total HR costs per year
        self.hr_costs_per_year = self.edited_hr_df.groupby('Start Year')['Total'].sum().reset_index()
        self.hr_costs_per_year.columns = ['Year', 'Total HR Costs']

        st.subheader('Total HR Costs per Year')
        st.dataframe(self.hr_costs_per_year.style.format({
            'Total HR Costs': '£{:,.0f}'
        }))

        st.subheader('Total HR Costs per Year, per Role')

        # Prepare data for the chart
        self.hr_costs_by_role = self.edited_hr_df.groupby(['Start Year', 'Role'])['Total'].sum().unstack()

        # Create the horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        self.hr_costs_by_role.plot(kind='barh', stacked=True, ax=ax)

        ax.set_xlabel('Total Cost (£)')
        ax.set_ylabel('Year')
        ax.set_title('Total HR Costs per Year, per Role')
        ax.legend(title='Role', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add value labels on the bars
        for i, year in enumerate(self.hr_costs_by_role.index):
            total = self.hr_costs_by_role.loc[year].sum()
            ax.text(total, i, f'£{total:,.0f}', va='center')

        plt.tight_layout()
        st.pyplot(fig)

        # Add a table with the exact figures
        st.subheader('HR Costs Breakdown by Role and Year')
        # Add a total column with the sum of all costs for each year
        self.hr_costs_by_role['Total'] = self.hr_costs_by_role.sum(axis=1)
        st.dataframe(self.hr_costs_by_role.style.format('£{:,.0f}'))

        # Calculate and display the percentage breakdown
        st.subheader('HR Costs Percentage Breakdown by Role and Year')
        self.hr_costs_percentage = self.hr_costs_by_role.apply(lambda x: x / x.sum() * 100, axis=1)
        st.dataframe(self.hr_costs_percentage.style.format('{:.2f}%'))


        # Headcount by Role and Year
        st.subheader('Headcount by Role and Year')

        # Prepare data for the chart
        self.hr_costs_by_role = self.edited_hr_df.groupby(['Start Year', 'Role'])['Total'].count().unstack()

        # Create the horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        self.hr_costs_by_role.plot(kind='barh', stacked=False, ax=ax)

        ax.set_xlabel('Quantity')
        ax.set_ylabel('Year')
        ax.set_title('Headcount by Role and Year')
        ax.legend(title='Role', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add value labels on the bars, showing the total headcount on each bar
        for container in ax.containers:
            ax.bar_label(container, label_type='edge')

        # Add separation space between bars
        ax.margins(y=0.2)

        plt.tight_layout()
        st.pyplot(fig)


        # Headcount by Role and Year Table. Add the number of customers for each year, as the second column.
        self.hr_costs_by_role['Customers'] = [int(self.customer_base[i]) for i in range(11, self.months, 12)]
        # reorder columns
        self.hr_costs_by_role = self.hr_costs_by_role[['Customers'] + [col for col in self.hr_costs_by_role.columns if col != 'Customers']]
        # Add a last column with the total headcount for each year, excluding the 'Customers' column
        self.hr_costs_by_role['Total Headcount'] = self.hr_costs_by_role.sum(axis=1) - self.hr_costs_by_role['Customers']

        st.dataframe(self.hr_costs_by_role)

        st.subheader('Total HR Costs per Year, by Business Area')

        # Define role to area mapping
        self.role_to_area = {
            'CEO': 'Administration',
            'COO': 'Administration',
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
        self.edited_hr_df['Area'] = self.edited_hr_df['Role'].map(self.role_to_area)

        # Prepare data for the chart
        self.hr_costs_by_area = self.edited_hr_df.groupby(['Start Year', 'Area'])['Total'].sum().unstack()

        # Create the horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        self.hr_costs_by_area.plot(kind='barh', stacked=True, ax=ax)

        ax.set_xlabel('Total Cost (£)')
        ax.set_ylabel('Year')
        ax.set_title('Total HR Costs per Year, by Business Area')
        ax.legend(title='Business Area', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add value labels on the bars
        for i, year in enumerate(self.hr_costs_by_area.index):
            total = self.hr_costs_by_area.loc[year].sum()
            ax.text(total, i, f'£{total:,.0f}', va='center')

        plt.tight_layout()
        st.pyplot(fig)

        # Add a table with the exact figures
        st.subheader('HR Costs Breakdown by Business Area and Year')
        # Add a total column with the sum of all costs for each year
        self.hr_costs_by_area['Total'] = self.hr_costs_by_area.sum(axis=1)
        st.dataframe(self.hr_costs_by_area.style.format('£{:,.0f}'))

        # Calculate and display the percentage breakdown
        st.subheader('HR Costs Percentage Breakdown by Business Area and Year')
        self.hr_costs_percentage_area = self.hr_costs_by_area.apply(lambda x: x / x.sum() * 100, axis=1)
        st.dataframe(self.hr_costs_percentage_area.style.format('{:.2f}%'))

        # Calculate Services Costs
        # employee_counts = edited_hr_df.groupby('Start Year')['Necessary Resources'].sum().tolist()
        self.employee_counts = self.hr_costs_by_role['Total Headcount'].tolist()
        self.customer_counts = [self.customer_base[i] for i in range(11, self.months, 12)]
        self.services_costs_df = self.calculate_services_costs(5, self.employee_counts, self.customer_counts)

    def render_services_costs(self):
        st.subheader('Services Costs')
        st.dataframe(self.services_costs_df.style.format({col: '£{:,.0f}' for col in self.services_costs_df.columns if col != 'Year'}))

    def render_5years_projection(self):
        # Prepare DataFrame for display
        self.df = pd.DataFrame({
            'Year': range(1, 6),
            'Customers': [self.customer_base[i] for i in range(11, self.months, 12)],
            'Revenue': self.annual_revenues,
            'COGS': [self.cogs[i] for i in range(11, self.months, 12)],#[self.customer_base[i] * self.cogs_per_customer for i in range(11, self.months, 12)],
            'HR Costs': self.hr_costs_per_year['Total HR Costs'],
            'Additional Costs': self.services_costs_df['Total Services Costs'],
            # 'Total Costs': self.total_costs,
        })

        self.df['Total Costs'] = self.df['COGS'] + self.df['HR Costs'] + self.df['Additional Costs']
        # Calculate EBITDA, Tax, and Net Profit
        self.df['EBITDA'] = self.df['Revenue'] - self.df['Total Costs']
        self.df['Tax'] = self.df['EBITDA'].apply(lambda x: self.calculate_uk_corporation_tax(x, 2023))

        # Subtract loss from previous year's tax, but only if the EBITDA was negative in the previous year
        self.df['Loss'] = self.df['EBITDA'].apply(lambda x: max(0, -x))
        self.df['Tax'] = self.df['Tax'] - self.df['Loss'].shift(1).fillna(0)
        self.df['Tax'] = self.df['Tax'].clip(lower=0)
        # Fill NaN values with 0
        self.df['Tax'] = self.df['Tax'].fillna(0)


        self.df['Net Profit'] = self.df['EBITDA'] - self.df['Tax']

        st.subheader('5-Year Financial Projection')
        st.dataframe(self.df.style.format({
            'Customers': '{:.0f}',
            'Revenue': '£{:,.0f}',
            'COGS': '£{:,.0f}',
            'HR Costs': '£{:,.0f}',
            'Total Costs': '£{:,.0f}',
            'Additional Costs': '£{:,.0f}',
            'EBITDA': '£{:,.0f}',
            'Loss': '£{:,.0f}',
            'Tax': '£{:,.0f}',
            'Net Profit': '£{:,.0f}'
        }))

        # Calculate NPV and IRR
        self.cash_flows = [-self.initial_investment] + self.df['Net Profit'].tolist()
        self.npv = self.calculate_npv(self.cash_flows, self.discount_rate)
        self.irr = self.calculate_irr(self.cash_flows)
        self.marketshare = self.df['Customers'].iloc[-1] / self.market_size

        st.subheader('Financial Indicators')

        # Calculate monthly growth rate (already defined in input parameters)
        self.monthly_growth = self.monthly_growth_rate * 100  # Convert to percentage

        # Calculate CAGR for customers and revenue
        self.customer_cagr = self.calculate_cagr(self.df['Customers'].iloc[0], self.df['Customers'].iloc[-1], 5)
        self.revenue_cagr = self.calculate_cagr(self.df['Revenue'].iloc[0], self.df['Revenue'].iloc[-1], 5)

        col1, col2 = st.columns(2)
        col1.metric('Net Present Value (NPV)', f'£{self.npv:,.0f}')
        col2.metric('Internal Rate of Return (IRR)', f'{self.irr:.2%}')

        col1, col2 = st.columns(2)
        col1.metric('Monthly Growth Rate', f'{self.monthly_growth:.2f}%')
        col2.metric('Customer CAGR (5 years)', f'{self.customer_cagr:.2%}')

        col1, col2 = st.columns(2)
        col1.metric('Revenue CAGR (5 years)', f'{self.revenue_cagr:.2%}')
        col2.metric('Number of Customers (5 years)', int(self.df['Customers'].iloc[-1]))

        col1.metric('Market Share (5 years)', f'{self.marketshare:.2%}')

    def render_growth_rates(self):
        # Growth rates table
        self.growth_rates = pd.DataFrame({
            'Metric': ['Monthly Growth Rate', 'Customer CAGR (5 years)', 'Revenue CAGR (5 years)'],
            'Rate': [self.monthly_growth, self.customer_cagr * 100, self.revenue_cagr * 100]
        })

        st.subheader('Growth Rates Summary')
        st.dataframe(self.growth_rates.style.format({
            'Rate': '{:.2f}%'
        }))

    def render_financial_indicators(self):
        # Monte Carlo simulation
        st.subheader('Monte Carlo Simulation')
        self.npv_results, self.irr_results = self.run_monte_carlo(self.initial_investment, np.array(self.annual_revenues), np.array(self.df['Total Costs']), self.discount_rate)

        col1, col2 = st.columns(2)
        col1.metric('NPV (Mean)', f'£{np.mean(self.npv_results):,.0f}')
        col2.metric('IRR (Mean)', f'{np.mean(self.irr_results):.2%}')

        # Visualizations
        st.subheader('Business Evolution and Scalability')

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(range(1, 61), self.customer_base, color='blue', label='Customers')
        ax1.set_xlabel('Months')
        ax1.set_ylabel('Number of Customers', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(range(1, 61), self.monthly_revenues, color='green', label='Monthly Revenue')
        ax2.set_ylabel('Monthly Revenue (£)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 1), ncol=2)
        plt.title('Business Evolution: Customer Base and Monthly Revenue')
        st.pyplot(fig)

        # NPV and IRR Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.hist(self.npv_results, bins=30, edgecolor='black')
        ax1.set_title('NPV Distribution')
        ax1.set_xlabel('NPV (£)')
        ax1.set_ylabel('Frequency')

        ax2.hist(self.irr_results, bins=30, edgecolor='black')
        ax2.set_title('IRR Distribution')
        ax2.set_xlabel('IRR')
        ax2.set_ylabel('Frequency')

        st.pyplot(fig)

    def render_profit_loss(self):
        # Profit vs Loss Chart
        st.subheader('Profit vs Loss Over 5 Years')

        fig, ax = plt.subplots(figsize=(12, 6))

        years = self.df['Year']
        revenue = self.df['Revenue']
        # total_costs = df['Total Costs'] + df['HR Costs']
        total_costs = self.df['Total Costs']
        profit_loss = self.df['Net Profit']
        profit_loss_percentage = profit_loss / revenue

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
            'Net Profit/Loss': profit_loss,
            'Net P&L %': profit_loss_percentage,
        })
        st.dataframe(yearly_breakdown.style.format({
            'Revenue': '£{:,.0f}',
            'Total Costs': '£{:,.0f}',
            'Net Profit/Loss': '£{:,.0f}',
            'Net P&L %': '{:.2%}'
        }))

    def render_anual_budget(self):
        st.subheader('Annual Budget Breakdown by Department')

        # Prepare data for the chart
        years = self.df['Year']
        self.sm_costs = [r * self.sales_marketing_pct for r in self.df['Revenue']]
        self.engineering_costs = [r * self.engineering_pct for r in self.df['Revenue']]
        self.operations_costs = [r * self.operations_pct for r in self.df['Revenue']]
        self.admin_costs = [r * self.admin_pct for r in self.df['Revenue']]

        # Create the stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(years, self.sm_costs, label='Sales & Marketing', alpha=0.8)
        ax.bar(years, self.engineering_costs, bottom=self.sm_costs, label='Engineering', alpha=0.8)
        ax.bar(years, self.operations_costs, bottom=[i+j for i,j in zip(self.sm_costs, self.engineering_costs)], label='Operations', alpha=0.8)
        ax.bar(years, self.admin_costs, bottom=[i+j+k for i,j,k in zip(self.sm_costs, self.engineering_costs, self.operations_costs)], label='Administration', alpha=0.8)

        ax.set_xlabel('Year')
        ax.set_ylabel('Budget (£)')
        ax.set_title('Annual Budget Breakdown by Department')
        ax.legend(loc='upper left')

        # Add value labels on the bars
        for i, year in enumerate(years):
            total = self.sm_costs[i] + self.engineering_costs[i] + self.operations_costs[i] + self.admin_costs[i]
            ax.text(year, total, f'£{total:,.0f}', ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig)

        # Add a table with the exact figures
        st.subheader('Annual Budget Breakdown (Detailed)')
        budget_breakdown = pd.DataFrame({
            'Year': years,
            'Sales & Marketing': self.sm_costs,
            'Engineering': self.engineering_costs,
            'Operations': self.operations_costs,
            'Administration': self.admin_costs,
            'Total Budget': [sum(x) for x in zip(self.sm_costs, self.engineering_costs, self.operations_costs, self.admin_costs)]
        })

        st.dataframe(budget_breakdown.style.format({col: '£{:,.0f}' for col in budget_breakdown.columns if col != 'Year'}))

    def render_5year_PandL_statement(self):
        st.subheader('5-Year Profit and Loss (P&L) Statement')

        pl_data = {
            'Category': [
                'Number of Customers',
                'Revenue (Sales)',
                'Cost of Goods Sold (COGS)',
                'Gross Profit',
                'Operating Expenses (OPEX)',
                'Operating Income (Operating Profit or EBIT)',
                'Other Income and Expenses',
                'Earnings Before Interest and Taxes (EBIT)',
                # 'Interest Expense',
                # 'Earnings Before Tax (EBT)',
                'Taxes',
                'Net Income (Net Profit or Net Earnings)',
                'Earnings Per Share (EPS)'
            ]
        }
        for i in range(5):
            year_data = [
                int(round(self.df['Customers'].iloc[i], 0)), # Number of Customers
                round(self.df['Revenue'].iloc[i], 2), # Revenue
                -round(self.df['Customers'].iloc[i], 0) * self.cogs_per_customer, # COGS
                0, # Gross Profit
                -(round(self.services_costs_df['Total Services Costs'].iloc[i], 2) +  round(self.df['HR Costs'].iloc[i], 2)), # Operating Expenses
                0, # Operating Income
                0, # Other Income and Expenses
                round(self.df['EBITDA'].iloc[i], 2),  # EBIT
                # 0, # Interest Expense
                -round(self.df['Tax'].iloc[i], 2), # Taxes
                round(self.df['Net Profit'].iloc[i], 2), # Net Profit
                round(self.df['Net Profit'].iloc[i], 2) / self.NUMBER_OF_SHARES # EPS
            ]
            
            year_data[3] = year_data[1] + year_data[2]  # Gross Profit
            year_data[5] = year_data[3] + year_data[4]
            
            pl_data[f'Year {i+1}'] = year_data

        pl_df = pd.DataFrame(pl_data)
        pl_df["Total"] = pl_df.sum(axis=1, numeric_only=True)

        st.dataframe(pl_df.style.format({
            'Year 1': '{:,.0f}',
            'Year 2': '{:,.0f}',
            'Year 3': '{:,.0f}',
            'Year 4': '{:,.0f}',
            'Year 5': '{:,.0f}',
            'Total': '£{:,.0f}',
        }))
        
        # Plot the P&L statement
        fig, ax = plt.subplots(figsize=(12, 6))
        pl_df.set_index('Category').drop(columns='Total').loc[['Revenue (Sales)']].T.plot(kind='bar', ax=ax)
        ax.set_ylabel('Amount (£)')
        ax.set_title('Revenue Over 5 Years')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Note:** This P&L statement is based on the projections and calculations from the financial model. All figures are in British Pounds (£).")


def main():
    model = SaaSFinancialModel()
    model.render_sidebar_parameters()
    model.calculate_projections()
    model.render_hr_resources()
    model.render_services_costs()
    model.render_5years_projection()
    model.render_growth_rates()
    model.render_financial_indicators()
    # model.render_profit_loss()
    # model.render_anual_budget()
    model.render_5year_PandL_statement()
    
if __name__ == "__main__":
    main()