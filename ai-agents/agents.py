import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import pandas as pd
from datetime import datetime
import re
from openpyxl.styles import Font


# Initialize OpenAI LLM
llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.5)

# Define specialized agent prompts
revenue_prompt = PromptTemplate(
    input_variables=["products", "business_models", "growth_rate", "prices"],
    template="""
    Generate a revenue projection based on the following:
    - Products: {products}
    - Business Models: {business_models}
    - Growth Rate: {growth_rate}
    - Prices: {prices}
    Output should include segmented projections for each product and each business model, showing year-by-year breakdowns.
    Please provide detailed customer numbers per year in your analysis.
    Format your response to clearly separate customer numbers and revenue figures.
    """
)
revenue_chain = LLMChain(llm=llm, prompt=revenue_prompt, output_key="revenue_projection")

cogs_prompt = PromptTemplate(
    input_variables=["revenue_projection", "customers_per_device", "device_cost"],
    template="""
    Calculate Cost of Goods Sold (COGS) based on the following:
    - Revenue Projections and Customer Data: {revenue_projection}
    - IoT Device Requirements: 1 device per {customers_per_device} customers
    - Device Cost: £{device_cost} per device
    
    Please calculate:
    1. Number of devices needed per year based on customer growth
    2. Total device costs per year
    3. Other direct costs (assume 40% of revenue for product costs and 20% for service delivery)
    4. Total COGS including both device costs and other direct costs
    
    Format your response with clear year-by-year breakdowns for each cost component.
    """
)
cogs_chain = LLMChain(llm=llm, prompt=cogs_prompt, output_key="cogs_projection")

hr_costs_prompt = PromptTemplate(
    input_variables=["employee_roles", "salaries", "hiring_schedule"],
    template="""
    Calculate HR costs based on the following:
    - Employee Roles: {employee_roles}
    - Average Salaries: {salaries}
    - Hiring Schedule: {hiring_schedule}
    Output should include total HR costs for each year.
    """
)
hr_costs_chain = LLMChain(llm=llm, prompt=hr_costs_prompt, output_key="hr_costs")

services_costs_prompt = PromptTemplate(
    input_variables=["services", "annual_costs"],
    template="""
    Estimate the annual cost of services based on:
    - Services used: {services}
    - Annual Cost per Service: {annual_costs}
    Include a detailed breakdown for each service.
    """
)
services_costs_chain = LLMChain(llm=llm, prompt=services_costs_prompt, output_key="services_costs")

operational_expenses_prompt = PromptTemplate(
    input_variables=["categories", "monthly_expenses"],
    template="""
    Calculate operational expenses based on the following:
    - Expense Categories: {categories}
    - Monthly Expenses for each category: {monthly_expenses}
    Provide an annual breakdown.
    """
)
operational_expenses_chain = LLMChain(llm=llm, prompt=operational_expenses_prompt, output_key="operational_expenses")

infrastructure_costs_prompt = PromptTemplate(
    input_variables=["equipment", "costs", "depreciation"],
    template="""
    Estimate the costs for infrastructure and equipment:
    - Equipment: {equipment}
    - Initial Cost: {costs}
    - Depreciation schedule: {depreciation}
    Provide detailed yearly estimates.
    """
)
infrastructure_costs_chain = LLMChain(llm=llm, prompt=infrastructure_costs_prompt, output_key="infrastructure_costs")

other_costs_prompt = PromptTemplate(
    input_variables=["miscellaneous_items", "annual_estimates"],
    template="""
    Provide an estimate for miscellaneous costs based on:
    - Items: {miscellaneous_items}
    - Annual Estimates: {annual_estimates}
    Include a detailed breakdown.
    """
)
other_costs_chain = LLMChain(llm=llm, prompt=other_costs_prompt, output_key="other_costs")

# Create SequentialChain
financial_forecast_chain = SequentialChain(
    chains=[
        revenue_chain,
        cogs_chain,
        hr_costs_chain,
        services_costs_chain,
        operational_expenses_chain,
        infrastructure_costs_chain,
        other_costs_chain
    ],
    input_variables=[
        "products", "business_models", "growth_rate", "prices",
        "customers_per_device", "device_cost",
        "employee_roles", "salaries", "hiring_schedule",
        "services", "annual_costs",
        "categories", "monthly_expenses",
        "equipment", "costs", "depreciation",
        "miscellaneous_items", "annual_estimates"
    ],
    output_variables=[
        "revenue_projection",
        "cogs_projection",
        "hr_costs",
        "services_costs",
        "operational_expenses",
        "infrastructure_costs",
        "other_costs"
    ]
)

# Sample inputs
input_data = {
    "products": "Product A, Product B",
    "business_models": "Subscription, One-time Purchase",
    "growth_rate": "10% annually",
    "prices": "Product A: £20, Product B: £50",
    "customers_per_device": "10",  # New parameter: customers per IoT device
    "device_cost": "30",  # New parameter: cost per IoT device
    "employee_roles": "Developer, Sales, Manager",
    "salaries": "£80,000, £60,000, £100,000",
    "hiring_schedule": "Q1: 2 Developers, Q2: 1 Sales",
    "services": "AWS, CRM Software, Marketing Tools",
    "annual_costs": "£10,000, £5,000, £8,000",
    "categories": "Rent, Utilities, Miscellaneous",
    "monthly_expenses": "£5,000, £1,500, £1,000",
    "equipment": "Servers, Computers, Office Furniture",
    "costs": "£30,000, £20,000, £10,000",
    "depreciation": "5 years",
    "miscellaneous_items": "Travel, Training",
    "annual_estimates": "£5,000, £2,000"
}

# Execute the chain
results = financial_forecast_chain(input_data)

# Print results
# print("Financial Forecast Report:")
# for key, value in results.items():
#     print(f"\n{key.replace('_', ' ').title()}:")
#     print(value)
    
def extract_number(text):
    """Extract numerical value from text containing currency and calculations."""
    if isinstance(text, (int, float)):
        return float(text)
    
    # Remove currency symbols and commas
    text = str(text).replace('$', '').replace('£', '').replace(',', '')
    
    # Try to find a simple number first
    number_match = re.search(r'\d+\.?\d*', text)
    if number_match:
        return float(number_match.group())
    
    # If no simple number, evaluate basic calculations
    try:
        # Remove any text, keeping only numbers and basic operators
        calc_text = re.sub(r'[^0-9\+\-\*\/\.\(\)\s]', '', text)
        if calc_text:
            return float(eval(calc_text))
    except:
        pass
    
    return 0.0

def parse_projection_data(text, prefix):
    """Parse year-by-year data from projection text."""
    years_data = [0] * 5  # Initialize 5 years of data
    lines = text.split('\n')
    for line in lines:
        if prefix in line.lower() and ('£' in line or '$' in line):
            # Try to find year indicators and values
            year_matches = re.finditer(r'year\s*(\d+)[:\s]*[£$]?\s*([\d,]+\.?\d*)', line.lower())
            for match in year_matches:
                year_idx = int(match.group(1)) - 1
                if 0 <= year_idx < 5:
                    years_data[year_idx] = extract_number(match.group(2))
    return years_data

def process_financial_forecast(results):
    # Initialize DataFrame structure
    columns = ['Category', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    
    # Parse revenue and customer data
    revenue_data = parse_projection_data(results['revenue_projection'], 'revenue')
    customers_data = parse_projection_data(results['revenue_projection'], 'customers')
    
    # Parse COGS data
    cogs_data = parse_projection_data(results['cogs_projection'], 'total cogs')
    device_costs = parse_projection_data(results['cogs_projection'], 'device costs')
    product_costs = parse_projection_data(results['cogs_projection'], 'product costs')
    service_costs = parse_projection_data(results['cogs_projection'], 'service costs')
    
    # Create P&L structure
    pl_structure = {
        'Revenue': {
            'Product Sales': [r * 0.7 for r in revenue_data],  # Assuming 70% product revenue
            'Service Revenue': [r * 0.3 for r in revenue_data]  # Assuming 30% service revenue
        },
        'Cost_of_Goods_Sold': {
            'Product Costs': product_costs,
            'Service Costs': service_costs,
            'IoT Device Costs': device_costs
        }
    }

    rows = []
    
    # Revenue section
    rows.append(['Revenue', '', '', '', '', ''])
    for category, values in pl_structure['Revenue'].items():
        rows.append([category] + [f'£{val:,.0f}' for val in values])
    total_revenue = [sum(x) for x in zip(*pl_structure['Revenue'].values())]
    rows.append(['Total Revenue'] + [f'£{val:,.0f}' for val in total_revenue])
    
    # COGS section
    rows.append(['Cost of Goods Sold (COGS)', '', '', '', '', ''])
    for category, values in pl_structure['Cost_of_Goods_Sold'].items():
        rows.append([category] + [f'£{val:,.0f}' for val in values])
    total_cogs = [sum(x) for x in zip(*pl_structure['Cost_of_Goods_Sold'].values())]
    rows.append(['Total COGS'] + [f'£{val:,.0f}' for val in total_cogs])
    
    # Gross Profit
    gross_profit = [r - c for r, c in zip(total_revenue, total_cogs)]
    rows.append(['Gross Profit'] + [f'£{val:,.0f}' for val in gross_profit])
    
    # Parse and add operating expenses
    operating_expenses = parse_projection_data(results['operational_expenses'], 'total')
    marketing_sales = parse_projection_data(results['operational_expenses'], 'marketing')
    rd_expenses = parse_projection_data(results['operational_expenses'], 'r&d')
    admin_costs = parse_projection_data(results['operational_expenses'], 'admin')
    
    rows.append(['Operating Expenses', '', '', '', '', ''])
    rows.append(['Marketing & Sales'] + [f'£{val:,.0f}' for val in marketing_sales])
    rows.append(['R&D Expenses'] + [f'£{val:,.0f}' for val in rd_expenses])
    rows.append(['Administrative Costs'] + [f'£{val:,.0f}' for val in admin_costs])
    rows.append(['Total Operating Expenses'] + [f'£{val:,.0f}' for val in operating_expenses])
    
    # Operating Profit (EBIT)
    operating_profit = [g - o for g, o in zip(gross_profit, operating_expenses)]
    rows.append(['Operating Profit (EBIT)'] + [f'£{val:,.0f}' for val in operating_profit])
    
    # Other Income/Expenses
    interest_expenses = [5000, 10000, 10000, 10000, 10000]  # Example values
    tax_rate = 0.20
    tax_expenses = [max(0, op * tax_rate) for op in operating_profit]
    
    rows.append(['Other Income/Expenses', '', '', '', '', ''])
    rows.append(['Interest Expenses'] + [f'£{val:,.0f}' for val in interest_expenses])
    rows.append(['Tax (20%)'] + [f'£{val:,.0f}' for val in tax_expenses])
    
    # Net Profit
    net_profit = [op - ie - te for op, ie, te in zip(operating_profit, interest_expenses, tax_expenses)]
    rows.append(['Net Profit'] + [f'£{val:,.0f}' for val in net_profit])
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # Save to Excel with formatting
    filename = f"financial_forecast_{datetime.now().strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='P&L', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['P&L']
        
        # Format columns
        worksheet.column_dimensions['A'].width = 25
        for col in ['B', 'C', 'D', 'E', 'F']:
            worksheet.column_dimensions[col].width = 15
        
        # Create bold font style
        bold_font = Font(bold=True)
        
        # Add formatting
        for row_idx, row in enumerate(worksheet.iter_rows(min_row=1, max_row=worksheet.max_row), 1):
            if not any(cell.value for cell in row[1:]):  # Section headers
                for cell in row:
                    cell.font = bold_font
            elif row[0].value in ['Total Revenue', 'Total COGS', 'Gross Profit', 
                                'Total Operating Expenses', 'Operating Profit (EBIT)', 
                                'Net Profit']:
                for cell in row:
                    cell.font = bold_font
    
    return filename

# Process results
filename = process_financial_forecast(results)
print(f"P&L saved to: {filename}")