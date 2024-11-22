import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
import pandas as pd
from datetime import datetime
import re

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
    """
)
revenue_chain = LLMChain(llm=llm, prompt=revenue_prompt, output_key="revenue_projection")

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
        hr_costs_chain,
        services_costs_chain,
        operational_expenses_chain,
        infrastructure_costs_chain,
        other_costs_chain
    ],
    input_variables=[
        "products", "business_models", "growth_rate", "prices",
        "employee_roles", "salaries", "hiring_schedule",
        "services", "annual_costs",
        "categories", "monthly_expenses",
        "equipment", "costs", "depreciation",
        "miscellaneous_items", "annual_estimates"
    ],
    output_variables=[
        "revenue_projection",
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
    "prices": "Product A: $20, Product B: $50",
    "employee_roles": "Developer, Sales, Manager",
    "salaries": "$80,000, $60,000, $100,000",
    "hiring_schedule": "Q1: 2 Developers, Q2: 1 Sales",
    "services": "AWS, CRM Software, Marketing Tools",
    "annual_costs": "$10,000, $5,000, $8,000",
    "categories": "Rent, Utilities, Miscellaneous",
    "monthly_expenses": "$5,000, $1,500, $1,000",
    "equipment": "Servers, Computers, Office Furniture",
    "costs": "$30,000, $20,000, $10,000",
    "depreciation": "5 years",
    "miscellaneous_items": "Travel, Training",
    "annual_estimates": "$5,000, $2,000"
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
    # Remove currency symbols and commas
    text = text.replace('$', '').replace('£', '').replace(',', '')
    
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

def process_financial_forecast(results):
    pl_data = {
        'Category': [],
        'Amount': []
    }
    
    # Process Revenue
    total_revenue = 0
    revenue_lines = results['revenue_projection'].split('\n')
    for line in revenue_lines:
        if ':' in line and ('$' in line or '£' in line):
            amount = extract_number(line.split(':')[1])
            if amount > 0:
                total_revenue += amount
                pl_data['Category'].append('Revenue - ' + line.split(':')[0].strip())
                pl_data['Amount'].append(amount)
    
    # Add Total Revenue
    pl_data['Category'].append('Total Revenue')
    pl_data['Amount'].append(total_revenue)
    
    # Process Costs
    total_costs = 0
    
    # Process each cost category
    cost_mappings = {
        'hr_costs': 'Staff Costs',
        'services_costs': 'Services & Software',
        'operational_expenses': 'Operating Expenses',
        'infrastructure_costs': 'Infrastructure & Equipment',
        'other_costs': 'Other Expenses'
    }
    
    for key, category in cost_mappings.items():
        amount = extract_number(results[key])
        pl_data['Category'].append(category)
        pl_data['Amount'].append(amount)
        total_costs += amount
    
    # Add Total Costs
    pl_data['Category'].append('Total Costs')
    pl_data['Amount'].append(total_costs)
    
    # Calculate Gross Profit
    gross_profit = total_revenue - total_costs
    pl_data['Category'].append('Gross Profit')
    pl_data['Amount'].append(gross_profit)
    
    # Create DataFrame
    df = pd.DataFrame(pl_data)
    
    # Format amounts
    df['Amount'] = df['Amount'].apply(lambda x: f"£{x:,.2f}")
    
    # Save to Excel with formatting
    filename = f"financial_forecast_{datetime.now().strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='P&L', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['P&L']
        
        worksheet.column_dimensions['A'].width = 30
        worksheet.column_dimensions['B'].width = 20
        
        # Add title
        worksheet.cell(row=1, column=1, value='Profit & Loss Statement')
        worksheet.cell(row=2, column=1, value='Category')
        worksheet.cell(row=2, column=2, value='Amount')
        
        # Adjust all data down one row due to title
        for row in range(worksheet.max_row, 2, -1):
            for col in range(1, 3):
                worksheet.cell(row=row+1, column=col, value=worksheet.cell(row=row, column=col).value)
    
    return filename

# Process results
filename = process_financial_forecast(results)
print(f"P&L saved to: {filename}")