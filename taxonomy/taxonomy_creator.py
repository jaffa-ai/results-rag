import json
import openpyxl
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def create_taxonomy_from_excel(file_path):

    workbook = openpyxl.load_workbook(file_path, data_only=True)
    sheet = workbook['Presentation']

    taxonomy = {}
    current_table_type = None
    skip_next_row = False  # Flag to skip column header row

    # Iterate over the rows in the sheet
    for row in sheet.iter_rows(min_row=2, values_only=True):
        prefix, name, label = row[0], row[1], row[2]

        # Skip the column header row after table header
        if skip_next_row:
            skip_next_row = False
            continue

        # Check if this is a header row (LinkRole in first column)
        if prefix == 'LinkRole':
            # Clean up the name string
            cleaned_name = name.lower().replace('[', '').replace(']', '').strip()

            # Identify table type from the cleaned name column
            if 'cash flow statement' in cleaned_name and 'indirect' in cleaned_name:
                current_table_type = 'CashFlowStatementIndirect'
            elif 'cash flow statement' in cleaned_name and 'direct' in cleaned_name:
                current_table_type = 'CashFlowStatementDirect'
            elif 'statement of assets and liabilities' in cleaned_name:
                current_table_type = 'BalanceSheet'
            elif 'financial result' in cleaned_name:
                current_table_type = 'ProfitAndLoss'
            elif 'general information' in cleaned_name:
                current_table_type = 'DisclosureOfGeneralInformationAboutCompany'
            skip_next_row = True  # Set flag to skip next row (column headers)
            continue

        # Skip empty rows
        if all(val is None or val == '' for val in (prefix, name, label)):
            continue

        # Skip abstract tags
        if label and '[Abstract]' in label:
            continue

        # Skip if no table type identified yet
        if current_table_type is None:
            continue

        # Create XML tag
        xml_tag = f"{prefix}:{name}"

        # Create common terms
        common_terms = label

        # Add to taxonomy
        if current_table_type not in taxonomy:
            taxonomy[current_table_type] = {}

        # Ensure each name is unique within its table type
        if name not in taxonomy[current_table_type]:
            taxonomy[current_table_type][name] = {
                "xml_tag": xml_tag,
                "description": label,
                "common_terms": [common_terms]
            }

    return taxonomy

excel_file_path = os.path.join(PROJECT_ROOT, 'financial_results_indAS', 'Ind AS Taxonomy-2020-03-31.xlsx')
output_file_path = os.path.join(PROJECT_ROOT, 'base_taxonomy.json')

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Create taxonomy
taxonomy = create_taxonomy_from_excel(excel_file_path)

# Save the taxonomy to file
with open(output_file_path, 'w') as f:
    json.dump(taxonomy, f, indent=2)