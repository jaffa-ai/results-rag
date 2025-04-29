from typing import Dict, List, Tuple, Optional
from pydantic_model import FinancialData, QuarterData
from financial_statements_taxonomy import ProfitAndLoss, SegmentInformation

def validate_profit_and_loss(profit_loss: Dict) -> List[Tuple[str, float, float, float]]:
    """
    Validates a profit and loss statement according to the rules defined in financial_statement_validations.md.
    
    Args:
        profit_loss: A dictionary containing profit and loss data
        
    Returns:
        List of tuples containing (validation_rule, expected_value, actual_value, difference)
    """
    validation_failures = []
    
    # Safe access helper function
    def safe_get(obj, keys, default=0.0):
        """Safely get a nested value from a dictionary with a default if not found"""
        if not obj or not isinstance(obj, dict):
            return default
            
        current = obj
        for key in keys:
            if isinstance(current, dict) and key in current and current[key] is not None:
                current = current[key]
            else:
                return default
        return current or default

    # Get values using safe dictionary access
    sales = safe_get(profit_loss, ['revenue', 'sales_from_products_services'])
    other_operating = safe_get(profit_loss, ['revenue', 'other_operating_revenue'])
    other_income = safe_get(profit_loss, ['revenue', 'other_income'])
    total_income = safe_get(profit_loss, ['revenue', 'total_income'])
    
    # 1. total_income = sales_from_products_services + other_operating_revenue + other_income
    expected_total_income = sales + other_operating + other_income
    if abs(expected_total_income - total_income) > 0.01:  # Allow for small rounding differences
        validation_failures.append(
            ("total_income = sales_from_products_services + other_operating_revenue + other_income",
             expected_total_income,
             total_income,
             expected_total_income - total_income)
        )
    
    # 2. profit_before_exceptional_items_and_tax = total_income - total_expenses
    # expected_profit_before_exceptional = profit_loss.total_income - profit_loss.total_expenses
    # if abs(expected_profit_before_exceptional - profit_loss.profit_before_exceptional_items_and_tax) > 0.01:
    #     validation_failures.append(
    #         ("profit_before_exceptional_items_and_tax = total_income - total_expenses",
    #          expected_profit_before_exceptional,
    #          profit_loss.profit_before_exceptional_items_and_tax,
    #          expected_profit_before_exceptional - profit_loss.profit_before_exceptional_items_and_tax)
    #     )
    
    # 3. profit_before_tax = profit_before_exceptional_items_and_tax - exceptional_items
    # expected_profit_before_tax = profit_loss.profit_before_exceptional_items_and_tax - profit_loss.exceptional_items
    # if abs(expected_profit_before_tax - profit_loss.profit_before_tax) > 0.01:
    #     validation_failures.append(
    #         ("profit_before_tax = profit_before_exceptional_items_and_tax - exceptional_items",
    #          expected_profit_before_tax,
    #          profit_loss.profit_before_tax,
    #          expected_profit_before_tax - profit_loss.profit_before_tax)
    #     )
    
    # Get tax expense values
    current_tax = safe_get(profit_loss, ['tax_expense', 'current_tax'])
    deferred_tax = safe_get(profit_loss, ['tax_expense', 'deferred_tax'])
    total_tax_expense = safe_get(profit_loss, ['tax_expense', 'total_tax_expense'], None)

    # Check if total_tax_expense is present in the extraction
    if total_tax_expense is not None:
        total_tax_expense = total_tax_expense or 0.0
        # 4. total_tax_expense = current_tax + deferred_tax
        expected_total_tax = current_tax + deferred_tax
        if abs(expected_total_tax - total_tax_expense) > 0.01:
            validation_failures.append( 
                ("total_tax_expense = current_tax + deferred_tax",
                 expected_total_tax,
                 total_tax_expense,
                 expected_total_tax - total_tax_expense)
            )
    else:
        # If total_tax_expense is not present, calculate it from components
        total_tax_expense = current_tax + deferred_tax

    # total_expense = all elt in expense + other metrics in expense
    
    # Get profit values
    profit_before_tax = safe_get(profit_loss, ['profit_before_tax', 'profit_before_tax'])
    profit_after_tax = safe_get(profit_loss, ['profit_after_tax', 'profit_after_tax'])
    
    # 5. profit_after_tax = profit_before_tax - total_tax_expense
    expected_profit_after_tax = profit_before_tax - total_tax_expense
    if abs(expected_profit_after_tax - profit_after_tax) > 0.01:
        validation_failures.append(
            ("profit_after_tax = profit_before_tax - total_tax_expense",
             expected_profit_after_tax,
             profit_after_tax,
             expected_profit_after_tax - profit_after_tax)
        )

    
    return validation_failures

def validate_segment_information(segment_info: Dict) -> List[Tuple[str, float, float, float]]:
    """
    Validates segment information for internal consistency.
    
    Args:
        segment_info: Dictionary containing segment information data
        
    Returns:
        List of tuples containing (validation_rule, expected_value, actual_value, difference)
    """
    validation_failures = []
    
    if not segment_info:
        return []  # Nothing to validate
        
    # Safe access helper function
    def safe_get(obj, keys, default=0.0):
        """Safely get a nested value from a dictionary with a default if not found"""
        if not obj or not isinstance(obj, dict):
            return default
            
        current = obj
        for key in keys:
            if isinstance(current, dict) and key in current and current[key] is not None:
                current = current[key]
            else:
                return default
        return current or default
    
    # 1. Check if total_revenue_from_operations is approximately equal to 
    # the sum of all values in segment_revenues
    try:
        if 'segment_revenues' in segment_info and segment_info['segment_revenues']:
            segment_revenue_sum = sum(segment_info['segment_revenues'].values())
            total_revenue = safe_get(segment_info, ['total_revenue_from_operations'])
            
            if abs(segment_revenue_sum - total_revenue) > 0.01:  # Allow for small rounding differences
                validation_failures.append(
                    ("Sum of segment_revenues equals total_revenue_from_operations",
                     segment_revenue_sum,
                     total_revenue,
                     segment_revenue_sum - total_revenue)
                )
    except Exception as e:
        # Append a tuple with error information
        validation_failures.append(
            (f"Error validating segment revenues: {str(e)}",
             0.0,  # placeholder for expected
             0.0,  # placeholder for actual
             0.0)  # placeholder for difference
        )
    
    # 2. Check if total_segment_profit_before_interest_tax matches sum of segment results
    try:
        if 'segment_results' in segment_info and segment_info['segment_results']:
            results_sum = sum(segment_info['segment_results'].values())
            total_results = safe_get(segment_info, ['total_segment_profit_before_interest_tax'])
            
            if abs(results_sum - total_results) > 0.01:  # Allow for small rounding differences
                validation_failures.append(
                    ("Sum of segment_results equals total_segment_profit_before_interest_tax",
                     results_sum,
                     total_results,
                     results_sum - total_results)
                )
    except Exception as e:
        validation_failures.append(
            (f"Error validating segment results: {str(e)}",
             0.0,  # placeholder for expected
             0.0,  # placeholder for actual
             0.0)  # placeholder for difference
        )
    
    # 3. Check if profit_before_tax is consistent with segment results and unallocated expenses
    try:
        # PBT should be total segment profit minus unallocated expenses (finance costs + corporate expenses)
        total_segment_profit = safe_get(segment_info, ['total_segment_profit_before_interest_tax'])
        finance_costs = safe_get(segment_info, ['finance_costs'])
        corporate_expenses = safe_get(segment_info, ['unallocable_corporate_expenses'])
        
        expected_pbt = total_segment_profit - (finance_costs + corporate_expenses)
        actual_pbt = safe_get(segment_info, ['profit_before_tax'])
        
        if abs(expected_pbt - actual_pbt) > 0.01:  # Allow for small rounding differences
            validation_failures.append(
                ("profit_before_tax equals total_segment_profit minus finance_costs and unallocable_corporate_expenses",
                 expected_pbt,
                 actual_pbt,
                 expected_pbt - actual_pbt)
            )
    except Exception as e:
        validation_failures.append(
            (f"Error validating profit before tax: {str(e)}",
             0.0,  # placeholder for expected
             0.0,  # placeholder for actual
             0.0)  # placeholder for difference
        )
    
    # 4. Check if total_assets equals the sum of segment_assets plus unallocated_assets
    try:
        if 'segment_assets' in segment_info and segment_info['segment_assets']:
            segment_assets_sum = sum(segment_info['segment_assets'].values())
            unallocated_assets = safe_get(segment_info, ['unallocated_assets'])
            total_assets = safe_get(segment_info, ['total_assets'])
            
            expected_total_assets = segment_assets_sum + unallocated_assets
            if abs(expected_total_assets - total_assets) > 0.01:  # Allow for small rounding differences
                validation_failures.append(
                    ("total_assets equals sum of segment_assets plus unallocated_assets",
                     expected_total_assets,
                     total_assets,
                     expected_total_assets - total_assets)
                )
    except Exception as e:
        validation_failures.append(
            (f"Error validating segment assets: {str(e)}",
             0.0,  # placeholder for expected
             0.0,  # placeholder for actual
             0.0)  # placeholder for difference
        )
    
    # 5. Check if total_liabilities equals the sum of segment_liabilities plus unallocated_liabilities
    try:
        if 'segment_liabilities' in segment_info and segment_info['segment_liabilities']:
            segment_liabilities_sum = sum(segment_info['segment_liabilities'].values())
            unallocated_liabilities = safe_get(segment_info, ['unallocated_liabilities'])
            total_liabilities = safe_get(segment_info, ['total_liabilities'])
            
            expected_total_liabilities = segment_liabilities_sum + unallocated_liabilities
            if abs(expected_total_liabilities - total_liabilities) > 0.01:  # Allow for small rounding differences
                validation_failures.append(
                    ("total_liabilities equals sum of segment_liabilities plus unallocated_liabilities",
                     expected_total_liabilities,
                     total_liabilities,
                     expected_total_liabilities - total_liabilities)
                )
    except Exception as e:
        validation_failures.append(
            (f"Error validating segment liabilities: {str(e)}",
             0.0,  # placeholder for expected
             0.0,  # placeholder for actual
             0.0)  # placeholder for difference
        )
    
    return validation_failures

def validate_financial_data(financial_data: Dict, statement_types: Optional[List[str]] = None) -> Dict[str, Dict[str, List[Tuple[str, float, float, float]]]]:
    """
    Validates all financial statements in the combined financial data object.
    
    Args:
        financial_data: A dictionary containing financial data (or FinancialData object)
        statement_types: Optional list of statement types to validate. If None, validates all statements.
                         Possible values: ['standalone_profit_and_loss', 'consolidated_profit_and_loss',
                                          'standalone_segment_information', 'consolidated_segment_information',
                                          'standalone_balance_sheet', 'consolidated_balance_sheet',
                                          'standalone_cashflow', 'consolidated_cashflow']
        
    Returns:
        Dictionary mapping period ids to dictionaries of validation failures by statement type
    """
    validation_results = {}
    
    # Get the root data if passing a FinancialData object
    data_to_validate = financial_data.get('root', financial_data)
    
    for period_id, quarter_data in data_to_validate.items():
        period_validations = {}
        
        # Validate standalone profit and loss if it exists and if in statement_types (or if statement_types is None)
        if 'standalone_profit_and_loss' in quarter_data and quarter_data['standalone_profit_and_loss'] and (statement_types is None or 'standalone_profit_and_loss' in statement_types):
            standalone_validations = validate_profit_and_loss(quarter_data['standalone_profit_and_loss'])
            if standalone_validations:
                period_validations["standalone_profit_and_loss"] = standalone_validations
        
        # Validate consolidated profit and loss if it exists and if in statement_types (or if statement_types is None)
        if 'consolidated_profit_and_loss' in quarter_data and quarter_data['consolidated_profit_and_loss'] and (statement_types is None or 'consolidated_profit_and_loss' in statement_types):
            consolidated_validations = validate_profit_and_loss(quarter_data['consolidated_profit_and_loss'])
            if consolidated_validations:
                period_validations["consolidated_profit_and_loss"] = consolidated_validations
        
        # Validate standalone segment information if it exists and if in statement_types (or if statement_types is None)
        if 'standalone_segment_information' in quarter_data and quarter_data['standalone_segment_information'] and (statement_types is None or 'standalone_segment_information' in statement_types):
            standalone_segment_validations = validate_segment_information(quarter_data['standalone_segment_information'])
            if standalone_segment_validations:
                period_validations["standalone_segment_information"] = standalone_segment_validations
        
        # Validate consolidated segment information if it exists and if in statement_types (or if statement_types is None)
        if 'consolidated_segment_information' in quarter_data and quarter_data['consolidated_segment_information'] and (statement_types is None or 'consolidated_segment_information' in statement_types):
            consolidated_segment_validations = validate_segment_information(quarter_data['consolidated_segment_information'])
            if consolidated_segment_validations:
                period_validations["consolidated_segment_information"] = consolidated_segment_validations
        
        # Add period validations to results if there were any failures
        if period_validations:
            validation_results[period_id] = period_validations
    
    return validation_results

def format_validation_results(validation_results: Dict[str, Dict[str, List[Tuple[str, float, float, float]]]]) -> Dict:
    """
    Formats validation results into a structured dictionary suitable for JSON.
    
    Args:
        validation_results: Dictionary of validation failures by period and statement type
        
    Returns:
        Structured dictionary with validation results
    """
    if not validation_results:
        return {"status": "success", "message": "All financial statements passed validation", "periods": []}
    
    formatted_results = {
        "status": "warning",
        "message": "Financial statement validation failures detected",
        "periods": []
    }
    
    for period_id, period_validations in validation_results.items():
        period_data = {
            "period": period_id,
            "statements": []
        }
        
        for statement_type, validations in period_validations.items():
            statement_data = {
                "statement_type": statement_type,
                "validations": []
            }
            
            for validation in validations:
                # Handle both tuple format from validation functions
                if isinstance(validation, tuple) and len(validation) == 4:
                    rule, expected, actual, difference = validation
                    validation_data = {
                        "rule": rule,
                        "expected": round(expected, 2),
                        "actual": round(actual, 2),
                        "difference": round(difference, 2),
                        "percentage_difference": round((difference / expected * 100 if expected != 0 else 0), 2)
                    }
                    statement_data["validations"].append(validation_data)
                else:
                    # Handle any other format (should not happen with updated functions)
                    statement_data["validations"].append({"error": str(validation)})
            
            period_data["statements"].append(statement_data)
        
        formatted_results["periods"].append(period_data)
    
    return formatted_results