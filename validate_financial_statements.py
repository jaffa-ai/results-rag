from typing import Dict, List, Tuple, Optional
from pydantic_model import FinancialData, QuarterData
from financial_statements_taxonomy import ProfitAndLoss, SegmentInformation

def validate_profit_and_loss(profit_loss: ProfitAndLoss) -> List[Tuple[str, float, float, float]]:
    """
    Validates a profit and loss statement according to the rules defined in financial_statement_validations.md.
    
    Args:
        profit_loss: A ProfitAndLoss object to validate
        
    Returns:
        List of tuples containing (validation_rule, expected_value, actual_value, difference)
    """
    validation_failures = []
    

    sales = profit_loss.revenue.sales_from_products_services or 0.0
    other_operating = profit_loss.revenue.other_operating_revenue or 0.0
    other_income = profit_loss.revenue.other_income or 0.0
    total_income = profit_loss.revenue.total_income or 0.0
    
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
    
    current_tax = profit_loss.tax_expense.current_tax or 0.0
    deferred_tax = profit_loss.tax_expense.deferred_tax or 0.0
    total_tax_expense = profit_loss.tax_expense.total_tax_expense

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
    
    profit_before_tax = profit_loss.profit_before_tax.profit_before_tax or 0.0
    profit_after_tax = profit_loss.profit_after_tax.profit_after_tax or 0.0
    
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

def validate_segment_information(segment_info: SegmentInformation) -> List[Tuple[str, float, float, float]]:
    """
    Validates segment information for internal consistency.
    
    Args:
        segment_info: The segment information to validate
        
    Returns:
        List of tuples containing (validation_rule, expected_value, actual_value, difference)
    """
    validation_failures = []
    
    if not segment_info:
        return []  # Nothing to validate
    
    # 1. Check if total_revenue_from_operations is approximately equal to 
    # the sum of all values in segment_revenues
    try:
        if segment_info.segment_revenues:
            segment_revenue_sum = sum(segment_info.segment_revenues.values())
            total_revenue = segment_info.total_revenue_from_operations or 0.0
            
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
        if segment_info.segment_results:
            results_sum = sum(segment_info.segment_results.values())
            total_results = segment_info.total_segment_profit_before_interest_tax or 0.0
            
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
        total_segment_profit = segment_info.total_segment_profit_before_interest_tax or 0.0
        finance_costs = segment_info.finance_costs or 0.0
        corporate_expenses = segment_info.unallocable_corporate_expenses or 0.0
        
        expected_pbt = total_segment_profit - (finance_costs + corporate_expenses)
        actual_pbt = segment_info.profit_before_tax or 0.0
        
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
        if segment_info.segment_assets:
            segment_assets_sum = sum(segment_info.segment_assets.values())
            unallocated_assets = segment_info.unallocated_assets or 0.0
            total_assets = segment_info.total_assets or 0.0
            
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
        if segment_info.segment_liabilities:
            segment_liabilities_sum = sum(segment_info.segment_liabilities.values())
            unallocated_liabilities = segment_info.unallocated_liabilities or 0.0
            total_liabilities = segment_info.total_liabilities or 0.0
            
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

def validate_financial_data(financial_data: FinancialData) -> Dict[str, Dict[str, List[Tuple[str, float, float, float]]]]:
    """
    Validates all financial statements in the combined financial data object.
    
    Args:
        financial_data: A FinancialData object containing multiple financial statements
        
    Returns:
        Dictionary mapping period ids to dictionaries of validation failures by statement type
    """
    validation_results = {}
    
    for period_id, quarter_data in financial_data.root.items():
        period_validations = {}
        
        # Validate standalone profit and loss if it exists
        if quarter_data.standalone_profit_and_loss:
            standalone_validations = validate_profit_and_loss(quarter_data.standalone_profit_and_loss)
            if standalone_validations:
                period_validations["standalone_profit_and_loss"] = standalone_validations
        
        # Validate consolidated profit and loss if it exists
        if quarter_data.consolidated_profit_and_loss:
            consolidated_validations = validate_profit_and_loss(quarter_data.consolidated_profit_and_loss)
            if consolidated_validations:
                period_validations["consolidated_profit_and_loss"] = consolidated_validations
        
        # Validate standalone segment information if it exists
        if quarter_data.standalone_segment_information:
            standalone_segment_validations = validate_segment_information(quarter_data.standalone_segment_information)
            if standalone_segment_validations:
                period_validations["standalone_segment_information"] = standalone_segment_validations
        
        # Validate consolidated segment information if it exists
        if quarter_data.consolidated_segment_information:
            consolidated_segment_validations = validate_segment_information(quarter_data.consolidated_segment_information)
            if consolidated_segment_validations:
                period_validations["consolidated_segment_information"] = consolidated_segment_validations
        
        # Add period validations to results if there were any failures
        if period_validations:
            validation_results[period_id] = period_validations
    
    return validation_results

def format_validation_results(validation_results: Dict[str, Dict[str, List[Tuple[str, float, float, float]]]]) -> str:
    """
    Formats validation results into a human-readable string.
    
    Args:
        validation_results: Dictionary of validation failures by period and statement type
        
    Returns:
        Formatted string with validation results
    """
    if not validation_results:
        return "All financial statements passed validation."
    
    result_lines = ["Financial statement validation failures:"]
    
    for period_id, period_validations in validation_results.items():
        result_lines.append(f"\nPeriod: {period_id}")
        
        for statement_type, validations in period_validations.items():
            result_lines.append(f"  Statement type: {statement_type}")
            
            for validation in validations:
                # Handle both tuple format from validation functions
                if isinstance(validation, tuple) and len(validation) == 4:
                    rule, expected, actual, difference = validation
                    result_lines.append(f"    Rule: {rule}")
                    result_lines.append(f"      Expected: {expected:.2f}")
                    result_lines.append(f"      Actual: {actual:.2f}")
                    result_lines.append(f"      Difference: {difference:.2f}")
                else:
                    # Handle any other format (should not happen with updated functions)
                    result_lines.append(f"    {validation}")
                result_lines.append("")
    
    return "\n".join(result_lines)