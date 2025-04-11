from pydantic import BaseModel, Field
from typing import List, Any, Optional, Literal, ClassVar, Dict
from enum import Enum, auto

# Define categories with their weights
class ProfitLossCategory(str, Enum):
    REVENUE_FROM_OPERATIONS = "REVENUE_FROM_OPERATIONS"  # +1
    INCOME = "INCOME"  # +1
    COST_OF_GOODS_SOLD = "COST_OF_GOODS_SOLD"  # -1
    EXPENSES = "EXPENSES"
    SHARE_IN_PROFIT_LOSS_ASSOCIATES = "SHARE_IN_PROFIT_LOSS_ASSOCIATES"  # +1/-1
    PROFIT_BEFORE_TAX = "PROFIT_BEFORE_TAX"  # +1
    TAX_EXPENSE = "TAX_EXPENSE"  # -1
    PROFIT_AFTER_TAX = "PROFIT_AFTER_TAX"  # +1

# Category weights dictionary
CATEGORY_WEIGHTS = {
    ProfitLossCategory.REVENUE_FROM_OPERATIONS: 1,
    ProfitLossCategory.INCOME: 1,
    ProfitLossCategory.COST_OF_GOODS_SOLD: -1,
    ProfitLossCategory.EXPENSES: -1,
    ProfitLossCategory.PROFIT_BEFORE_TAX: 1,
    ProfitLossCategory.TAX_EXPENSE: -1,
    ProfitLossCategory.PROFIT_AFTER_TAX: 1,
}

def taxonomy_field(
    default: Any = ...,
    description: str = None,
    alias: str = None,
    common_terms: List[str] = None,
    category: ProfitLossCategory = None,
    **kwargs
) -> Field:
    """Create a Field with taxonomy metadata including common terms and category."""
    extra = kwargs.pop("json_schema_extra", {})
    if common_terms:
        extra["common_terms"] = common_terms
    if category:
        extra["category"] = category.value
        extra["weight"] = CATEGORY_WEIGHTS[category]
    
    return Field(
        default,
        description=description,
        alias=alias,
        json_schema_extra=extra,
        **kwargs
    )

# New model for other metrics that don't match predefined fields
class OtherMetric(BaseModel):
    name: str = Field(description="The name of the metric as it appears in the financial statement presented in lowercase with underscore as separators, e.g. 'revenue_from_operations'")
    description: str = Field(description="A description of the metric as it appears in the financial statement")
    value: Optional[float] = Field(None, description="The value of the metric as it appears in the financial statement")
    
# Define category models with layout information
class Revenue(BaseModel):
    layout_info: ClassVar[str] = """
    - Indicative layout information to understand portion of financial statement to extract from:
        -- terms that appear before this section (and NOT to be included in the extracted information): None
        -- terms that appear after this section (and NOT to be included in the extracted information): Expenses
    - Sales from products and services is usually the largest line item, or is reported as Revenue from Operations
    - Total income is often a subtotal that combines Revenue from Operations and Other Income, and marks the end of this category
    """
    sales_from_products_services: float = taxonomy_field(
        ..., 
        description="Revenue from products and services", 
        alias="RevenueFromProductsAndServices",
        common_terms=["Revenue from products and services", "Operating revenue from products and services", "Revenue from Operations"],
        category=ProfitLossCategory.REVENUE_FROM_OPERATIONS
    )
    other_operating_revenue: Optional[float] = taxonomy_field(
        None, 
        description="Other operating revenue", 
        alias="OtherOperatingRevenue",
        common_terms=["Other operating revenue"],
        category=ProfitLossCategory.REVENUE_FROM_OPERATIONS
    )
    other_income: float = taxonomy_field(
        ..., 
        description="Other income", 
        alias="OtherIncome",
        common_terms=["Other income"],
        category=ProfitLossCategory.INCOME
    )
    total_income: float = taxonomy_field(
        ..., 
        description="Total income", 
        alias="Income",
        common_terms=["Total income", "Total Income"],
        category=ProfitLossCategory.INCOME
    )
    # Add dictionary for other metrics in this category
    other_metrics: Dict[str, OtherMetric] = Field(default_factory=dict)

class Expenses(BaseModel):
    layout_info: ClassVar[str] = """
    - Indicative layout information to understand portion of financial statement to extract from:
        -- terms that appear before this section (and NOT to be included in the extracted information): Total Income
        -- terms that appear after this section (and NOT to be included in the extracted information): Profit Before Tax
    - These are usually the expense items in the P&L statement. 
    - Total expenses is often a subtotal that combines all expense items, and marks the end of this category.
    """
    
    cost_of_materials_consumed: Optional[float] = taxonomy_field(
        None, 
        description="Cost of materials consumed; Cost of equipment/license/software", 
        alias="CostOfMaterialsConsumed",
        common_terms=["Cost of materials consumed"],
        category=ProfitLossCategory.COST_OF_GOODS_SOLD
    )
    purchases_of_stock_in_trade: Optional[float] = taxonomy_field(
        None, 
        description="Purchases of stock-in-trade", 
        alias="PurchasesOfStockInTrade",
        common_terms=["Purchases of stock-in-trade"],
        category=ProfitLossCategory.COST_OF_GOODS_SOLD
    )
    changes_in_inventories: Optional[float] = taxonomy_field(
        None, 
        description="Changes in inventories of finished goods, work-in-progress and stock-in-trade", 
        alias="ChangesInInventoriesOfFinishedGoodsWorkInProgressAndStockInTrade",
        common_terms=["Changes in inventories of finished goods, work-in-progress and stock-in-trade"],
        category=ProfitLossCategory.COST_OF_GOODS_SOLD
    )
    employee_benefit_expense: float = taxonomy_field(
        ..., 
        description="Employee benefit expense", 
        alias="EmployeeBenefitExpense",
        common_terms=["Employee benefit expense", "Employee Benefits Expense"],
        category=ProfitLossCategory.EXPENSES
    )
    finance_costs: float = taxonomy_field(
        ..., 
        description="Finance costs", 
        alias="FinanceCosts",
        common_terms=["Finance costs"],
        category=ProfitLossCategory.EXPENSES
    )
    depreciation_and_amortisation: float = taxonomy_field(
        ..., 
        description="Depreciation, depletion and amortisation expense", 
        alias="DepreciationDepletionAndAmortisationExpense",
        common_terms=["Depreciation, depletion and amortisation expense", "Depreciation and Amortisation Expense"],
        category=ProfitLossCategory.EXPENSES
    )
    other_expenses: float = taxonomy_field(
        ..., 
        description="Other expenses", 
        alias="OtherExpenses",
        common_terms=["Other expenses"],
        category=ProfitLossCategory.EXPENSES
    )
    exceptional_extraordinary_items: Optional[float] = taxonomy_field(
        None, 
        description="Exceptional and extraordinary items", 
        alias="ExceptionalExtraordinaryItems",
        common_terms=["Exceptional and extraordinary items"],
        category=ProfitLossCategory.EXPENSES
    )
    total_expenses: float = taxonomy_field(
        ..., 
        description="Total Expenses", 
        alias="Expenses",
        common_terms=["Total Expenses"],
        category=ProfitLossCategory.EXPENSES
    )
    # Add dictionary for other metrics in this category
    other_metrics: Dict[str, OtherMetric] = Field(default_factory=dict)

class ProfitBeforeTax(BaseModel):
    layout_info: ClassVar[str] = """
    - Indicative layout information to understand portion of financial statement to extract from:
        -- terms that appear before this section (and NOT to be included in the extracted information): Total Expenses
        -- terms that appear after this section (and NOT to be included in the extracted information): Tax Expense
    - Profit before tax is often a subtotal that combines all profit before tax items, and marks the end of this category.
    - Profit before tax is a key metric that analysts focus on, so it's prominently displayed.
    """
    
    profit_before_share_in_associates: Optional[float] = taxonomy_field(
        None,
        description="Profit before share in profit/(loss) of associates/joint ventures and tax",
        alias="ProfitBeforeShareInAssociates",
        common_terms=["Profit before share in profit/(loss) of associates/joint ventures and tax"],
        category=ProfitLossCategory.PROFIT_BEFORE_TAX
    )
    share_in_profit_loss_associates_joint_ventures: Optional[float] = taxonomy_field(
        None, 
        description="Share in profit (loss) of associates and joint ventures", 
        alias="ShareInProfitLossOfAssociatesAndJointVentures",
        common_terms=["Share in profit (loss) of associates and joint ventures", "Share in Profit / (Loss) of Associates / Joint Venture"],
        category=ProfitLossCategory.PROFIT_BEFORE_TAX
    )
    profit_before_exceptional_extraordinary_items_and_tax: Optional[float] = taxonomy_field(
        None, 
        description="Profit before exceptional items and tax", 
        alias="ProfitBeforeExceptionalItemsAndTax",
        common_terms=["Profit before exceptional items and tax"],
        category=ProfitLossCategory.PROFIT_BEFORE_TAX
    )
    profit_before_tax: float = taxonomy_field(
        ..., 
        description="Total profit before tax", 
        alias="ProfitBeforeTax",
        common_terms=["Total profit before tax", "PBT", "EBT", "Profit before Tax"],
        category=ProfitLossCategory.PROFIT_BEFORE_TAX
    )
    # Add dictionary for other metrics in this category
    other_metrics: Dict[str, OtherMetric] = Field(default_factory=dict)

class TaxExpense(BaseModel):
    layout_info: ClassVar[str] = """
    - Indicative layout information to understand portion of financial statement to extract from:
        -- terms that appear before this section (and NOT to be included in the extracted information): Profit Before Tax
        -- terms that appear after this section (and NOT to be included in the extracted information): Profit After Tax
    - Usually has 2-3 line items breaking down different types of taxes
    - Current tax is usually the largest component.
    - Total tax is a subtotal that combines all tax items, and marks the end of this category.
    """
    
    current_tax: float = taxonomy_field(
        ..., 
        description="Current tax", 
        alias="CurrentTax",
        common_terms=["Current tax"],
        category=ProfitLossCategory.TAX_EXPENSE
    )
    deferred_tax: float = taxonomy_field(
        0.0, 
        description="Deferred tax", 
        alias="DeferredTax",
        common_terms=["Deferred tax"],
        category=ProfitLossCategory.TAX_EXPENSE
    )
    total_tax_expense: Optional[float] = taxonomy_field(
        None, 
        description="Total tax expense", 
        alias="TaxExpense",
        common_terms=["Total tax expense", "Total Tax Expense"],
        category=ProfitLossCategory.TAX_EXPENSE
    )
    # Add dictionary for other metrics in this category
    other_metrics: Dict[str, OtherMetric] = Field(default_factory=dict)

class ProfitAfterTax(BaseModel):
    layout_info: ClassVar[str] = """
    - Indicative layout information to understand portion of financial statement to extract from:
        -- terms that appear before this section (and NOT to be included in the extracted information): Profit Before Tax, Tax Expense
        -- terms that appear after this section (and NOT to be included in the extracted information): Other Comprehensive Income
    - This is usually the final figure in the main income statement before any other comprehensive income.
    - This is one of the most important metrics and is prominently displayed.
    - This figure is calculated as Profit Before Tax minus Total Tax Expense.
    """
    
    profit_after_tax: float = taxonomy_field(
        ..., 
        description="Net profit (loss) for period from continuing operations", 
        alias="ProfitLossForPeriodFromContinuingOperations",
        common_terms=["Net profit (loss) for period from continuing operations", "PAT", "Profit after tax", "Profit for the period"],
        category=ProfitLossCategory.PROFIT_AFTER_TAX
    )
    # Add dictionary for other metrics in this category
    other_metrics: Dict[str, OtherMetric] = Field(default_factory=dict)

# Main ProfitAndLoss class with mandatory category classes
class ProfitAndLoss(BaseModel):
    revenue: Revenue
    expenses: Expenses
    profit_before_tax: ProfitBeforeTax
    tax_expense: TaxExpense
    profit_after_tax: ProfitAfterTax



# +++++++++++++++++++SEGMENT STARTS++++++++++++++++++++++++++++++


class SegmentInformation(BaseModel):
    # Segment Revenue
    segment_revenues: dict[str, float] = taxonomy_field(
        ...,
        description="Revenue from each business segment",
        alias="SegmentRevenues",
        common_terms=["Segment Revenue", "Revenue by segment"]
    )
    total_revenue_from_operations: float = taxonomy_field(
        ...,
        description="Total revenue from operations across all segments",
        alias="RevenueFromOperations",
        common_terms=["Revenue from operations", "Total segment revenue", "Net revenue"]
    )
    
    # Segment Results
    segment_results: Optional[dict[str, float]] = taxonomy_field(
        ...,
        description="Results (profit/loss) at EBIT level from each business segment (Earnings Before Interest and Tax)",
        alias="SegmentResults",
        common_terms=["Segment Results", "Segment profit", "Results by segment", "EBIT by segment"]
    )
    total_segment_profit_before_interest_tax: Optional[float] = taxonomy_field(
        ...,
        description="Total Segment EBIT (Earnings Before Interest and Tax) and unallocable expenses/income",
        alias="TotalSegmentProfitBeforeInterestTax",
        common_terms=["Total Segment Profit before Interest and Tax", "Segment profit before interest and tax", "Total segment EBIT"]
    )
    finance_costs: Optional[float] = taxonomy_field(
        ...,
        description="Finance costs",
        alias="FinanceCosts",
        common_terms=["Finance Costs", "Interest expenses"]
    )
    unallocable_corporate_expenses: Optional[float] = taxonomy_field(
        ...,
        description="Unallocable Corporate Expenses (Net of unallocable income)",
        alias="UnallocableCorporateExpenses",
        common_terms=["Unallocable Corporate Expenses", "Corporate expenses"]
    )
    profit_before_tax: Optional[float]  = taxonomy_field(
        ...,
        description="Profit before Tax",
        alias="ProfitBeforeTax",
        common_terms=["Profit before Tax", "PBT"]
    )
    
    # Segment Assets
    segment_assets: Optional[dict[str, float]] = taxonomy_field(
        ...,
        description="Assets allocated to each business segment",
        alias="SegmentAssets",
        common_terms=["Segment Assets", "Assets by segment"]
    )
    unallocated_assets: Optional[float] = taxonomy_field(
        0.0,
        description="Assets not allocated to any specific segment",
        alias="UnallocatedAssets",
        common_terms=["Unallocated Assets", "Corporate assets"]
    )
    total_assets: Optional[float] = taxonomy_field(
        ...,
        description="Total assets across all segments",
        alias="TotalAssets",
        common_terms=["TOTAL ASSETS", "Total segment assets"]
    )
    
    # Segment Liabilities
    segment_liabilities: Optional[dict[str, float]] = taxonomy_field(
        ...,
        description="Liabilities allocated to each business segment",
        alias="SegmentLiabilities",
        common_terms=["Segment Liabilities", "Liabilities by segment"]
    )
    unallocated_liabilities: Optional[float] = taxonomy_field(
        0.0,
        description="Liabilities not allocated to any specific segment",
        alias="UnallocatedLiabilities",
        common_terms=["Unallocated Liabilities", "Corporate liabilities"]
    )
    total_liabilities: Optional[float] = taxonomy_field(
        ...,
        description="Total liabilities across all segments",
        alias="TotalLiabilities",
        common_terms=["TOTAL LIABILITIES", "Total segment liabilities"]
    )
    
    # Other metrics not explicitly defined in the model
    other_metrics: Dict[str, OtherMetric] = taxonomy_field(
        default_factory=dict,
        description="Additional segment metrics not captured by other fields",
        alias="OtherMetrics",
        common_terms=["Other metrics", "Additional metrics"]
    )

# ======================================BALANCE SHEET=====================================
class BalanceSheet(BaseModel):
    # Non-current Assets
    property_plant_equipment: float = taxonomy_field(
        ...,
        description="Property, plant and equipment",
        alias="PropertyPlantAndEquipment",
        common_terms=["Property, plant and equipment"]
    )
    capital_work_in_progress: float = taxonomy_field(
        0.0,
        description="Capital work-in-progress",
        alias="CapitalWorkInProgress",
        common_terms=["Capital work-in-progress"]
    )
    investment_property: float = taxonomy_field(
        0.0,
        description="Investment property",
        alias="InvestmentProperty",
        common_terms=["Investment property"]
    )
    goodwill: float = taxonomy_field(
        0.0,
        description="Goodwill",
        alias="Goodwill",
        common_terms=["Goodwill"]
    )
    other_intangible_assets: float = taxonomy_field(
        0.0,
        description="Other intangible assets",
        alias="OtherIntangibleAssets",
        common_terms=["Other intangible assets"]
    )
    intangible_assets_under_development: float = taxonomy_field(
        0.0,
        description="Intangible assets under development",
        alias="IntangibleAssetsUnderDevelopment",
        common_terms=["Intangible assets under development"]
    )
    non_current_investments: float = taxonomy_field(
        0.0,
        description="Non-current investments",
        alias="NoncurrentInvestments",
        common_terms=["Non-current investments"]
    )
    deferred_tax_assets: float = taxonomy_field(
        0.0,
        description="Deferred tax assets (net)",
        alias="DeferredTaxAssetsNet",
        common_terms=["Deferred tax assets (net)"]
    )
    other_non_current_assets: float = taxonomy_field(
        0.0,
        description="Other non-current assets",
        alias="OtherNoncurrentAssets",
        common_terms=["Other non-current assets"]
    )
    total_non_current_assets: float = taxonomy_field(
        ...,
        description="Total non-current assets",
        alias="NoncurrentAssets",
        common_terms=["Total non-current assets"]
    )
    
    # Current Assets
    inventories: float = taxonomy_field(
        ...,
        description="Inventories",
        alias="Inventories",
        common_terms=["Inventories"]
    )
    current_investments: float = taxonomy_field(
        0.0,
        description="Current investments",
        alias="CurrentInvestments",
        common_terms=["Current investments"]
    )
    trade_receivables: float = taxonomy_field(
        ...,
        description="Trade receivables, current",
        alias="TradeReceivablesCurrent",
        common_terms=["Trade receivables, current"]
    )
    cash_and_cash_equivalents: float = taxonomy_field(
        ...,
        description="Cash and cash equivalents",
        alias="CashAndCashEquivalents",
        common_terms=["Cash and cash equivalents"]
    )
    bank_balance_other_than_cash: float = taxonomy_field(
        0.0,
        description="Bank balance other than cash and cash equivalents",
        alias="BankBalanceOtherThanCashAndCashEquivalents",
        common_terms=["Bank balance other than cash and cash equivalents"]
    )
    current_loans: float = taxonomy_field(
        0.0,
        description="Loans, current",
        alias="LoansCurrent",
        common_terms=["Loans, current"]
    )
    other_current_financial_assets: float = taxonomy_field(
        0.0,
        description="Other current financial assets",
        alias="OtherCurrentFinancialAssets",
        common_terms=["Other current financial assets"]
    )
    other_current_assets: float = taxonomy_field(
        0.0,
        description="Other current assets",
        alias="OtherCurrentAssets",
        common_terms=["Other current assets"]
    )
    total_current_assets: float = taxonomy_field(
        ...,
        description="Total current assets",
        alias="CurrentAssets",
        common_terms=["Total current assets"]
    )
    
    # Total Assets
    total_assets: float = taxonomy_field(
        ...,
        description="Total assets",
        alias="Assets",
        common_terms=["Total assets"]
    )
    
    # Equity
    equity_share_capital: float = taxonomy_field(
        ...,
        description="Equity share capital",
        alias="EquityShareCapital",
        common_terms=["Equity share capital"]
    )
    other_equity: float = taxonomy_field(
        ...,
        description="Other equity",
        alias="OtherEquity",
        common_terms=["Other equity", "Reserves and surplus"]
    )
    total_equity: float = taxonomy_field(
        ...,
        description="Total equity",
        alias="Equity",
        common_terms=["Total equity"]
    )
    
    # Non-current Liabilities
    non_current_borrowings: float = taxonomy_field(
        0.0,
        description="Borrowings, non-current",
        alias="BorrowingsNoncurrent",
        common_terms=["Borrowings, non-current", "Long term borrowings"]
    )
    non_current_provisions: float = taxonomy_field(
        0.0,
        description="Provisions, non-current",
        alias="ProvisionsNoncurrent",
        common_terms=["Provisions, non-current"]
    )
    deferred_tax_liabilities: float = taxonomy_field(
        0.0,
        description="Deferred tax liabilities (net)",
        alias="DeferredTaxLiabilitiesNet",
        common_terms=["Deferred tax liabilities (net)"]
    )
    other_non_current_liabilities: float = taxonomy_field(
        0.0,
        description="Other non-current liabilities",
        alias="OtherNoncurrentLiabilities",
        common_terms=["Other non-current liabilities"]
    )
    total_non_current_liabilities: float = taxonomy_field(
        ...,
        description="Total non-current liabilities",
        alias="NoncurrentLiabilities",
        common_terms=["Total non-current liabilities"]
    )
    
    # Current Liabilities
    current_borrowings: float = taxonomy_field(
        0.0,
        description="Borrowings, current",
        alias="BorrowingsCurrent",
        common_terms=["Borrowings, current", "Short term borrowings"]
    )
    trade_payables: float = taxonomy_field(
        ...,
        description="Trade payables, current",
        alias="TradePayablesCurrent",
        common_terms=["Trade payables, current"]
    )
    other_current_financial_liabilities: float = taxonomy_field(
        0.0,
        description="Other current financial liabilities",
        alias="OtherCurrentFinancialLiabilities",
        common_terms=["Other current financial liabilities"]
    )
    other_current_liabilities: float = taxonomy_field(
        ...,
        description="Other current liabilities",
        alias="OtherCurrentLiabilities",
        common_terms=["Other current liabilities"]
    )
    current_provisions: float = taxonomy_field(
        0.0,
        description="Provisions, current",
        alias="ProvisionsCurrent",
        common_terms=["Provisions, current"]
    )
    current_tax_liabilities: float = taxonomy_field(
        0.0,
        description="Current tax liabilities",
        alias="CurrentTaxLiabilities",
        common_terms=["Current tax liabilities"]
    )
    total_current_liabilities: float = taxonomy_field(
        ...,
        description="Total current liabilities",
        alias="CurrentLiabilities",
        common_terms=["Total current liabilities"]
    )
    
    # Total Liabilities
    total_liabilities: float = taxonomy_field(
        ...,
        description="Total liabilities",
        alias="Liabilities",
        common_terms=["Total liabilities"]
    )
    
    # Total Equity and Liabilities
    total_equity_and_liabilities: float = taxonomy_field(
        ...,
        description="Total equity and liabilities",
        alias="EquityAndLiabilities",
        common_terms=["Total equity and liabilities"]
    )



# =================================CASH FLOW===================================

class CashFlow(BaseModel):
    # Operating Activities
    profit_before_tax: float = taxonomy_field(
        ...,
        description="Profit before tax",
        alias="ProfitLossBeforeTax",
        common_terms=["Profit before tax", "PBT"]
    )
    adjustments_for_depreciation: float = taxonomy_field(
        ...,
        description="Adjustments for depreciation and amortisation expense",
        alias="AdjustmentsForDepreciationAndAmortisationExpense",
        common_terms=["Adjustments for depreciation and amortisation expense"]
    )
    adjustments_for_finance_costs: float = taxonomy_field(
        0.0,
        description="Adjustments for finance costs",
        alias="AdjustmentsForFinanceCosts",
        common_terms=["Adjustments for finance costs"]
    )
    adjustments_for_interest_income: float = taxonomy_field(
        0.0,
        description="Adjustments for interest income",
        alias="AdjustmentsForInterestIncome",
        common_terms=["Adjustments for interest income"]
    )
    operating_profit_before_working_capital_changes: float = taxonomy_field(
        ...,
        description="Operating profit before working capital changes",
        alias="OperatingProfitBeforeWorkingCapitalChanges",
        common_terms=["Operating profit before working capital changes"]
    )
    adjustments_for_increase_in_trade_receivables: float = taxonomy_field(
        0.0,
        description="Adjustments for increase (decrease) in trade receivables",
        alias="AdjustmentsForIncreaseDecreaseInTradeReceivables",
        common_terms=["Adjustments for increase (decrease) in trade receivables"]
    )
    adjustments_for_increase_in_inventories: float = taxonomy_field(
        0.0,
        description="Adjustments for increase (decrease) in inventories",
        alias="AdjustmentsForIncreaseDecreaseInInventories",
        common_terms=["Adjustments for increase (decrease) in inventories"]
    )
    adjustments_for_increase_in_trade_payables: float = taxonomy_field(
        0.0,
        description="Adjustments for increase (decrease) in trade payables",
        alias="AdjustmentsForIncreaseDecreaseInTradePayables",
        common_terms=["Adjustments for increase (decrease) in trade payables"]
    )
    adjustments_for_other_financial_assets: float = taxonomy_field(
        0.0,
        description="Adjustments for other financial assets",
        alias="AdjustmentsForOtherFinancialAssets",
        common_terms=["Adjustments for other financial assets"]
    )
    adjustments_for_other_financial_liabilities: float = taxonomy_field(
        0.0,
        description="Adjustments for other financial liabilities",
        alias="AdjustmentsForOtherFinancialLiabilities",
        common_terms=["Adjustments for other financial liabilities"]
    )
    adjustments_for_other_assets: float = taxonomy_field(
        0.0,
        description="Adjustments for other assets",
        alias="AdjustmentsForOtherAssets",
        common_terms=["Adjustments for other assets"]
    )
    adjustments_for_other_liabilities: float = taxonomy_field(
        0.0,
        description="Adjustments for other liabilities",
        alias="AdjustmentsForOtherLiabilities",
        common_terms=["Adjustments for other liabilities"]
    )
    cash_generated_from_operations: float = taxonomy_field(
        ...,
        description="Cash generated from operations",
        alias="CashGeneratedFromOperations",
        common_terms=["Cash generated from operations"]
    )
    income_taxes_paid: float = taxonomy_field(
        ...,
        description="Income taxes paid (refund)",
        alias="IncomeTaxesPaidRefund",
        common_terms=["Income taxes paid (refund)", "Direct taxes paid"]
    )
    net_cash_from_operating_activities: float = taxonomy_field(
        ...,
        description="Net cash flows from (used in) operating activities",
        alias="NetCashFlowsFromUsedInOperatingActivities",
        common_terms=["Net cash flows from (used in) operating activities", "Cash from operations"]
    )
    
    # Investing Activities
    purchase_of_property_plant_equipment: float = taxonomy_field(
        0.0,
        description="Purchase of property, plant and equipment",
        alias="PurchaseOfPropertyPlantAndEquipment",
        common_terms=["Purchase of property, plant and equipment", "Capital expenditure"]
    )
    proceeds_from_sales_of_property_plant_equipment: float = taxonomy_field(
        0.0,
        description="Proceeds from sales of property, plant and equipment",
        alias="ProceedsFromSalesOfPropertyPlantAndEquipment",
        common_terms=["Proceeds from sales of property, plant and equipment"]
    )
    purchase_of_investments: float = taxonomy_field(
        0.0,
        description="Purchase of investments",
        alias="PurchaseOfInvestments",
        common_terms=["Purchase of investments"]
    )
    proceeds_from_sales_of_investments: float = taxonomy_field(
        0.0,
        description="Proceeds from sales of investments",
        alias="ProceedsFromSalesOfInvestments",
        common_terms=["Proceeds from sales of investments"]
    )
    interest_received: float = taxonomy_field(
        0.0,
        description="Interest received",
        alias="InterestReceived",
        common_terms=["Interest received"]
    )
    net_cash_from_investing_activities: float = taxonomy_field(
        ...,
        description="Net cash flows from (used in) investing activities",
        alias="NetCashFlowsFromUsedInInvestingActivities",
        common_terms=["Net cash flows from (used in) investing activities", "Cash from investing"]
    )
    
    # Financing Activities
    proceeds_from_borrowings: float = taxonomy_field(
        0.0,
        description="Proceeds from borrowings",
        alias="ProceedsFromBorrowings",
        common_terms=["Proceeds from borrowings"]
    )
    repayments_of_borrowings: float = taxonomy_field(
        0.0,
        description="Repayments of borrowings",
        alias="RepaymentsOfBorrowings",
        common_terms=["Repayments of borrowings"]
    )
    dividends_paid: float = taxonomy_field(
        0.0,
        description="Dividends paid",
        alias="DividendsPaid",
        common_terms=["Dividends paid"]
    )
    interest_paid: float = taxonomy_field(
        0.0,
        description="Interest paid",
        alias="InterestPaid",
        common_terms=["Interest paid"]
    )
    net_cash_from_financing_activities: float = taxonomy_field(
        ...,
        description="Net cash flows from (used in) financing activities",
        alias="NetCashFlowsFromUsedInFinancingActivities",
        common_terms=["Net cash flows from (used in) financing activities", "Cash from financing"]
    )
    
    # Net Cash Flow
    net_increase_in_cash: float = taxonomy_field(
        ...,
        description="Net increase (decrease) in cash and cash equivalents",
        alias="NetIncreaseDecreaseInCashAndCashEquivalents",
        common_terms=["Net increase (decrease) in cash and cash equivalents", "Net cash flow"]
    )
    cash_at_beginning_of_period: float = taxonomy_field(
        ...,
        description="Cash and cash equivalents cash flow statement at beginning of period",
        alias="CashAndCashEquivalentsCashFlowStatementAtBeginningOfPeriod",
        common_terms=["Cash and cash equivalents at beginning of period", "Opening cash balance"]
    )
    cash_at_end_of_period: float = taxonomy_field(
        ...,
        description="Cash and cash equivalents cash flow statement at end of period",
        alias="CashAndCashEquivalentsCashFlowStatementAtEndOfPeriod",
        common_terms=["Cash and cash equivalents at end of period", "Closing cash balance"]
    )
