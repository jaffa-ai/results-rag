This documents the following:  
- creation of pydantic extraction schemas from Ind AS taxonomy for quarterly filings
- calculation of custom metrics and ratios
- run validations on financial statements

## Taxonomy files

We create our custom taxonomy files from IndAS taxonomy (taxonomy/financial_results_indAS/Ind AS Taxonomy 2020-03-31.xslx)

- **base_taxonomy**: lists items from Ind AS taxonomy that are extracted using LLMs. 

The fields captured are:
1. xml_tag: tag used in Ind AS taxonomy
2. description: natural language explanation of the term
3. common_terms: other common terms and abbreviations used

Note: base taxonomy may not have all terms from IndAS taxonomy, those terms that we prefer to calculate instead would be part of calculated taxonomy.

- **calculated_taxonomy**: lists calculated items from Ind AS taxonomy (these will have xml_tag) and other metrics and ratios by type of financial statement.

In addition to fields in base taxonomy, fields are:  
4. calculation: formula of metric or ratio using other metrics  
5. after: denotes the mteric after which it is to be presented  
6. kind: simple/ratio, used to find type of change (in % or bps)  

## Pyandic models for extraction using base taxonomy

### Profit and loss

| Taxonomy Item | Pydantic Item | Change | Reason for Inclusion/Deletion |
|---------------|---------------|--------|-------------------------------|
| RevenueFromOperations | sales_from_products_services | Renamed | to indicate revenue from product/services |
| NA | other_operating_revenue | Added | to provide more granular classification of revenue not from product/services |
| OtherIncome | other_income | | |
| Income | total_income | | |
| CostOfMaterialsConsumed | cost_of_materials_consumed | | |
| PurchasesOfStockInTrade | purchases_of_stock_in_trade | | |
| ChangesInInventoriesOfFinishedGoodsWorkInProgressAndStockInTrade | changes_in_inventories | | |
| EmployeeBenefitExpense | employee_benefit_expense | | |
| FinanceCosts | finance_costs | | |
| DepreciationDepletionAndAmortisationExpense | depreciation_and_amortisation | | |
| OtherExpenses | other_expenses | Removed | moved to calculated_taxonomy.json as a derived field, so as to avoid any mismatch if there are other expenses that we don't capture in pydantic model |
| Expenses | total_expenses | | |
| ProfitBeforeExceptionalItemsAndTax | profit_before_exceptional_extraordinary_items_and_tax | Removed | moved to calculated_taxonomy.json |
| ExceptionalItemsBeforeTax | exceptional_extraordinary_items | Removed | moved to calculated_taxonomy.json |
| ProfitBeforeTax | profit_before_tax | | |
| CurrentTax | current_tax | | |
| DeferredTax | deferred_tax | | |
| TaxExpense | total_tax_expense | | |
| ProfitLossForPeriodFromContinuingOperations | profit_after_tax | | |

## Base calculated items 

Some items from Ind AS are re-calculated, these are given in calculated taxonomy and have associated xml_tags.

1. **profit and loss**: other_expenses, profit_before_exceptional_extraordinary_items_and_tax, exceptional_extraordinary_items

## Validations

We run all possible validations to ensure that the extraction is complete and correct.

1. **profit and loss**: 

| Rule | Validation | Status |
|------|------------|--------|
| 1 | total_income = sales_from_products_services + other_operating_revenue + other_income | Active |
| 2 | total_tax_expense = current_tax + deferred_tax | Active |
| 3 | profit_after_tax = profit_before_tax - total_tax_expense | Active |
| 4 | profit_before_exceptional_items_and_tax = total_income - total_expenses | Inactive (Commented) |
| 5 | profit_before_tax = profit_before_exceptional_items_and_tax - exceptional_items | Inactive (Commented) |
