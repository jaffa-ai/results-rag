from pydantic import BaseModel, Field, RootModel
from typing import Dict, Optional, List, Any
from langchain.output_parsers import PydanticOutputParser
from financial_statements_taxonomy import ProfitAndLoss, CashFlow, BalanceSheet, SegmentInformation

# Consolidated and Standalone Financial Statements

class ConsolidatedProfitAndLoss(ProfitAndLoss):
    pass


class StandaloneProfitAndLoss(ProfitAndLoss):
    pass


class ConsolidatedSegmentInformation(SegmentInformation):
    pass

class StandaloneSegmentInformation(SegmentInformation):
    pass

class ConsolidatedCashFlow(CashFlow):
    pass


class StandaloneCashFlow(CashFlow):
    pass


class ConsolidatedBalanceSheet(BalanceSheet):
    pass


class StandaloneBalanceSheet(BalanceSheet):
    pass


# Quarter Data Model

class QuarterData(BaseModel):
    company: str
    period: str
    periodStart: str
    periodEnd: str
    currency: str
    unit: str

    consolidated_profit_and_loss: Optional[ConsolidatedProfitAndLoss] = None
    standalone_profit_and_loss: Optional[StandaloneProfitAndLoss] = None

    consolidated_segment_information: Optional[ConsolidatedSegmentInformation] = None
    standalone_segment_information: Optional[StandaloneSegmentInformation] = None

    consolidated_cashflow: Optional[ConsolidatedCashFlow] = None
    standalone_cashflow: Optional[StandaloneCashFlow] = None

    consolidated_balance_sheet: Optional[ConsolidatedBalanceSheet] = None
    standalone_balance_sheet: Optional[StandaloneBalanceSheet] = None


class FinancialData(RootModel[Dict[str, QuarterData]]):
    pass

class EnhancedPydanticOutputParser(PydanticOutputParser):
    def get_format_instructions(self) -> str:
        base_instructions = super().get_format_instructions()
        
        # Add common terms to format instructions
        common_terms_info = "\n\nField descriptions and common terms:\n"
        for field_name, field in self.pydantic_object.model_fields.items():
            field_info = f"- {field_name}: "
            
            # Add description if available
            if field.description:
                field_info += f"{field.description}"
            
            # Add common terms if available
            if field.json_schema_extra and "common_terms" in field.json_schema_extra:
                terms = ", ".join(field.json_schema_extra["common_terms"])
                field_info += f" (Common terms: {terms})"
            
            common_terms_info += field_info + "\n"
        
        return base_instructions + common_terms_info