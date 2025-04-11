import os
import json
import asyncio
import random
import time
import re
import logging

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langchain_core.callbacks import AsyncCallbackManagerForChainRun
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from financial_statements_taxonomy import (
    Expenses,
    OtherMetric,
    ProfitAfterTax,
    ProfitBeforeTax,
    Revenue,
    SegmentInformation,
    TaxExpense,
)

from pydantic_model import (
    ConsolidatedProfitAndLoss,
    ConsolidatedSegmentInformation,
    EnhancedPydanticOutputParser,
    FinancialData,
    QuarterData,
    StandaloneProfitAndLoss,
    StandaloneSegmentInformation,
)

from validate_financial_statements import validate_financial_data, format_validation_results


# Get logger from main application
logger = logging.getLogger(__name__)

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_DEPLOYMENT')

def read_output_md(file_path: str) -> str:
    """
    Reads the content of the specified markdown file and returns it as a string.

    Args:
        file_path (str): The path to the markdown file.

    Returns:
        str: The content of the markdown file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

def generate_period_info(quarter: str):
    """
    Dynamically generates period information based on the given quarter.
    
    Args:
        quarter (str): Period in formats Q[y]FY[xx], 9MFY[xx], 6MFY[xx], or FY[xx]
        
    Returns:
        dict: Dictionary mapping period identifiers to readable period descriptions
    """
    period_info = {}

    fy_position = quarter.find("FY")
    fiscal_year = int(quarter[fy_position+2:])
    prev_fiscal_year = fiscal_year - 1
    
    # Handle different period formats
    if quarter.startswith("Q"):

        quarter_num = int(quarter[1:2])

        # Month-end dates for each quarter (assuming April-March fiscal year)
        quarter_end_dates = {
            1: {"month": "06", "day": "30"},
            2: {"month": "09", "day": "30"},
            3: {"month": "12", "day": "31"},
            4: {"month": "03", "day": "31"}
        }

        if quarter_num == 1:
            period_info[quarter] = f"3 (three) months or quarter ended {quarter_end_dates[quarter_num]['day']}-{quarter_end_dates[quarter_num]['month']}-{prev_fiscal_year} ({quarter})"
        elif quarter_num == 2:
            period_info[quarter] = f"3 (three) months or quarter ended {quarter_end_dates[quarter_num]['day']}-{quarter_end_dates[quarter_num]['month']}-{prev_fiscal_year} ({quarter})"
        elif quarter_num == 3:
            period_info[quarter] = f"3 (three) months or quarter ended {quarter_end_dates[quarter_num]['day']}-{quarter_end_dates[quarter_num]['month']}-{prev_fiscal_year} ({quarter})"
        elif quarter_num == 4:
            period_info[quarter] = f"3 (three) months or quarter ended {quarter_end_dates[quarter_num]['day']}-{quarter_end_dates[quarter_num]['month']}-{fiscal_year} ({quarter})"
        
    elif quarter.startswith("9M"):      
        period_info[quarter] = f"9 (nine) months ended 31-12-{prev_fiscal_year} ({quarter})"

    elif quarter.startswith("6M"):
        period_info[quarter] = f"6 (six) months or half year ended 30-09-{prev_fiscal_year} ({quarter})"
        
    elif quarter.startswith("FY"):
        period_info[quarter] = f"12 (twelve) months or FY ended 31-03-{fiscal_year} ({quarter})"
    
    # logger.info(f"Period info: {period_info}")

    return period_info

async def create_extraction_pipeline_async(statement_type, period):
    """
    Create an asynchronous extraction pipeline for a financial statement type.
    
    Args:
        statement_type: The type of financial statement (e.g., StandaloneProfitAndLoss, ConsolidatedSegmentInformation)
        period: The period to extract (e.g., 'Q3FY25')
        
    Returns:
        An async langchain extraction pipeline
    """
    os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_retries=10
    )
    
    # Get the statement class name to use in the template
    statement_name = statement_type.__name__
    
    # Determine if this is standalone or consolidated
    statement_desc = "standalone" if "Standalone" in statement_name else "consolidated"
    
    # Create a Pydantic parser for this statement type
    parser = EnhancedPydanticOutputParser(pydantic_object=statement_type)
    format_instructions = parser.get_format_instructions()
    
    # Generate period info for the template
    period_info = generate_period_info(period)
    
    # Template for extraction
    template = f"""You are a financial analyst assistant that is analyzing financial reports.
    From the following text extract the {statement_desc} financial information for {period_info.get(period, period)}.
    
    Text to analyze:
    {{text}}
    
    Information to extract:
    {format_instructions}
    
    IMPORTANT:
    1. Only extract values that are clearly stated in the text. If the requested information is not available in the text, return the default values from the schema.
    2. Make sure all values match exactly what is stated in the text.
    3. IMPORTANT: When providing numeric values, do NOT use commas as thousand separators. For example, use 140148 instead of 140,148.
    All numbers must be in standard JSON format without commas.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the extraction chain using LCEL with async support
    chain = (
        {"text": RunnablePassthrough()} 
        | prompt 
        | llm 
        | parser
    )
    
    return chain


@retry(
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=2, min=2, max=200),
    retry=retry_if_exception_type((Exception)),
    reraise=True
)
async def invoke_chain_with_retry(chain, content):
    """
    Invokes a chain with retry logic to handle transient errors.
    
    Args:
        chain: The chain to invoke
        content: The content to process
        
    Returns:
        The result of the chain invocation
    """
    try:
        # Format content properly based on whether it's already a dict or just text
        formatted_content = content if isinstance(content, dict) else {"text": content}
        result = await chain.ainvoke(formatted_content)
        return result
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        # Add a small random delay before retrying to avoid rate limits
        await asyncio.sleep(random.uniform(1, 2))
        raise


async def create_category_extraction_pipeline_async(statement_type, category_class, period):
    """
    Creates an asynchronous extraction pipeline for a specific financial statement category.
    
    Args:
        statement_type: Either StandaloneProfitAndLoss or ConsolidatedProfitAndLoss
        category_class: A category class from financial_statements_taxonomy (e.g., Revenue)
        period: The period to extract (e.g., 'Q3FY25')
        
    Returns:
        An async langchain extraction pipeline specific to the category
    """
    os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_retries=10
    )

    # Create a parser for the specified category class
    parser = EnhancedPydanticOutputParser(pydantic_object=category_class)
    
    # Generate dynamic period information
    period_info = generate_period_info(period)
    
    # Get the category name for more context
    category_name = category_class.__name__
    
    # Get layout info for this category
    layout_info = getattr(category_class, 'layout_info', "No specific layout information available")
    
    # Determine if this is standalone or consolidated
    statement_desc = "standalone" if isinstance(statement_type, (StandaloneProfitAndLoss, StandaloneSegmentInformation)) else "consolidated"

    # print(f"Category name: {category_name}")
    # print(f"--- Layout info for {category_name}: \n {layout_info}")
    # print(f"--- Format instructions: \n {parser.get_format_instructions()}")
    
    # Template focused on the specific category and statement type
    template = f"""You are a financial analyst assistant that is required to extract {category_name} information from {statement_desc} financial results.
    From this text extract the {category_name} data for {period_info.get(period, period)} from the {statement_desc} statement ONLY.
    
    Text to analyze:
    <text>
    {{text}}
    </text>
    
    Information to extract for {category_name}:
    <financial_metrics_to_extract>
    {{format_instructions}}
    </financial_metrics_to_extract>

    The layout information associated with {category_name} is as follows:
    <layout_info>
    {{layout_info}}
    </layout_info>

    NOTE: 
    1. Any metrics not mentioned below in <financial_metrics_to_extract>, but present in <text> and 
    determined to be a metric for this category as per logic in <layout_info> should still be extracted 
    and saved into the schema for 'other_metric' fields.
    2. IMPORTANT: For segment information, make sure to capture ALL metrics not explicitly defined in the model fields in the 'other_metrics' dictionary. This includes any segment-specific KPIs, ratios, or other financial metrics mentioned in the text.
    3. Only extract values that are clearly stated in the text. If the requested information is not available in the text, return the default values from the schema.
    4. IMPORTANT: When providing numeric values, do NOT use commas as thousand separators. For example, use 140148 instead of 140,148. 
    All numbers must be in standard JSON format without commas.
    5. IMPORTANT FOR SEGMENT INFORMATION: When extracting segment information, carefully ensure you're extracting segment assets, segment liabilities and segment results values for the same business segments that appear in segment revenues. Segment information may span multiple pages, with part of segment revenues, results, assets and liabilities on seperate pages. Look for the same segment names (e.g., "Oil to Chemicals (O2C)", "Oil and Gas", "Retail" etc.) and ensure you're mapping values correctly across the entire segment information model.
    6. CRITICAL FOR SEGMENT RESULTS: Always interpret segment_results as EBIT (Earnings Before Interest and Tax) level results by default unless explicitly stated otherwise in the text. If the text doesn't specify the level of segment results, assume they are at the EBIT level.
    7. CRITICAL: In your JSON output, use the exact field names provided in the format instructions, including the case. The JSON should use PascalCase field names like "SegmentRevenues", "TotalAssets", etc., NOT snake_case names like "segment_revenues".
    8. AVOID DUPLICATION: Do not duplicate segment information between segment_revenues/segment_result/segment_assets/segment_liabilities and other_metrics. For example: if a segment's revenue is already included in segment_revenues (e.g., "Oil to Chemicals (O2C)": 123704), DO NOT include the same information in other_metrics. 
    9. DUPLICATE CHECK: Any entries in financial statements that start with '- SegmentName' or bullet SegmentName and match values already present in segment_revenues/segment_result/segment_assets/segment_liabilities should NOT be added to other_metrics. For example, if "Oil to Chemicals (O2C)" appears in segment_revenues, do not add an entry like "- Oil to Chemicals (O2C)" with the same value to other_metrics.
    10. FOR SEGMENT VALUE INFORMATION: Values labeled as "Segment Value of Sales and Services" or similar that match segment revenue values should NOT be included in other_metrics. These are duplicates of the segment_revenues data.
    """
    
    prompt = ChatPromptTemplate.from_template(template).partial(
        format_instructions=parser.get_format_instructions(),
        layout_info=layout_info
    )

    # logger.info(f"--- Prompt: \n {prompt}")
    
    # Create the extraction chain using LCEL with async support
    chain = (
        {"text": RunnablePassthrough()} 
        | prompt 
        | llm 
        | parser
    )
    
    return chain

async def extract_financial_statement_by_categories(statement_type, period: str, content=None):
    """
    Extracts a financial statement by extracting each category in parallel and then combining them.
    
    Args:
        statement_type: The financial statement type (StandaloneProfitAndLoss, ConsolidatedProfitAndLoss,
                        StandaloneSegmentInformation, or ConsolidatedSegmentInformation)
        period: The period to extract (e.g., 'Q3FY25')
        content: The pre-filtered text content containing relevant pages
        
    Returns:
        The complete extracted financial statement
    """
    # Get period info for extraction
    period_info = generate_period_info(period)
    
    # Determine statement description for logging
    if statement_type == StandaloneProfitAndLoss:
        statement_desc = "standalone profit and loss"
    elif statement_type == ConsolidatedProfitAndLoss:
        statement_desc = "consolidated profit and loss"
    elif statement_type == StandaloneSegmentInformation:
        statement_desc = "standalone segment information"
    elif statement_type == ConsolidatedSegmentInformation:
        statement_desc = "consolidated segment information"
    else:
        statement_desc = statement_type.__name__
    
    # print(f"Extracting {statement_desc} for {period_info.get(period, period)}")
    
    # Define categories based on statement type
    if statement_type in (StandaloneProfitAndLoss, ConsolidatedProfitAndLoss):
        # For profit and loss statements
        categories = [
            Revenue,
            Expenses, 
            ProfitBeforeTax,
            TaxExpense,
            ProfitAfterTax
        ]
    elif statement_type in (StandaloneSegmentInformation, ConsolidatedSegmentInformation):
        # For segment information - don't use categories since SegmentInformation is a single model
        # The model already contains all required fields
        categories = [SegmentInformation]
    else:
        raise ValueError(f"Unsupported statement type: {statement_type.__name__}")
    
    # Define an async function to extract a single category
    async def extract_category(category_class):
        category_name = category_class.__name__
        try:
            # print(f"  Started extracting {category_name}...")
            # Create the extraction pipeline for this category
            chain = await create_category_extraction_pipeline_async(statement_type, category_class, period)
            
            # Extract the category data
            result = await invoke_chain_with_retry(chain, content)
            
            # print(f"  ✓ Extracted {category_name}")
            # print(f"  ✓ Result: \n {result}")
            return category_name.lower(), result
        except Exception as e:
            print(f"  ✗ Failed to extract {category_name}: {e}")
            # Return default instance on error
            return category_name.lower(), category_class()
    
    # Create extraction tasks for each category
    extraction_tasks = [extract_category(category_class) for category_class in categories]
    
    # Execute all extraction tasks in parallel
    results_list = await asyncio.gather(*extraction_tasks)
    
    # Convert results list to dictionary
    results = dict(results_list)

    # print(f"--- Results: for {period_info.get(period, period)} for {statement_desc}: \n {results}")
    
    # Combine all categories into a single financial statement
    try:
        if statement_type in (StandaloneProfitAndLoss, ConsolidatedProfitAndLoss):
            # Create profit and loss statement
            statement = statement_type(
                revenue=results.get('revenue'),
                expenses=results.get('expenses'),
                profit_before_tax=results.get('profitbeforetax'),
                tax_expense=results.get('taxexpense'),
                profit_after_tax=results.get('profitaftertax')
            )
        elif statement_type in (StandaloneSegmentInformation, ConsolidatedSegmentInformation):
            # For segment information, use the directly extracted model
            if 'segmentinformation' in results:
                segment_info = results.get('segmentinformation')
                
                # Handle both snake_case and PascalCase attribute names
                # First, try to get PascalCase attributes, fall back to snake_case if not available
                statement = statement_type(
                    SegmentRevenues=getattr(segment_info, 'SegmentRevenues', getattr(segment_info, 'segment_revenues', None)),
                    RevenueFromOperations=getattr(segment_info, 'RevenueFromOperations', getattr(segment_info, 'total_revenue_from_operations', None)),
                    SegmentResults=getattr(segment_info, 'SegmentResults', getattr(segment_info, 'segment_results', None)),
                    TotalSegmentProfitBeforeInterestTax=getattr(segment_info, 'TotalSegmentProfitBeforeInterestTax', getattr(segment_info, 'total_segment_profit_before_interest_tax', None)),
                    FinanceCosts=getattr(segment_info, 'FinanceCosts', getattr(segment_info, 'finance_costs', None)),
                    UnallocableCorporateExpenses=getattr(segment_info, 'UnallocableCorporateExpenses', getattr(segment_info, 'unallocable_corporate_expenses', None)),
                    ProfitBeforeTax=getattr(segment_info, 'ProfitBeforeTax', getattr(segment_info, 'profit_before_tax', None)),
                    SegmentAssets=getattr(segment_info, 'SegmentAssets', getattr(segment_info, 'segment_assets', None)),
                    UnallocatedAssets=getattr(segment_info, 'UnallocatedAssets', getattr(segment_info, 'unallocated_assets', None)),
                    TotalAssets=getattr(segment_info, 'TotalAssets', getattr(segment_info, 'total_assets', None)),
                    SegmentLiabilities=getattr(segment_info, 'SegmentLiabilities', getattr(segment_info, 'segment_liabilities', None)),
                    UnallocatedLiabilities=getattr(segment_info, 'UnallocatedLiabilities', getattr(segment_info, 'unallocated_liabilities', None)),
                    TotalLiabilities=getattr(segment_info, 'TotalLiabilities', getattr(segment_info, 'total_liabilities', None)),
                    OtherMetrics=getattr(segment_info, 'OtherMetrics', getattr(segment_info, 'other_metrics', {}))
                )
            else:
                # If extraction didn't work, initialize with empty model
                statement = statement_type()
        
        return statement
    
    except Exception as e:
        print(f"Error combining categories into {statement_desc}: {e}")
        raise

# Update the existing extraction function to use the new category-based approach
async def extract_information_async(statement_type, period: str, content=None):
    """
    Asynchronously extracts financial data for any statement type from the given content.
    
    Args:
        statement_type: The type of financial data to extract
        period (str): The period to extract (e.g., 'Q3FY25')
        content: The pre-filtered text content containing relevant pages
        
    Returns:
        The extracted financial data
    """
    try:
        # Use the category-based approach for all statement types
        if statement_type in (StandaloneProfitAndLoss, ConsolidatedProfitAndLoss, 
                             StandaloneSegmentInformation, ConsolidatedSegmentInformation):
            try:
                return await extract_financial_statement_by_categories(statement_type, period, content)
            except Exception as e:
                # If we get a JSON parsing error, try to fix the format of the numbers in the response
                error_str = str(e).lower()
                if "invalid json output" in error_str or "invalid control character" in error_str:
                    print(f"JSON parsing error detected, retrying with preprocessing: {e}")
                    # Modify the create_category_extraction_pipeline_async function to instruct the LLM 
                    # not to use commas in numeric values
                    return await extract_financial_statement_by_categories_with_preprocessing(statement_type, period, content)
                else:
                    raise
        else:
            # For any other type, fall back to the original pipeline approach
            chain = await create_extraction_pipeline_async(statement_type, period)
            result = await invoke_chain_with_retry(chain, content)
            return result
    
    except Exception as e:
        print(f"Error in extraction pipeline for {statement_type.__name__} in {period}: {e}")
        raise

async def create_category_extraction_pipeline_with_preprocessing_async(statement_type, category_class, period):
    """
    Creates an extraction pipeline with preprocessing to handle numeric values correctly.
    This version explicitly instructs the model not to use commas in numeric values.
    """
    os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_retries=10
    )

    # Create a parser for the specified category class
    parser = EnhancedPydanticOutputParser(pydantic_object=category_class)
    
    # Generate dynamic period information
    period_info = generate_period_info(period)
    
    # Get the category name for more context
    category_name = category_class.__name__
    
    # Get layout info for this category
    layout_info = getattr(category_class, 'layout_info', "No specific layout information available")
    
    # Determine if this is standalone or consolidated
    statement_desc = "standalone" if isinstance(statement_type, (StandaloneProfitAndLoss, StandaloneSegmentInformation)) else "consolidated"
    
    # Template with very explicit numeric formatting instructions
    template = f"""You are a financial analyst assistant that is required to extract {category_name} information from {statement_desc} financial results.
    From this text extract the {category_name} data for {period_info.get(period, period)} from the {statement_desc} statement ONLY.
    
    Text to analyze:
    <text>
    {{text}}
    </text>
    
    Information to extract for {category_name}:
    <financial_metrics_to_extract>
    {{format_instructions}}
    </financial_metrics_to_extract>

    The layout information associated with {category_name} is as follows:
    <layout_info>
    {{layout_info}}
    </layout_info>

    NOTE: 
    1. Any metrics not mentioned below in <financial_metrics_to_extract>, but present in <text> and 
    determined to be a metric for this category as per logic in <layout_info> should still be extracted 
    and saved into the schema for 'other_metric' fields.
    2. IMPORTANT: For segment information, make sure to capture ALL metrics not explicitly defined in the model fields in the 'other_metrics' dictionary. This includes any segment-specific KPIs, ratios, or other financial metrics mentioned in the text.
    3. Only extract values that are clearly stated in the text. If the requested information is not available in the text, return the default values from the schema.
    4. IMPORTANT: When providing numeric values, do NOT use commas as thousand separators. For example, use 140148 instead of 140,148. 
    All numbers must be in standard JSON format without commas.
    5. IMPORTANT FOR SEGMENT INFORMATION: When extracting segment information, carefully ensure you're extracting segment assets and segment liabilities values for the same business segments that appear in segment revenues and segment results. Segment information may span multiple pages, with segment revenues/results on one page and segment assets/liabilities on another page. Look for the same segment names (e.g., "Oil to Chemicals (O2C)", "Oil and Gas", etc.) and ensure you're mapping values correctly across the entire segment information model.
    6. CRITICAL FOR SEGMENT RESULTS: Always interpret segment_results as EBIT (Earnings Before Interest and Tax) level results by default unless explicitly stated otherwise in the text. If the text doesn't specify the level of segment results, assume they are at the EBIT level.
    7. Do NOT confuse "Segment Value of Sales and Services" with "Segment Assets" - they are different categories.
    8. CRITICAL: In your JSON output, use the exact field names provided in the format instructions, including the case. The JSON should use PascalCase field names like "SegmentRevenues", "TotalAssets", etc., NOT snake_case names like "segment_revenues".
    9. AVOID DUPLICATION: Do not duplicate segment revenue information between segment_revenues and other_metrics. If a segment's revenue is already included in segment_revenues (e.g., "Oil to Chemicals (O2C)": 123704), DO NOT include the same information in other_metrics. 
    10. DUPLICATE CHECK: Any entries in financial statements that start with '- SegmentName' and match values already present in segment_revenues should NOT be added to other_metrics. For example, if "Oil to Chemicals (O2C)" appears in segment_revenues, do not add an entry like "- Oil to Chemicals (O2C)" with the same value to other_metrics.
    11. FOR SEGMENT VALUE INFORMATION: Values labeled as "Segment Value of Sales and Services" or similar that match segment revenue values should NOT be included in other_metrics. These are duplicates of the segment_revenues data.
    """
    
    prompt = ChatPromptTemplate.from_template(template).partial(
        format_instructions=parser.get_format_instructions(),
        layout_info=layout_info
    )
    
    # Create the extraction chain using LCEL with async support
    chain = (
        {"text": RunnablePassthrough()} 
        | prompt 
        | llm 
        | parser
    )
    
    return chain

async def extract_financial_statement_by_categories_with_preprocessing(statement_type, period: str, content=None):
    """
    Extracts a financial statement by extracting each category in parallel and then combining them.
    This version includes preprocessing to handle numeric values correctly.
    
    Args:
        statement_type: The financial statement type (StandaloneProfitAndLoss, ConsolidatedProfitAndLoss,
                        StandaloneSegmentInformation, or ConsolidatedSegmentInformation)
        period: The period to extract (e.g., 'Q3FY25')
        content: The pre-filtered text content containing relevant pages
        
    Returns:
        The complete extracted financial statement
    """
    # Get period info for extraction
    period_info = generate_period_info(period)
    
    # Determine statement description for logging
    if statement_type == StandaloneProfitAndLoss:
        statement_desc = "standalone profit and loss"
    elif statement_type == ConsolidatedProfitAndLoss:
        statement_desc = "consolidated profit and loss"
    elif statement_type == StandaloneSegmentInformation:
        statement_desc = "standalone segment information"
    elif statement_type == ConsolidatedSegmentInformation:
        statement_desc = "consolidated segment information"
    else:
        statement_desc = statement_type.__name__
    
    # print(f"Extracting {statement_desc} for {period_info.get(period, period)} with preprocessing")
    
    # Define categories based on statement type
    if statement_type in (StandaloneProfitAndLoss, ConsolidatedProfitAndLoss):
        # For profit and loss statements
        categories = [
            Revenue,
            Expenses, 
            ProfitBeforeTax,
            TaxExpense,
            ProfitAfterTax
        ]
    elif statement_type in (StandaloneSegmentInformation, ConsolidatedSegmentInformation):
        # For segment information - don't use categories since SegmentInformation is a single model
        # The model already contains all required fields
        categories = [SegmentInformation]
    else:
        raise ValueError(f"Unsupported statement type: {statement_type.__name__}")
    
    # Define an async function to extract a single category
    async def extract_category(category_class):
        category_name = category_class.__name__
        try:
            # print(f"  Started extracting {category_name}...")
            # Create the extraction pipeline for this category
            chain = await create_category_extraction_pipeline_with_preprocessing_async(statement_type, category_class, period)
            
            # Extract the category data
            result = await invoke_chain_with_retry(chain, content)
            
            # print(f"  ✓ Extracted {category_name}")
            # print(f"  ✓ Result: \n {result}")
            return category_name.lower(), result
        except Exception as e:
            print(f"  ✗ Failed to extract {category_name}: {e}")
            # Return default instance on error
            return category_name.lower(), category_class()
    
    # Create extraction tasks for each category
    extraction_tasks = [extract_category(category_class) for category_class in categories]
    
    # Execute all extraction tasks in parallel
    results_list = await asyncio.gather(*extraction_tasks)
    
    # Convert results list to dictionary
    results = dict(results_list)

    # print(f"--- Results: for {period_info.get(period, period)} for {statement_desc}: \n {results}")
    
    # Combine all categories into a single financial statement
    try:
        if statement_type in (StandaloneProfitAndLoss, ConsolidatedProfitAndLoss):
            # Create profit and loss statement
            statement = statement_type(
                revenue=results.get('revenue'),
                expenses=results.get('expenses'),
                profit_before_tax=results.get('profitbeforetax'),
                tax_expense=results.get('taxexpense'),
                profit_after_tax=results.get('profitaftertax')
            )
        elif statement_type in (StandaloneSegmentInformation, ConsolidatedSegmentInformation):
            # For segment information, use the directly extracted model
            if 'segmentinformation' in results:
                segment_info = results.get('segmentinformation')
                
                # Handle both snake_case and PascalCase attribute names
                # First, try to get PascalCase attributes, fall back to snake_case if not available
                statement = statement_type(
                    SegmentRevenues=getattr(segment_info, 'SegmentRevenues', getattr(segment_info, 'segment_revenues', None)),
                    RevenueFromOperations=getattr(segment_info, 'RevenueFromOperations', getattr(segment_info, 'total_revenue_from_operations', None)),
                    SegmentResults=getattr(segment_info, 'SegmentResults', getattr(segment_info, 'segment_results', None)),
                    TotalSegmentProfitBeforeInterestTax=getattr(segment_info, 'TotalSegmentProfitBeforeInterestTax', getattr(segment_info, 'total_segment_profit_before_interest_tax', None)),
                    FinanceCosts=getattr(segment_info, 'FinanceCosts', getattr(segment_info, 'finance_costs', None)),
                    UnallocableCorporateExpenses=getattr(segment_info, 'UnallocableCorporateExpenses', getattr(segment_info, 'unallocable_corporate_expenses', None)),
                    ProfitBeforeTax=getattr(segment_info, 'ProfitBeforeTax', getattr(segment_info, 'profit_before_tax', None)),
                    SegmentAssets=getattr(segment_info, 'SegmentAssets', getattr(segment_info, 'segment_assets', None)),
                    UnallocatedAssets=getattr(segment_info, 'UnallocatedAssets', getattr(segment_info, 'unallocated_assets', None)),
                    TotalAssets=getattr(segment_info, 'TotalAssets', getattr(segment_info, 'total_assets', None)),
                    SegmentLiabilities=getattr(segment_info, 'SegmentLiabilities', getattr(segment_info, 'segment_liabilities', None)),
                    UnallocatedLiabilities=getattr(segment_info, 'UnallocatedLiabilities', getattr(segment_info, 'unallocated_liabilities', None)),
                    TotalLiabilities=getattr(segment_info, 'TotalLiabilities', getattr(segment_info, 'total_liabilities', None)),
                    OtherMetrics=getattr(segment_info, 'OtherMetrics', getattr(segment_info, 'other_metrics', {}))
                )
            else:
                # If extraction didn't work, initialize with empty model
                statement = statement_type()
        
        return statement
    
    except Exception as e:
        print(f"Error combining categories into {statement_desc}: {e}")
        raise

def create_financial_data_object(company_name, quarter, standalone_pl=None, consolidated_pl=None, standalone_segment=None, consolidated_segment=None):
    """
    Creates a FinancialData object with the extracted profit and loss statements.
    
    Args:
        company_name (str): The name of the company
        quarter (str): The quarter (e.g., 'Q3FY25')
        standalone_pl: The standalone profit and loss statement
        consolidated_pl: The consolidated profit and loss statement
        standalone_segment: The standalone segment information
        consolidated_segment: The consolidated segment information
    Returns:
        FinancialData: A FinancialData object with the extracted data
    """
    # Parse quarter and determine date range
    quarter_type = quarter[0:2]  # e.g., "Q3", "9M", "FY"
    
    # Extract fiscal year correctly
    if quarter.startswith("FY"):
        fiscal_year = int(quarter[2:])
    else:
        fy_position = quarter.find("FY")
        fiscal_year = int(quarter[fy_position+2:]) if fy_position != -1 else int(quarter[2:])
    
    # Convert to full years
    fiscal_year_end = 2000 + fiscal_year
    
    # Quarter date mappings dynamically generated
    start_date = ""
    end_date = ""
    
    # Fiscal year starts on April 1
    if quarter.startswith("Q1"):
        start_date = f"{fiscal_year_end-1}-04-01"
        end_date = f"{fiscal_year_end-1}-06-30"
    elif quarter.startswith("Q2"):
        start_date = f"{fiscal_year_end-1}-07-01"
        end_date = f"{fiscal_year_end-1}-09-30"
    elif quarter.startswith("Q3"):
        start_date = f"{fiscal_year_end-1}-10-01"
        end_date = f"{fiscal_year_end-1}-12-31"
    elif quarter.startswith("Q4"):
        start_date = f"{fiscal_year_end}-01-01"
        end_date = f"{fiscal_year_end}-03-31"
    elif quarter.startswith("6M"):
        start_date = f"{fiscal_year_end-1}-04-01"
        end_date = f"{fiscal_year_end-1}-09-30"
    elif quarter.startswith("9M"):
        start_date = f"{fiscal_year_end-1}-04-01"
        end_date = f"{fiscal_year_end-1}-12-31"
    elif quarter.startswith("FY"):
        start_date = f"{fiscal_year_end-1}-04-01"
        end_date = f"{fiscal_year_end}-03-31"
    
    # Create QuarterData instance
    quarter_data = QuarterData(
        company=company_name,
        period=quarter,
        periodStart=start_date,
        periodEnd=end_date,
        currency="INR",
        unit="crores",
        standalone_profit_and_loss=standalone_pl,
        consolidated_profit_and_loss=consolidated_pl,
        standalone_segment_information=standalone_segment,
        consolidated_segment_information=consolidated_segment
    )
    
    # Create FinancialData instance
    financial_data = FinancialData(root={quarter: quarter_data})
    
    return financial_data

async def run_extractions_async(company_name, quarter, content=None):
    """
    Runs extractions for multiple quarters and statement types asynchronously.
    
    Args:
        company_name (str): Name of the company
        quarter (str): The main quarter to extract (e.g., 'Q3FY25')
        content: The content to extract information from
        
    Returns:
        FinancialData: A combined financial data object with all extracted data
    """
    # Parse the input quarter to determine which periods to extract
    quarter_num = int(quarter[1:2])
    fy_position = quarter.find("FY")
    fiscal_year = int(quarter[fy_position+2:])
    prev_fiscal_year = fiscal_year - 1
    
    # Generate periods based on input quarter
    periods = []
    
    # 1. Always include current quarter
    periods.append(quarter)
    
    # 2. Include same quarter in previous year
    periods.append(f"Q{quarter_num}FY{prev_fiscal_year}")
    
    # 3. Include previous quarter if not Q1
    if quarter_num > 1:
        periods.append(f"Q{quarter_num-1}FY{fiscal_year}")
    if quarter_num == 1:
        periods.append(f"Q4FY{prev_fiscal_year}")
    
    # 4. Include other periods based on quarter number
    if quarter_num == 1:
        # For Q1, include previous fiscal year
        periods.append(f"FY{prev_fiscal_year}")
    elif quarter_num == 2:
        # For Q2, include 6-month periods and previous fiscal year
        periods.append(f"6MFY{fiscal_year}")
        periods.append(f"6MFY{prev_fiscal_year}")
        periods.append(f"FY{prev_fiscal_year}")
    elif quarter_num == 3:
        # For Q3, include 9-month periods and previous fiscal year
        periods.append(f"9MFY{fiscal_year}")
        periods.append(f"9MFY{prev_fiscal_year}")
        periods.append(f"FY{prev_fiscal_year}")
    elif quarter_num == 4:
        # For Q4, include current and previous fiscal years
        periods.append(f"FY{fiscal_year}")
        periods.append(f"FY{prev_fiscal_year}")
    
    print(f"Periods to extract: {periods}")
    
    # Create a combined financial data object
    combined_financial_data = FinancialData(root={})
    
    # Create tasks for all periods
    extraction_tasks = []
    
    for period in periods:
        print(f"\nCreating extraction tasks for {period}...")
        
        # Create tasks for all statement types for this period
        standalone_task = extract_information_async(StandaloneProfitAndLoss, period, content)
        consolidated_task = extract_information_async(ConsolidatedProfitAndLoss, period, content)
        standalone_segment_task = extract_information_async(StandaloneSegmentInformation, period, content)
        consolidated_segment_task = extract_information_async(ConsolidatedSegmentInformation, period, content)
        
        # Add tasks to our list with period information
        extraction_tasks.append((period, standalone_task, consolidated_task, standalone_segment_task, consolidated_segment_task))
    
    # Process all periods concurrently
    print(f"\nProcessing {len(periods)} periods concurrently...")

    results = await asyncio.gather(
        *[asyncio.gather(standalone_task, consolidated_task, standalone_segment_task, consolidated_segment_task, return_exceptions=True) 
          for period, standalone_task, consolidated_task, standalone_segment_task, consolidated_segment_task in extraction_tasks],
        return_exceptions=True
    )
    
    # Process results for each period
    for i, period in enumerate(periods):
        if isinstance(results[i], Exception):
            print(f"Failed to process period {period}: {results[i]}")
            continue
            
        standalone_result, consolidated_result, standalone_segment_result, consolidated_segment_result = results[i]
        
        # Process standalone P&L result
        standalone_pl = None
        if not isinstance(standalone_result, Exception):
            standalone_pl = standalone_result
            print(f"Successfully extracted standalone P&L for {period}")
        else:
            print(f"Failed to extract standalone P&L for {period}: {standalone_result}")
        
        # Process consolidated P&L result
        consolidated_pl = None
        if not isinstance(consolidated_result, Exception):
            consolidated_pl = consolidated_result
            print(f"Successfully extracted consolidated P&L for {period}")
        else:
            print(f"Failed to extract consolidated P&L for {period}: {consolidated_result}")
        
        # Process standalone segment information result
        standalone_segment = None
        if not isinstance(standalone_segment_result, Exception):
            standalone_segment = standalone_segment_result
            print(f"Successfully extracted standalone segment information for {period}")
        else:
            print(f"Failed to extract standalone segment information for {period}: {standalone_segment_result}")

        # Process consolidated segment information result
        consolidated_segment = None
        if not isinstance(consolidated_segment_result, Exception):
            consolidated_segment = consolidated_segment_result
            print(f"Successfully extracted consolidated segment information for {period}")
        else:
            print(f"Failed to extract consolidated segment information for {period}: {consolidated_segment_result}")

        # Create and add to the combined financial data object
        if standalone_pl or consolidated_pl or standalone_segment or consolidated_segment:
            financial_data = create_financial_data_object(
                company_name, 
                period, 
                standalone_pl=standalone_pl, 
                consolidated_pl=consolidated_pl,
                standalone_segment=standalone_segment,
                consolidated_segment=consolidated_segment 
            )

            # Add this quarter's data to the combined financial data
            combined_financial_data.root.update(financial_data.root)

    # Validate the extracted financial data
    print("\nValidating financial statements...")
    validation_results = validate_financial_data(combined_financial_data)
    
    if validation_results:
        formatted_results = format_validation_results(validation_results)
        print(formatted_results)
        
        # Add validation results to the combined financial data as metadata
        if not hasattr(combined_financial_data, 'metadata'):
            combined_financial_data.metadata = {}
        combined_financial_data.metadata['validation_results'] = validation_results
    else:
        print("All financial statements passed validation.")

    return combined_financial_data


def create_vector_store_from_pages(pages: list):
    """
    Creates a vector store from individual pages using Google Generative AI embeddings.
    
    Args:
        pages (list): List of page contents
        
    Returns:
        InMemoryVectorStore: A vector store containing the embedded pages
    """
    # Initialize Google Generative AI embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv('GEMINI_API_KEY')
    )
    
    # Create documents from individual pages
    documents = [Document(page_content=page) for page in pages]
    
    # Create and return the vector store
    return InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )

def preprocess_json_output(json_str):
    """
    Preprocesses JSON output from the LLM to remove commas in numeric values.
    
    Args:
        json_str (str): The JSON string to preprocess
        
    Returns:
        str: The preprocessed JSON string
    """
    # Define a pattern to match numeric values with commas
    # This pattern looks for digits followed by a comma followed by more digits
    # within the context of a JSON key-value pair
    pattern = r'(:\s*)(\d{1,3}(?:,\d{3})+)(\s*[,}])'
    
    # Define a replacement function that removes commas from the matched number
    def replace_commas(match):
        before, number, after = match.groups()
        clean_number = number.replace(',', '')
        return f'{before}{clean_number}{after}'
    
    # Apply the replacement to the JSON string
    preprocessed_str = re.sub(pattern, replace_commas, json_str)
    return preprocessed_str

# Custom parser that preprocesses the JSON output before parsing
class EnhancedPydanticOutputParser(EnhancedPydanticOutputParser):
    def parse(self, response):
        """Parse the output of an LLM call with preprocessing for numeric values."""
        json_str = ""
        if isinstance(response, str):
            json_str = response
        elif isinstance(response, BaseMessage):
            json_str = response.content
        else:
            raise ValueError(f"Got unexpected type {type(response)}")
            
        # Extract JSON string if it's wrapped in markdown code blocks
        if "```json" in json_str and "```" in json_str:
            match = re.search(r"```json\n(.*?)```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Try without the json specifier
                match = re.search(r"```\n(.*?)```", json_str, re.DOTALL)
                if match:
                    json_str = match.group(1)
        
        # Preprocess the JSON string to remove commas in numeric values
        json_str = preprocess_json_output(json_str)
        
        # Attempt to parse the preprocessed JSON
        try:
            json_object = json.loads(json_str)
            
            # Handle field naming convention mismatch
            # If the model has field aliases, convert snake_case keys to their aliased names
            field_alias_map = {}
            for field_name, field in self.pydantic_object.model_fields.items():
                if field.alias:
                    field_alias_map[field_name] = field.alias
            
            # If we have a flat JSON object, check and update keys
            if isinstance(json_object, dict):
                # Create a new dict with aliased keys
                aliased_json_object = {}
                for key, value in json_object.items():
                    # If the key is a snake_case field name that has an alias, use the alias
                    if key in field_alias_map:
                        aliased_json_object[field_alias_map[key]] = value
                    else:
                        aliased_json_object[key] = value
                
                json_object = aliased_json_object
            
            return self.pydantic_object.parse_obj(json_object)
        except json.JSONDecodeError as e:
            raise OutputParserException(f"Invalid json output: {json_str}") from e
        except ValidationError as e:
            raise OutputParserException(f"Failed to parse {json_str} as {self.pydantic_object.__name__}") from e
