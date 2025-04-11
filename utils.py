import logging
import asyncio
import os
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import re

from extract_results import create_financial_data_object, extract_information_async
from pydantic_model import (
    StandaloneProfitAndLoss, ConsolidatedProfitAndLoss,
    StandaloneSegmentInformation, ConsolidatedSegmentInformation
)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set up logger
logger = logging.getLogger(__name__)

def get_periods_to_extract(quarter: str):

    # Generate periods dynamically based on the input quarter
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

    return periods

# async def retrieve_and_extract_financial_information(
#     information_type: str,
#     period: str,
#     model_class,
#     bm25_index,
#     table_docs,
#     doc_to_page_map,
#     pages,
#     faiss_vectorstore
# ):
#     """
#     A generalized function to retrieve and extract financial information.
    
#     Args:
#         information_type: Type of information to extract (e.g., 'standalone profit and loss statement')
#         period: The financial period (e.g., 'Q3FY24')
#         model_class: The Pydantic model class to use for extraction
#         bm25_index: BM25Okapi index for search
#         table_docs: List of table documents
#         doc_to_page_map: Mapping from document index to page index
#         pages: List of page metadata objects
#         faiss_vectorstore: FAISS vector store for semantic search
        
#     Returns:
#         The extracted financial information or None if no relevant pages found
#     """
#     # Form the query
#     query = f"{information_type} for {period}"
    
#     # 1. Use BM25 on table metadata
#     bm25_pages = search_with_bm25(bm25_index, table_docs, doc_to_page_map, pages, query)
    
#     # 2. Use FAISS for semantic retrieval
#     faiss_pages = search_with_faiss(faiss_vectorstore, pages, query)
    
#     # 3. Combine results (avoiding duplicates)
#     combined_pages = []
#     seen_page_numbers = set()
    
#     # Add BM25 results first
#     for page in bm25_pages:
#         page_num = page['page_number']
#         if page_num not in seen_page_numbers:
#             combined_pages.append(page)
#             seen_page_numbers.add(page_num)
    
#     # Add semantic results next
#     for page in faiss_pages:
#         page_num = page['page_number']
#         if page_num not in seen_page_numbers:
#             combined_pages.append(page)
#             seen_page_numbers.add(page_num)
    
#     if combined_pages:
#         # Log the relevant pages
#         logger.info(f"Combined retrieval found {len(combined_pages)} pages for {information_type} in {period}: {[page['page_number'] for page in combined_pages]}")
        
#         # Prompt user to modify the list of combined_pages
#         user_input = input("Please enter the page numbers to include, separated by commas (e.g., 1,2,3): ")
#         user_page_numbers = set(map(int, user_input.split(',')))

#         # Filter combined_pages based on user input
#         filtered_pages = [page for page in combined_pages if page['page_number'] in user_page_numbers]

#         # Log the filtered pages
#         logger.info(f"User selected {len(filtered_pages)} pages: {[page['page_number'] for page in filtered_pages]}")
        
#         # Use filtered pages for further processing
#         combined_pages = filtered_pages
        
#         # Combine the relevant pages
#         content = "\n\n".join([page.get('original_content', '') for page in combined_pages])
        
#         # Extract information
#         extracted_info = await extract_information_async(model_class, period, content)
#         logger.info(f"Extracted {information_type} for {period}")
        
#         return extracted_info
    
#     return None

async def get_relevant_pages_for_statement(
    information_type: str,
    bm25_index,
    table_docs,
    doc_to_page_map,
    pages,
    faiss_vectorstore
):
    """
    Performs hybrid search to find relevant pages for a statement type.
    
    Args:
        information_type: Type of information to extract (e.g., 'standalone profit and loss statement')
        bm25_index: BM25Okapi index for search
        table_docs: List of table documents
        doc_to_page_map: Mapping from document index to page index
        pages: List of page metadata objects
        faiss_vectorstore: FAISS vector store for semantic search
        
    Returns:
        List of relevant pages
    """
    # Form the query without period information
    query = f"{information_type['description']}"

    # logger.info(f"Searching for {query}")
    
    # 1. Use BM25 on table metadata
    bm25_pages = search_with_bm25(bm25_index, table_docs, doc_to_page_map, pages, query)
    
    # 2. Use FAISS for semantic retrieval
    faiss_pages = search_with_faiss(faiss_vectorstore, pages, query)
    
    # 3. Combine results (avoiding duplicates)
    combined_pages = []
    seen_page_numbers = set()
    
    # Add BM25 results first
    for page in bm25_pages:
        page_num = page['page_number']
        if page_num not in seen_page_numbers:
            combined_pages.append(page)
            seen_page_numbers.add(page_num)

    # logger.info(f"BM25 found {len(bm25_pages)} pages for {information_type}: {[page['page_number'] for page in bm25_pages]}")
    
    # Add semantic results next
    for page in faiss_pages:
        page_num = page['page_number']
        if page_num not in seen_page_numbers:
            combined_pages.append(page)
            seen_page_numbers.add(page_num)
    
    # logger.info(f"FAISS found {len(faiss_pages)} pages for {information_type}: {[page['page_number'] for page in faiss_pages]}")

    if combined_pages:
        # logger.info(f"Combined retrieval found {len(combined_pages)} pages for {information_type}: {[page['page_number'] for page in combined_pages]}")
        
        # Prompt user to modify the list of combined_pages
        user_input = input(f"Please enter the page numbers to include for {information_type['name']}, from {[page['page_number'] for page in combined_pages]}: ")
        user_page_numbers = set(map(int, user_input.split(',')))

        # Include ALL user-specified pages, not just ones in combined_pages
        filtered_pages = []
        for page_num in user_page_numbers:
            # Try to find the page in combined_pages first
            matching_pages = [p for p in combined_pages if p['page_number'] == page_num]
            if matching_pages:
                filtered_pages.append(matching_pages[0])
            else:
                # Find the page in the complete pages list
                matching_pages = [p for p in pages if p['page_number'] == page_num]
                if matching_pages:
                    filtered_pages.append(matching_pages[0])
                    
        return filtered_pages
    
    return []

async def extract_financial_information(
    company_name: str,
    periods: list[str],
    bm25_index,
    table_docs,
    doc_to_page_map,
    pages,
    faiss_vectorstore,
    statement_types: list[str] = None
) -> dict:
    """Modified to perform search once per statement type"""
    
    # Define available statement types and their configurations
    available_types = {
        'standalone_pl': {
            "name": "standalone profit and loss statement",
            'description': """standalone profit and loss statement; statement of standalone financial results;
            particulars -  revenue from operations, other income, total income, total expenses, profit before tax, tax expense, profit after tax""",
            'model_class': StandaloneProfitAndLoss
        },
        'consolidated_pl': {
            "name": "consolidated profit and loss statement",
            'description': """consolidated profit and loss statement; statement of consolidated financial results;
            particulars -  revenue from operations, other income, total income, total expenses, profit before tax, tax expense, profit after tax""",
            'model_class': ConsolidatedProfitAndLoss
        },
        'standalone_segment': {
            "name": "standalone segment information",
            'description': """standalone segment information; segment information (statement of standalone financial results);
            particulars -  segment revenue, segment profit/loss, segment assets, segment liabilities""",
            'model_class': StandaloneSegmentInformation
        },
        'consolidated_segment': {
            "name": "consolidated segment information",
            'description': """consolidated segment information; segment information (statement of consolidated financial results);
            particulars -  segment revenue, segment profit/loss, segment assets, segment liabilities""",
            'model_class': ConsolidatedSegmentInformation
        }
    }

    # If statement_types is not provided, use all available types
    types_to_extract = statement_types if statement_types else list(available_types.keys())
    
    # First, get relevant pages for each statement type
    statement_pages = {}

    logger.info(f"Getting relevant pages for {types_to_extract}")

    for type_key in types_to_extract:
        if type_key in available_types:
            type_config = available_types[type_key]
            relevant_pages = await get_relevant_pages_for_statement(
                type_config,
                bm25_index,
                table_docs,
                doc_to_page_map,
                pages,
                faiss_vectorstore
            )
            if relevant_pages:
                statement_pages[type_key] = relevant_pages

    # For segment information types, analyze if information is split across pages
    for type_key in types_to_extract:
        if type_key in statement_pages and type_key.endswith('_segment'):
            pages = statement_pages[type_key]
            
            # Check what's available on each page
            segment_content_map = {}
            for page in pages:
                content = page.get('original_content', '').lower()
                segment_content_map[page['page_number']] = {
                    "revenue": "segment revenue" in content or "segment revenues" in content,
                    "results": "segment result" in content or "segment results" in content,
                    "assets": "segment asset" in content or "segment assets" in content,
                    "liabilities": "segment liabilit" in content
                }
            
            # If we're missing fields on some pages, ensure all related pages are used together
            if any(pages) and not all(all(fields.values()) for fields in segment_content_map.values()):
                # Sort pages by page number to maintain document order
                statement_pages[type_key] = sorted(pages, key=lambda p: p['page_number'])
                logger.info(f"Detected split segment information across pages: {[p['page_number'] for p in statement_pages[type_key]]}")
    
    # Now process each period using the pre-selected pages
    all_tasks = []
    period_task_map = {}
    
    for period in periods:
        period_tasks = []
        for type_key in types_to_extract:
            if type_key in statement_pages:  # Only process if we have relevant pages
                type_config = available_types[type_key]
                content = "\n\n".join([
                    page.get('original_content', '') 
                    for page in statement_pages[type_key]
                ])
                task = extract_information_async(
                    type_config['model_class'],
                    period,
                    content
                )
                period_tasks.append((type_key, task))
        
        all_tasks.extend([task for _, task in period_tasks])
        period_task_map[period] = period_tasks

    # Run extraction tasks concurrently
    all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    # Process results
    extracted_data = {}
    task_index = 0
    
    for period in periods:
        period_tasks = period_task_map[period]
        results = all_results[task_index:task_index + len(period_tasks)]
        task_index += len(period_tasks)
        
        period_results = {}
        for (type_key, _), result in zip(period_tasks, results):
            period_results[type_key] = None if isinstance(result, Exception) else result
        
        if any(result is not None for result in period_results.values()):
            period_data = create_financial_data_object(
                company_name,
                period,
                standalone_pl=period_results.get('standalone_pl'),
                consolidated_pl=period_results.get('consolidated_pl'),
                standalone_segment=period_results.get('standalone_segment'),
                consolidated_segment=period_results.get('consolidated_segment')
            )
            extracted_data[period] = period_data.root[period]
    
    # Apply standardization before returning
    extracted_data = standardize_financial_data(extracted_data)
    return extracted_data

def create_bm25_index_from_table_metadata(pages):
    """
    Create a BM25 index from table metadata in pages.
    
    Args:
        pages: List of page metadata objects
        
    Returns:
        Tuple of (BM25Okapi index, list of table documents, mapping from doc index to page)
    """
    # Extract table metadata
    table_docs = []
    doc_to_page_map = {}  # Maps document index to the original page
    
    for page_idx, page in enumerate(pages):
        tables = page.get('tables', [])
        for table in tables:
            # Create a document from table caption and summary
            table_text = f"{table.get('table_caption', '')} {table.get('table_summary', '')}"
            if table_text.strip():  # Only add non-empty documents
                table_docs.append(table_text)
                doc_to_page_map[len(table_docs) - 1] = page_idx
    
    # Tokenize documents for BM25
    tokenized_docs = [doc.lower().split() for doc in table_docs]
    
    # Create and return BM25 index
    return BM25Okapi(tokenized_docs), table_docs, doc_to_page_map

def search_with_bm25(bm25_index, table_docs, doc_to_page_map, pages, query, top_k=3):
    """
    Search for relevant pages using BM25 on table metadata.
    
    Args:
        bm25_index: BM25Okapi index
        table_docs: List of table documents
        doc_to_page_map: Mapping from document index to page index
        pages: List of page metadata objects
        query: Search query
        top_k: Number of top results to return
        
    Returns:
        List of relevant pages
    """
    is_consolidated_query = "consolidated" in query.lower()
    is_standalone_query = "standalone" in query.lower()
    is_segment_query = "segment" in query.lower()
    
    # Tokenize query for BM25
    tokenized_query = query.lower().split()    
    # Get BM25 scores
    doc_scores = bm25_index.get_scores(tokenized_query)
    
    filtered_indices = []
    for idx, score in enumerate(doc_scores):
        if score > 0:  # Only consider non-zero scores
            doc_text = table_docs[idx].lower()

            match_found = False
            
            # Check for consolidated/standalone
            if is_consolidated_query and is_segment_query:
                match_found = ("consolidated" in doc_text and "segment" in doc_text)
            elif is_standalone_query and is_segment_query:
                match_found = ("standalone" in doc_text and "segment" in doc_text)
            elif is_consolidated_query:
                match_found = "consolidated" in doc_text and "profit" in doc_text
            elif is_standalone_query:
                match_found = "standalone" in doc_text and "profit" in doc_text
            else:
                # For general queries
                match_found = True
                
            if match_found:
                filtered_indices.append(idx)
            # Check if document matches the required consolidation type
            # if (is_consolidated_query and "consolidated results" in doc_text) or \
            #    (is_standalone_query and "standalone results" in doc_text) or \
            #    (not is_consolidated_query and not is_standalone_query):  # For general queries
            #     filtered_indices.append(idx)
    
    # Sort filtered indices by score
    filtered_indices.sort(key=lambda idx: doc_scores[idx], reverse=True)
    
    # Take top-k results from filtered list
    top_indices = filtered_indices[:top_k]
    
    # Get the corresponding pages (no need for score > 0 check as we already did it)
    page_indices = set(doc_to_page_map[idx] for idx in top_indices) 
    
    # Return relevant pages
    relevant_pages = [pages[idx] for idx in page_indices]
    
    
    # logger.info(f"BM25 found {len(relevant_pages)} relevant pages for query '{query}': {[page['page_number'] for page in relevant_pages]}")
    
    return relevant_pages


def create_faiss_index_from_pages(pages):
    """
    Create a FAISS vector index from all pages using dense embeddings.
    
    Args:
        pages: List of page metadata objects
        
    Returns:
        FAISS vector store and a mapping to original pages
    """
    # Initialize the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv('GEMINI_API_KEY')
    )
    
    # Extract text content from pages
    texts = []
    metadatas = []
    
    for i, page in enumerate(pages):
        # Get the page content, summary, and table info
        page_number = page.get('page_number', i+1)
        page_summary = page.get('page_summary', '')
        original_content = page.get('original_content', '')
        
        # Extract table information
        table_info = ""
        for table in page.get('tables', []):
            table_info += f"{table.get('table_caption', '')} {table.get('table_summary', '')} "
        
        # Create a rich text representation for embedding
        # Include enough context for semantic understanding but not too much to dilute relevance
        text = f"Page {page_number}. Summary: {page_summary}. Tables: {table_info}. Content: {original_content[:500]}"
        
        texts.append(text)
        metadatas.append({"page_index": i, "page_number": page_number})
    
    # Create FAISS index from texts
    faiss_vectorstore = FAISS.from_texts(
        texts, embeddings, metadatas=metadatas
    )
    
    return faiss_vectorstore

def search_with_faiss(faiss_vectorstore, pages, query, top_k=3):
    """
    Search for relevant pages using FAISS semantic search.
    
    Args:
        faiss_vectorstore: FAISS vector store
        pages: List of page metadata objects
        query: Query text
        top_k: Number of top results to return
        
    Returns:
        List of relevant pages
    """

    is_consolidated_query = "consolidated" in query.lower()
    is_standalone_query = "standalone" in query.lower()
    is_segment_query = "segment" in query.lower()

    # Create retriever with specified k
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # Get relevant documents
    docs = faiss_retriever.get_relevant_documents(query)
    
    # Map back to original pages
    relevant_pages = []
    for doc in docs:
        page_index = doc.metadata.get("page_index")
        if page_index is not None and 0 <= page_index < len(pages):
            page = pages[page_index]
            
            # Check if page contains tables with matching consolidation type
            if is_consolidated_query or is_standalone_query:
                tables = page.get('tables', [])
                page_summary = page.get('page_summary', '').lower()
                page_has_right_type = False
                
                for table in tables:
                    table_caption = table.get('table_caption', '').lower()
                    
                    if is_consolidated_query and is_segment_query:
                        page_has_right_type = ("consolidated" in page_summary and "segment" in page_summary)
                    elif is_standalone_query and is_segment_query:
                        page_has_right_type = ("standalone" in page_summary and "segment" in page_summary)
                    elif is_consolidated_query:
                        page_has_right_type = "consolidated" in page_summary and "profit" in page_summary
                    elif is_standalone_query:
                        page_has_right_type = "standalone" in page_summary and "profit" in page_summary
                    else:
                        # For general queries
                        page_has_right_type = True
                    # if (is_consolidated_query and "consolidated results" in table_caption) or \
                    #    (is_standalone_query and "standalone results" in table_caption):
                    #     page_has_right_type = True
                    #     break
                
                if page_has_right_type:
                    relevant_pages.append(page)
                    # logger.info(f"FAISS found matching page {page.get('page_number')} for query '{query}'")
            else:
                # For general queries without specific consolidation type requirements
                relevant_pages.append(page)
                # logger.info(f"FAISS found page {page.get('page_number')} for query '{query}'")
            
            # Stop once we have enough results
            if len(relevant_pages) >= top_k:
                break
    
    return relevant_pages


# Helper function to extract tables from markdown content
def extract_table_from_content(content: str) -> List[List[str]]:
    """
    Extracts markdown tables from the page content.
    
    Args:
        content: The page content in markdown format
        
    Returns:
        A list of rows, where each row is a list of cell values
    """
    tables = []
    current_table = []
    in_table = False
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            # This looks like a table row
            if not in_table and '---' not in line:
                in_table = True
            
            # Skip separator rows
            if '---' in line:
                continue
                
            # Split by | and clean up cells
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            
            # Only add non-empty rows
            if any(cells):
                current_table.append(cells)
        elif in_table and not line:
            # Empty line after table - end of table
            if current_table:
                tables.append(current_table)
                current_table = []
            in_table = False
    
    # Don't forget the last table if there's no empty line after it
    if in_table and current_table:
        tables.append(current_table)
    
    # If we found no tables but the content looks like it might contain tabular data,
    # try a more aggressive approach
    if not tables:
        # Look for patterns of numbers and text arranged in columns
        numeric_pattern = re.compile(r'\d+[\d,.]*')
        
        # Group lines that might form a table
        potential_table = []
        current_group = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                if current_group:
                    potential_table.append(current_group)
                    current_group = []
                continue
                
            # Count numbers in the line
            numbers = numeric_pattern.findall(line)
            if len(numbers) >= 2:  # Line with multiple numbers is likely a table row
                # Split by whitespace and clean up
                cells = [cell.strip() for cell in re.split(r'\s{2,}', line)]
                current_group.append(cells)
        
        # Add the last group if needed
        if current_group:
            potential_table.append(current_group)
        
        # If we found any potential tables with a reasonable structure, use the first one
        for group in potential_table:
            if len(group) >= 2:  # At least two rows
                tables.append(group)
                break
    
    # Return the first table if any were found, otherwise empty list
    return tables[0] if tables else []

def table_to_markdown(table_content):
    """
    Convert a table (list of lists) to markdown format.
    
    Args:
        table_content: List of rows, where each row is a list of cell values
        
    Returns:
        Markdown representation of the table
    """
    if not table_content:
        return ""
    
    # Calculate column widths
    col_widths = [0] * len(table_content[0])
    for row in table_content:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Generate the markdown table
    markdown_lines = []
    
    # Header row
    header = "| " + " | ".join([str(cell).ljust(col_widths[i]) for i, cell in enumerate(table_content[0])]) + " |"
    markdown_lines.append(header)
    
    # Separator row
    separator = "| " + " | ".join(["-" * col_widths[i] for i in range(len(col_widths))]) + " |"
    markdown_lines.append(separator)
    
    # Data rows
    for row in table_content[1:]:
        data_row = "| " + " | ".join([str(cell).ljust(min(col_widths[i], len(col_widths))) 
                                      for i, cell in enumerate(row) if i < len(col_widths)]) + " |"
        markdown_lines.append(data_row)
    
    # Add an empty line after the table
    markdown_lines.append("")
    
    return "\n".join(markdown_lines)

def standardize_segment_information(segment_data):
    """
    Standardize segment information by removing duplicate fields from other_metrics
    and ensuring consistent field naming across all periods.
    
    Args:
        segment_data: Segment information data containing standard fields and other_metrics
        
    Returns:
        Standardized segment information data
    """
    if not segment_data or not isinstance(segment_data, dict):
        return segment_data
        
    # Create a copy to avoid modifying the original data
    result = {**segment_data}
    
    # If there's no other_metrics, just return the original data
    if 'other_metrics' not in result or not result['other_metrics']:
        return result
    
    # Get other_metrics
    other_metrics = dict(result.get('other_metrics', {}))
    
    # Define standard fields and their possible variations in other_metrics
    standard_field_variations = {
        'unallocated_assets': [
            'unallocated_assets', 'segment_assets_unallocated', 
            'unallocable_assets', 'segment_unallocated_assets',
            'assets_unallocated'
        ],
        'unallocated_liabilities': [
            'unallocated_liabilities', 'segment_liabilities_unallocated',
            'unallocable_liabilities', 'segment_unallocated_liabilities',
            'liabilities_unallocated'
        ]
    }
    
    # Check each standard field
    for standard_field, variations in standard_field_variations.items():
        # Remove variations of this standard field from other_metrics
        keys_to_remove = []
        for key in other_metrics:
            if key in variations:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            other_metrics.pop(key, None)
    
    # Update the other_metrics field
    result['other_metrics'] = other_metrics
    return result

# This function needs to be called after creating the financial data object
# or as part of the extraction process
def standardize_financial_data(financial_data):
    """Standardize all financial data across periods"""
    if not financial_data or not isinstance(financial_data, dict):
        return financial_data
        
    for period, period_data in financial_data.items():
        # Standardize standalone segment information
        if 'standalone_segment_information' in period_data and period_data['standalone_segment_information']:
            period_data['standalone_segment_information'] = standardize_segment_information(
                period_data['standalone_segment_information']
            )
        
        # Standardize consolidated segment information
        if 'consolidated_segment_information' in period_data and period_data['consolidated_segment_information']:
            period_data['consolidated_segment_information'] = standardize_segment_information(
                period_data['consolidated_segment_information']
            )
    
    return financial_data

def standardize_llm_response(response_text):
    """Standardize field names in raw LLM JSON response"""
    # Define standardized field mappings
    field_mappings = {
        '"segment_assets_unallocated"': '"unallocated_assets"',
        '"segment_liabilities_unallocated"': '"unallocated_liabilities"',
        '"finance_cost"': '"finance_costs"',
        '"value_of_sales_&_services"': '"value_of_sales_services"',
        '"segment_results_ebitda_': '"segment_ebitda_'
    }
    
    # Apply string replacements to standardize field names
    standardized_text = response_text
    for old_name, new_name in field_mappings.items():
        standardized_text = standardized_text.replace(old_name, new_name)
    
    return standardized_text