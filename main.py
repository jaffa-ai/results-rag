import asyncio
import io
import json
import logging
import os
import time
from io import BytesIO
from typing import List, Optional

import aiohttp
import httpx
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from openai import AsyncAzureOpenAI
from llama_index.core.schema import TextNode
from llama_parse import LlamaParse
from azure.storage.blob import BlobServiceClient


from pydantic_model import FinancialData
from utils import get_periods_to_extract, extract_financial_information, create_bm25_index_from_table_metadata, create_faiss_index_from_pages, extract_table_from_content, table_to_markdown
from models.page_metadata import PageSummary, TableMetadata
from validate_financial_statements import validate_financial_data, format_validation_results


docs_uuid = "2ff08be1-6c70-480d-a630-45c3bad77fcc"
docs_url = f"/docs-{docs_uuid}"
app = FastAPI(title = "Results RAG", docs_url=docs_url)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# logging.getLogger("httpx").setLevel(logging.WARNING)

# Add these lines after the basic logging configuration
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)


logger.info(f"Docs URL: {docs_url}")

# Step 1: Define the function first
def is_numeric(value):
    """Check if a string represents a number (handles commas in Indian format)"""
    if not isinstance(value, str):
        return False
    # Remove commas and spaces
    value = value.replace(',', '').replace(' ', '').strip()
    try:
        float(value)
        return True
    except ValueError:
        return False

# Step 2: Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Step 3: Register filter after the function is defined
templates.env.filters["is_numeric"] = is_numeric

# Step 4: Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Blob Storage configuration
try:
    
    AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_BLOB_STORAGE_CONNECTION_STRING')
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("AZURE_BLOB_STORAGE_CONNECTION_STRING environment variable is not set")
        
    CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')
    CONTAINER_NAME_MVP = os.getenv('AZURE_CONTAINER_NAME_MVP')
    
    if not CONTAINER_NAME or not CONTAINER_NAME_MVP:
        raise ValueError("Container name environment variables are not set")
    
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    container_client_mvp = blob_service_client.get_container_client(CONTAINER_NAME_MVP)
    
except Exception as e:
    print(f"Error initializing Azure Storage: {str(e)}")
    raise

@app.get("/extract-document")
async def extract_document(company_name: str, quarter: str, document_type: str, regenerate: bool = False):
    """
    Common function to extract and parse documents using LlamaParse
    
    Args:
        company_name: Company symbol/name
        quarter: Quarter period (e.g., Q1FY24)
        document_type: Type of document
        regenerate: Whether to regenerate the output even if it already exists
    
    Returns:
        Dict containing the parsed document pages and SAS URL
    """
    try:
        # No need to validate document_type as it's enforced by the Enum
        report_type = f"Quarterly{document_type}"

        print(f"Extracting {report_type} document for {company_name} {quarter}")
        
        # Set the correct output folder
        if document_type == "InvestorPresentation":
            output_folder = "investor_ppt"
        else:  # FinancialResult
            output_folder = "financial_result"
        
        # Call documents/get_metadata to verify document exists
        mvp_backend_url = os.getenv("MVP_BACKEND_URL")
        if not mvp_backend_url:
            raise ValueError("MVP_BACKEND_URL environment variable not set")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{mvp_backend_url}/documents/get_metadata",
                params={
                    "company_symbol": company_name,
                    "period": quarter,
                    "report_type": report_type
                }
            )

            print(f"Response: {response.json()}")

            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"{document_type} document not found"
                )
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Error fetching document metadata"
                )

            metadata = response.json()
            
            # Find the document metadata and SAS URL
            period_type = "quarters" if quarter.startswith("Q") else "years"
            period_data = metadata["company_docs"][period_type].get(quarter, {})
            
            doc_filename = f"{quarter}_{company_name}_{report_type}"
            doc_metadata = period_data.get(doc_filename)
            
            if not doc_metadata or not doc_metadata.get("exists"):
                raise HTTPException(
                    status_code=404,
                    detail=f"{document_type} document not found"
                )

            sas_url = doc_metadata.get("sas_url")
            if not sas_url:
                raise HTTPException(
                    status_code=500,
                    detail="SAS URL not found in metadata"
                )
            

            # Check if summary already exists in blob storage
            output_blob_path = f"{company_name}/outputs/quarters/{quarter}/{output_folder}/summary.jsonl"
            blob_client = container_client_mvp.get_blob_client(output_blob_path)

            if blob_client.exists() and not regenerate:
                summary_data = blob_client.download_blob().readall()
                pages = json.loads(summary_data)
                return {
                    "status": "success", 
                    "company_name": company_name, 
                    "quarter": quarter, 
                    "document_type": document_type,
                    "pages": pages,
                    "sas_url": sas_url  # Include the SAS URL in the response
                }

            else:
                # Get appropriate parsing instructions based on document type
                if document_type == "InvestorPresentation":
                    parsing_instruction = """
                    You are given pages from an investor presentation of a company.
                    For any graphs, try to create a 2D table of relevant values, along with a description of the graph.
                    For any schematic diagrams, MAKE SURE to describe a list of all components and their connections to each other.
                    Make sure that you always parse out the text with the correct reading order.
                    """
                else: 
                    parsing_instruction = """
                    You are given pages from a quarterly financial results document of a company.
                    For any graphs, try to create a 2D table of relevant values, along with a description of the graph.
                    For any schematic diagrams, MAKE SURE to describe a list of all components and their connections to each other.
                    Make sure that you always parse out the text with the correct reading order.
                    For any tables, always capture the table caption especially noting the type of table and level of consolidation i.e. either standalone or consolidated.
                    """


                # Create parser in synchronous context
                parser = LlamaParse(
                    api_key=os.getenv("LLAMA_PARSE_API_KEY"),
                    result_type="markdown",
                    use_vendor_multimodal_model=True,
                    vendor_multimodal_model_name="gemini-2.0-flash-001",
                    vendor_multimodal_api_key=os.getenv("GEMINI_API_KEY"),
                    invalidate_cache=True,
                    parsing_instruction=parsing_instruction,
                )

                
                # Run the parsing in a separate thread to avoid blocking
                def run_parser():
                    return parser.get_json_result(sas_url)

                # Use asyncio.to_thread to run the synchronous code in a separate thread
                json_objs = await asyncio.to_thread(run_parser)
                logger.info(f"Parsed {len(json_objs[0]['pages'])} pages")
                
                # Save job metadata separately if needed
                job_metadata_blob_path = f"{company_name}/outputs/quarters/{quarter}/{output_folder}/parsing_job_metadata.json"
                container_client_mvp.upload_blob(
                    job_metadata_blob_path,
                    json.dumps(json_objs[0]["job_metadata"], indent=2),
                    overwrite=True
                )

                def get_text_nodes(json_list: List[dict]):
                    text_nodes = []
                    for idx, page in enumerate(json_list):
                        text_node = TextNode(text=page["md"], metadata={"page": page["page"]})
                        text_nodes.append(text_node)
                    return text_nodes
                
                json_list = json_objs[0]["pages"]
                docs = get_text_nodes(json_list)

                # Convert docs to list of pages with content
                pages = []
                pages_metadata = []
                for doc in docs:
                    pages.append(doc.get_content(metadata_mode="all"))

                for i, page in enumerate(pages):
                    # Get the page number from the corresponding doc's metadata or use index+1
                    page_number = docs[i].metadata.get("page", i + 1)
                    page_metadata = await extract_page_metadata(page, page_number)
                    pages_metadata.append(page_metadata)

                # Save the results to blob storage
                container_client_mvp.upload_blob(
                    output_blob_path,
                    json.dumps(pages_metadata, indent=2),
                    overwrite=True
                )
                
                return {
                    "company_name": company_name,
                    "quarter": quarter,
                    "document_type": document_type,
                    "pages": pages_metadata,
                    "sas_url": sas_url  # Include the SAS URL in the response
                }
            

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error extracting document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/review-tables/{doc_id}", response_class=HTMLResponse)
async def review_tables(
    request: Request, 
    doc_id: str,
    regenerate: bool = False
):
    """
    Render a page showing all extracted tables for review and correction.
    
    Args:
        request: The FastAPI request object
        doc_id: Format should be {company_name}_{quarter}_{document_type}
        regenerate: Whether to force re-extraction
        
    Returns:
        HTML template with editable tables
    """
    try:
        # Parse the doc_id
        parts = doc_id.split('_')
        if len(parts) < 3:
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
            
        # Last part is document_type, the part before that is quarter, and the rest is company_name
        document_type = parts[-1]
        quarter = parts[-2]
        company_name = '_'.join(parts[:-2])
        
        # Set the correct output folder
        output_folder = "investor_ppt" if document_type == "InvestorPresentation" else "financial_result"
        
        # Define the blob path
        blob_path = f"{company_name}/outputs/quarters/{quarter}/{output_folder}/summary.jsonl"
        
        # Check if we need to re-extract the document
        if regenerate:
            # Call the extract_document endpoint
            extraction_result = await extract_document(
                company_name=company_name, 
                quarter=quarter, 
                document_type=document_type, 
                regenerate=True
            )
            logger.info(f"Re-extracted document: {company_name}_{quarter}_{document_type}")
            
        # Get the blob client
        blob_client = container_client_mvp.get_blob_client(blob_path)
        
        # Check if the blob exists
        if not blob_client.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Extracted document not found for {doc_id}"
            )
        
        # Download and parse the data
        summary_data = blob_client.download_blob().readall().decode("utf-8")
        pages = json.loads(summary_data)
        
        # Build table list with page information
        tables_by_page = []
        
        for page in pages:
            page_number = page.get("page_number")
            page_summary = page.get("page_summary", "")
            tables = page.get("tables", [])
            
            # Extract tables from content
            table_content = extract_table_from_content(page.get("original_content", ""))
            
            # Add page and table information to our list
            if tables:
                for i, table in enumerate(tables):
                    table_id = f"{page_number}_{i}"
                    
                    # Use extracted table content if available, otherwise use empty table
                    table_data = table_content if table_content and i == 0 else [[]]
                    
                    tables_by_page.append({
                        "page_number": page_number,
                        "page_summary": page_summary,
                        "table_id": table_id,
                        "table_caption": table.get("table_caption", ""),
                        "table_summary": table.get("table_summary", ""),
                        "table_content": table_data
                    })
        
        # Add is_numeric as a template filter
        templates.env.filters["is_numeric"] = is_numeric
        
        # Render the template with table data
        return templates.TemplateResponse(
            "review_tables.html",
            {
                "request": request,
                "doc_id": doc_id,
                "company_name": company_name,
                "quarter": quarter,
                "document_type": document_type,
                "tables_by_page": tables_by_page
            }
        )
    
    except Exception as e:
        logger.error(f"Error in review tables: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-table/{doc_id}/{table_id}", include_in_schema=False)
async def update_table(
    doc_id: str,
    table_id: str,
    updated_table: dict
):
    """
    Update a specific table with corrected values.
    
    Args:
        doc_id: Format should be {company_name}_{quarter}_{document_type}
        table_id: Format should be {page_number}_{table_index}
        updated_table: The corrected table data
        
    Returns:
        Confirmation of the update
    """
    try:
        # Parse the doc_id
        parts = doc_id.split('_')
        if len(parts) < 3:
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
            
        document_type = parts[-1]
        quarter = parts[-2]
        company_name = '_'.join(parts[:-2])
        
        # Parse the table_id
        table_parts = table_id.split('_')
        if len(table_parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid table_id format")
            
        page_number = int(table_parts[0])
        table_index = int(table_parts[1])
        
        # Set the correct output folder
        output_folder = "investor_ppt" if document_type == "InvestorPresentation" else "financial_result"
        
        # Define the blob paths
        blob_path = f"{company_name}/outputs/quarters/{quarter}/{output_folder}/summary.jsonl"
        backup_path = f"{company_name}/outputs/quarters/{quarter}/{output_folder}/summary_backup_{int(time.time())}.jsonl"
        
        # Get the blob client
        blob_client = container_client_mvp.get_blob_client(blob_path)
        
        # Check if the blob exists
        if not blob_client.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Extracted document summary not found for {doc_id}"
            )
        
        # Download and parse the existing data
        summary_data = blob_client.download_blob().readall().decode("utf-8")
        pages = json.loads(summary_data)
        
        # Make a backup first
        backup_blob_client = container_client_mvp.get_blob_client(backup_path)
        backup_blob_client.upload_blob(summary_data, overwrite=True)
        
        # Find the specific page
        target_page = None
        page_index = -1
        for i, page in enumerate(pages):
            if page.get("page_number") == page_number:
                target_page = page
                page_index = i
                break
        
        if not target_page:
            raise HTTPException(
                status_code=404,
                detail=f"Page {page_number} not found in the extracted document"
            )
        
        # Update the specific table
        if table_index >= len(target_page.get("tables", [])):
            raise HTTPException(
                status_code=404,
                detail=f"Table index {table_index} not found in page {page_number}"
            )
        
        # Get the existing table
        existing_table = target_page["tables"][table_index]
        
        # Update page summary if provided
        if "page_summary" in updated_table:
            target_page["page_summary"] = updated_table.get("page_summary")
        
        # Update table metadata
        existing_table["table_caption"] = updated_table.get("table_caption", existing_table.get("table_caption", ""))
        existing_table["table_summary"] = updated_table.get("table_summary", existing_table.get("table_summary", ""))
        
        # Also update the original_content with the table data
        if "table_content" in updated_table and updated_table["table_content"]:
            # Convert the table content to markdown format
            markdown_table = table_to_markdown(updated_table["table_content"])
            
            # Try to locate and replace the table in the original content
            original_content = target_page.get("original_content", "")
            
            # This is a simplified approach - for complex documents you might need
            # a more sophisticated algorithm to locate and replace the exact table
            lines = original_content.split('\n')
            in_table = False
            table_start = -1
            table_end = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('|') and line.strip().endswith('|'):
                    if not in_table:
                        in_table = True
                        table_start = i
                elif in_table and not line.strip():
                    table_end = i
                    break
            
            # If we found a table, replace it
            if table_start >= 0:
                if table_end < 0:
                    table_end = len(lines)
                
                # Replace the table
                new_lines = lines[:table_start] + markdown_table.split('\n') + lines[table_end:]
                target_page["original_content"] = '\n'.join(new_lines)
        
        # Update the page in the document
        pages[page_index] = target_page
        
        # Save the updated data
        container_client_mvp.upload_blob(
            blob_path,
            json.dumps(pages, indent=2),
            overwrite=True
        )
        
        return {
            "status": "success",
            "message": f"Table {table_id} updated for {doc_id}",
            "backup_created": backup_path
        }
    
    except Exception as e:
        logger.error(f"Error updating table: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



async def extract_page_metadata(page_content, page_number):
    """
    Extract metadata for a page using Azure OpenAI and Pydantic models.
    
    Args:
        page_content: Content of the page
        page_number: Page number of the document
    
    Returns:
        Dictionary with page_number, page_summary, tables, and original_content
    """
    
    # Initialize OpenAI client
    client = AsyncAzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_DEPLOYMENT")
    )
    
    
    # # Set up the system prompt
    # system_prompt = """You are a financial analyst specialized in extracting structured information from financial documents.
    # Analyze the given page from a financial report and extract the following information:
    # 1. A concise summary of the page content (2-3 sentences).
    # 2. For any tables found on the page:
     #    - IMPORTANT: Explicitly identify and tag each table as either "Standalone Results" or "Consolidated Results" in the table caption
    #    - If there's no clear indication of consolidation level, default to "Standalone Results" and mention this in the caption.
    #    - A brief description of what the table represents

    system_prompt = """You are a financial analyst specialized in extracting structured information from financial documents.
    Analyze the given page from a financial report and extract the following information:
    1. A concise summary of the page content (2-3 sentences).
    2. For any tables found on the page:
        - The table caption or title
        - **Classification Tags:**  
            a) **Table Type:**  
                - If the table contains **segment-related data**, explicitly tag it as `"Segment Information"`.  
                - If the table is **not segment-related**, classify it as either `"Standalone Results"` or `"Consolidated Results"`.  
            b) **Consolidation Level:**  
                - If the table clearly indicates `"Standalone"` or `"Consolidated"`, apply the correct tag.  
                - If no clear consolidation level is mentioned:  
                    - If the table is **segment-related**, default to `"Standalone Segment Information"` and mention this in the caption.  
                    - If the table is **not segment-related**, default to `"Standalone Results"` and mention this in the caption.  
        - A brief description of what the table represents

    
    Format your response as a JSON object with the following structure:
    {
        "page_summary": "...",
        "tables": [
            {
                "table_caption": "...",  
                "table_summary": "..."
            }
        ]
    }
    
    If no tables are present on the page, return an empty array for "tables".
    """
    
    try:
        response = await client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Page content:\n\n{page_content}"}
            ],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        metadata_json = json.loads(response.choices[0].message.content)
        
        # Create PageSummary object using the Pydantic model
        page_summary = PageSummary(
            page_number=page_number,
            page_summary=metadata_json.get("page_summary", "No summary available"),
            tables=[
                TableMetadata(
                    table_caption=table.get("table_caption", "No caption available"),
                    table_summary=table.get("table_summary", "No description available")
                )
                for table in metadata_json.get("tables", [])
            ],
            original_content=page_content
        )
                # Convert to dictionary for JSON serialization
        return page_summary.model_dump()
        
    except Exception as e:
        return {
            "page_number": page_number,
            "page_summary": "Error extracting metadata",
            "tables": [],
            "original_content": page_content
        }

@app.get("/extract-financial-statements")
async def extract_financial_statements_endpoint(
    company_name: str, 
    quarter: str,
    regenerate: bool = False,
    statement_types: str = None
):
    """
    Extract financial statements from parsed document.
    
    Args:
        company_name: Company symbol/name
        quarter: Quarter period (e.g., Q1FY24)
        regenerate: Whether to regenerate the output even if it already exists
        statement_types: Comma-separated list of statement types to extract. Options include:
                       'standalone_pl', 'consolidated_pl', 'standalone_segment', 'consolidated_segment'
                       If None, will extract all available statements.
    
    Returns:
        Dict containing the extracted financial data
    """
    try:
        # Define paths
        input_blob_path = f"{company_name}/outputs/quarters/{quarter}/financial_result/summary.jsonl"
        output_blob_path = f"{company_name}/outputs/quarters/{quarter}/financial_result/{company_name}_{quarter}_financial_statements.json"
        
        # Process statement_types parameter (convert from comma-separated string to list if provided)
        statement_types_list = statement_types.split(',') if statement_types else None
        
        # Check if output already exists
        blob_client = container_client_mvp.get_blob_client(output_blob_path)
        
        if blob_client.exists() and not regenerate:
            # Return existing financial statements
            financial_data = blob_client.download_blob().readall().decode("utf-8")
            return {
                "status": "success",
                "company_name": company_name,
                "quarter": quarter,
                "financial_data": json.loads(financial_data)
            }
        
        # Check if input exists
        input_blob_client = container_client_mvp.get_blob_client(input_blob_path)
        if not input_blob_client.exists():
            # If input doesn't exist, try to generate it
            logger.info(f"Summary file not found. Attempting to extract document first.")
            await extract_document(company_name, quarter, "FinancialResult", regenerate)
            
            # Check again if input exists after extraction
            if not input_blob_client.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Summary file not found for {company_name} {quarter}"
                )
        
        # Download the summary.jsonl file
        summary_data = input_blob_client.download_blob().readall().decode("utf-8")
        pages = json.loads(summary_data)

        # logger.info(f"Pages: {pages[0]}")
        
        # vector_store = create_vector_index_from_pages(pages)
        faiss_vectorstore = create_faiss_index_from_pages(pages)
        bm25_index, table_docs, doc_to_page_map = create_bm25_index_from_table_metadata(pages)
        
        periods = get_periods_to_extract(quarter)
        logger.info(f"Periods to extract: {periods}")
        
        # Extract financial information concurrently, passing statement_types parameter
        extracted_data = await extract_financial_information(
            company_name,
            periods,
            bm25_index,
            table_docs,
            doc_to_page_map,
            pages,
            faiss_vectorstore,
            statement_types=statement_types_list
        )
        
        # Clean up duplicate segment information data
        for period, period_data in extracted_data.items():
            # Process standalone segment information if it exists
            if "standalone_segment_information" in period_data and period_data["standalone_segment_information"]:
                period_data["standalone_segment_information"] = remove_duplicate_segment_data(period_data["standalone_segment_information"])
            
            # Process consolidated segment information if it exists
            if "consolidated_segment_information" in period_data and period_data["consolidated_segment_information"]:
                period_data["consolidated_segment_information"] = remove_duplicate_segment_data(period_data["consolidated_segment_information"])
        
        # Combine all extracted data
        financial_data = FinancialData(root=extracted_data)

        
        # Save to blob storage and return
        container_client_mvp.upload_blob(
            output_blob_path,
            json.dumps(financial_data.model_dump(by_alias=True), indent=2),
            overwrite=True
        )
        
        return {
            "company_name": company_name,
            "quarter": quarter,
            "financial_data": financial_data.model_dump(by_alias=True)
        }
        
    except Exception as e:
        logger.error(f"Error extracting financial statements: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


            
@app.get("/review-financial-statements/{doc_id}", response_class=HTMLResponse)
async def review_financial_statements(
    request: Request,
    doc_id: str,
    regenerate: bool = False
):
    """
    Render a page showing financial statements for editing and correction.
    
    Args:
        request: The FastAPI request object
        doc_id: Format should be {company_name}_{quarter}
        regenerate: Whether to force re-extraction
        
    Returns:
        HTML template with editable financial statements
    """
    try:
        # Parse the doc_id
        parts = doc_id.split('_')
        if len(parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
            
        # Last part is quarter, and the rest is company_name
        quarter = parts[-1]
        company_name = '_'.join(parts[:-1])
        
        # Define the blob path
        blob_path = f"{company_name}/outputs/quarters/{quarter}/financial_result/{company_name}_{quarter}_financial_statements.json"
        
        # Check if we need to re-extract the financial statements
        if regenerate:
            # Call the extract_financial_statements endpoint
            extraction_result = await extract_financial_statements_endpoint(
                company_name=company_name,
                quarter=quarter,
                regenerate=True
            )
            logger.info(f"Re-extracted financial statements: {company_name}_{quarter}")
            
        # Get the blob client
        blob_client = container_client_mvp.get_blob_client(blob_path)
        
        # Check if the blob exists
        if not blob_client.exists():
            # Try to extract financial statements if they don't exist
            await extract_financial_statements_endpoint(
                company_name=company_name,
                quarter=quarter,
                regenerate=False
            )
            
            # Check again if the blob exists after extraction
            if not blob_client.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Financial statements not found for {doc_id}"
                )
        
        # Download and parse the data
        financial_data_json = blob_client.download_blob().readall().decode("utf-8")
        financial_data = json.loads(financial_data_json)
        
        # Get all available periods
        available_periods = list(financial_data.keys())
        
        # Render the template with all financial data periods
        return templates.TemplateResponse(
            "review_financial_statements.html",
            {
                "request": request,
                "doc_id": doc_id,
                "company_name": company_name,
                "quarter": quarter,
                "financial_data": financial_data,
                "available_periods": available_periods
            }
        )
    
    except Exception as e:
        logger.error(f"Error in edit financial statements: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-financial-statement/{doc_id}/{statement_type}", include_in_schema=False)
async def update_financial_statement(
    doc_id: str,
    statement_type: str,
    updated_data: dict
):
    """
    Update a specific financial statement with corrected values.
    
    Args:
        doc_id: Format should be {company_name}_{period}
        statement_type: Type of financial statement (e.g., standalone_profit_and_loss)
        updated_data: The corrected financial statement data
        
    Returns:
        Confirmation of the update
    """
    try:
        # Parse the doc_id
        parts = doc_id.split('_')
        if len(parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid doc_id format")
            
        # This is the period we're updating (might be a different period than the main quarter)
        update_period = parts[-1]
        company_name = '_'.join(parts[:-1])
        
        # Validate statement_type
        valid_statement_types = [
            "standalone_profit_and_loss", 
            "consolidated_profit_and_loss",
            "standalone_segment_information",
            "consolidated_segment_information",
            "standalone_balance_sheet",
            "consolidated_balance_sheet",
            "standalone_cashflow",
            "consolidated_cashflow"
        ]
        
        if statement_type not in valid_statement_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid statement type. Must be one of: {', '.join(valid_statement_types)}"
            )
        
        # If this is a segment information update, remove duplicate revenue data from other_metrics
        if statement_type in ["standalone_segment_information", "consolidated_segment_information"]:
            updated_data = remove_duplicate_segment_data(updated_data)
        
        # Get the main quarter for the file path
        # In this implementation, we assume all periods are stored in the same 
        # file in the main quarter directory (typically Q3FY25 or whatever primary quarter was extracted)
        
        # First try to access the file directly
        main_quarter = update_period  # Start by assuming the update_period is the main quarter
        blob_path = f"{company_name}/outputs/quarters/{main_quarter}/financial_result/{company_name}_{main_quarter}_financial_statements.json"
        blob_client = container_client_mvp.get_blob_client(blob_path)
        
        # If the file doesn't exist for the update period, we need to find the main quarter
        if not blob_client.exists():
            # List all quarters under the company's outputs directory to find where the data is stored
            quarters_list = []
            quarter_path_prefix = f"{company_name}/outputs/quarters/"
            for blob in container_client_mvp.list_blobs(name_starts_with=quarter_path_prefix):
                path_parts = blob.name.split('/')
                if len(path_parts) > 3:  # Should be company/outputs/quarters/Q3FY25/...
                    quarters_list.append(path_parts[3])
            
            quarters_list = list(set(quarters_list))  # Remove duplicates
            
            # Try to find a quarter that has the financial statements file
            found_main_quarter = False
            for q in quarters_list:
                test_path = f"{company_name}/outputs/quarters/{q}/financial_result/{company_name}_{q}_financial_statements.json"
                test_client = container_client_mvp.get_blob_client(test_path)
                if test_client.exists():
                    main_quarter = q
                    blob_path = test_path
                    blob_client = test_client
                    found_main_quarter = True
                    break
            
            if not found_main_quarter:
                raise HTTPException(
                    status_code=404,
                    detail=f"Financial statements file not found for {company_name}"
                )
        
        # Download and parse the existing data
        financial_data_json = blob_client.download_blob().readall().decode("utf-8")
        financial_data = json.loads(financial_data_json)
        
        # Make a backup first
        backup_path = f"{company_name}/outputs/quarters/{main_quarter}/financial_result/{company_name}_{main_quarter}_financial_statements_backup_{int(time.time())}.json"
        backup_blob_client = container_client_mvp.get_blob_client(backup_path)
        backup_blob_client.upload_blob(financial_data_json, overwrite=True)
        
        # Check if the period to update exists in the financial data
        if update_period not in financial_data:
            raise HTTPException(
                status_code=404,
                detail=f"Period {update_period} not found in financial data"
            )
        
        # Get the period data object
        period_data = financial_data[update_period]
        
        # Update the specific statement
        if statement_type not in period_data:
            raise HTTPException(
                status_code=404,
                detail=f"Statement type {statement_type} not found in period {update_period}"
            )
        
        # Update the statement with the new data
        period_data[statement_type] = updated_data
        
        # Save the updated data
        container_client_mvp.upload_blob(
            blob_path,
            json.dumps(financial_data, indent=2),
            overwrite=True
        )
        
        # Try to validate the updated data to make sure it's consistent
        try:
            # Convert to FinancialData model
            financial_data_model = FinancialData(root=financial_data)
            
            # Save with consistent field names
            container_client_mvp.upload_blob(
                blob_path,
                json.dumps(financial_data_model.model_dump(by_alias=True), indent=2),
                overwrite=True
            )
        except Exception as e:
            # Log but don't fail the update - original data was saved already
            logger.warning(f"Could not standardize field names after update: {str(e)}")
        
        return {
            "status": "success",
            "message": f"Financial statement {statement_type} updated for {doc_id}",
            "backup_created": backup_path
        }
    
    except Exception as e:
        logger.error(f"Error updating financial statement: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def remove_duplicate_segment_data(segment_data):
    """
    Remove duplicated segment revenue data from other_metrics.
    
    This function removes entries from other_metrics that duplicate data already present in segment_revenues.
    
    Args:
        segment_data: The segment information data containing segment_revenues and other_metrics
        
    Returns:
        Cleaned segment information data with duplicates removed
    """
    # If there's no other_metrics, just return the original data
    if not segment_data.get('other_metrics'):
        return segment_data
    
    # Create a copy of the data to avoid modifying the original
    cleaned_data = {**segment_data}
    other_metrics = dict(cleaned_data.get('other_metrics', {}))
    segment_revenues = cleaned_data.get('segment_revenues', {})
    segment_results = cleaned_data.get('segment_results', {})
    segment_assets = cleaned_data.get('segment_assets', {})
    segment_liabilities = cleaned_data.get('segment_liabilities', {})
    
    # Keys to remove
    keys_to_remove = []
    
    # Check for pattern: segment_value_of_sales_and_services_revenue_*
    # And other patterns that indicate segment revenue duplication
    for key, metric in other_metrics.items():
        # Check if the key follows the pattern of segment_value_of_sales_and_services_revenue_*
        if key.startswith('segment_value_of_sales_and_services_revenue_'):
            keys_to_remove.append(key)
            continue
            
        # Check if description starts with '- ' (indicating segment name)
        description = metric.get('description', '')
        if description.startswith('- '):
            # Extract segment name from the description
            segment_name = description[2:].strip()  # Remove the '- ' prefix
            
            # Check if this segment exists in segment_revenues with the same value
            for segment, value in segment_revenues.items():
                # The segment names might not match exactly, so do a fuzzy match
                if (segment_name.lower() in segment.lower() or segment.lower() in segment_name.lower()) and \
                   abs(metric.get('value', 0) - value) < 0.01:  # Allow for small floating point differences
                    keys_to_remove.append(key)
                    break
            
            # Check if this segment exists in segment_results with the same value
            for segment, value in segment_results.items():
                if (segment_name.lower() in segment.lower() or segment.lower() in segment_name.lower()) and \
                   abs(metric.get('value', 0) - value) < 0.01:  # Allow for small floating point differences
                    keys_to_remove.append(key)
                    break
            
            # Check if this segment exists in segment_assets with the same value
            for segment, value in segment_assets.items():
                if (segment_name.lower() in segment.lower() or segment.lower() in segment_name.lower()) and \
                   abs(metric.get('value', 0) - value) < 0.01:  # Allow for small floating point differences
                    keys_to_remove.append(key)
                    break   
            
            # Check if this segment exists in segment_liabilities with the same value
            for segment, value in segment_liabilities.items():
                if (segment_name.lower() in segment.lower() or segment.lower() in segment_name.lower()) and \
                   abs(metric.get('value', 0) - value) < 0.01:  # Allow for small floating point differences
                    keys_to_remove.append(key)
                    break
            
    
    # Remove duplicates
    for key in keys_to_remove:
        other_metrics.pop(key, None)
    
    cleaned_data['other_metrics'] = other_metrics
    return cleaned_data

@app.get("/validate-financial-statements")
async def validate_financial_statements_endpoint(
    company_name: str,
    quarter: str,
    statement_type: str = None
):
    """
    Validate financial statements for a company and quarter.
    
    Args:
        company_name: Company symbol/name
        quarter: Quarter period (e.g., Q1FY24)
        statement_type: Which statement to validate. Options: 'standalone_pl', 'consolidated_pl', 
                      'standalone_segment', 'consolidated_segment', or 'all' if not specified
        
    Returns:
        Structured validation results for the specified statement type(s)
    """
    try:
        # Define the blob path to retrieve financial data
        blob_path = f"{company_name}/outputs/quarters/{quarter}/financial_result/{company_name}_{quarter}_financial_statements.json"
        
        # Get the blob client
        blob_client = container_client_mvp.get_blob_client(blob_path)
        
        # Check if the blob exists
        if not blob_client.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Financial statements not found for {company_name} {quarter}"
            )
        
        # Download and parse the financial data
        financial_data_json = blob_client.download_blob().readall().decode("utf-8")
        financial_data_raw = json.loads(financial_data_json)

        # Map statement_type parameter to the model
        statement_type_map = {
            "standalone_pl": "standalone_profit_and_loss",
            "consolidated_pl": "consolidated_profit_and_loss",
            "standalone_segment": "standalone_segment_information", 
            "consolidated_segment": "consolidated_segment_information"
        }

        # Skip Pydantic validation for now and directly pass the raw data to validate_financial_data
        if statement_type and statement_type.lower() != "all":
            if statement_type not in statement_type_map:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid statement_type. Must be one of: {', '.join(statement_type_map.keys())} or 'all'"
                )
            
            # Filter the data to include only the specified statement type
            filtered_data = {}
            
            for period, period_data in financial_data_raw.items():
                filtered_period_data = {}
                mapped_type = statement_type_map[statement_type]
                
                if mapped_type in period_data:
                    filtered_period_data[mapped_type] = period_data[mapped_type]
                    filtered_data[period] = filtered_period_data
            
            
            # Run validation on filtered data
            validation_results = validate_financial_data({"root": filtered_data})
            logger.info(f"Validating only {statement_type}")
            logger.info(f"Validation results: {validation_results}")
        else:
            # Run validation on all data
            validation_results = validate_financial_data({"root": financial_data_raw})
            logger.info("Validating all statement types")

        # Format the validation results
        formatted_results = format_validation_results(validation_results)
        
        # Add metadata to the response
        formatted_results["company_name"] = company_name
        formatted_results["quarter"] = quarter
        formatted_results["statement_type"] = statement_type or "all"
        
        return formatted_results
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error validating financial statements: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/download-tables-excel")
async def download_tables_excel(
    company_name: str,
    quarter: str,
    document_type: str,
    page_number: str  # Changed to str to accept 'all'
):
    """
    Download tables from a specific page or all pages of a document as an Excel file.
    
    Args:
        company_name: Company symbol/name
        quarter: Quarter period (e.g., Q1FY24)
        document_type: Type of document (FinancialResult or InvestorPresentation)
        page_number: The page number to extract tables from, or 'all' for all pages
        
    Returns:
        Excel file containing tables from the specified page(s)
    """
    try:
        # Set the correct output folder
        output_folder = "investor_ppt" if document_type == "InvestorPresentation" else "financial_result"
        
        # Define the blob path
        blob_path = f"{company_name}/outputs/quarters/{quarter}/{output_folder}/summary.jsonl"
        
        # Get the blob client
        blob_client = container_client_mvp.get_blob_client(blob_path)
        
        # Check if the blob exists
        if not blob_client.exists():
            # Try to extract the document first
            await extract_document(company_name, quarter, document_type, regenerate=False)
            
            # Check again if the blob exists after extraction
            if not blob_client.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found for {company_name} {quarter} {document_type}"
                )
        
        # Download and parse the data
        summary_data = blob_client.download_blob().readall().decode("utf-8")
        pages = json.loads(summary_data)
        
        # Create Excel file in memory
        output = BytesIO()
        
        # Create ExcelWriter object
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Determine if we should process all pages or just one
            if page_number.lower() == 'all':
                # Process all pages with tables
                target_pages = [page for page in pages if page.get("tables", [])]
                
                if not target_pages:
                    # Create a sheet with a message if no tables found
                    pd.DataFrame({"Message": ["No tables found in this document"]}).to_excel(
                        writer, sheet_name="Info", index=False
                    )
            else:
                # Try to convert page_number to integer
                try:
                    page_num_int = int(page_number)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid page_number. Must be a number or 'all'"
                    )
                
                # Find the specific page
                target_pages = [page for page in pages if page.get("page_number") == page_num_int]
                
                if not target_pages:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Page {page_number} not found in the document or has no tables"
                    )
            
            # Process each target page
            for target_page in target_pages:
                page_num = target_page.get("page_number")
                sheet_name = f"Page {page_num}"
                
                # Get tables from the page
                tables = target_page.get("tables", [])
                
                # If the page has no tables, skip it
                if not tables:
                    continue
                
                # Extract table content from the page
                table_content = extract_table_from_content(target_page.get("original_content", ""))
                
                # Get the workbook and create a new worksheet
                workbook = writer.book
                worksheet = workbook.add_worksheet(sheet_name)
                
                # Start from row 0
                current_row = 0
                max_col_width = {}  # To track column widths
                
                # Process each table
                for i, table in enumerate(tables):
                    # # Get table caption
                    # table_caption = table.get("table_caption", f"Table {i+1}")
                    
                    # # Write the table caption with merged cells
                    # bold_format = workbook.add_format({'bold': True})
                    # worksheet.write(current_row, 0, table_caption, bold_format)
                    # current_row += 1
                    
                    # Get table data (use extracted table_content if available for first table)
                    data = table_content if table_content and i == 0 else [[]]
                    
                    # Only process the table if it has data
                    if data and len(data) > 0 and len(data[0]) > 0:
                        # Write headers with bold format
                        headers = data[0]
                        header_format = workbook.add_format({'bold': True, 'border': 1})
                        
                        for col, header in enumerate(headers):
                            worksheet.write(current_row, col, header, header_format)
                            # Track column width based on header length
                            max_col_width[col] = max(max_col_width.get(col, 0), len(str(header)))
                        
                        current_row += 1
                        
                        # Write data
                        for row_idx, row_data in enumerate(data[1:]):
                            for col, cell in enumerate(row_data):
                                if col < len(headers):  # Only write if within header length
                                    worksheet.write(current_row, col, cell)
                                    # Track column width
                                    max_col_width[col] = max(max_col_width.get(col, 0), len(str(cell)))
                            current_row += 1
                    else:
                        # Write a "No data" message for empty tables
                        worksheet.write(current_row, 0, "No data available")
                        current_row += 1
                    
                    # Add a gap of 3 rows between tables (only if not the last table)
                    if i < len(tables) - 1:
                        current_row += 3
                
                # Set column widths
                for col, width in max_col_width.items():
                    worksheet.set_column(col, col, width + 2)  # Add padding
        
        # Set pointer at the beginning of the stream
        output.seek(0)
        
        # Set filename
        if page_number.lower() == 'all':
            filename = f"{company_name}_{quarter}_{document_type}_all_tables.xlsx"
        else:
            filename = f"{company_name}_{quarter}_{document_type}_page{page_number}_tables.xlsx"
        
        # Return Excel file
        return StreamingResponse(
            output, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error downloading tables as Excel: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Usage example
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
