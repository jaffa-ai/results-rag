from pydantic import BaseModel, Field
from typing import List, Optional

class TableMetadata(BaseModel):
    """Metadata for tables found in a page"""
    table_caption: str = Field(..., description="Caption or title of the table")
    table_summary: str = Field(..., description="Brief description of what the table represents")

class PageSummary(BaseModel):
    """Summary and metadata for a single page"""
    page_number: int = Field(..., description="Page number of the page")
    page_summary: str = Field(..., description="Concise summary of the page content")
    tables: Optional[List[TableMetadata]] = Field(default_factory=list, description="List of tables found in the page (if any)")
    original_content: str = Field(..., description="Original extracted content of the page")