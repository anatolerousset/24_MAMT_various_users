"""Hybrid chunker with image support, leveraging both doc structure & token awareness."""

import warnings
import logging
import re
from typing import Any, ClassVar, Final, Iterable, Iterator, Literal, Optional, Union
from pathlib import Path
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, PositiveInt, StringConstraints, TypeAdapter, model_validator, field_validator
from typing_extensions import Annotated, Self

try:
    import semchunk
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    raise RuntimeError(
        "Module requires 'chunking' extra; to install, run: "
        "`pip install 'docling-core[chunking]'`"
    )

from pandas import DataFrame

from docling_core.search.package import VERSION_PATTERN
from docling_core.transforms.chunker import BaseChunk, BaseChunker, BaseMeta
from docling_core.types import DoclingDocument as DLDocument


from parser_and_chunker.msword_backend_modif_metadataplusv250408 import (
    DocumentOriginMod,
)

from docling_core.types.doc.document import (
    DocItem,
    #DocumentOrigin,
    LevelNumber,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    PictureItem,
    GroupItem,
)

from docling_core.types.doc.labels import DocItemLabel, GroupLabel


from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.base import ImageRefMode

from tabulate import tabulate

_VERSION: Final = "1.0.0"

_KEY_SCHEMA_NAME = "schema_name"
_KEY_VERSION = "version"
_KEY_DOC_ITEMS = "doc_items"
_KEY_HEADINGS = "headings"
_KEY_CAPTIONS = "captions"
_KEY_ORIGIN = "origin"
_KEY_IMAGE_PATHS = "image_paths"

_logger = logging.getLogger(__name__)


class DocMeta(BaseMeta):
    """Data model for Hierarchical Chunker chunk metadata."""

    schema_name: Literal["docling_core.transforms.chunker.DocMeta"] = Field(
        default="docling_core.transforms.chunker.DocMeta",
        alias=_KEY_SCHEMA_NAME,
    )
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = (
        Field(
            default=_VERSION,
            alias=_KEY_VERSION,
        )
    )
    doc_items: list[DocItem] = Field(
        alias=_KEY_DOC_ITEMS,
        min_length=1,
    )
    headings: Optional[list[str]] = Field(
        default=None,
        alias=_KEY_HEADINGS,
        min_length=1,
    )
    captions: Optional[list[str]] = Field(
        default=None,
        alias=_KEY_CAPTIONS,
        min_length=1,
    )
    origin: Optional[DocumentOriginMod] = Field(
        default=None,
        alias=_KEY_ORIGIN,
    )
    image_paths: Optional[list[str]] = Field(
        default=None,
        alias=_KEY_IMAGE_PATHS,
        min_length=1,
    )

    excluded_embed: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]
    excluded_llm: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]

    @field_validator(_KEY_VERSION)
    @classmethod
    def check_version_is_compatible(cls, v: str) -> str:
        """Check if this meta item version is compatible with current version."""
        current_match = re.match(VERSION_PATTERN, _VERSION)
        doc_match = re.match(VERSION_PATTERN, v)
        if (
            doc_match is None
            or current_match is None
            or doc_match["major"] != current_match["major"]
            or doc_match["minor"] > current_match["minor"]
        ):
            raise ValueError(f"incompatible version {v} with schema version {_VERSION}")
        else:
            return _VERSION


class DocChunk(BaseChunk):
    """Data model for document chunks."""

    meta: DocMeta

class HybridChunkerWithImages(BaseChunker):
    r"""Chunker doing tokenization-aware refinements on top of document layout chunking.
    
    This chunker extends the HybridChunker functionality with support for image paths and
    preserves Markdown formatting for tables and lists.

    Args:
        tokenizer: The tokenizer to use; either instantiated object or name or path of
            respective pretrained model
        max_tokens: The maximum number of tokens per chunk. If not set, limit is
            resolved from the tokenizer
        merge_peers: Whether to merge undersized chunks sharing same relevant metadata
        merge_list_items (bool): Whether to merge successive list items. 
            Defaults to True.
        delim (str): Delimiter to use for merging text. Defaults to "\n".
        handle_pictures (bool): Whether to handle picture items.
            Defaults to True.
        image_mode (ImageRefMode): Mode for handling image references.
            Defaults to ImageRefMode.REFERENCED.
        image_output_dir (Union[str, Path]): Directory to save extracted images.
            Defaults to "extracted_images".
        keep_table_md (bool): Whether to keep tables in Markdown format instead 
            of triplet serialization. Defaults to True.
        escaping_underscores (bool): Whether to escape underscores in text content.
            Defaults to True.
        preserve_headers (bool): Whether to include header formatting in chunks.
            Defaults to True.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Union[PreTrainedTokenizerBase, str] = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = None  
    merge_peers: bool = True
    merge_list_items: bool = True
    handle_pictures: bool = True
    image_mode: ImageRefMode = ImageRefMode.REFERENCED
    delim: str = "\n"
    image_output_dir: Union[str, Path] = "extracted_images"
    keep_table_md: bool = True  
    escaping_underscores: bool = False  
    preserve_headers: bool = False 

    @model_validator(mode="after")
    def _patch_tokenizer_and_max_tokens(self) -> Self:
        self._tokenizer = (
            self.tokenizer
            if isinstance(self.tokenizer, PreTrainedTokenizerBase)
            else AutoTokenizer.from_pretrained(self.tokenizer)
        )
        if self.max_tokens is None:
            self.max_tokens = TypeAdapter(PositiveInt).validate_python(
                self._tokenizer.model_max_length
            )
        
        # Ensure image_output_dir is a Path object
        if isinstance(self.image_output_dir, str):
            self.image_output_dir = Path(self.image_output_dir)
            
        return self


    # Utility function to escape underscores but preserve them in URLs and LaTeX
    def _escape_underscores(self, text: str) -> str:
        """Escape underscores but leave them intact in URLs and LaTeX equations."""
        if not self.escaping_underscores:
            return text
            
        # Identify URL patterns and LaTeX equations
        url_pattern = r"!\[.*?\]\((.*?)\)"
        latex_pattern = r"\$\$?(?:\\.|[^$\\])*\$\$?"
        combined_pattern = f"({url_pattern})|({latex_pattern})"

        parts = []
        last_end = 0

        for match in re.finditer(combined_pattern, text):
            # Text to add before the special pattern (needs to be escaped)
            before_match = text[last_end : match.start()]
            parts.append(re.sub(r"(?<!\\)_", r"\_", before_match))

            # Add the special pattern part (do not escape)
            parts.append(match.group(0))
            last_end = match.end()

        # Add the final part of the text (which needs to be escaped)
        if last_end < len(text):
            parts.append(re.sub(r"(?<!\\)_", r"\_", text[last_end:]))

        return "".join(parts)
    
    def _markdown_serialize(self, table_df: DataFrame) -> str:
        """Serialize a table DataFrame into a Markdown table.
        
        This method follows a similar approach to export_to_markdown, replacing
        newlines in cells and using tabulate for cleaner formatting.
        """        
        # Convert DataFrame to a list of lists (similar to the grid in export_to_markdown)
        table = []
        
        # Then add rows
        for _, row in table_df.iterrows():
            tmp = []
            for val in row:
                # Replace newlines in cell text to prevent breaking markdown tables
                text = str(val).strip()
                text = text.replace("\n", " ")
                tmp.append(text)
            table.append(tmp)
        
        # Generate the markdown table
        if len(table) > 0 and len(table[0]) > 0:
            try:
                # First attempt with default settings
                md_table = tabulate(table, headers=table[0], tablefmt="github")
            except ValueError:
                # Fall back to disable_numparse if the first attempt fails
                md_table = tabulate(
                    table,
                    headers=table[0],
                    tablefmt="github",
                    disable_numparse=True
                )
        else:
            md_table = ""
        
        return md_table

    def _process_list_items(self, list_items: list[TextItem], indent: int = 4) -> str:
        """Process list items to maintain proper Markdown formatting.
        
        This method formats a list of TextItems as Markdown, respecting
        list nesting levels and indentation similar to export_to_markdown.
        """
        if not list_items:
            #print("No list items to process")
            return ""
            
        #print(f"Processing {len(list_items)} list items")
        
        md_lines = []
        list_nesting_level = 1  # Start with level 1 for simplicity
        
        # Group items by their parent list and level
        for i, item in enumerate(list_items):
            # Calculate indent based on list_nesting_level
            list_indent = " " * (indent * (list_nesting_level - 1))
            
            marker = ""
            if isinstance(item, ListItem) and item.enumerated:
                marker = item.marker
                #print(f"Item {i}: Enumerated with marker '{marker}'")
            else:
                marker = "-"  # Markdown uses dash as item marker
                #print(f"Item {i}: Using default marker '-'")
                
            text = f"{list_indent}{marker} {item.text}"
            md_lines.append(text)
            #print(f"Formatted: '{text}'")
        
        result = self.delim.join(md_lines)
        #print(f"Final list text: '{result}'")
        return result

    def _process_document(self, dl_doc: DLDocument) -> Iterator[DocChunk]:
        """Process the document and generate initial chunks."""
        heading_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []
        
        # Initialize list-related variables
        in_list = False
        list_nesting_level = 0
        current_list_group = None
        
        for item, level in dl_doc.iterate_items():
            captions = None
            image_paths = None
            
            if isinstance(item, DocItem):
                
                # Handle list items merging
                if self.merge_list_items:
                    # Handle list groups to track nesting
                    if isinstance(item, GroupItem) and item.label in [
                        GroupLabel.LIST,
                        GroupLabel.ORDERED_LIST,
                    ]:
                        # Entering a new list
                        list_nesting_level += 1
                        in_list = True
                        continue
                    
                    # Check if we're exiting a list group based on level change
                    if in_list and level < list_nesting_level:
                        # Calculate how many levels we've exited
                        level_difference = list_nesting_level - level
                        # Decrement list_nesting_level for each level we've exited
                        list_nesting_level = max(0, list_nesting_level - level_difference)
                        
                        # If we've completely exited all lists, process the accumulated items
                        if list_nesting_level == 0 and list_items:
                            # Format list items with proper markdown
                            list_text = self._process_list_items(list_items, indent=4)
                            yield DocChunk(
                                text=list_text,
                                meta=DocMeta(
                                    doc_items=list_items,
                                    headings=[
                                        heading_by_level[k]
                                        for k in sorted(heading_by_level)
                                    ]
                                    or None,
                                    origin=dl_doc.origin,
                                ),
                            )
                            list_items = []  # reset
                            in_list = False  # Reset in_list after processing
                    
                    # Handle individual list items
                    if isinstance(
                        item, ListItem
                    ) or (
                        isinstance(item, TextItem)
                        and item.label == DocItemLabel.LIST_ITEM
                    ):
                        list_items.append(item)
                        in_list = True  # Set in_list to True when processing list items
                        continue
                    elif list_items:  # need to yield
                        # Format list items with proper markdown
                        list_text = self._process_list_items(list_items, indent=4)
                        yield DocChunk(
                            text=list_text,
                            meta=DocMeta(
                                doc_items=list_items,
                                headings=[
                                    heading_by_level[k]
                                    for k in sorted(heading_by_level)
                                ]
                                or None,
                                origin=dl_doc.origin,
                            ),
                        )
                        list_items = []  # reset
                        in_list = False  # Reset in_list after processing

                # Handle section headers
                if (isinstance(item, SectionHeaderItem) or 
                    (isinstance(item, TextItem) and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE])):
                    
                    # If we were in a list, flush it before the header
                    if in_list and list_items:
                        list_text = self._process_list_items(list_items, indent=4)
                        yield DocChunk(
                            text=list_text,
                            meta=DocMeta(
                                doc_items=list_items.copy(),
                                headings=[heading_by_level[k] for k in sorted(heading_by_level)] or None,
                                origin=dl_doc.origin,
                            ),
                        )
                        # Reset list state
                        list_items = []
                        in_list = False
                        list_nesting_level = 0
                        current_list_group = None
                    
                    header_level = (
                        item.level
                        if isinstance(item, SectionHeaderItem)
                        else (0 if item.label == DocItemLabel.TITLE else 1)
                    )
                    heading_by_level[header_level] = item.text

                    # Remove headings of higher level as they went out of scope
                    keys_to_del = [k for k in heading_by_level if k > header_level]
                    for k in keys_to_del:
                        heading_by_level.pop(k, None)
                    
                    """
                    # If preserve_headers is enabled, yield a chunk for this header
                    if self.preserve_headers:
                        if isinstance(item, SectionHeaderItem):
                            # Ensure at least 2 '#' for consistency with export_to_markdown
                            marker = "#" * max(2, header_level)
                            formatted_text = f"{marker} {item.text}"
                        elif item.label == DocItemLabel.TITLE:
                            formatted_text = f"# {item.text}"
                        else:
                            marker = "#" * max(2, header_level)
                            formatted_text = f"{marker} {item.text}"
                            
                        yield DocChunk(
                            text=formatted_text,
                            meta=DocMeta(
                                doc_items=[item],
                                headings=[heading_by_level[k] for k in sorted(heading_by_level)] or None,
                                origin=dl_doc.origin,
                            ),
                        )
                    """
                    continue
                    
                # Process different item types
                if isinstance(item, TextItem) or (
                    (not self.merge_list_items) and isinstance(item, ListItem)
                ):
                    text = item.text
                    #print("hors list",item.label, item.text)
                    # Apply underscore escaping if needed
                    if self.escaping_underscores:
                        text = self._escape_underscores(text)
                    
                elif isinstance(item, TableItem):
                    table_df = item.export_to_dataframe()
                    if table_df.shape[0] < 1 or table_df.shape[1] < 2:
                        # at least two cols needed, as first column contains row headers
                        continue
                    
                    # Always use markdown format if keep_table_md is True
                    if self.keep_table_md:
                        text = self._markdown_serialize(table_df=table_df)
                    else:
                        text = self._triplet_serialize(table_df=table_df)
                        
                    captions = [
                        c.text for c in [r.resolve(dl_doc) for r in item.captions]
                    ] or None
                    
                elif self.handle_pictures and isinstance(item, PictureItem):
                    # Handle picture items with proper markdown image format
                    caption_text = item.caption_text(dl_doc) if len(item.captions) > 0 else ""
                    img_path = self._get_image_path(item)
                    
                    if img_path:
                        # Use proper markdown image syntax
                        alt_text = caption_text if caption_text else "Image"
                        text = f"![{alt_text}]({img_path})"
                        image_paths = [img_path]
                    else:
                        text = f"[Image: {caption_text}]" if caption_text else "[Image]"
                        
                    # Extract captions
                    captions = [
                        c.text for c in [r.resolve(dl_doc) for r in item.captions]
                    ] or None
                    
                else:
                    continue

                # Create and yield chunk
                yield DocChunk(
                    text=text,
                    meta=DocMeta(
                        doc_items=[item],
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                        or None,
                        captions=captions,
                        image_paths=image_paths,
                        origin=dl_doc.origin,
                    ),
                )

        # Process any remaining list items
        if self.merge_list_items and list_items:
            list_text = self._process_list_items(list_items, indent=4)
            yield DocChunk(
                text=list_text,
                meta=DocMeta(
                    doc_items=list_items,
                    headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                    or None,
                    origin=dl_doc.origin,
                ),
            )

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        r"""Chunk the provided document with token awareness and image handling.

        Args:
            dl_doc (DLDocument): document to chunk
            **kwargs: Additional keyword arguments, can include:
                - image_output_dir: Override the default image output directory
                - escaping_underscores: Override the default underscore escaping behavior
                - include_headings_in_chunks: Override whether to include header formatting in chunks

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        # Override image_output_dir if provided in kwargs
        if "image_output_dir" in kwargs:
            original_output_dir = self.image_output_dir
            if isinstance(kwargs["image_output_dir"], str):
                self.image_output_dir = Path(kwargs["image_output_dir"])
            else:
                self.image_output_dir = kwargs["image_output_dir"]
                
        # Override escaping_underscores if provided in kwargs
        original_escaping_underscores = None
        if "escaping_underscores" in kwargs:
            original_escaping_underscores = self.escaping_underscores
            self.escaping_underscores = kwargs["escaping_underscores"]
            
        # Override include_headings_in_chunks if provided in kwargs
        original_include_headings = None
        if "include_headings_in_chunks" in kwargs:
            original_include_headings = self.include_headings_in_chunks
            self.include_headings_in_chunks = kwargs["include_headings_in_chunks"]
        
        try:
            # First, process document to get initial chunks with proper markdown formatting
            initial_chunks = list(self._process_document(dl_doc))
            
            # Then apply token-aware processing
            # 1. Split by doc items if needed
            res = [x for c in initial_chunks for x in self._split_by_doc_items(c)]
            
            # 2. Split using plain text if still needed
            res = [x for c in res for x in self._split_using_plain_text(c)]
            
            # 3. Merge chunks with matching metadata if desired
            if self.merge_peers:
                res = self._merge_chunks_with_matching_metadata(res)
                
            # 4. Filter out empty chunks and normalize newlines
            filtered_res = []
            for chunk in res:
                if isinstance(chunk, DocChunk):
                    # Skip chunks with empty text
                    if self._is_empty_text(chunk.text):
                        continue
                    
                    # Normalize newlines in text
                    chunk.text = self._normalize_newlines(chunk.text)
                    
                    filtered_res.append(chunk)
                else:
                    # For any other types of chunks, keep as is
                    filtered_res.append(chunk)
                
            return iter(filtered_res)
        finally:
            # Restore original settings if they were overridden
            if "image_output_dir" in kwargs:
                self.image_output_dir = original_output_dir
                
            if original_escaping_underscores is not None:
                self.escaping_underscores = original_escaping_underscores
                
            if original_include_headings is not None:
                self.include_headings_in_chunks = original_include_headings


    def _count_text_tokens(self, text: Optional[Union[str, list[str]]]) -> int:
        """Count tokens in text."""
        if text is None:
            return 0
        elif isinstance(text, list):
            total = 0
            for t in text:
                total += self._count_text_tokens(t)
            return total
        return len(self._tokenizer.tokenize(text, max_length=None))

    class _ChunkLengthInfo(BaseModel):
        """Information about chunk token lengths."""
        total_len: int
        text_len: int
        other_len: int

    def _count_chunk_tokens(self, doc_chunk: DocChunk) -> int:
        """Count tokens in a chunk."""
        ser_txt = self.serialize(chunk=doc_chunk)
        return len(self._tokenizer.tokenize(text=ser_txt, max_length=None))

    def _doc_chunk_length(self, doc_chunk: DocChunk) -> _ChunkLengthInfo:
        """Get token length information for a chunk."""
        text_length = self._count_text_tokens(doc_chunk.text)
        total = self._count_chunk_tokens(doc_chunk=doc_chunk)
        return self._ChunkLengthInfo(
            total_len=total,
            text_len=text_length,
            other_len=total - text_length,
        )

    def _make_chunk_from_doc_items(
        self, doc_chunk: DocChunk, window_start: int, window_end: int
    ) -> DocChunk:
        """Create a new chunk from a subset of doc items, preserving list formatting."""
        doc_items = doc_chunk.meta.doc_items[window_start : window_end + 1]
        meta = DocMeta(
            doc_items=doc_items,
            headings=doc_chunk.meta.headings,
            captions=doc_chunk.meta.captions,
            origin=doc_chunk.meta.origin,
            image_paths=doc_chunk.meta.image_paths,  # Preserve image paths
        )
        
        # Check if this chunk contains list items
        contains_list_items = any(
            isinstance(item, ListItem) or 
            (isinstance(item, TextItem) and item.label == DocItemLabel.LIST_ITEM)
            for item in doc_items
        )
        
        if contains_list_items:
            # Use _process_list_items for list formatting
            list_text = self._process_list_items(
                [item for item in doc_items if isinstance(item, TextItem)], 
                indent=4
            )
            window_text = list_text
        else:
            # Use regular joining for non-list items
            window_text = (
                doc_chunk.text
                if len(doc_chunk.meta.doc_items) == 1
                else self.delim.join(
                    [
                        doc_item.text
                        for doc_item in doc_items
                        if isinstance(doc_item, TextItem)
                    ]
                )
            )
        
        new_chunk = DocChunk(text=window_text, meta=meta)
        return new_chunk

    def _split_by_doc_items(self, doc_chunk: DocChunk) -> list[DocChunk]:
        """Split a chunk by document items while respecting token limits."""
        chunks = []
        window_start = 0
        window_end = 0  # an inclusive index
        num_items = len(doc_chunk.meta.doc_items)
        while window_end < num_items:
            new_chunk = self._make_chunk_from_doc_items(
                doc_chunk=doc_chunk,
                window_start=window_start,
                window_end=window_end,
            )
            if self._count_chunk_tokens(doc_chunk=new_chunk) <= self.max_tokens:
                if window_end < num_items - 1:
                    window_end += 1
                    # Still room left to add more to this chunk AND still at least one
                    # item left
                    continue
                else:
                    # All the items in the window fit into the chunk and there are no
                    # other items left
                    window_end = num_items  # signalizing the last loop
            elif window_start == window_end:
                # Only one item in the window and it doesn't fit into the chunk. So
                # we'll just make it a chunk for now and it will get split in the
                # plain text splitter.
                window_end += 1
                window_start = window_end
            else:
                # Multiple items in the window but they don't fit into the chunk.
                # However, the existing items must have fit or we wouldn't have
                # gotten here. So we put everything but the last item into the chunk
                # and then start a new window INCLUDING the current window end.
                new_chunk = self._make_chunk_from_doc_items(
                    doc_chunk=doc_chunk,
                    window_start=window_start,
                    window_end=window_end - 1,
                )
                window_start = window_end
            chunks.append(new_chunk)
        return chunks

    def _split_using_plain_text(self, doc_chunk: DocChunk) -> list[DocChunk]:
        """Split a chunk into smaller chunks based on token limits while preserving list formatting."""
        lengths = self._doc_chunk_length(doc_chunk)
        if lengths.total_len <= self.max_tokens:
            return [DocChunk(**doc_chunk.export_json_dict())]
        
        # Check if this is a list chunk by examining if it contains list items
        contains_list_items = any(
            isinstance(item, ListItem) or 
            (isinstance(item, TextItem) and item.label == DocItemLabel.LIST_ITEM)
            for item in doc_chunk.meta.doc_items
        )
        
        # If this chunk contains list items, we need special handling
        if contains_list_items:
            # Check if we can identify list markers in the text
            lines = doc_chunk.text.split('\n')
            list_pattern = re.compile(r'^\s*(-|\d+\.)\s')
            
            # If most lines look like list items, use line-based splitting
            list_line_count = sum(1 for line in lines if list_pattern.match(line))
            if list_line_count > 0 and list_line_count >= len(lines) / 2:
                print(f"Found {list_line_count} list lines out of {len(lines)} total lines")
                
                # Split the text by lines while preserving list markers
                available_length = self.max_tokens - lengths.other_len
                chunks = []
                current_chunk_lines = []
                current_tokens = 0
                
                for line in lines:
                    # Skip empty lines
                    if not line.strip():
                        if current_chunk_lines:
                            current_chunk_lines.append(line)
                        continue
                        
                    line_tokens = self._count_text_tokens(line + '\n')
                    
                    # If adding this line would exceed token limit, create a new chunk
                    if current_tokens + line_tokens > available_length and current_chunk_lines:
                        chunk_text = '\n'.join(current_chunk_lines)
                        chunks.append(DocChunk(text=chunk_text, meta=doc_chunk.meta))
                        current_chunk_lines = []
                        current_tokens = 0
                    
                    # Add line to current chunk
                    current_chunk_lines.append(line)
                    current_tokens += line_tokens
                
                # Add any remaining lines as the last chunk
                if current_chunk_lines:
                    chunk_text = '\n'.join(current_chunk_lines)
                    chunks.append(DocChunk(text=chunk_text, meta=doc_chunk.meta))
                    
                print(f"Created {len(chunks)} chunks from list")
                for i, chunk in enumerate(chunks):
                    print(f"Chunk {i}: {chunk.text[:100]}...")
                
                return chunks
        
        # For non-list chunks or if list detection failed, use default semantic chunking
        available_length = self.max_tokens - lengths.other_len
        sem_chunker = semchunk.chunkerify(
            self._tokenizer, chunk_size=available_length
        )
        if available_length <= 0:
            warnings.warn(
                f"Headers and captions for this chunk are longer than the total amount of size for the chunk, chunk will be ignored: {doc_chunk.text=}"  # noqa
            )
            return []
        text = doc_chunk.text
        segments = sem_chunker.chunk(text)
        chunks = [DocChunk(text=s, meta=doc_chunk.meta) for s in segments]
        return chunks
            
    def _merge_chunks_with_matching_metadata(self, chunks: list[DocChunk]) -> list[DocChunk]:
        """Merge adjacent chunks that share metadata and fit within token limits."""
        output_chunks = []
        window_start = 0
        window_end = 0  # an inclusive index
        num_chunks = len(chunks)
        while window_end < num_chunks:
            chunk = chunks[window_end]
            # Only consider headings and captions for metadata matching, not image_paths
            metadata_keys = (
                chunk.meta.headings, 
                chunk.meta.captions
            )
            ready_to_append = False
            if window_start == window_end:
                current_metadata_keys = metadata_keys
                current_image_paths = chunk.meta.image_paths
                window_end += 1
                first_chunk_of_window = chunk
            else:
                chks = chunks[window_start : window_end + 1]
                doc_items = [it for chk in chks for it in chk.meta.doc_items]
                
                # Combine image paths from all chunks in the window
                combined_image_paths = current_image_paths
                if chunk.meta.image_paths:
                    if combined_image_paths:
                        # Create a new list with all image paths
                        combined_image_paths = list(combined_image_paths) + list(chunk.meta.image_paths)
                    else:
                        combined_image_paths = chunk.meta.image_paths
                
                candidate = DocChunk(
                    text=self.delim.join([chk.text for chk in chks]),
                    meta=DocMeta(
                        doc_items=doc_items,
                        headings=current_metadata_keys[0],
                        captions=current_metadata_keys[1],
                        image_paths=combined_image_paths,  # Use combined image paths
                        origin=chunk.meta.origin,
                    ),
                )
                if (
                    metadata_keys == current_metadata_keys
                    and self._count_chunk_tokens(doc_chunk=candidate) <= self.max_tokens
                ):
                    # there is room to include the new chunk so add it to the window and
                    # continue
                    window_end += 1
                    new_chunk = candidate
                    # Update the current_image_paths for future combinations
                    current_image_paths = combined_image_paths
                else:
                    ready_to_append = True
            if ready_to_append or window_end == num_chunks:
                # no more room OR the start of new metadata.  Either way, end the block
                # and use the current window_end as the start of a new block
                if window_start + 1 == window_end:
                    # just one chunk so use it as is
                    output_chunks.append(first_chunk_of_window)
                else:
                    output_chunks.append(new_chunk)
                # no need to reset window_text, etc. because that will be reset in the
                # next iteration in the if window_start == window_end block
                window_start = window_end

        return output_chunks
                
    def _triplet_serialize(self, table_df: DataFrame) -> str:
        """Serialize a table DataFrame into a string of triplets."""
        # copy header as first row and shift all rows by one
        table_df.loc[-1] = table_df.columns  # type: ignore[call-overload]
        table_df.index = table_df.index + 1
        table_df = table_df.sort_index()

        rows = [str(item).strip() for item in table_df.iloc[:, 0].to_list()]
        cols = [str(item).strip() for item in table_df.iloc[0, :].to_list()]

        nrows = table_df.shape[0]
        ncols = table_df.shape[1]
        texts = [
            f"{rows[i]}, {cols[j]} = {str(table_df.iloc[i, j]).strip()}"
            for i in range(1, nrows)
            for j in range(1, ncols)
        ]
        output_text = ". ".join(texts)

        return output_text

    
    def _get_image_path(self, picture_item: PictureItem) -> Optional[str]:
        """Extract image path from a PictureItem."""
        if not picture_item.image:
            return None
            
        image_ref = picture_item.image
        
        # Create output directory for images using the configurable path
        output_dir = self.image_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle data URIs by saving to disk
        if isinstance(image_ref.uri, AnyUrl) and image_ref.uri.scheme == "data":
            # Use the underlying PIL image to save
            if image_ref.pil_image:
                # Generate a unique filename
                hex_hash = picture_item._image_to_hexhash()
                if hex_hash:
                    image_path = f"image_{hex_hash}.png"
                    image_ref.pil_image.save(output_dir /image_path)
                    return str(image_path)
        
        # For file:// URIs, Path objects, etc.
        if isinstance(image_ref.uri, AnyUrl):
            if image_ref.uri.scheme == "file" and image_ref.uri.path:
                from urllib.parse import unquote
                return unquote(str(image_ref.uri.path))
        elif isinstance(image_ref.uri, Path):
            return str(image_ref.uri)
        
        return None

    def _is_empty_text(self, text: str) -> bool:
        """Check if text is empty or contains only whitespace/newlines."""
        if not text:
            return True
        # Check if string contains only whitespace characters
        return text.isspace()
    
    def _normalize_newlines(self, text: str) -> str:
        """Normalize multiple consecutive newlines to a maximum of two."""
        # Replace 3 or more consecutive newlines with just 2
        import re
        return re.sub(r'\n{3,}', '\n\n', text)