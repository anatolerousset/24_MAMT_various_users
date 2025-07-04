"""Duplicate document management for vector stores."""

import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
import tempfile

from config.config import Config
from utils.blob_utils import BlobStorageManager

_log = logging.getLogger(__name__)

class DuplicateManager:
    """Manages duplicate detection and removal in vector stores."""
    
    def __init__(self, qdrant_url: str = None):
        """
        Initialize the duplicate manager.
        
        Args:
            qdrant_url: URL for Qdrant instance
        """
        self.qdrant_url = qdrant_url or Config.QDRANT_URL
        self.client = QdrantClient(url=self.qdrant_url)
        self.batch_size = Config.DUPLICATE_BATCH_SIZE
        self.blob_manager = BlobStorageManager()
    
    def parse_date(self, date_str: str) -> datetime:
        """
        Parse date string into datetime object.
        Handle multiple date formats that might be present in the metadata.
        """
        if not date_str:
            return datetime.min
        
        # Common date formats to try
        date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If all formats fail, try to extract just the date part
        try:
            date_part = date_str.split('T')[0] if 'T' in date_str else date_str.split(' ')[0]
            return datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            _log.warning(f"Could not parse date: {date_str}")
            return datetime.min

    def get_document_date(self, metadata: Dict[str, Any]) -> datetime:
        """
        Extract the most relevant date from document metadata.
        Priority: last_modified_date > origin.last_modified_date > created_date > datetime.min
        """
        dl_meta = metadata.get('dl_meta', {})
        
        # Try last_modified_date first (direct field)
        if 'last_modified_date' in dl_meta:
            return self.parse_date(dl_meta['last_modified_date'])
        
        # Try origin.last_modified_date as backup
        if 'origin' in dl_meta:
            origin = dl_meta['origin']
            if isinstance(origin, dict):
                # Check for last_modified_date field in origin
                if 'last_modified_date' in origin:
                    return self.parse_date(str(origin['last_modified_date']))
                # Also check for other possible date fields in origin
                for date_field in ['last_modified', 'modified_date', 'date']:
                    if date_field in origin:
                        return self.parse_date(str(origin[date_field]))
            elif isinstance(origin, str):
                # Try to parse origin as date if it's a string
                return self.parse_date(origin)
        
        # Try other common date fields in dl_meta
        for date_field in ['created_date', 'date', 'timestamp']:
            if date_field in dl_meta:
                return self.parse_date(dl_meta[date_field])
        
        # Try date fields in root metadata (outside dl_meta)
        for date_field in ['last_modified_date', 'created_date', 'date', 'timestamp']:
            if date_field in metadata:
                return self.parse_date(metadata[date_field])
        
        _log.warning(f"No date found in metadata: {metadata}")
        return datetime.min

    def get_all_points(self, collection_name: str) -> List[models.Record]:
        """
        Retrieve all points from the collection with their vectors and metadata.
        """
        _log.info(f"Retrieving all points from collection: {collection_name}")
        
        all_points = []
        offset = None
        
        while True:
            # Get batch of points
            response = self.client.scroll(
                collection_name=collection_name,
                limit=self.batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True
            )
            
            points, next_offset = response
            all_points.extend(points)
            
            if next_offset is None:
                break
            offset = next_offset
            
            _log.info(f"Retrieved {len(all_points)} points so far...")
        
        _log.info(f"Total points retrieved: {len(all_points)}")
        return all_points

    def find_duplicate_groups(self, points: List[models.Record], similarity_threshold: float = 0.95) -> List[List[int]]:
        """
        Find groups of duplicate documents based on cosine similarity of finetuned_dense vectors.
        
        Args:
            points: List of document points
            similarity_threshold: Cosine similarity threshold for duplicates
            
        Returns:
            List of lists, where each inner list contains indices of duplicate documents
        """
        _log.info(f"Finding duplicate groups with similarity threshold: {similarity_threshold}")
        
        # Extract finetuned_dense vectors
        vectors = []
        valid_indices = []
        
        for i, point in enumerate(points):
            if 'finetuned_dense' in point.vector:
                vectors.append(point.vector['finetuned_dense'])
                valid_indices.append(i)
            else:
                _log.warning(f"Point {point.id} missing finetuned_dense vector")
        
        if not vectors:
            _log.error("No valid finetuned_dense vectors found!")
            return []
        
        vectors = np.array(vectors)
        _log.info(f"Computing cosine similarity for {len(vectors)} vectors...")
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(vectors)
        
        # Find duplicate groups
        duplicate_groups = []
        processed = set()
        
        for i in range(len(similarity_matrix)):
            if valid_indices[i] in processed:
                continue
            
            # Find all documents similar to this one
            similar_indices = []
            for j in range(len(similarity_matrix)):
                if similarity_matrix[i][j] >= similarity_threshold:
                    similar_indices.append(valid_indices[j])
            
            if len(similar_indices) > 1:
                duplicate_groups.append(similar_indices)
                processed.update(similar_indices)
        
        _log.info(f"Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups

    def select_documents_to_keep(self, points: List[models.Record], duplicate_groups: List[List[int]]) -> Tuple[List[str], List[str], List[Dict]]:
        """
        For each duplicate group, select the most recent document to keep.
        
        Args:
            points: List of document points
            duplicate_groups: Groups of duplicate document indices
            
        Returns:
            Tuple of (ids_to_keep, ids_to_delete, duplicate_groups_info)
        """
        _log.info("Selecting documents to keep based on recency...")
        
        ids_to_delete = []
        ids_to_keep = []
        duplicate_groups_info = []
        
        # Track all points in duplicate groups
        points_in_groups = set()
        for group in duplicate_groups:
            points_in_groups.update(group)
        
        # Process each duplicate group
        for group_idx, group in enumerate(duplicate_groups):
            _log.info(f"Processing duplicate group {group_idx + 1}/{len(duplicate_groups)} with {len(group)} documents")
            
            # Get documents in this group with their dates
            group_docs = []
            for point_idx in group:
                point = points[point_idx]
                doc_date = self.get_document_date(point.payload.get('metadata', {}))
                metadata = point.payload.get('metadata', {})
                source = metadata.get('source', 'unknown')
                group_docs.append((point.id, doc_date, point_idx, source))
            
            # Sort by date (most recent first)
            group_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Keep the most recent, delete the rest
            most_recent_id = group_docs[0][0]
            ids_to_keep.append(most_recent_id)
            
            # Prepare group info for report
            group_info = {
                "group_id": group_idx + 1,
                "total_documents": len(group_docs),
                "kept_document": {
                    "id": group_docs[0][0],
                    "date": group_docs[0][1].isoformat() if group_docs[0][1] != datetime.min else "unknown",
                    "source": group_docs[0][3]
                },
                "deleted_documents": []
            }
            
            for doc_id, doc_date, _, source in group_docs[1:]:
                ids_to_delete.append(doc_id)
                group_info["deleted_documents"].append({
                    "id": doc_id,
                    "date": doc_date.isoformat() if doc_date != datetime.min else "unknown",
                    "source": source
                })
            
            duplicate_groups_info.append(group_info)
            
            _log.info(f"Group {group_idx + 1}: Keeping {most_recent_id} (date: {group_docs[0][1]}), deleting {len(group_docs) - 1} duplicates")
        
        # Keep all documents that are not part of any duplicate group
        for i, point in enumerate(points):
            if i not in points_in_groups:
                ids_to_keep.append(point.id)
        
        _log.info(f"Total documents to keep: {len(ids_to_keep)}")
        _log.info(f"Total documents to delete: {len(ids_to_delete)}")
        
        return ids_to_keep, ids_to_delete, duplicate_groups_info

    def delete_duplicate_documents(self, collection_name: str, ids_to_delete: List[str], dry_run: bool = True):
        """
        Delete duplicate documents from the collection.
        
        Args:
            collection_name: Name of the collection
            ids_to_delete: List of document IDs to delete
            dry_run: If True, only simulate the deletion without actually deleting
        """
        if not ids_to_delete:
            _log.info("No documents to delete")
            return
        
        if dry_run:
            _log.info(f"DRY RUN: Would delete {len(ids_to_delete)} documents")
            _log.info("Sample IDs to delete:")
            for i, doc_id in enumerate(ids_to_delete[:5]):
                _log.info(f"  {i+1}. {doc_id}")
            if len(ids_to_delete) > 5:
                _log.info(f"  ... and {len(ids_to_delete) - 5} more")
            return
        
        _log.info(f"Deleting {len(ids_to_delete)} duplicate documents...")
        
        # Delete in batches to avoid overwhelming the server
        batch_size = Config.DUPLICATE_BATCH_SIZE
        for i in tqdm(range(0, len(ids_to_delete), batch_size), desc="Deleting duplicates"):
            batch_ids = ids_to_delete[i:i+batch_size]
            
            try:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=batch_ids)
                )
                _log.info(f"Deleted batch {i//batch_size + 1}: {len(batch_ids)} documents")
            except Exception as e:
                _log.error(f"Error deleting batch {i//batch_size + 1}: {e}")
                # Continue with next batch instead of failing completely
        
        _log.info("Deletion complete!")

    def save_deletion_report(self, ids_to_keep: List[str], ids_to_delete: List[str], 
                           points: List[models.Record], duplicate_groups_info: List[Dict], 
                           filename: str = None):
        """
        Save a focused report showing duplicate groups to blob storage.
        
        Args:
            ids_to_keep: List of document IDs to keep
            ids_to_delete: List of document IDs to delete
            points: List of all document points
            duplicate_groups_info: Information about duplicate groups
            filename: Name of the report file
        """
        if filename is None:
            filename = f"deletion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        _log.info(f"Saving deletion report to blob storage: {filename}")
        
        report = {
            "summary": {
                "total_documents": len(points),
                "documents_to_keep": len(ids_to_keep),
                "documents_to_delete": len(ids_to_delete),
                "duplicate_groups_found": len(duplicate_groups_info),
                "deletion_percentage": (len(ids_to_delete) / len(points)) * 100 if points else 0
            },
            "duplicate_groups": duplicate_groups_info
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(report, temp_file, indent=2, ensure_ascii=False)
            temp_path = Path(temp_file.name)
        
        try:
            # Upload to blob storage
            self.blob_manager.upload_file_to_blob(
                temp_path, 
                Config.REPORTS_CONTAINER, 
                filename
            )
            _log.info(f"Deletion report saved to blob storage: {Config.REPORTS_CONTAINER}/{filename}")
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def remove_duplicates(self, collection_name: str, similarity_threshold: float = 0.95, 
                         dry_run: bool = True, save_report: bool = True) -> Dict[str, Any]:
        """
        Main function to remove near duplicates from Qdrant collection.
        [Implementation remains the same, just calls updated save_deletion_report]
        """
        _log.info("Starting duplicate removal process...")
        _log.info(f"Collection: {collection_name}")
        _log.info(f"Similarity threshold: {similarity_threshold}")
        _log.info(f"Dry run mode: {dry_run}")
        
        # Check if collection exists
        if not self.client.collection_exists(collection_name):
            _log.error(f"Collection {collection_name} does not exist!")
            return {"success": False, "error": f"Collection {collection_name} does not exist"}
        
        # Get collection info
        collection_info = self.client.get_collection(collection_name)
        _log.info(f"Collection info: {collection_info}")
        
        # Step 1: Retrieve all points
        points = self.get_all_points(collection_name)
        if not points:
            _log.error("No points found in collection!")
            return {"success": False, "error": "No points found in collection"}
        
        # Step 2: Find duplicate groups
        duplicate_groups = self.find_duplicate_groups(points, similarity_threshold)
        if not duplicate_groups:
            _log.info("No duplicates found!")
            return {
                "success": True, 
                "duplicates_found": False,
                "total_documents": len(points),
                "duplicate_groups": 0
            }
        
        # Step 3: Select which documents to keep
        ids_to_keep, ids_to_delete, duplicate_groups_info = self.select_documents_to_keep(points, duplicate_groups)
        
        # Step 4: Save deletion report to blob storage
        if save_report:
            self.save_deletion_report(ids_to_keep, ids_to_delete, points, duplicate_groups_info)
        
        # Step 5: Delete duplicates
        self.delete_duplicate_documents(collection_name, ids_to_delete, dry_run=dry_run)
        
        _log.info("Duplicate removal process completed!")
        
        if dry_run:
            _log.info("This was a dry run. Set dry_run=False to perform actual deletion.")
        
        return {
            "success": True,
            "duplicates_found": True,
            "total_documents": len(points),
            "documents_to_keep": len(ids_to_keep),
            "documents_to_delete": len(ids_to_delete),
            "duplicate_groups": len(duplicate_groups),
            "deletion_percentage": (len(ids_to_delete) / len(points)) * 100,
            "dry_run": dry_run
        }