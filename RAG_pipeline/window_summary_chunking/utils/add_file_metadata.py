"""
Add file-level metadata to chunks from generate_summary/metadata/ folder
Adds categories, user rights, etc. to chunk_text_with_metadata
"""
import os
import json


def format_metadata_fields(metadata, fields_to_include):
    """
    Format selected metadata fields into a readable string
    
    Args:
        metadata: Dictionary containing file-level metadata
        fields_to_include: List of field names to include
    
    Returns:
        Formatted string with metadata
    """
    lines = []
    
    for field in fields_to_include:
        if field not in metadata:
            continue
        
        value = metadata[field]
        
        # Format field name (convert snake_case to Title Case)
        field_name = field.replace('_', ' ').title()
        
        # Format value based on type
        if isinstance(value, list):
            if value:  # Only add if list is not empty
                lines.append(f"{field_name}:")
                for item in value:
                    lines.append(f"  - {item}")
        elif isinstance(value, dict):
            if value:  # Only add if dict is not empty
                lines.append(f"{field_name}:")
                for key, val in value.items():
                    lines.append(f"  {key}: {val}")
        elif isinstance(value, str) and value:  # Only add if string is not empty
            lines.append(f"{field_name}: {value}")
    
    return "\n".join(lines)


def add_file_metadata_to_chunks(
    chunks_folder,
    metadata_folder,
    output_folder,
    fields_to_include=None
):
    """
    Add file-level metadata to chunks
    
    Args:
        chunks_folder: Folder with chunk JSON files
        metadata_folder: Folder with metadata JSON files
        output_folder: Output folder for chunks with file metadata
        fields_to_include: List of metadata fields to include (None = use defaults)
    """
    
    # Default fields to include
    if fields_to_include is None:
        fields_to_include = [
            'categories_of_information_collected',
            'user_rights',
            'purposes_of_processing',
            'data_recipients',
            'security_measures',
            'retention_policies'
        ]
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all chunk files
    chunk_files = [f for f in os.listdir(chunks_folder) if f.endswith('.json')]
    
    print(f"Processing {len(chunk_files)} chunk files")
    print("=" * 70)
    
    for chunk_file in chunk_files:
        # Extract document name from chunk filename
        # e.g., "23andMe_with_summaries.json" -> "23andMe"
        doc_name = chunk_file.replace('_with_summaries.json', '').replace('.json', '')
        
        # Find corresponding metadata file
        metadata_file = os.path.join(metadata_folder, f"{doc_name}_metadata.json")
        
        if not os.path.exists(metadata_file):
            print(f"[WARNING] No metadata found for {doc_name}, skipping...")
            continue
        
        # Load metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Format file-level metadata
        file_metadata_text = format_metadata_fields(metadata, fields_to_include)
        
        # Load chunks
        chunk_path = os.path.join(chunks_folder, chunk_file)
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        # Add file-level metadata to each chunk
        for chunk_id, chunk_data in chunks.items():
            # Get original chunk_text (without any existing metadata)
            chunk_text = chunk_data.get('chunk_text', '')
            
            # Remove any existing metadata tags from chunk_text_with_metadata if it exists
            # This ensures we start fresh
            
            # Create NEW metadata section with ONLY file-level metadata
            enhanced_metadata = "<metadata>\n"
            
            # Add file-level metadata
            if file_metadata_text:
                enhanced_metadata += file_metadata_text + "\n"
            
            enhanced_metadata += "</metadata>"
            
            # Update chunk - use clean chunk_text + new metadata
            chunk_data['chunk_text_with_metadata'] = f"{chunk_text}\n\n{enhanced_metadata}"
            chunk_data['file_metadata'] = {k: metadata.get(k) for k in fields_to_include if k in metadata}
            
            # Remove window_chunks property if it exists
            if 'window_chunks' in chunk_data:
                del chunk_data['window_chunks']
        
        # Save updated chunks
        output_path = os.path.join(output_folder, chunk_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] {chunk_file}: {len(chunks)} chunks updated")
    
    print("\n" + "=" * 70)
    print("[OK] All chunks updated with file-level metadata!")


if __name__ == "__main__":
    import sys
    
    # Default paths
    chunks_folder = "RAG_pipeline/window_summary_chunking/chunks_with_summaries"
    metadata_folder = "RAG_pipeline/generate_summary/metadata"
    output_folder = "RAG_pipeline/window_summary_chunking/chunks_with_full_metadata"
    
    # Allow custom paths
    if len(sys.argv) > 1:
        chunks_folder = sys.argv[1]
    if len(sys.argv) > 2:
        metadata_folder = sys.argv[2]
    if len(sys.argv) > 3:
        output_folder = sys.argv[3]
    
    print("Adding File-Level Metadata to Chunks")
    print("=" * 70)
    print(f"Chunks folder: {chunks_folder}")
    print(f"Metadata folder: {metadata_folder}")
    print(f"Output folder: {output_folder}")
    print()
    
    # Fields to include - ONLY categories and user rights
    fields = [
        'categories_of_information_collected',
        'user_rights'
    ]
    
    add_file_metadata_to_chunks(
        chunks_folder=chunks_folder,
        metadata_folder=metadata_folder,
        output_folder=output_folder,
        fields_to_include=fields
    )
