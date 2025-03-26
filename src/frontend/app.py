"""
Gradio web interface for the document processing system.
"""
import os
import io
import json
import logging
import tempfile
import requests
from typing import Dict, Any, List, Optional, Tuple

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")

def process_document(
    file,
    perform_ner,
    perform_classification,
    perform_summarization,
    summary_length,
    summary_method,
    preserve_structure,
    clean_text
) -> Tuple:
    """
    Process a document using the API and format the results for display.
    
    Args:
        file: Uploaded file
        perform_ner: Whether to perform NER
        perform_classification: Whether to perform classification
        perform_summarization: Whether to perform summarization
        summary_length: Summary length
        summary_method: Summarization method
        preserve_structure: Whether to preserve structure
        clean_text: Whether to clean text
        
    Returns:
        Formatted outputs for Gradio interface
    """
    if file is None:
        return (
            "### Error: Please upload a document file",
            None,
            None,
            None,
            None,
            None,
            None
        )
    
    try:
        # Prepare options
        options = {
            "perform_ner": perform_ner,
            "perform_classification": perform_classification,
            "perform_summarization": perform_summarization,
            "summary_length": summary_length,
            "summary_method": summary_method,
            "preserve_structure": preserve_structure,
            "clean_text": clean_text
        }
        
        # Log what we're sending to the API for debugging
        logger.info(f"Processing file: {os.path.basename(file.name)} with options: {options}")
        
        # Make API request with more detailed error handling
        try:
            response = requests.post(
                f"{API_URL}/api/process",
                files={"file": (os.path.basename(file.name), file, "application/octet-stream")},
                data={"options": json.dumps(options)}
            )
            
            logger.info(f"API response status: {response.status_code}")
            
            # Try to get response content for debugging
            try:
                response_content = response.json()
                logger.info(f"API response: {json.dumps(response_content)[:500]}...")
            except:
                logger.error(f"Could not parse response as JSON. Raw response: {response.text[:500]}...")
                response_content = {"error": "Invalid response from API", "raw": response.text[:500]}
                
            if response.status_code != 200:
                error_msg = response_content.get("error", f"API error: {response.status_code}")
                return (
                    f"### Error: {error_msg}\n\nDetails: {response.text[:500]}",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None
                )
            
            # Parse response
            result = response_content
        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}", exc_info=True)
            return (
                f"### Error: Failed to connect to API\n\nDetails: {str(e)}",
                None,
                None,
                None,
                None,
                None,
                None
            )
        
        # Check if result contains error before display
        if not result.get("success", False) or "error" in result:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"API returned error: {error_msg}")
            return (
                f"### Error: {error_msg}\n\nResponse details: {json.dumps(result, indent=2)[:500]}...",
                None,
                None,
                None,
                None,
                None,
                None
            )
        
        # Format and return the results for display
        return display_results(result)
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        import traceback
        tb = traceback.format_exc()
        return (
            f"### Error: {str(e)}\n\n```\n{tb}\n```",
            None,
            None,
            None,
            None,
            None,
            None
        )

def display_results(result: Dict[str, Any]) -> Tuple:
    """
    Format and display processing results.
    
    Args:
        result: Processing result dictionary
        
    Returns:
        Formatted outputs for Gradio interface
    """
    if "error" in result:
        return (
            f"### Error: {result['error']}",
            None,
            None,
            None,
            None,
            None,
            None
        )
    
    if not result.get("success", False):
        return (
            f"### Error: {result.get('error', 'Unknown error')}",
            None,
            None,
            None,
            None,
            None,
            None
        )
    
    # Format metadata
    metadata = result.get("metadata", {})
    metadata_text = "### Document Metadata\n\n"
    if metadata:
        metadata_text += "| Field | Value |\n"
        metadata_text += "| --- | --- |\n"
        for key, value in metadata.items():
            if value is not None:
                metadata_text += f"| {key} | {value} |\n"
    else:
        metadata_text += "No metadata available."
    
    # Format classification results
    classification = result.get("classification", {})
    classification_text = "### Document Classification\n\n"
    if classification:
        classification_text += f"**Document Type:** {classification.get('document_type')} "
        classification_text += f"(Confidence: {classification.get('document_type_confidence', 0):.2f})\n\n"
        classification_text += f"**Priority:** {classification.get('priority')} "
        classification_text += f"(Confidence: {classification.get('priority_confidence', 0):.2f})\n\n"
        
        # Add detailed classification results if available
        if classification.get('document_type_all'):
            classification_text += "#### Document Type Confidence Scores\n\n"
            for item in classification.get('document_type_all', []):
                classification_text += f"- {item['label']}: {item['confidence']:.2f}\n"
    else:
        classification_text += "No classification results available."
    
    # Format summary results
    summary = result.get("summary", {})
    summary_text = "### Document Summary\n\n"
    if summary:
        summary_text += f"{summary.get('summary', '')}\n\n"
        summary_text += f"*Summary method: {summary.get('method')}, Length: {summary.get('length')}*\n\n"
        
        metrics = summary.get('metrics', {})
        if metrics:
            summary_text += "| Metric | Value |\n"
            summary_text += "| --- | --- |\n"
            summary_text += f"| Compression Ratio | {metrics.get('compression_ratio', 0):.2f} |\n"
            summary_text += f"| Original Length | {metrics.get('original_length', 0)} words |\n"
            summary_text += f"| Summary Length | {metrics.get('summary_length', 0)} words |\n"
    else:
        summary_text += "No summary available."
    
    # Format entity results
    entities = result.get("entities", [])
    entity_text = "### Named Entities\n\n"
    if entities:
        entity_text += "| Entity | Type | Confidence |\n"
        entity_text += "| --- | --- | --- |\n"
        
        # Limit to top 20 entities for display
        for entity in entities[:20]:
            entity_text += f"| {entity.get('text')} | {entity.get('type')} | {entity.get('confidence', 0):.2f} |\n"
        
        if len(entities) > 20:
            entity_text += f"\n*Showing 20 of {len(entities)} entities*"
    else:
        entity_text += "No entities found."
    
    # Entity type visualization
    entity_types = result.get("entity_types", {})
    entity_plot = None
    if entity_types:
        try:
            # Create entity type count data
            type_counts = {k: len(v) for k, v in entity_types.items()}
            
            # Sort by count
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            types = [t[0] for t in sorted_types]
            counts = [t[1] for t in sorted_types]
            
            ax.bar(types, counts, color='skyblue')
            ax.set_xlabel('Entity Type')
            ax.set_ylabel('Count')
            ax.set_title('Named Entity Counts by Type')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            entity_plot = buf
        except Exception as e:
            logger.error(f"Error creating entity plot: {str(e)}")
            entity_plot = None
    
    # Create entity dataframe for display
    entity_df = None
    if entities:
        try:
            entity_data = []
            for entity in entities:
                entity_data.append({
                    "Text": entity.get('text'),
                    "Type": entity.get('type'),
                    "Confidence": f"{entity.get('confidence', 0):.2f}"
                })
            
            entity_df = pd.DataFrame(entity_data)
        except Exception as e:
            logger.error(f"Error creating entity dataframe: {str(e)}")
    
    # Features visualization
    features = result.get("features", {})
    features_text = "### Document Features\n\n"
    if features:
        features_text += "| Feature | Value |\n"
        features_text += "| --- | --- |\n"
        for key, value in features.items():
            if key == "top_terms":
                features_text += f"| Top Terms | {', '.join(value[:10])} |\n"
            elif isinstance(value, bool):
                features_text += f"| {key} | {'Yes' if value else 'No'} |\n"
            else:
                features_text += f"| {key} | {value} |\n"
    else:
        features_text += "No feature information available."
    
    return (
        f"## Document Processing Results\n\n{metadata_text}",
        classification_text,
        summary_text,
        entity_text,
        entity_plot,
        entity_df,
        features_text
    )

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    with gr.Blocks(title="NLP Document Processor") as app:
        gr.Markdown("# NLP Document Processing System")
        gr.Markdown("Upload a document (PDF, DOCX, or TXT) for processing with NLP techniques.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                file_input = gr.File(
                    label="Upload Document", 
                    file_types=[".pdf", ".docx", ".doc", ".txt"]
                )
                
                # Processing options
                gr.Markdown("### Processing Options")
                with gr.Row():
                    perform_ner = gr.Checkbox(
                        label="Extract Entities", 
                        value=True
                    )
                    perform_classification = gr.Checkbox(
                        label="Classify Document", 
                        value=True
                    )
                    perform_summarization = gr.Checkbox(
                        label="Generate Summary", 
                        value=True
                    )
                
                summary_length = gr.Radio(
                    label="Summary Length",
                    choices=["short", "medium", "detailed"],
                    value="medium"
                )
                
                summary_method = gr.Radio(
                    label="Summary Method",
                    choices=["abstractive", "extractive", "hybrid"],
                    value="abstractive"
                )
                
                with gr.Accordion("Advanced Options", open=False):
                    preserve_structure = gr.Checkbox(
                        label="Preserve Document Structure", 
                        value=True
                    )
                    clean_text = gr.Checkbox(
                        label="Clean/Normalize Text", 
                        value=True
                    )
                
                # Process button
                process_btn = gr.Button("Process Document", variant="primary")
                
            with gr.Column(scale=2):
                # Output section - using tabs for organization
                with gr.Tabs():
                    with gr.TabItem(label="Overview"):
                        overview_output = gr.Markdown()
                    
                    with gr.TabItem(label="Classification"):
                        classification_output = gr.Markdown()
                    
                    with gr.TabItem(label="Summary"):
                        summary_output = gr.Markdown()
                    
                    with gr.TabItem(label="Entities"):
                        entity_output = gr.Markdown()
                        entity_plot_output = gr.Image(label="Entity Distribution")
                        entity_table_output = gr.DataFrame(label="Extracted Entities")
                    
                    with gr.TabItem(label="Features"):
                        features_output = gr.Markdown()
        
        # Set up processing click event
        process_btn.click(
            fn=process_document,
            inputs=[
                file_input,
                perform_ner,
                perform_classification,
                perform_summarization,
                summary_length,
                summary_method,
                preserve_structure,
                clean_text
            ],
            outputs=[
                overview_output,
                classification_output, 
                summary_output,
                entity_output,
                entity_plot_output,
                entity_table_output,
                features_output
            ],
            show_progress="full"
        )
    
    return app

def main():
    """Main function to launch the Gradio app."""
    app = create_gradio_interface()
    
    # Launch the app
    app.launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("SHARE", "False").lower() == "true"
    )

if __name__ == "__main__":
    main()