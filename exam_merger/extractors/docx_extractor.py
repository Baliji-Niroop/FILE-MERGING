import os

try:
    import docx
except ImportError:
    docx = None

from ..models.chunk import Chunk

def docx_table_to_markdown(table) -> str:
    rows = []
    for row in table.rows:
        row_cells = []
        for cell in row.cells:
            row_cells.append(cell.text.replace("\n", " ").strip())
        if any(cell != "" for cell in row_cells):
            rows.append(row_cells)
            
    if not rows:
        return ""
        
    cols_count = len(rows[0])
    md_lines = []
    # Header
    md_lines.append("| " + " | ".join(rows[0]) + " |")
    # Separator
    md_lines.append("| " + " | ".join(["---"] * cols_count) + " |")
    # Rows
    for row in rows[1:]:
        if len(row) < cols_count:
            row.extend([""] * (cols_count - len(row)))
        elif len(row) > cols_count:
            row = row[:cols_count]
        md_lines.append("| " + " | ".join(row) + " |")
        
    return "\n".join(md_lines)

def extract(path: str, describe_diagrams: bool = False) -> list[Chunk]:
    chunks = []
    filename = os.path.basename(path)

    if docx is None:
        return []
    
    try:
        doc = docx.Document(path)
    except Exception as e:
        print(f"Error reading {path} via python-docx: {e}")
        return []
        
    active_headings = {}
    current_body_paras = []
    current_body_len = 0
    
    def get_heading_path():
        if not active_headings:
            return ""
        sorted_keys = sorted(active_headings.keys())
        return " > ".join(active_headings[k] for k in sorted_keys)
        
    def flush_body_text():
        nonlocal current_body_paras, current_body_len
        if not current_body_paras:
            return
        
        combined_text = "\n\n".join(current_body_paras)
        heading_path = get_heading_path()
        location = f"Section: {heading_path}" if heading_path else "Body"
        
        chunks.append(Chunk(
            text=combined_text,
            source_file=filename,
            location=location,
            chunk_type="text"
        ))
        
        current_body_paras = []
        current_body_len = 0

    # Iterate elements in document order
    for child in doc.element.body:
        if child.tag.endswith('p'):
            p = docx.text.paragraph.Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
                
            style_name = p.style.name if p.style else ""
            if style_name and style_name.startswith("Heading"):
                # Flush existing text
                flush_body_text()
                
                # Extract level
                parts = style_name.split()
                level = 1
                if len(parts) > 1 and parts[1].isdigit():
                    level = int(parts[1])
                
                # Update current active headings hierarchy
                active_headings = {k: v for k, v in active_headings.items() if k < level}
                active_headings[level] = text
                
                # Add heading chunk itself
                chunks.append(Chunk(
                    text=text,
                    source_file=filename,
                    location=f"Heading {level}",
                    chunk_type="text"
                ))
            else:
                # Regular paragraph
                current_body_paras.append(text)
                current_body_len += len(text) + 2
                
                # Chunk threshold: let's group paragraphs up to ~2000 chars
                if current_body_len >= 2000:
                    flush_body_text()
                    
        elif child.tag.endswith('tbl'):
            tbl = docx.table.Table(child, doc)
            flush_body_text()
            
            table_md = docx_table_to_markdown(tbl)
            if table_md:
                heading_path = get_heading_path()
                location = f"Table under Section: {heading_path}" if heading_path else "Table"
                
                chunks.append(Chunk(
                    text=table_md,
                    source_file=filename,
                    location=location,
                    chunk_type="table"
                ))
                
    # Flush remaining text
    flush_body_text()
    
    return chunks
