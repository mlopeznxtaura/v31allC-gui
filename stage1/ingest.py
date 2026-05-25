import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

class Stage1Ingest:
    """Ingest raw weights/text into Stage1 JSONL schema."""
    
    @staticmethod
    def detect_format(content: str) -> str:
        """Guess file format."""
        content = content[:500].strip()
        if content.startswith('[') or content.startswith('{'):
            return 'json'
        elif '\t' in content or ',' in content:
            return 'csv'
        elif content.startswith('array') or content.startswith('tensor'):
            return 'numpy'
        return 'text'
    
    @staticmethod
    def parse_json(content: str) -> List[Dict]:
        """Parse JSON array or JSONL."""
        lines = content.strip().split('\n')
        records = []
        for line in lines:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except:
                    pass
        return records if records else [json.loads(content)]
    
    @staticmethod
    def parse_csv(content: str) -> List[Dict]:
        """Parse CSV into dicts."""
        lines = content.strip().split('\n')
        reader = csv.DictReader(lines)
        return list(reader)
    
    @staticmethod
    def parse_text_weights(content: str) -> List[Dict]:
        """Parse raw weight text (one per line)."""
        records = []
        for i, line in enumerate(content.strip().split('\n')):
            if line.strip():
                try:
                    val = float(line)
                    records.append({
                        'weight': val,
                        'index': i,
                        'type': 'scalar'
                    })
                except:
                    records.append({
                        'text': line.strip(),
                        'index': i,
                        'type': 'text'
                    })
        return records
    
    @staticmethod
    def ingest_file(file_content: str, filename: str) -> tuple[List[Dict], str]:
        """Main ingest: auto-detect format and convert."""
        fmt = Stage1Ingest.detect_format(file_content)
        
        if fmt == 'json':
            parsed = Stage1Ingest.parse_json(file_content)
        elif fmt == 'csv':
            parsed = Stage1Ingest.parse_csv(file_content)
        else:
            parsed = Stage1Ingest.parse_text_weights(file_content)
        
        return parsed, fmt
    
    @staticmethod
    def to_stage1_jsonl(parsed_records: List[Dict], filename: str, entry_offset: int = 0) -> str:
        """Convert any parsed format to Stage1 JSONL."""
        from stage1.core.record_builder import RecordBuilder
        
        rb = RecordBuilder(
            total_entries=len(parsed_records),
            codebase_hash=filename[:16],
            mission=f"ingest_{filename}"
        )
        
        jsonl_lines = []
        for idx, rec in enumerate(parsed_records):
            # Extract text if available
            binary_text = str(rec.get('binary', rec.get('text', 'ingested')[:80]))
            geometry_text = f"Entry {idx+1+entry_offset}/{len(parsed_records)}"
            language_text = json.dumps(rec)[:200]
            
            jsonl_line = rb.build_jsonl_line(
                binary_text=binary_text,
                geometry_text=geometry_text,
                language_text=language_text,
            )
            jsonl_lines.append(jsonl_line)
        
        return '\n'.join(jsonl_lines)

