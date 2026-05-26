"""stage1/ingest.py — Parse various data formats"""

import json
import csv
from io import StringIO


class Stage1Ingest:
    """Ingest raw data in various formats"""
    
    @staticmethod
    def detect_format(content: str) -> str:
        """Guess file format from content"""
        # Strip markdown code fences first
        content = content.strip()
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])
        
        content = content.strip()
        first_line = content.split('\n')[0].strip()
        
        # Check for JSONL: single-line objects, multiple lines with { at start
        if first_line.startswith('{'):
            # Count newlines with { at start
            json_lines = sum(1 for line in content.split('\n') if line.strip().startswith('{'))
            total_lines = len([l for l in content.split('\n') if l.strip()])
            if json_lines > 1 and total_lines > 1:
                return 'jsonl'
            else:
                return 'json'
        
        # Try JSON array
        if content.startswith('['):
            return 'json'
        
        # Check CSV (no { or [ at start, has comma or tab)
        if '\t' in content or ',' in content:
            return 'csv'
        
        return 'text'
    
    @staticmethod
    def parse_jsonl(content: str) -> list:
        """Parse JSONL (one JSON object per line)"""
        # Strip markdown code fences
        content = content.strip()
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])
        
        records = []
        for line in content.strip().split('\n'):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except:
                    pass
        return records
    
    @staticmethod
    def parse_json(content: str) -> list:
        """Parse JSON array or single object"""
        # Strip markdown code fences
        content = content.strip()
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])
        
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            else:
                return [parsed]
        except:
            return []
    
    @staticmethod
    def parse_csv(content: str) -> list:
        """Parse CSV into list of dicts"""
        reader = csv.DictReader(StringIO(content))
        return list(reader)
    
    @staticmethod
    def parse_text(content: str) -> list:
        """Parse plain text as single record"""
        return [{"text": content}]
    
    @staticmethod
    def parse_format(content: str, fmt: str) -> list:
        """Route to appropriate parser"""
        if fmt == 'jsonl':
            return Stage1Ingest.parse_jsonl(content)
        elif fmt == 'json':
            return Stage1Ingest.parse_json(content)
        elif fmt == 'csv':
            return Stage1Ingest.parse_csv(content)
        else:
            return Stage1Ingest.parse_text(content)
