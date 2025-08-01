#!/usr/bin/env python3
"""
Helper script to capture and analyze debug logs
"""

import re
import json
from datetime import datetime

def extract_debug_sections(log_file_path):
    """Extract debug sections from log file"""
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file_path}")
        return
    
    # Find all debug sections
    sections = {
        'requests': re.findall(r'=== CHAT COMPLETION REQUEST ===(.*?)(?==== |$)', content, re.DOTALL),
        'prompts': re.findall(r'=== GENERATED PROMPT ===(.*?)(?==== |$)', content, re.DOTALL),
        'responses': re.findall(r'=== RAW MODEL RESPONSE ===(.*?)(?==== |$)', content, re.DOTALL),
        'parsing': re.findall(r'=== PARSING RESULTS ===(.*?)(?==== |$)', content, re.DOTALL),
        'final': re.findall(r'=== FINAL RESPONSE ===(.*?)(?==== |$)', content, re.DOTALL)
    }
    
    print(f"🔍 Debug Log Analysis")
    print(f"=" * 50)
    print(f"Found {len(sections['requests'])} requests")
    print(f"Found {len(sections['responses'])} model responses")
    print(f"Found {len(sections['parsing'])} parsing results")
    
    # Show the most recent interaction
    if sections['requests']:
        print(f"\n📥 Most Recent Request:")
        print(sections['requests'][-1][:500] + "..." if len(sections['requests'][-1]) > 500 else sections['requests'][-1])
    
    if sections['responses']:
        print(f"\n🤖 Most Recent Model Response:")
        print(sections['responses'][-1][:500] + "..." if len(sections['responses'][-1]) > 500 else sections['responses'][-1])
    
    if sections['parsing']:
        print(f"\n🔧 Most Recent Parsing Results:")
        print(sections['parsing'][-1])
    
    return sections

def save_debug_dump(log_file_path, output_file="debug_dump.json"):
    """Save all debug info to a JSON file for analysis"""
    
    sections = extract_debug_sections(log_file_path)
    if not sections:
        return
    
    # Create structured dump
    dump = {
        "timestamp": datetime.now().isoformat(),
        "log_file": log_file_path,
        "summary": {
            "total_requests": len(sections['requests']),
            "total_responses": len(sections['responses']),
            "total_parsing_results": len(sections['parsing'])
        },
        "latest_interaction": {
            "request": sections['requests'][-1] if sections['requests'] else None,
            "prompt_info": sections['prompts'][-1] if sections['prompts'] else None,
            "model_response": sections['responses'][-1] if sections['responses'] else None,
            "parsing_results": sections['parsing'][-1] if sections['parsing'] else None,
            "final_response": sections['final'][-1] if sections['final'] else None
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Debug dump saved to: {output_file}")
    print(f"Use: cat {output_file} | jq . (if you have jq installed)")

if __name__ == "__main__":
    import sys
    
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/qwen3_server.log"
    
    print(f"📖 Reading logs from: {log_file}")
    extract_debug_sections(log_file)
    save_debug_dump(log_file)