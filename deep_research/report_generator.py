"""
Report generation for deep research.
Creates comprehensive multi-page Markdown reports.
"""
import json
import os
from datetime import datetime
from pathlib import Path
import config
from core.llm import call_model
from deep_research.mindmap import get_mindmap_text

REPORT_SECTIONS = [
    ("executive_summary", "Executive Summary"),
    ("introduction", "Introduction"),
    ("detailed_findings", "Detailed Findings"),
    ("relationships", "Relationships & Connections"),
    ("contradictions", "Contradictions & Open Questions"),
    ("conclusions", "Conclusions"),
    ("references", "References & Sources"),
]

def _generate_section_prompt(section_key, section_title, research_id, facts, chunks, mindmap_text):
    context = f"Mindmap:\n{mindmap_text}\n\nFacts ({len(facts)}):\n"
    for f in facts[:100]:
        context += f"- {f.get('fact_text','')} (source: {f.get('source_span','')}, conf: {f.get('confidence',0)})\n"
    if chunks:
        context += "\nRelevant excerpts:\n"
        for _, _, doc_hash, text in chunks[:10]:
            context += f"- [{doc_hash}] {text[:500]}\n"
    prompt = f"""
You are writing a section of a comprehensive research report.
Topic: {section_title}
Section: {section_title}

Using the provided context, write a detailed, multi-paragraph section (at least 500 words) covering this aspect.
Use Markdown formatting with subheadings and bullet points where appropriate.
Be thorough and include all relevant information, but do not invent facts.

Context:
{context}

Write the section content now (start with '## {section_title}'):
"""
    return prompt

def generate_report(research_id, facts, chunks, output_dir):
    """Generate a full report and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    mindmap_text = get_mindmap_text(research_id)
    report_parts = []
    report_parts.append(f"# Research Report: {research_id}\n")
    report_parts.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report_parts.append("\n---\n")

    for section_key, section_title in REPORT_SECTIONS:
        prompt = _generate_section_prompt(section_key, section_title, research_id, facts, chunks, mindmap_text)
        section_content = call_model(prompt, max_tokens=2000)
        if section_content:
            report_parts.append(section_content.strip())
        else:
            report_parts.append(f"## {section_title}\n\n[Section could not be generated]")
        report_parts.append("\n---\n")

    report_md = "\n".join(report_parts)
    filename = f"research_{research_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = Path(output_dir) / filename
    filepath.write_text(report_md, encoding="utf-8")
    if config.REPORT_COHERENCE_PASS:
        print("  Running coherence pass...")
        report_md = coherence_pass(report_md, research_id)
        filepath.write_text(report_md, encoding="utf-8")
    return str(filepath)


def coherence_pass(report_md, query):
    """Run a final check for contradictions and missing sections."""
    prompt = f"""
You are a quality assurance editor.
Review the following research report for contradictions, missing important sections, or unsupported claims.
Fix any issues you find. Return the complete revised report in Markdown.

Report:
{report_md}

Revised report:
"""
    revised = call_model(prompt, max_tokens=4000)
    return revised if revised else report_md
