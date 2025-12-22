#!/usr/bin/env python3
"""Test ML Service extraction in Django context."""
import os
import sys

print("=== Starting Test ===")

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cardiodetect.settings')
import django
django.setup()

print("=== Django setup complete ===")

from services.ml_service import MLService
ml = MLService()
print(f"Initial loaded: {ml.loaded}")
print(f"Pipeline: {ml.pipeline}")

# Force load pipeline
print("\n=== Loading Pipeline ===")
ml._load_pipeline()
print(f"After load - loaded: {ml.loaded}")
print(f"After load - pipeline: {ml.pipeline}")

# Call process_document
print("\n=== Processing Document ===")
result = ml.process_document('/Users/prajanv/CardioDetect/Milestone_2/Medical_report/Synthetic_report/SYN-002.png')
print(f"Fields count: {len(result.get('fields', {}))}")
print(f"Quality: {result.get('quality')}")
print(f"Error: {result.get('error')}")
if result.get('fields'):
    print(f"Age: {result['fields'].get('age')}")
