import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extraction.entity_normalizer import normalize_date

def test_iso_date():
    assert normalize_date("March 15, 2023") == "2023-03-15"

def test_day_month_year():
    assert normalize_date("15 March 2023") == "2023-03-15"

def test_year_only():
    assert normalize_date("The event happened in 2020.") == "2020"
