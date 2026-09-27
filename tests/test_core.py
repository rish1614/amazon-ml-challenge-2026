import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code' / 'business_entity_resolution'))
from src.metrics import entity_f05
from src.normalize import normalize_text, name_without_legal_suffix

def test_singleton():
    assert entity_f05(set(), set()) == 1.0
    assert entity_f05(set(), {'S2-1'}) == 0.0

def test_f05_example():
    score = entity_f05({'A','B'}, {'A','B','C'})
    assert round(score,3) == 0.714

def test_normalization():
    assert normalize_text('A.B.C.   Pvt. Ltd') == 'a b c pvt ltd'
    assert name_without_legal_suffix('ABC Pvt Ltd') == 'abc'
