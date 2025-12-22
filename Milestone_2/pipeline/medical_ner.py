"""
Medical Named Entity Recognition (NER) Module
==============================================
Uses SpaCy for extracting medical entities from OCR text
"""

import re
from typing import Dict, Any, List, Optional

# Try to import SpaCy
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    spacy = None

# Try to import SciSpacy medical model
nlp = None
HAS_MEDICAL_MODEL = False
if HAS_SPACY:
    try:
        nlp = spacy.load("en_core_sci_sm")
        HAS_MEDICAL_MODEL = True
    except:
        try:
            nlp = spacy.load("en_core_web_sm")
            HAS_MEDICAL_MODEL = True
        except:
            pass


class MedicalNER:
    """
    Medical Named Entity Recognition for extracting:
    - Drug names
    - Medical conditions
    - Lab values
    - Procedures
    """
    
    def __init__(self):
        self.nlp = nlp
        self.has_spacy = HAS_SPACY
        self.has_medical = HAS_MEDICAL_MODEL
        
        # Common drug patterns for fallback
        self.drug_patterns = [
            r'\b(aspirin|metformin|lisinopril|atorvastatin|amlodipine|metoprolol|omeprazole|losartan|gabapentin|hydrochlorothiazide)\b',
            r'\b(simvastatin|levothyroxine|azithromycin|amoxicillin|pantoprazole|furosemide|prednisone|tramadol|clopidogrel|carvedilol)\b',
            r'\b(warfarin|digoxin|diltiazem|verapamil|ramipril|enalapril|captopril|spironolactone|rosuvastatin|pravastatin)\b',
        ]
        
        # Medical condition patterns
        self.condition_patterns = [
            r'\b(hypertension|diabetes|hyperlipidemia|coronary artery disease|heart failure|atrial fibrillation)\b',
            r'\b(myocardial infarction|angina|stroke|tia|peripheral vascular disease|chronic kidney disease)\b',
            r'\b(hypercholesterolemia|obesity|metabolic syndrome|cardiomyopathy|arrhythmia|valvular disease)\b',
        ]
        
        # Lab test patterns
        self.lab_patterns = [
            r'\b(hemoglobin|hematocrit|wbc|rbc|platelet|creatinine|bun|egfr)\b',
            r'\b(hba1c|a1c|ldl|hdl|triglycerides|cholesterol|glucose|fasting glucose)\b',
            r'\b(troponin|bnp|pro-bnp|inr|pt|ptt|ast|alt|alkaline phosphatase)\b',
        ]
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract medical entities from text.
        
        Returns:
            Dictionary with drugs, conditions, labs, and other entities
        """
        entities = {
            'drugs': [],
            'conditions': [],
            'labs': [],
            'procedures': [],
            'other': []
        }
        
        text_lower = text.lower()
        
        # Try SpaCy medical model first
        if self.has_medical and self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    }
                    
                    # Categorize based on label
                    if ent.label_ in ['CHEMICAL', 'DRUG']:
                        entities['drugs'].append(entity_info)
                    elif ent.label_ in ['DISEASE', 'CONDITION']:
                        entities['conditions'].append(entity_info)
                    else:
                        entities['other'].append(entity_info)
            except Exception as e:
                print(f"SpaCy error: {e}")
        
        # Supplement with regex patterns (always run for completeness)
        for pattern in self.drug_patterns:
            for match in re.finditer(pattern, text_lower):
                drug = match.group(1).capitalize()
                if drug not in [d.get('text', d) for d in entities['drugs']]:
                    entities['drugs'].append({'text': drug, 'label': 'DRUG', 'source': 'regex'})
        
        for pattern in self.condition_patterns:
            for match in re.finditer(pattern, text_lower):
                condition = match.group(1).title()
                if condition not in [c.get('text', c) for c in entities['conditions']]:
                    entities['conditions'].append({'text': condition, 'label': 'CONDITION', 'source': 'regex'})
        
        for pattern in self.lab_patterns:
            for match in re.finditer(pattern, text_lower):
                lab = match.group(1).upper()
                if lab not in [l.get('text', l) for l in entities['labs']]:
                    entities['labs'].append({'text': lab, 'label': 'LAB_TEST', 'source': 'regex'})
        
        # Extract procedures (common cardiac procedures)
        procedure_patterns = [
            r'\b(echocardiogram|ecg|ekg|stress test|angiogram|catheterization|pci|cabg|ablation)\b',
            r'\b(holter monitor|event monitor|cardiac mri|ct angiography|nuclear stress test)\b',
        ]
        for pattern in procedure_patterns:
            for match in re.finditer(pattern, text_lower):
                proc = match.group(1).upper()
                if proc not in [p.get('text', p) for p in entities['procedures']]:
                    entities['procedures'].append({'text': proc, 'label': 'PROCEDURE', 'source': 'regex'})
        
        return {
            'entities': entities,
            'counts': {
                'drugs': len(entities['drugs']),
                'conditions': len(entities['conditions']),
                'labs': len(entities['labs']),
                'procedures': len(entities['procedures']),
            },
            'using_spacy': self.has_medical,
            'total': sum(len(v) for v in entities.values())
        }
    
    def extract_medications_with_dosage(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medications with their dosages.
        Pattern: "Drug 100mg" or "Drug 100 mg daily"
        """
        medications = []
        
        # Pattern: Drug name followed by dosage
        pattern = r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\s*(?:(daily|bid|tid|qid|prn|once|twice|weekly))?'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            medications.append({
                'drug': match.group(1),
                'dose': float(match.group(2)),
                'unit': match.group(3).lower(),
                'frequency': match.group(4) if match.group(4) else 'unspecified'
            })
        
        return medications
    
    def get_risk_relevant_info(self, text: str) -> Dict[str, Any]:
        """
        Extract information specifically relevant to cardiovascular risk.
        """
        result = {
            'on_statins': False,
            'on_antihypertensives': False,
            'on_antiplatelet': False,
            'on_diabetes_meds': False,
            'has_known_cad': False,
            'has_prior_mi': False,
            'has_heart_failure': False,
        }
        
        text_lower = text.lower()
        
        # Statins
        statins = ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin', 'fluvastatin']
        result['on_statins'] = any(s in text_lower for s in statins)
        
        # Antihypertensives
        antihtn = ['lisinopril', 'losartan', 'amlodipine', 'metoprolol', 'atenolol', 'carvedilol', 'hydrochlorothiazide', 'hctz']
        result['on_antihypertensives'] = any(a in text_lower for a in antihtn)
        
        # Antiplatelet
        antiplatelet = ['aspirin', 'clopidogrel', 'plavix', 'ticagrelor', 'prasugrel']
        result['on_antiplatelet'] = any(a in text_lower for a in antiplatelet)
        
        # Diabetes meds
        dm_meds = ['metformin', 'glipizide', 'glyburide', 'insulin', 'januvia', 'jardiance', 'ozempic']
        result['on_diabetes_meds'] = any(d in text_lower for d in dm_meds)
        
        # Conditions
        result['has_known_cad'] = any(c in text_lower for c in ['coronary artery disease', 'cad', 'coronary disease'])
        result['has_prior_mi'] = any(m in text_lower for m in ['myocardial infarction', 'heart attack', 'mi', 'stemi', 'nstemi'])
        result['has_heart_failure'] = any(h in text_lower for h in ['heart failure', 'hf', 'chf', 'cardiomyopathy'])
        
        return result


# Singleton
_ner = None

def get_ner() -> MedicalNER:
    global _ner
    if _ner is None:
        _ner = MedicalNER()
    return _ner


def extract_medical_entities(text: str) -> Dict[str, Any]:
    """Convenience function"""
    return get_ner().extract_entities(text)


if __name__ == '__main__':
    ner = MedicalNER()
    
    test_text = """
    Patient is a 55-year-old male with history of hypertension, diabetes mellitus type 2,
    and hyperlipidemia. Currently on metformin 1000mg BID, lisinopril 20mg daily, 
    and atorvastatin 40mg at bedtime. Recent labs show HbA1c 7.2%, LDL 98 mg/dL.
    Previous stress test was negative. ECG shows normal sinus rhythm.
    No history of myocardial infarction or coronary artery disease.
    """
    
    result = ner.extract_entities(test_text)
    
    print("=== Medical NER Results ===")
    print(f"Using SpaCy: {result['using_spacy']}")
    print(f"\nDrugs found: {result['counts']['drugs']}")
    for d in result['entities']['drugs']:
        print(f"  - {d['text']}")
    
    print(f"\nConditions found: {result['counts']['conditions']}")
    for c in result['entities']['conditions']:
        print(f"  - {c['text']}")
    
    print(f"\nLabs found: {result['counts']['labs']}")
    for l in result['entities']['labs']:
        print(f"  - {l['text']}")
    
    print(f"\nProcedures found: {result['counts']['procedures']}")
    for p in result['entities']['procedures']:
        print(f"  - {p['text']}")
    
    print("\n=== Risk-Relevant Info ===")
    risk_info = ner.get_risk_relevant_info(test_text)
    for k, v in risk_info.items():
        if v:
            print(f"  ✓ {k}")
