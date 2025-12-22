"""
Medical Domain Spell Checker
=============================
Corrects OCR errors in medical text using domain-specific dictionary
Uses SymSpell-like algorithm for fast fuzzy matching
"""

from typing import Dict, List, Tuple, Optional
import re

class MedicalSpellChecker:
    """
    Domain-specific spell checker for medical OCR text.
    Handles common OCR errors like:
    - 'Cholestercl' -> 'Cholesterol'
    - 'Dlabetes' -> 'Diabetes'
    - 'Systoiic' -> 'Systolic'
    """
    
    def __init__(self):
        # Medical dictionary with correct spellings
        self.medical_terms = self._build_dictionary()
        
        # Common OCR error patterns (char substitutions)
        self.ocr_errors = {
            'l': ['1', 'i', '|'],
            'o': ['0', 'O'],
            'i': ['1', 'l', '!'],
            '0': ['o', 'O'],
            '1': ['l', 'i', '|'],
            's': ['5', '$'],
            '5': ['s', 'S'],
            'e': ['c', '3'],
            'c': ['e', '('],
            'g': ['9', 'q'],
            '9': ['g', 'q'],
            'a': ['@', '4'],
            'b': ['6', '8'],
            'd': ['cl', 'ol'],
            'm': ['rn', 'nn'],
            'w': ['vv', 'uu'],
        }
        
        # Pre-compute lowercase terms for matching
        self.terms_lower = {t.lower(): t for t in self.medical_terms}
    
    def _build_dictionary(self) -> set:
        """Build comprehensive medical dictionary"""
        terms = set()
        
        # Cardiovascular terms
        cardio = [
            'age', 'sex', 'male', 'female', 'gender',
            'systolic', 'diastolic', 'blood', 'pressure', 'mmhg',
            'cholesterol', 'hdl', 'ldl', 'triglycerides', 'lipid', 'lipids',
            'glucose', 'fasting', 'random', 'sugar', 'diabetes', 'diabetic',
            'hypertension', 'hypertensive', 'hyperlipidemia', 'hypercholesterolemia',
            'cardiovascular', 'cardiac', 'heart', 'disease', 'risk', 'chd',
            'smoking', 'smoker', 'cigarettes', 'tobacco', 'nonsmoker',
            'bmi', 'weight', 'height', 'obesity', 'obese', 'overweight',
            'pulse', 'rate', 'heartrate', 'bpm', 'rhythm', 'sinus',
            'ecg', 'ekg', 'electrocardiogram', 'echocardiogram', 'echo',
            'stress', 'test', 'treadmill', 'exercise', 'tolerance',
            'angina', 'chest', 'pain', 'pectoris', 'discomfort',
            'infarction', 'myocardial', 'attack', 'mi', 'stemi', 'nstemi',
            'coronary', 'artery', 'arteries', 'cad', 'atherosclerosis',
            'stroke', 'tia', 'cerebrovascular', 'ischemic',
            'murmur', 'valve', 'valvular', 'mitral', 'aortic', 'stenosis',
            'fibrillation', 'atrial', 'flutter', 'arrhythmia', 'afib',
            'ejection', 'fraction', 'lvef', 'dysfunction', 'systolic',
            'failure', 'hf', 'chf', 'congestive', 'cardiomyopathy',
            'peripheral', 'vascular', 'pvd', 'claudication',
        ]
        
        # Lab values
        labs = [
            'hemoglobin', 'hematocrit', 'wbc', 'rbc', 'platelet', 'platelets',
            'creatinine', 'bun', 'egfr', 'gfr', 'kidney', 'renal',
            'hba1c', 'a1c', 'glycated', 'hemoglobin',
            'troponin', 'bnp', 'ntprobnp', 'proBNP',
            'ast', 'alt', 'alkaline', 'phosphatase', 'bilirubin', 'liver',
            'sodium', 'potassium', 'chloride', 'bicarbonate', 'electrolytes',
            'calcium', 'magnesium', 'phosphorus',
            'tsh', 'thyroid', 't3', 't4', 'thyroxine',
            'inr', 'pt', 'ptt', 'coagulation', 'clotting',
            'uric', 'acid', 'urate', 'gout',
            'albumin', 'protein', 'globulin',
        ]
        
        # Medications
        medications = [
            'aspirin', 'metformin', 'lisinopril', 'losartan', 'amlodipine',
            'atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin',
            'metoprolol', 'carvedilol', 'atenolol', 'bisoprolol',
            'furosemide', 'hydrochlorothiazide', 'hctz', 'spironolactone',
            'warfarin', 'coumadin', 'clopidogrel', 'plavix', 'ticagrelor',
            'digoxin', 'amiodarone', 'diltiazem', 'verapamil',
            'nitroglycerin', 'isosorbide', 'nitrate',
            'insulin', 'glipizide', 'glyburide', 'januvia', 'jardiance',
            'ramipril', 'enalapril', 'captopril', 'benazepril',
            'prednisone', 'prednisolone', 'dexamethasone',
            'omeprazole', 'pantoprazole', 'lansoprazole',
            'gabapentin', 'pregabalin', 'tramadol',
            'therapy', 'treatment', 'medication', 'drug', 'dose', 'dosage',
            'mg', 'mcg', 'ml', 'units', 'daily', 'twice', 'bid', 'tid', 'qid',
        ]
        
        # Clinical terms
        clinical = [
            'patient', 'history', 'medical', 'clinical', 'report',
            'diagnosis', 'diagnosed', 'assessment', 'impression',
            'symptoms', 'signs', 'findings', 'examination', 'physical',
            'normal', 'abnormal', 'elevated', 'low', 'high', 'borderline',
            'positive', 'negative', 'present', 'absent', 'noted',
            'mild', 'moderate', 'severe', 'chronic', 'acute',
            'family', 'smoking', 'alcohol', 'exercise', 'diet', 'lifestyle',
            'followup', 'referral', 'consultation', 'recommend',
            'years', 'months', 'weeks', 'days', 'old', 'year',
        ]
        
        # Add all terms
        for term_list in [cardio, labs, medications, clinical]:
            terms.update(term_list)
        
        return terms
    
    def _edit_distance(self, s1: str, s2: str, max_dist: int = 2) -> int:
        """Calculate edit distance with early termination"""
        if abs(len(s1) - len(s2)) > max_dist:
            return max_dist + 1
        
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                insert = prev[j + 1] + 1
                delete = curr[j] + 1
                replace = prev[j] + (0 if c1 == c2 else 1)
                curr.append(min(insert, delete, replace))
            prev = curr
            if min(prev) > max_dist:
                return max_dist + 1
        return prev[-1]
    
    def find_correction(self, word: str, max_distance: int = 2) -> Optional[str]:
        """Find the best correction for a misspelled word"""
        word_lower = word.lower()
        
        # Exact match
        if word_lower in self.terms_lower:
            return self.terms_lower[word_lower]
        
        # Find candidates within edit distance
        candidates = []
        for term_lower, term_orig in self.terms_lower.items():
            # Quick length filter
            if abs(len(term_lower) - len(word_lower)) > max_distance:
                continue
            
            # Quick first char filter (most OCR errors preserve first char)
            if term_lower[0] != word_lower[0] and max_distance < 2:
                continue
            
            dist = self._edit_distance(word_lower, term_lower, max_distance)
            if dist <= max_distance:
                candidates.append((term_orig, dist))
        
        if not candidates:
            return None
        
        # Return closest match (prefer shorter distance, then alphabetical)
        candidates.sort(key=lambda x: (x[1], x[0]))
        return candidates[0][0]
    
    def correct_text(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Correct OCR errors in text.
        Only corrects words that are likely medical terms (5+ chars).
        
        Returns:
            Tuple of (corrected_text, list of corrections made)
        """
        corrections = []
        # Only target longer words (5+ chars) to avoid false positives on common words
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text)
        
        corrected_text = text
        for word in set(words):
            word_lower = word.lower()
            # Skip if it's a common English word (not in our medical dictionary)
            common_words = {'about', 'after', 'again', 'being', 'below', 'between', 
                          'could', 'found', 'every', 'first', 'where', 'which', 
                          'while', 'their', 'there', 'these', 'those', 'through',
                          'under', 'using', 'would', 'years', 'results', 'signs',
                          'vital', 'vitals', 'shows', 'history'}
            if word_lower in common_words:
                continue
            
            # Only correct if not already in dictionary
            if word_lower not in self.terms_lower:
                correction = self.find_correction(word, max_distance=1)  # More strict
                if correction and correction.lower() != word_lower:
                    edit_dist = self._edit_distance(word_lower, correction.lower())
                    conf = 1 - (edit_dist / len(word))
                    # Only accept high confidence corrections
                    if conf >= 0.7:
                        corrections.append({
                            'original': word,
                            'corrected': correction,
                            'confidence': conf
                        })
                        # Replace in text (case-insensitive)
                        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                        corrected_text = pattern.sub(correction, corrected_text)
        
        return corrected_text, corrections
    
    def correct_field_names(self, text: str) -> str:
        """
        Specifically correct common field name OCR errors.
        These are the most critical for extraction.
        """
        # Common OCR mistakes for field names
        field_corrections = {
            # Cholesterol variants
            'cholestercl': 'cholesterol',
            'cholesteroi': 'cholesterol',
            'cho1esterol': 'cholesterol',
            'choiesterol': 'cholesterol',
            'choles terol': 'cholesterol',
            
            # Blood pressure
            'systoiic': 'systolic',
            'systo1ic': 'systolic',
            'diastollc': 'diastolic',
            'diasto1ic': 'diastolic',
            'b1ood': 'blood',
            'biood': 'blood',
            'pressur e': 'pressure',
            
            # Glucose/Diabetes
            'g1ucose': 'glucose',
            'giucose': 'glucose',
            'dlabetes': 'diabetes',
            'diabctes': 'diabetes',
            'd1abetes': 'diabetes',
            
            # HDL/LDL
            'hd1': 'hdl',
            'ld1': 'ldl',
            '1dl': 'ldl',
            'triglycerldes': 'triglycerides',
            
            # Other common
            'hypertens1on': 'hypertension',
            'card1ac': 'cardiac',
            'card1ovascular': 'cardiovascular',
            'smok1ng': 'smoking',
            'pat1ent': 'patient',
            'med1cal': 'medical',
        }
        
        corrected = text
        for wrong, right in field_corrections.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            corrected = pattern.sub(right, corrected)
        
        return corrected


# Singleton instance
_spell_checker = None

def get_spell_checker() -> MedicalSpellChecker:
    global _spell_checker
    if _spell_checker is None:
        _spell_checker = MedicalSpellChecker()
    return _spell_checker


def correct_ocr_text(text: str) -> Tuple[str, List[Dict]]:
    """Convenience function for spell checking OCR text"""
    checker = get_spell_checker()
    # First apply known field corrections
    text = checker.correct_field_names(text)
    # Then general spell check
    return checker.correct_text(text)


if __name__ == '__main__':
    checker = MedicalSpellChecker()
    
    # Test with OCR errors
    test_text = """
    Pat1ent is a 55 year o1d ma1e with history of hypertens1on, 
    dlabetes me11itus type 2, and hyperlipidemia.
    
    Lab Results:
    - Cho1estero1: 245 mg/dL
    - HD1: 38 mg/dL
    - LD1: 165 mg/dL
    - G1ucose (fasting): 128 mg/dL
    - HbA1c: 7.2%
    
    Vita1 Signs:
    - B1ood Pressure: 142/88 mmHg (Systo1ic/Diasto1ic)
    - Heart Rate: 78 bpm
    """
    
    print("=== Original Text ===")
    print(test_text)
    
    corrected, corrections = checker.correct_text(
        checker.correct_field_names(test_text)
    )
    
    print("\n=== Corrected Text ===")
    print(corrected)
    
    print("\n=== Corrections Made ===")
    for c in corrections:
        print(f"  '{c['original']}' -> '{c['corrected']}' (conf: {c['confidence']:.2f})")
