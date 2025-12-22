"""
CardioDetect Clinical Advisor Module
=====================================
Evidence-based clinical decision support integrating major cardiovascular guidelines.

╔══════════════════════════════════════════════════════════════════════════════╗
║  GUIDELINES INTEGRATED                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • ACC/AHA 2017 - Hypertension Management                                     ║
║  • ACC/AHA 2018 - Cholesterol & Statin Therapy                                ║
║  • ACC/AHA 2019 - Primary Prevention of Cardiovascular Disease                ║
║  • WHO 2020 - Physical Activity and Sedentary Behaviour                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

DISCLAIMER: This tool provides clinical decision support based on published
guidelines. It does NOT replace clinical judgment. All recommendations should
be reviewed by a qualified healthcare provider before implementation.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class UrgencyLevel(Enum):
    ROUTINE = "Routine"
    SOON = "Soon"
    URGENT = "Urgent"
    EMERGENCY = "Emergency"


class GradeOfRecommendation(Enum):
    """ACC/AHA Recommendation Grades"""
    I = "I"      # Strong benefit, should be performed
    IIA = "IIa"  # Moderate benefit, reasonable to perform
    IIB = "IIb"  # Weak benefit, may be considered
    III = "III"  # No benefit or harmful


@dataclass
class Recommendation:
    """Structured clinical recommendation"""
    category: str
    action: str
    rationale: str
    grade: GradeOfRecommendation
    urgency: UrgencyLevel
    guideline_source: str  # e.g., "ACC/AHA 2017"
    details: Optional[str] = None
    contraindicated_if: Optional[List[str]] = None  # Allergies/conditions
    treatment_consideration: Optional[str] = None   # If already on treatment


# =============================================================================
# KNOWN MEDICATION ALLERGIES & CONTRAINDICATIONS
# =============================================================================

STATIN_CONTRAINDICATIONS = {
    "statins": ["Active liver disease", "Pregnancy", "Breastfeeding"],
    "simvastatin": ["Concurrent strong CYP3A4 inhibitors", "Cyclosporine"],
    "atorvastatin": ["Active liver disease"],
    "rosuvastatin": ["Severe renal impairment (if high dose)"]
}

ANTIHYPERTENSIVE_CONTRAINDICATIONS = {
    "ace_inhibitors": ["Angioedema history", "Pregnancy", "Bilateral renal artery stenosis"],
    "arbs": ["Pregnancy", "Bilateral renal artery stenosis"],
    "thiazides": ["Severe hyponatremia", "Symptomatic hyperuricemia/gout"],
    "ccbs": ["Severe aortic stenosis (dihydropyridines)", "Heart block (non-dihydropyridines)"],
    "beta_blockers": ["Severe bradycardia", "AV block", "Uncontrolled heart failure", "Asthma (non-selective)"]
}


# =============================================================================
# ACC/AHA 2017 BLOOD PRESSURE GUIDELINES
# =============================================================================

BP_CATEGORIES = {
    "Normal": {
        "sbp_range": (0, 120), "dbp_range": (0, 80),
        "action": "Promote healthy lifestyle habits",
        "urgency": UrgencyLevel.ROUTINE,
        "grade": GradeOfRecommendation.I
    },
    "Elevated": {
        "sbp_range": (120, 130), "dbp_range": (0, 80),
        "action": "Non-pharmacological therapy (lifestyle modifications)",
        "urgency": UrgencyLevel.ROUTINE,
        "grade": GradeOfRecommendation.I
    },
    "Stage 1 Hypertension": {
        "sbp_range": (130, 140), "dbp_range": (80, 90),
        "action": "Lifestyle modifications; consider medication if 10y ASCVD risk ≥10% or CVD/DM/CKD",
        "urgency": UrgencyLevel.SOON,
        "grade": GradeOfRecommendation.I
    },
    "Stage 2 Hypertension": {
        "sbp_range": (140, 180), "dbp_range": (90, 120),
        "action": "Lifestyle modifications + antihypertensive medication",
        "urgency": UrgencyLevel.SOON,
        "grade": GradeOfRecommendation.I
    },
    "Hypertensive Crisis": {
        "sbp_range": (180, 999), "dbp_range": (120, 999),
        "action": "🚨 IMMEDIATE medical evaluation for target organ damage",
        "urgency": UrgencyLevel.EMERGENCY,
        "grade": GradeOfRecommendation.I
    }
}

FIRST_LINE_ANTIHYPERTENSIVES = [
    {"class": "Thiazide Diuretics", "examples": "Chlorthalidone, HCTZ", 
     "preferred_for": "General population, elderly", "avoid_if": ANTIHYPERTENSIVE_CONTRAINDICATIONS["thiazides"]},
    {"class": "ACE Inhibitors", "examples": "Lisinopril, Enalapril", 
     "preferred_for": "CKD, Diabetes, Heart failure", "avoid_if": ANTIHYPERTENSIVE_CONTRAINDICATIONS["ace_inhibitors"]},
    {"class": "ARBs", "examples": "Losartan, Valsartan", 
     "preferred_for": "ACEi intolerant, CKD, Diabetes", "avoid_if": ANTIHYPERTENSIVE_CONTRAINDICATIONS["arbs"]},
    {"class": "CCBs", "examples": "Amlodipine, Nifedipine", 
     "preferred_for": "General population, elderly", "avoid_if": ANTIHYPERTENSIVE_CONTRAINDICATIONS["ccbs"]}
]


# =============================================================================
# ACC/AHA 2018 CHOLESTEROL GUIDELINES
# =============================================================================

STATIN_INTENSITY = {
    "High": {
        "ldl_reduction": "≥50%",
        "agents": ["Atorvastatin 40-80 mg", "Rosuvastatin 20-40 mg"]
    },
    "Moderate": {
        "ldl_reduction": "30-49%",
        "agents": ["Atorvastatin 10-20 mg", "Rosuvastatin 5-10 mg", "Simvastatin 20-40 mg"]
    },
    "Low": {
        "ldl_reduction": "<30%",
        "agents": ["Simvastatin 10 mg", "Pravastatin 10-20 mg"]
    }
}

STATIN_ELIGIBILITY_RULES = [
    {"condition": "LDL-C ≥190 mg/dL", "check": lambda f: f.get("ldl", 0) >= 190,
     "therapy": "High-Intensity Statin", "grade": GradeOfRecommendation.I,
     "rationale": "Severe hypercholesterolemia warrants aggressive LDL-C lowering"},
    {"condition": "Diabetes (Age 40-75)", "check": lambda f: f.get("diabetes", 0) == 1 and 40 <= f.get("age", 0) <= 75,
     "therapy": "Moderate-Intensity Statin (High if risk enhancers)", "grade": GradeOfRecommendation.I,
     "rationale": "Diabetes is a major ASCVD risk equivalent"},
    {"condition": "10y ASCVD Risk ≥20%", "check": lambda f: f.get("ascvd_risk_10y", 0) >= 20,
     "therapy": "High-Intensity Statin", "grade": GradeOfRecommendation.I,
     "rationale": "High 10-year ASCVD risk requires intensive lipid therapy"},
    {"condition": "10y ASCVD Risk 7.5-19.9%", "check": lambda f: 7.5 <= f.get("ascvd_risk_10y", 0) < 20,
     "therapy": "Moderate-to-High Intensity Statin", "grade": GradeOfRecommendation.IIA,
     "rationale": "Intermediate risk; consider risk enhancers for intensity decision"},
]


# =============================================================================
# WHO 2020 PHYSICAL ACTIVITY GUIDELINES
# =============================================================================

PHYSICAL_ACTIVITY_TARGETS = {
    "adults_18_64": {
        "aerobic": "150-300 min/week moderate OR 75-150 min/week vigorous",
        "strength": "Muscle-strengthening 2+ days/week",
        "sedentary": "Limit sedentary time; replace with any activity"
    },
    "older_adults_65plus": {
        "aerobic": "Same as adults (150-300 min moderate OR 75-150 min vigorous)",
        "strength": "Muscle-strengthening 2+ days/week",
        "balance": "Balance and functional training 3+ days/week",
        "sedentary": "Limit sedentary time"
    },
    "chronic_conditions": {
        "aerobic": "As tolerated; avoid prolonged inactivity",
        "strength": "Muscle-strengthening 2+ days/week as tolerated",
        "note": "⚠️ Consult healthcare provider before starting exercise program"
    }
}


# =============================================================================
# EMERGENCY PROTOCOLS
# =============================================================================

EMERGENCY_PROTOCOLS = {
    "hypertensive_crisis": {
        "criteria": "SBP ≥180 mmHg OR DBP ≥120 mmHg",
        "immediate_action": "🚨 SEEK IMMEDIATE MEDICAL ATTENTION",
        "instructions": [
            "Call emergency services (911) or go to nearest ER",
            "Do not drive yourself",
            "Sit or lie down in a comfortable position",
            "If prescribed, take any rescue BP medication",
            "Monitor for symptoms: severe headache, chest pain, vision changes, confusion"
        ],
        "organ_damage_signs": ["Chest pain", "Shortness of breath", "Neurological symptoms", 
                               "Severe headache", "Vision changes", "Confusion"]
    },
    "suspected_mi": {
        "criteria": "Chest pain with risk factors",
        "immediate_action": "🚨 CALL 911 IMMEDIATELY",
        "instructions": [
            "Chew 325 mg aspirin (if not allergic)",
            "Sit or lie down",
            "Loosen tight clothing",
            "Be prepared to perform CPR if needed"
        ]
    }
}


# =============================================================================
# CLINICAL ADVISOR ENGINE
# =============================================================================

class ClinicalAdvisor:
    """
    Evidence-based clinical decision support engine.
    Generates personalized recommendations based on ACC/AHA & WHO guidelines.
    """
    
    def __init__(self):
        self.recommendations: List[Recommendation] = []
        self.warnings: List[str] = []
        self.emergency_alert: Optional[Dict] = None
    
    def classify_bp(self, sbp: float, dbp: float) -> str:
        """Classify BP per ACC/AHA 2017"""
        if sbp >= 180 or dbp >= 120:
            return "Hypertensive Crisis"
        elif sbp >= 140 or dbp >= 90:
            return "Stage 2 Hypertension"
        elif sbp >= 130 or dbp >= 80:
            return "Stage 1 Hypertension"
        elif sbp >= 120:
            return "Elevated"
        return "Normal"
    
    def check_allergies(self, allergies: List[str], med_class: str) -> List[str]:
        """Check for medication contraindications"""
        conflicts = []
        allergy_lower = [a.lower() for a in allergies]
        
        # Check statin allergies
        if med_class == "statin":
            for allergy in allergy_lower:
                if "statin" in allergy or "atorvastatin" in allergy or "rosuvastatin" in allergy:
                    conflicts.append(f"Patient allergic to {allergy}")
        
        # Check ACE inhibitor allergies
        if med_class == "ace_inhibitor":
            for allergy in allergy_lower:
                if "lisinopril" in allergy or "enalapril" in allergy or "ace" in allergy or "angioedema" in allergy:
                    conflicts.append(f"Contraindicated: {allergy}")
        
        return conflicts
    
    def get_bp_recommendation(self, sbp: float, dbp: float, 
                               has_high_risk: bool = False,
                               on_treatment: bool = False,
                               allergies: List[str] = None) -> Recommendation:
        """Generate BP management recommendation (ACC/AHA 2017)"""
        category = self.classify_bp(sbp, dbp)
        bp_info = BP_CATEGORIES[category]
        
        # Check for emergency
        if category == "Hypertensive Crisis":
            self.emergency_alert = EMERGENCY_PROTOCOLS["hypertensive_crisis"]
        
        action = bp_info["action"]
        treatment_note = None
        
        if category == "Stage 1 Hypertension" and has_high_risk:
            action = "Lifestyle + antihypertensive medication (high-risk patient)"
        
        if on_treatment:
            treatment_note = "⚠️ Patient on existing BP therapy - consider dose adjustment or adding agent"
            if category in ["Stage 1 Hypertension", "Stage 2 Hypertension"]:
                action = f"Current therapy may be inadequate. {action}"
        
        contraindicated = []
        if allergies:
            contraindicated = self.check_allergies(allergies, "ace_inhibitor")
        
        return Recommendation(
            category="Blood Pressure Management",
            action=action,
            rationale=f"BP {sbp}/{dbp} mmHg = {category}",
            grade=bp_info["grade"],
            urgency=bp_info["urgency"],
            guideline_source="ACC/AHA 2017 Hypertension Guideline",
            details=f"Target: <130/80 mmHg",
            contraindicated_if=contraindicated if contraindicated else None,
            treatment_consideration=treatment_note
        )
    
    def get_statin_recommendation(self, features: Dict[str, Any],
                                   on_statin: bool = False,
                                   allergies: List[str] = None) -> Optional[Recommendation]:
        """Evaluate statin eligibility (ACC/AHA 2018)"""
        for rule in STATIN_ELIGIBILITY_RULES:
            if rule["check"](features):
                treatment_note = None
                if on_statin:
                    treatment_note = "Patient already on statin therapy - verify intensity matches guideline"
                
                contraindicated = []
                if allergies:
                    contraindicated = self.check_allergies(allergies, "statin")
                    if contraindicated:
                        self.warnings.append(f"⚠️ Statin contraindicated: {', '.join(contraindicated)}")
                        return Recommendation(
                            category="Statin Therapy",
                            action="CONSULT SPECIALIST - Standard statin therapy contraindicated",
                            rationale=f"{rule['rationale']}, but patient has contraindication",
                            grade=rule["grade"],
                            urgency=UrgencyLevel.SOON,
                            guideline_source="ACC/AHA 2018 Cholesterol Guideline",
                            details="Consider alternative lipid-lowering therapy (e.g., PCSK9 inhibitor, ezetimibe)",
                            contraindicated_if=contraindicated,
                            treatment_consideration=treatment_note
                        )
                
                return Recommendation(
                    category="Statin Therapy",
                    action=rule["therapy"],
                    rationale=rule["rationale"],
                    grade=rule["grade"],
                    urgency=UrgencyLevel.SOON,
                    guideline_source="ACC/AHA 2018 Cholesterol Guideline",
                    details=f"Condition met: {rule['condition']}",
                    treatment_consideration=treatment_note
                )
        return None
    
    def get_activity_recommendation(self, age: int, 
                                     has_chronic_condition: bool = False) -> Recommendation:
        """Generate physical activity recommendation (WHO 2020)"""
        if has_chronic_condition:
            targets = PHYSICAL_ACTIVITY_TARGETS["chronic_conditions"]
            population = "Adults with chronic conditions"
        elif age >= 65:
            targets = PHYSICAL_ACTIVITY_TARGETS["older_adults_65plus"]
            population = "Older adults (65+)"
        else:
            targets = PHYSICAL_ACTIVITY_TARGETS["adults_18_64"]
            population = "Adults (18-64)"
        
        action = f"Aerobic: {targets['aerobic']}; Strength: {targets['strength']}"
        if "balance" in targets:
            action += f"; Balance: {targets['balance']}"
        
        return Recommendation(
            category="Physical Activity",
            action=action,
            rationale=f"WHO 2020 guidelines for {population}",
            grade=GradeOfRecommendation.I,
            urgency=UrgencyLevel.ROUTINE,
            guideline_source="WHO 2020 Physical Activity Guidelines",
            details=targets.get("note", "Any movement is better than none")
        )
    
    def get_smoking_recommendation(self, is_smoker: bool) -> Optional[Recommendation]:
        """Generate smoking cessation recommendation"""
        if is_smoker:
            return Recommendation(
                category="Smoking Cessation",
                action="Complete cessation strongly advised",
                rationale="CVD risk decreases within weeks; approaches non-smoker levels in 10-15 years",
                grade=GradeOfRecommendation.I,
                urgency=UrgencyLevel.SOON,
                guideline_source="ACC/AHA 2019 Primary Prevention Guideline",
                details="Offer: behavioral counseling + pharmacotherapy (varenicline, NRT, bupropion)"
            )
        return None
    
    def generate_recommendations(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive clinical recommendations.
        
        Args:
            features: Patient data with optional keys:
                - age, sex_code, systolic_bp, diastolic_bp
                - total_cholesterol, ldl, hdl, diabetes, smoking
                - ascvd_risk_10y
                - on_bp_meds: bool - currently on BP medication
                - on_statin: bool - currently on statin
                - allergies: List[str] - known allergies
        """
        recommendations = []
        self.warnings = []
        self.emergency_alert = None
        
        # Extract patient info
        sbp = features.get("systolic_bp", 120)
        dbp = features.get("diastolic_bp", 80)
        age = features.get("age", 50)
        allergies = features.get("allergies", [])
        on_bp_meds = features.get("on_bp_meds", False)
        on_statin = features.get("on_statin", False)
        
        has_high_risk = (features.get("ascvd_risk_10y", 0) >= 10 or 
                         features.get("diabetes", 0) == 1)
        
        # 1. Blood Pressure (ACC/AHA 2017)
        bp_rec = self.get_bp_recommendation(sbp, dbp, has_high_risk, on_bp_meds, allergies)
        recommendations.append(bp_rec)
        
        # 2. Statin Therapy (ACC/AHA 2018)
        statin_rec = self.get_statin_recommendation(features, on_statin, allergies)
        if statin_rec:
            recommendations.append(statin_rec)
        
        # 3. Physical Activity (WHO 2020)
        has_chronic = features.get("diabetes", 0) == 1
        activity_rec = self.get_activity_recommendation(age, has_chronic)
        recommendations.append(activity_rec)
        
        # 4. Smoking Cessation (ACC/AHA 2019)
        smoking_rec = self.get_smoking_recommendation(features.get("smoking", 0) == 1)
        if smoking_rec:
            recommendations.append(smoking_rec)
        
        # 5. Diet (ACC/AHA 2019)
        diet_rec = Recommendation(
            category="Dietary Pattern",
            action="DASH or Mediterranean diet",
            rationale="Evidence-based patterns reduce CVD events by 25-30%",
            grade=GradeOfRecommendation.I,
            urgency=UrgencyLevel.ROUTINE,
            guideline_source="ACC/AHA 2019 Primary Prevention Guideline",
            details="Sodium <2300 mg/day; emphasize fruits, vegetables, whole grains, lean protein"
        )
        recommendations.append(diet_rec)
        
        # Determine overall urgency
        urgency_summary = self._get_urgency_summary(recommendations)
        
        # Build output
        output = {
            "generated_at": datetime.now().isoformat(),
            "patient_summary": {
                "age": age,
                "bp": f"{sbp}/{dbp} mmHg",
                "bp_category": self.classify_bp(sbp, dbp),
                "diabetes": "Yes" if features.get("diabetes", 0) == 1 else "No",
                "smoker": "Yes" if features.get("smoking", 0) == 1 else "No",
                "on_bp_medication": "Yes" if on_bp_meds else "No",
                "on_statin": "Yes" if on_statin else "No",
                "allergies": allergies if allergies else "None reported"
            },
            "urgency_summary": urgency_summary,
            "recommendations": [self._format_recommendation(r) for r in recommendations],
            "warnings": self.warnings if self.warnings else None,
            "emergency_alert": self.emergency_alert,
            "guideline_sources": [
                "ACC/AHA 2017 - Hypertension",
                "ACC/AHA 2018 - Cholesterol", 
                "ACC/AHA 2019 - Primary Prevention",
                "WHO 2020 - Physical Activity"
            ],
            "disclaimer": "⚠️ This is clinical decision support only. All recommendations must be reviewed by a qualified healthcare provider."
        }
        
        return output
    
    def _format_recommendation(self, r: Recommendation) -> Dict:
        """Format recommendation for output"""
        rec = {
            "category": r.category,
            "action": r.action,
            "rationale": r.rationale,
            "grade": f"Class {r.grade.value}",
            "urgency": r.urgency.value,
            "source": r.guideline_source,
            "details": r.details
        }
        if r.contraindicated_if:
            rec["⚠️ contraindications"] = r.contraindicated_if
        if r.treatment_consideration:
            rec["treatment_note"] = r.treatment_consideration
        return rec
    
    def _get_urgency_summary(self, recommendations: List[Recommendation]) -> str:
        """Determine overall urgency"""
        urgencies = [r.urgency for r in recommendations]
        if UrgencyLevel.EMERGENCY in urgencies:
            return "🚨 EMERGENCY - SEEK IMMEDIATE MEDICAL ATTENTION"
        elif UrgencyLevel.URGENT in urgencies:
            return "⚠️ URGENT - See doctor within 24-48 hours"
        elif UrgencyLevel.SOON in urgencies:
            return "📋 SOON - Schedule appointment within 1-2 weeks"
        return "✅ ROUTINE - Continue preventive care"


# =============================================================================
# OUTPUT FORMATTER
# =============================================================================

def format_recommendations_text(result: Dict[str, Any]) -> str:
    """Format recommendations as readable clinical report"""
    lines = []
    lines.append("=" * 70)
    lines.append("  CARDIODETECT CLINICAL ADVISORY REPORT")
    lines.append("  Evidence-Based Cardiovascular Risk Management")
    lines.append("=" * 70)
    lines.append(f"  Generated: {result.get('generated_at', 'N/A')}")
    
    # Emergency Alert
    if result.get("emergency_alert"):
        ea = result["emergency_alert"]
        lines.append("\n" + "!" * 70)
        lines.append(f"  {ea['immediate_action']}")
        lines.append("!" * 70)
        for instr in ea["instructions"]:
            lines.append(f"  • {instr}")
        lines.append("")
    
    # Patient Summary
    ps = result["patient_summary"]
    lines.append(f"\n📋 PATIENT SUMMARY")
    lines.append(f"   Age: {ps['age']} years")
    lines.append(f"   Blood Pressure: {ps['bp']} ({ps['bp_category']})")
    lines.append(f"   Diabetes: {ps['diabetes']}")
    lines.append(f"   Smoker: {ps['smoker']}")
    lines.append(f"   On BP Medication: {ps['on_bp_medication']}")
    lines.append(f"   On Statin: {ps['on_statin']}")
    lines.append(f"   Allergies: {ps['allergies']}")
    
    # Urgency
    lines.append(f"\n{result['urgency_summary']}")
    
    # Warnings
    if result.get("warnings"):
        lines.append("\n⚠️ WARNINGS:")
        for w in result["warnings"]:
            lines.append(f"   {w}")
    
    # Recommendations
    lines.append(f"\n📌 CLINICAL RECOMMENDATIONS")
    lines.append("-" * 50)
    
    for i, rec in enumerate(result["recommendations"], 1):
        lines.append(f"\n{i}. {rec['category'].upper()}")
        lines.append(f"   Action: {rec['action']}")
        lines.append(f"   Rationale: {rec['rationale']}")
        lines.append(f"   Evidence: {rec['source']} ({rec['grade']})")
        lines.append(f"   Urgency: {rec['urgency']}")
        if rec.get('details'):
            lines.append(f"   Details: {rec['details']}")
        if rec.get('⚠️ contraindications'):
            lines.append(f"   ⚠️ Contraindications: {', '.join(rec['⚠️ contraindications'])}")
        if rec.get('treatment_note'):
            lines.append(f"   📝 {rec['treatment_note']}")
    
    # Footer
    lines.append("\n" + "-" * 50)
    lines.append("📚 GUIDELINE SOURCES:")
    for src in result["guideline_sources"]:
        lines.append(f"   • {src}")
    
    lines.append(f"\n{result['disclaimer']}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    advisor = ClinicalAdvisor()
    
    # Test 1: Emergency case
    print("\n" + "="*70)
    print("TEST 1: HYPERTENSIVE CRISIS")
    print("="*70)
    crisis_patient = {
        "age": 65,
        "systolic_bp": 195,
        "diastolic_bp": 125,
        "diabetes": 1,
        "smoking": 1,
        "on_bp_meds": True,
        "allergies": ["lisinopril", "aspirin"]
    }
    result1 = advisor.generate_recommendations(crisis_patient)
    print(format_recommendations_text(result1))
    
    # Test 2: Patient on treatment with allergies
    print("\n" + "="*70)
    print("TEST 2: PATIENT ON TREATMENT WITH ALLERGIES")
    print("="*70)
    treated_patient = {
        "age": 58,
        "systolic_bp": 142,
        "diastolic_bp": 88,
        "ldl": 165,
        "diabetes": 0,
        "smoking": 0,
        "on_bp_meds": True,
        "on_statin": True,
        "allergies": ["atorvastatin"],
        "ascvd_risk_10y": 12.5
    }
    result2 = advisor.generate_recommendations(treated_patient)
    print(format_recommendations_text(result2))
