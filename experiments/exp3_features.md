# Experiment 3: Engineered Features

This file documents the 10 engineered features used in
`exp3_feature_engineered_rf.py`.

1. **bp_age_ratio** = `systolic_bp / (age + 1)`  
   Blood pressure scaled by age.

2. **metabolic_load** = `(bmi / 25) * (glucose / 100) * (cholesterol / 200)`  
   Combined metabolic burden score.

3. **risk_score** = `age * 0.3 + systolic_bp * 0.2 + bmi * 0.15`  
   Simple linear risk proxy combining age, BP, and BMI.

4. **bp_control** = `hypertension * (1 - bpmeds)`  
   Indicates uncontrolled hypertension when treatment is absent.

5. **age_squared** = `age ** 2`  
   Captures non-linear age effects.

6. **bmi_cholesterol_interaction** = `bmi * cholesterol`  
   Interaction between body mass and lipid levels.

7. **diabetes_glucose_interaction** = `diabetes * glucose`  
   Emphasizes high glucose values in diabetic patients.

8. **smoking_age_interaction** = `smoking * age`  
   Higher values for older smokers.

9. **pulse_pressure_squared** = `pulse_pressure ** 2`  
   Non-linear effect of arterial stiffness proxy.

10. **metabolic_syndrome_count** = sum of
    `hypertension_flag`, `high_cholesterol_flag`, `high_glucose_flag`,
    `obesity_flag` (when available; otherwise set to 0)  
    Counts the number of major metabolic risk flags.
