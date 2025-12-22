# CardioDetect User Manual
## Comprehensive Guide for Patients, Doctors, and Administrators

**Version:** 1.0  
**Date:** December 2025

---

# Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
   - [Registration & Login](#registration--login)
   - [Password Recovery](#password-recovery)
3. [Patient User Guide](#3-patient-user-guide)
   - [Dashboard Overview](#dashboard-overview)
   - [Running a Risk Assessment](#running-a-risk-assessment)
   - [Viewing History](#viewing-history)
   - [Managing Your Profile](#managing-your-profile)
4. [Doctor User Guide](#4-doctor-user-guide)
   - [Doctor Dashboard](#doctor-dashboard)
   - [Patient Management](#patient-management)
   - [Uploading Medical Reports (OCR)](#uploading-medical-reports-ocr)
   - [Analyzing Results](#analyzing-results)
5. [Administrator User Guide](#5-administrator-user-guide)
   - [Admin Panel Overview](#admin-panel-overview)
   - [User Management](#user-management)
   - [Approving Profile Changes](#approving-profile-changes)
   - [System Analytics](#system-analytics)
6. [Troubleshooting & Support](#6-troubleshooting--support)

---

# 1. Introduction

Welcome to **CardioDetect**, an advanced AI-powered platform designed to assess cardiovascular disease risk early and accurately. This manual will guide you through the features specific to your role.

**Key Features:**
*   **AI Risk Prediction:** Instant analysis using advanced machine learning.
*   **Medical Report OCR:** Upload PDF/Image reports for automatic data extraction.
*   **Secure Platform:** HIPAA-compliant security with role-based access.

---

# 2. Getting Started

## Registration & Login

1.  Navigate to the **Home Page**.
2.  Click **"Get Started"** or **"Register"**.
3.  Fill in your details:
    *   **Name:** Full legal name.
    *   **Email:** Valid email address (used for verification).
    *   **Password:** Strong password (min 8 chars, mixed case, numbers).
    *   **Role:** Select "Patient" or "Doctor" (Doctor accounts require admin verification).
4.  Click **"Create Account"**.
5.  Check your email for a **Verification Link** and click it to activate your account.

## Password Recovery

If you forget your password:
1.  Click **"Forgot Password?"** on the Login screen.
2.  Enter your registered email.
3.  Click the link in the password reset email.
4.  Set a new secure password.

---

# 3. Patient User Guide

As a patient, you can monitor your heart health, run assessments, and track changes over time.

## Dashboard Overview
Your command center showing:
*   **Latest Risk Score:** Your most recent heart health assessment.
*   **Quick Actions:** Buttons to "New Assessment" or "View History".
*   **Health Trends:** Simple charts showing your progress.

## Running a Risk Assessment
1.  Click **"Assess My Risk"**.
2.  **Manual Entry:** Fill in the health form (Age, Blood Pressure, Cholesterol, etc.).
    *   *Tip:* If you don't know a value, use the default average provided or leave it blank if allowed.
3.  Click **"Calculate Risk"**.
4.  **View Results:**
    *   **Risk Level:** Low (Green), Moderate (Yellow), High (Red).
    *   **Probability:** Percentage chance of heart issues in the next 10 years.
    *   **Recommendations:** Personalized lifestyle advice based on your inputs.
5.  **Download Report:** Click "Download PDF" to save a copy for your records.

## Viewing History
*   Click **"History"** to see a list of all your past assessments.
*   Use the **Trends Graph** to see if your risk is increasing or decreasing.
*   Click on any past entry to view full details.

## Managing Your Profile
*   Go to **"Profile"** to update personal details (Phone, Address).
*   **Sensitive Changes:** Changing critical health data (like Date of Birth) may require **Admin Approval** for security. You will be notified when approved.
*   **Data Rights:** You can request a "Data Export" or "Account Deletion" from the Settings tab (GDPR compliance).

---

# 4. Doctor User Guide

Doctors have powerful tools to manage patients and analyze medical documents efficiently.

## Doctor Dashboard
*   **Patient List:** Quick access to your assigned patients.
*   **Recent Uploads:** Status of processed medical reports.
*   **Risk Alerts:** Notifications for patients flagged as "High Risk".

## Patient Management
1.  Go to **"My Patients"**.
2.  **Search:** Find patients by name or ID.
3.  **Assign Patient:** Use the "Add Patient" button to link a patient to your care (requires patient consent/code).
4.  **View Profile:** Click a patient to see their full assessment history and trends.

## Uploading Medical Reports (OCR)
Eliminate manual data entry by uploading scanned reports.

1.  Click **"Upload Report"**.
2.  Select a patient.
3.  **Drag & Drop** a file (PDF, JPG, PNG).
    *   *Supported:* Lab reports, clinical summaries.
    *   *Max Size:* 10MB.
4.  Click **"Analyze"**.
5.  **Review Extracted Data:** The AI will extract values (Cholesterol, BP, etc.).
    *   *Verify:* Check the "Confidence Score".
    *   *Edit:* Correct any misread numbers before submitting.
6.  **Confirm:** The system runs the risk prediction automatically using the extracted data.

## Analyzing Results
*   **Clinical Report:** Generated PDFs include:
    *   Patient Demographics.
    *   Extracted Vitals.
    *   AI Risk Assessment.
    *   **SHAP Analysis:** "Why this prediction?" (Explains top contributing factors like 'High Systolic BP').

---

# 5. Administrator User Guide

Admins oversee the platform's security, user base, and data integrity.

## Admin Panel Overview
Access the comprehensive dashboard at `/admin`.
*   **System Stats:** Total users, daily assessments, error rates.
*   **Pending Actions:** Approval requests, user verifications.

## User Management
*   **View Users:** List all registered Patients and Doctors.
*   **Actions:**
    *   **Deactivate:** Suspend suspicious accounts.
    *   **Reset MFA:** Assist users locked out of accounts.
    *   **Assign Role:** Upgrade a user to "Doctor" or "Admin" status.

## Approving Profile Changes
For data integrity, sensitive profile updates (e.g., Name, DOB) enter a "Pending" state.
1.  Go to **"Approvals"**.
2.  Review the **"Pending Changes"** list.
3.  Compare "Old Value" vs. "New Value".
4.  Click **"Approve"** to update the potentially or **"Reject"** to discard.

## System Analytics
*   **Prediction Logs:** detailed audit trail of every prediction run.
*   **Performance Metrics:** Monitor API latency and ML model accuracy.
*   **Security Logs:** View failed login attempts and potential brute-force attacks.

---

# 6. Troubleshooting & Support

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| **Login Failed** | Incorrect password or unverified email | Check "Caps Lock" or check email for verification link. |
| **OCR Reading Error** | Low quality image or handwriting | Use a scanned PDF or clear photo. Ensure text is typed, not handwritten. |
| **"Access Denied"** | Insufficient permissions | Contact Admin if you need Doctor access. |
| **Report Download Fails** | Pop-up blocker | Allow pop-ups for CardioDetect. |

**Still need help?** Contact support at `support@cardiodetect.com`.
