/**
 * CardioDetect - Main JavaScript
 * Handles animations, form interactions, and API calls
 */

document.addEventListener('DOMContentLoaded', function () {
    console.log('CardioDetect initialized');

    // Auto-hide messages after 5 seconds
    const messages = document.querySelectorAll('.message');
    messages.forEach(msg => {
        setTimeout(() => {
            msg.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => msg.remove(), 300);
        }, 5000);
    });
});

/**
 * Unit Conversion Functions
 */
const UnitConverter = {
    // Cholesterol: mg/dL <-> mmol/L
    cholesterolToMmol: (mgdl) => (mgdl / 38.67).toFixed(2),
    cholesterolToMgdl: (mmol) => Math.round(mmol * 38.67),

    // Glucose: mg/dL <-> mmol/L
    glucoseToMmol: (mgdl) => (mgdl / 18.0).toFixed(2),
    glucoseToMgdl: (mmol) => Math.round(mmol * 18.0),

    // Weight: kg <-> lbs
    kgToLbs: (kg) => (kg * 2.205).toFixed(1),
    lbsToKg: (lbs) => (lbs / 2.205).toFixed(1),

    // Height: cm <-> feet
    cmToFeet: (cm) => (cm / 30.48).toFixed(1),
    feetToCm: (feet) => Math.round(feet * 30.48),

    // BMI Calculator
    calculateBMI: (weight, height, weightUnit = 'kg', heightUnit = 'cm') => {
        let w = weightUnit === 'lbs' ? weight / 2.205 : weight;
        let h = heightUnit === 'ft' ? height * 30.48 : height;
        return (w / ((h / 100) ** 2)).toFixed(1);
    }
};

/**
 * Form Validation
 */
function validatePredictionForm(form) {
    const requiredFields = ['age', 'sex', 'systolic_bp', 'cholesterol', 'smoking'];
    let isValid = true;

    requiredFields.forEach(field => {
        const input = form.querySelector(`[name="${field}"]`);
        if (!input || !input.value) {
            isValid = false;
            input?.classList.add('error');
        } else {
            input?.classList.remove('error');
        }
    });

    return isValid;
}

/**
 * API Helpers
 */
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'include'
    };

    // Add CSRF token for non-GET requests
    if (method !== 'GET') {
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
        if (csrfToken) {
            options.headers['X-CSRFToken'] = csrfToken;
        }
    }

    if (data) {
        options.body = JSON.stringify(data);
    }

    const response = await fetch(endpoint, options);
    return response.json();
}

/**
 * File Upload Handler
 */
function setupFileUpload(dropZoneId, inputId, previewId) {
    const dropZone = document.getElementById(dropZoneId);
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);

    if (!dropZone || !input) return;

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(event => {
        dropZone.addEventListener(event, () => {
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length) {
            input.files = files;
            showFilePreview(files[0], preview);
        }
    });

    input.addEventListener('change', () => {
        if (input.files.length) {
            showFilePreview(input.files[0], preview);
        }
    });
}

function showFilePreview(file, previewEl) {
    if (!previewEl) return;

    const fileInfo = `
        <div class="file-preview">
            <i class="fas fa-file-medical"></i>
            <div class="file-info">
                <span class="file-name">${file.name}</span>
                <span class="file-size">${(file.size / 1024).toFixed(1)} KB</span>
            </div>
        </div>
    `;
    previewEl.innerHTML = fileInfo;
}

/**
 * Risk Color Helper
 */
function getRiskColor(category) {
    const colors = {
        'LOW': '#00D9C4',
        'MODERATE': '#FFB800',
        'HIGH': '#FF4757'
    };
    return colors[category] || '#808080';
}

// Export for global use
window.CardioDetect = {
    UnitConverter,
    validatePredictionForm,
    apiCall,
    setupFileUpload,
    getRiskColor
};
