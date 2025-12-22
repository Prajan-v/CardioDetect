/**
 * Centralized API Client for CardioDetect
 * All API endpoints should use this module instead of hardcoded URLs
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

/**
 * Get the full API URL for an endpoint
 */
export const getApiUrl = (endpoint: string): string => {
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
    return `${API_BASE}/${cleanEndpoint}`;
};

/**
 * API Endpoints - centralized configuration
 */
export const API_ENDPOINTS = {
    // Auth endpoints
    auth: {
        login: () => getApiUrl('auth/login/'),
        register: () => getApiUrl('auth/register/'),
        logout: () => getApiUrl('auth/logout/'),
        profile: () => getApiUrl('auth/profile/'),
        profileChanges: () => getApiUrl('auth/profile-changes/'),
        profileChangesSubmit: () => getApiUrl('auth/profile-changes/submit/'),
        passwordChange: () => getApiUrl('auth/password-change/'),
        passwordReset: () => getApiUrl('auth/password-reset/'),
        passwordResetConfirm: () => getApiUrl('auth/password-reset/confirm/'),
        verifyEmail: (email: string, token: string) =>
            getApiUrl(`auth/verify-email/?email=${encodeURIComponent(email)}&token=${encodeURIComponent(token)}`),
        resendVerification: () => getApiUrl('auth/resend-verification/'),
        adminPendingChanges: () => getApiUrl('auth/admin/pending-changes/'),
        adminApprove: (changeId: number) => getApiUrl(`auth/admin/approve/${changeId}/`),
        adminReject: (changeId: number) => getApiUrl(`auth/admin/reject/${changeId}/`),
    },

    // Prediction endpoints
    predict: {
        manual: () => getApiUrl('predict/manual/'),
        ocr: () => getApiUrl('predict/ocr/'),
        base: () => getApiUrl('predict/'),
    },

    // Doctor endpoints
    doctor: {
        dashboard: () => getApiUrl('doctor/dashboard/'),
        patients: () => getApiUrl('doctor/patients/'),
        patientDetail: (patientId: string) => getApiUrl(`doctor/patients/${patientId}/`),
        exportPatientsExcel: () => getApiUrl('doctor/patients/export/excel/'),
    },

    // Admin endpoints
    admin: {
        stats: () => getApiUrl('admin/stats/'),
    },

    // Activity/History endpoints
    activity: {
        history: () => getApiUrl('history/'),
        statistics: () => getApiUrl('statistics/'),
        patientDashboard: () => getApiUrl('patient/dashboard/'),
        exportExcel: () => getApiUrl('history/export/excel/'),
    },

    // Prediction export/PDF
    predictions: {
        detail: (id: string) => getApiUrl(`predictions/${id}/`),
        pdf: (id: string) => getApiUrl(`predictions/${id}/pdf/`),
    },

    // Other endpoints
    notifications: () => getApiUrl('notifications/'),
    notificationsRead: () => getApiUrl('notifications/read/'),
    health: () => getApiUrl('health/'),
} as const;

/**
 * Fetch wrapper with authentication
 */
export async function apiFetch(
    endpoint: string,
    options: RequestInit = {}
): Promise<Response> {
    const token = typeof window !== 'undefined'
        ? localStorage.getItem('auth_token')
        : null;

    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...(options.headers as Record<string, string>),
    };

    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    return fetch(endpoint, {
        ...options,
        headers,
    });
}

export default API_ENDPOINTS;
