/**
 * Authentication Service
 * Connects frontend to Django JWT authentication
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Export for use in other components
export const getApiBase = () => API_BASE;
export const getApiUrl = (endpoint: string) => `${API_BASE}${endpoint}`;


export interface User {
    id: string;
    email: string;
    full_name: string;
    first_name?: string;
    last_name?: string;
    role: 'patient' | 'doctor' | 'admin';
    email_verified?: boolean;
    // Doctor specific
    license_number?: string;
    specialization?: string;
    hospital?: string;
}


export interface AuthTokens {
    access: string;
    refresh: string;
}

export interface LoginCredentials {
    email: string;
    password: string;
}

export interface RegisterData {
    email: string;
    password: string;
    full_name: string;
    role: 'patient' | 'doctor';
    license_number?: string;
}


// Token storage keys
const ACCESS_TOKEN_KEY = 'auth_token';
const REFRESH_TOKEN_KEY = 'refresh_token';
const USER_KEY = 'user';

/**
 * Store tokens in localStorage
 */
export function setTokens(tokens: AuthTokens): void {
    localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access);
    localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh);
}

/**
 * Get access token
 */
export function getAccessToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(ACCESS_TOKEN_KEY);
}

// Alias for getAccessToken
export const getToken = getAccessToken;


/**
 * Get refresh token
 */
export function getRefreshToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(REFRESH_TOKEN_KEY);
}

/**
 * Store user in localStorage
 */
export function setUser(user: User): void {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
}

/**
 * Get current user
 */
export function getUser(): User | null {
    if (typeof window === 'undefined') return null;
    const userStr = localStorage.getItem(USER_KEY);
    return userStr ? JSON.parse(userStr) : null;
}

/**
 * Clear all auth data
 */
export function clearAuth(): void {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
    return !!getAccessToken();
}

/**
 * Login user
 */
export async function login(credentials: LoginCredentials): Promise<{ success: boolean; error?: string; user?: User }> {
    try {
        // Clear any existing auth data first
        clearAuth();

        const response = await fetch(`${API_BASE}/auth/login/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(credentials),
        });

        const data = await response.json();

        if (response.ok) {            // Backend returns: { access, refresh, user } - NOT { tokens: {access, refresh} }
            if (data.access || data.tokens) {
                // Handle both formats
                const tokens = data.tokens || { access: data.access, refresh: data.refresh };
                setTokens(tokens);
                if (data.user) {
                    setUser(data.user);
                }
                return { success: true, user: data.user };
            }

            return { success: false, error: 'Invalid response from server' };
        } else {
            return { success: false, error: data.error || data.detail || 'Login failed' };
        }
    } catch (error) {
        console.error('Login error:', error);
        return { success: false, error: 'Network error - is the server running?' };
    }
}


/**
 * Register new user
 */
export async function register(data: RegisterData): Promise<{ success: boolean; error?: string }> {
    try {
        // Split full_name into first_name and last_name
        const nameParts = data.full_name.trim().split(' ');
        const firstName = nameParts[0] || '';
        const lastName = nameParts.slice(1).join(' ') || '';

        // Build the request body with all required fields
        const requestBody: Record<string, unknown> = {
            email: data.email,
            password: data.password,
            password_confirm: data.password,  // Backend requires this
            first_name: firstName,
            last_name: lastName,
            role: data.role,
            terms_accepted: true,
            privacy_accepted: true,
        };

        // Add license_number for doctors
        if (data.role === 'doctor' && data.license_number) {
            requestBody.license_number = data.license_number;
        }


        const response = await fetch(`${API_BASE}/auth/register/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });

        const result = await response.json();

        if (response.ok) {
            return { success: true };
        } else {
            // Format error message
            if (result.errors) {
                const errorMessages = Object.values(result.errors).flat();
                return { success: false, error: (errorMessages as string[]).join('. ') };
            }
            return { success: false, error: result.error || result.detail || 'Registration failed' };
        }
    } catch (error) {
        console.error('Register error:', error);
        return { success: false, error: 'Network error' };
    }
}


/**
 * Logout user
 */
export function logout(): void {
    clearAuth();
    window.location.href = '/login';
}

/**
 * Refresh access token
 */
export async function refreshAccessToken(): Promise<boolean> {
    const refreshToken = getRefreshToken();
    if (!refreshToken) return false;

    try {
        const response = await fetch(`${API_BASE}/auth/token/refresh/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh: refreshToken }),
        });

        if (response.ok) {
            const data = await response.json();
            localStorage.setItem(ACCESS_TOKEN_KEY, data.access);
            return true;
        } else {
            clearAuth();
            return false;
        }
    } catch {
        return false;
    }
}

/**
 * Fetch with authentication
 */
export async function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
    const token = getAccessToken();

    const headers = new Headers(options.headers);
    if (token) {
        headers.set('Authorization', `Bearer ${token}`);
    }

    const response = await fetch(url, { ...options, headers });

    // If unauthorized, try to refresh token
    if (response.status === 401) {
        const refreshed = await refreshAccessToken();
        if (refreshed) {
            headers.set('Authorization', `Bearer ${getAccessToken()}`);
            return fetch(url, { ...options, headers });
        }
    }

    return response;
}
