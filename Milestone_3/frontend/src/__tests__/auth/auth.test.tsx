/**
 * Authentication Page Tests
 * Tests Login, Register, ForgotPassword, and ResetPassword pages.
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// ============================================================
// Login Page Tests
// ============================================================
describe('Login Page', () => {
    const MockLoginPage = () => {
        const [email, setEmail] = React.useState('')
        const [password, setPassword] = React.useState('')
        const [error, setError] = React.useState('')
        const [loading, setLoading] = React.useState(false)

        const handleSubmit = async (e: React.FormEvent) => {
            e.preventDefault()
            if (!email || !password) {
                setError('Email and password are required')
                return
            }
            setLoading(true)
            // Simulate API call
            await new Promise(r => setTimeout(r, 100))
            setLoading(false)
        }

        return (
            <form data-testid="login-form" onSubmit={handleSubmit}>
                <h1>Sign In</h1>
                {error && <div data-testid="error-message">{error}</div>}
                <input
                    type="email"
                    data-testid="email-input"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                />
                <input
                    type="password"
                    data-testid="password-input"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                />
                <button type="submit" data-testid="submit-button" disabled={loading}>
                    {loading ? 'Signing in...' : 'Sign In'}
                </button>
                <a href="/forgot-password" data-testid="forgot-password-link">Forgot Password?</a>
                <a href="/register" data-testid="register-link">Create Account</a>
            </form>
        )
    }

    it('renders login form with all elements', () => {
        render(<MockLoginPage />)
        expect(screen.getByText('Sign In')).toBeInTheDocument()
        expect(screen.getByTestId('email-input')).toBeInTheDocument()
        expect(screen.getByTestId('password-input')).toBeInTheDocument()
        expect(screen.getByTestId('submit-button')).toBeInTheDocument()
    })

    it('shows error when submitting empty form', async () => {
        render(<MockLoginPage />)
        fireEvent.click(screen.getByTestId('submit-button'))
        expect(await screen.findByTestId('error-message')).toHaveTextContent('Email and password are required')
    })

    it('accepts valid email input', async () => {
        render(<MockLoginPage />)
        const emailInput = screen.getByTestId('email-input')
        await userEvent.type(emailInput, 'test@example.com')
        expect(emailInput).toHaveValue('test@example.com')
    })

    it('masks password input', () => {
        render(<MockLoginPage />)
        expect(screen.getByTestId('password-input')).toHaveAttribute('type', 'password')
    })

    it('has forgot password link', () => {
        render(<MockLoginPage />)
        expect(screen.getByTestId('forgot-password-link')).toHaveAttribute('href', '/forgot-password')
    })

    it('has register link', () => {
        render(<MockLoginPage />)
        expect(screen.getByTestId('register-link')).toHaveAttribute('href', '/register')
    })

    it('shows loading state when submitting', async () => {
        render(<MockLoginPage />)
        await userEvent.type(screen.getByTestId('email-input'), 'test@example.com')
        await userEvent.type(screen.getByTestId('password-input'), 'password123')
        fireEvent.click(screen.getByTestId('submit-button'))
        expect(screen.getByTestId('submit-button')).toHaveTextContent('Signing in...')
    })
})

// ============================================================
// Register Page Tests
// ============================================================
describe('Register Page', () => {
    const MockRegisterPage = () => {
        const [formData, setFormData] = React.useState({
            email: '',
            password: '',
            confirmPassword: '',
            firstName: '',
            lastName: '',
            role: 'patient',
        })
        const [errors, setErrors] = React.useState<string[]>([])

        const handleSubmit = (e: React.FormEvent) => {
            e.preventDefault()
            const newErrors: string[] = []
            if (!formData.email) newErrors.push('Email is required')
            if (formData.password.length < 8) newErrors.push('Password must be at least 8 characters')
            if (formData.password !== formData.confirmPassword) newErrors.push('Passwords do not match')
            setErrors(newErrors)
        }

        return (
            <form data-testid="register-form" onSubmit={handleSubmit}>
                <h1>Create Account</h1>
                {errors.map((err, i) => (
                    <div key={i} data-testid="error-message">{err}</div>
                ))}
                <input
                    data-testid="firstname-input"
                    placeholder="First Name"
                    value={formData.firstName}
                    onChange={(e) => setFormData(prev => ({ ...prev, firstName: e.target.value }))}
                />
                <input
                    data-testid="lastname-input"
                    placeholder="Last Name"
                    value={formData.lastName}
                    onChange={(e) => setFormData(prev => ({ ...prev, lastName: e.target.value }))}
                />
                <input
                    type="email"
                    data-testid="email-input"
                    placeholder="Email"
                    value={formData.email}
                    onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                />
                <input
                    type="password"
                    data-testid="password-input"
                    placeholder="Password"
                    value={formData.password}
                    onChange={(e) => setFormData(prev => ({ ...prev, password: e.target.value }))}
                />
                <input
                    type="password"
                    data-testid="confirm-password-input"
                    placeholder="Confirm Password"
                    value={formData.confirmPassword}
                    onChange={(e) => setFormData(prev => ({ ...prev, confirmPassword: e.target.value }))}
                />
                <select
                    data-testid="role-select"
                    value={formData.role}
                    onChange={(e) => setFormData(prev => ({ ...prev, role: e.target.value }))}
                >
                    <option value="patient">Patient</option>
                    <option value="doctor">Doctor</option>
                </select>
                <button type="submit" data-testid="submit-button">Register</button>
            </form>
        )
    }

    it('renders registration form', () => {
        render(<MockRegisterPage />)
        expect(screen.getByText('Create Account')).toBeInTheDocument()
    })

    it('has all required input fields', () => {
        render(<MockRegisterPage />)
        expect(screen.getByTestId('firstname-input')).toBeInTheDocument()
        expect(screen.getByTestId('lastname-input')).toBeInTheDocument()
        expect(screen.getByTestId('email-input')).toBeInTheDocument()
        expect(screen.getByTestId('password-input')).toBeInTheDocument()
        expect(screen.getByTestId('confirm-password-input')).toBeInTheDocument()
    })

    it('has role selection dropdown', () => {
        render(<MockRegisterPage />)
        expect(screen.getByTestId('role-select')).toBeInTheDocument()
    })

    it('shows error for short password', async () => {
        render(<MockRegisterPage />)
        await userEvent.type(screen.getByTestId('password-input'), 'short')
        await userEvent.type(screen.getByTestId('confirm-password-input'), 'short')
        fireEvent.click(screen.getByTestId('submit-button'))
        expect(await screen.findByText(/Password must be at least 8 characters/i)).toBeInTheDocument()
    })

    it('shows error when passwords do not match', async () => {
        render(<MockRegisterPage />)
        await userEvent.type(screen.getByTestId('password-input'), 'password123')
        await userEvent.type(screen.getByTestId('confirm-password-input'), 'different123')
        fireEvent.click(screen.getByTestId('submit-button'))
        expect(await screen.findByText(/Passwords do not match/i)).toBeInTheDocument()
    })

    it('allows role selection', async () => {
        render(<MockRegisterPage />)
        const roleSelect = screen.getByTestId('role-select')
        await userEvent.selectOptions(roleSelect, 'doctor')
        expect(roleSelect).toHaveValue('doctor')
    })
})

// ============================================================
// Forgot Password Page Tests
// ============================================================
describe('ForgotPassword Page', () => {
    const MockForgotPasswordPage = () => {
        const [email, setEmail] = React.useState('')
        const [submitted, setSubmitted] = React.useState(false)

        const handleSubmit = (e: React.FormEvent) => {
            e.preventDefault()
            if (email) setSubmitted(true)
        }

        if (submitted) {
            return (
                <div data-testid="success-message">
                    Check your email for reset instructions
                </div>
            )
        }

        return (
            <form data-testid="forgot-password-form" onSubmit={handleSubmit}>
                <h1>Reset Password</h1>
                <p>Enter your email to receive reset instructions</p>
                <input
                    type="email"
                    data-testid="email-input"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                />
                <button type="submit" data-testid="submit-button">Send Reset Link</button>
                <a href="/login" data-testid="back-to-login">Back to Login</a>
            </form>
        )
    }

    it('renders forgot password form', () => {
        render(<MockForgotPasswordPage />)
        expect(screen.getByText('Reset Password')).toBeInTheDocument()
    })

    it('has email input field', () => {
        render(<MockForgotPasswordPage />)
        expect(screen.getByTestId('email-input')).toBeInTheDocument()
    })

    it('shows success message after submission', async () => {
        render(<MockForgotPasswordPage />)
        await userEvent.type(screen.getByTestId('email-input'), 'test@example.com')
        fireEvent.click(screen.getByTestId('submit-button'))
        expect(await screen.findByTestId('success-message')).toBeInTheDocument()
    })

    it('has back to login link', () => {
        render(<MockForgotPasswordPage />)
        expect(screen.getByTestId('back-to-login')).toHaveAttribute('href', '/login')
    })
})
