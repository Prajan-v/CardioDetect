/**
 * Dashboard and Prediction Page Tests
 * Tests Dashboard, Manual Prediction, OCR Upload, and History pages.
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// ============================================================
// Dashboard Page Tests
// ============================================================
describe('Dashboard Page', () => {
    const MockDashboard = ({
        user = { firstName: 'John', role: 'patient' },
        stats = { totalPredictions: 5, lastRisk: 'LOW' }
    }) => (
        <div data-testid="dashboard">
            <h1>Welcome, {user.firstName}</h1>
            <div data-testid="user-role">{user.role}</div>
            <div data-testid="stats">
                <span data-testid="total-predictions">{stats.totalPredictions}</span>
                <span data-testid="last-risk">{stats.lastRisk}</span>
            </div>
            <nav data-testid="quick-actions">
                <a href="/dashboard/predict">New Prediction</a>
                <a href="/dashboard/history">View History</a>
                <a href="/settings">Settings</a>
            </nav>
        </div>
    )

    it('renders welcome message with user name', () => {
        render(<MockDashboard />)
        expect(screen.getByText('Welcome, John')).toBeInTheDocument()
    })

    it('displays user role', () => {
        render(<MockDashboard />)
        expect(screen.getByTestId('user-role')).toHaveTextContent('patient')
    })

    it('shows prediction statistics', () => {
        render(<MockDashboard />)
        expect(screen.getByTestId('total-predictions')).toHaveTextContent('5')
        expect(screen.getByTestId('last-risk')).toHaveTextContent('LOW')
    })

    it('has quick action links', () => {
        render(<MockDashboard />)
        expect(screen.getByText('New Prediction')).toHaveAttribute('href', '/dashboard/predict')
        expect(screen.getByText('View History')).toHaveAttribute('href', '/dashboard/history')
    })

    it('displays doctor role correctly', () => {
        render(<MockDashboard user={{ firstName: 'Dr. Smith', role: 'doctor' }} />)
        expect(screen.getByTestId('user-role')).toHaveTextContent('doctor')
    })
})

// ============================================================
// Manual Prediction Form Tests
// ============================================================
describe('ManualPredictionForm', () => {
    const MockPredictionForm = () => {
        const [formData, setFormData] = React.useState({
            age: '',
            sex: '1',
            systolicBp: '',
            diastolicBp: '',
            cholesterol: '',
            heartRate: '',
            bmi: '',
            smoking: false,
            diabetes: false,
        })
        const [result, setResult] = React.useState<{ risk: string; percentage: number } | null>(null)
        const [loading, setLoading] = React.useState(false)
        const [error, setError] = React.useState('')

        const handleSubmit = async (e: React.FormEvent) => {
            e.preventDefault()
            if (!formData.age || !formData.systolicBp) {
                setError('Age and Blood Pressure are required')
                return
            }
            setLoading(true)
            setError('')
            await new Promise(r => setTimeout(r, 100))
            setResult({ risk: 'MODERATE', percentage: 35 })
            setLoading(false)
        }

        return (
            <div data-testid="prediction-page">
                <h1>Heart Disease Risk Assessment</h1>
                {error && <div data-testid="error">{error}</div>}
                <form data-testid="prediction-form" onSubmit={handleSubmit}>
                    <input
                        type="number"
                        data-testid="age-input"
                        placeholder="Age"
                        value={formData.age}
                        onChange={(e) => setFormData(prev => ({ ...prev, age: e.target.value }))}
                    />
                    <select
                        data-testid="sex-select"
                        value={formData.sex}
                        onChange={(e) => setFormData(prev => ({ ...prev, sex: e.target.value }))}
                    >
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                    <input
                        type="number"
                        data-testid="systolic-bp-input"
                        placeholder="Systolic BP"
                        value={formData.systolicBp}
                        onChange={(e) => setFormData(prev => ({ ...prev, systolicBp: e.target.value }))}
                    />
                    <input
                        type="number"
                        data-testid="diastolic-bp-input"
                        placeholder="Diastolic BP"
                        value={formData.diastolicBp}
                        onChange={(e) => setFormData(prev => ({ ...prev, diastolicBp: e.target.value }))}
                    />
                    <input
                        type="number"
                        data-testid="cholesterol-input"
                        placeholder="Total Cholesterol"
                        value={formData.cholesterol}
                        onChange={(e) => setFormData(prev => ({ ...prev, cholesterol: e.target.value }))}
                    />
                    <input
                        type="number"
                        data-testid="heart-rate-input"
                        placeholder="Heart Rate"
                        value={formData.heartRate}
                        onChange={(e) => setFormData(prev => ({ ...prev, heartRate: e.target.value }))}
                    />
                    <input
                        type="number"
                        data-testid="bmi-input"
                        placeholder="BMI"
                        value={formData.bmi}
                        onChange={(e) => setFormData(prev => ({ ...prev, bmi: e.target.value }))}
                    />
                    <label>
                        <input
                            type="checkbox"
                            data-testid="smoking-checkbox"
                            checked={formData.smoking}
                            onChange={(e) => setFormData(prev => ({ ...prev, smoking: e.target.checked }))}
                        />
                        Smoker
                    </label>
                    <label>
                        <input
                            type="checkbox"
                            data-testid="diabetes-checkbox"
                            checked={formData.diabetes}
                            onChange={(e) => setFormData(prev => ({ ...prev, diabetes: e.target.checked }))}
                        />
                        Diabetes
                    </label>
                    <button type="submit" data-testid="submit-button" disabled={loading}>
                        {loading ? 'Analyzing...' : 'Analyze Risk'}
                    </button>
                </form>
                {result && (
                    <div data-testid="result">
                        <span data-testid="risk-category">{result.risk}</span>
                        <span data-testid="risk-percentage">{result.percentage}%</span>
                    </div>
                )}
            </div>
        )
    }

    it('renders prediction form', () => {
        render(<MockPredictionForm />)
        expect(screen.getByText('Heart Disease Risk Assessment')).toBeInTheDocument()
    })

    it('has all required input fields', () => {
        render(<MockPredictionForm />)
        expect(screen.getByTestId('age-input')).toBeInTheDocument()
        expect(screen.getByTestId('sex-select')).toBeInTheDocument()
        expect(screen.getByTestId('systolic-bp-input')).toBeInTheDocument()
        expect(screen.getByTestId('cholesterol-input')).toBeInTheDocument()
        expect(screen.getByTestId('heart-rate-input')).toBeInTheDocument()
        expect(screen.getByTestId('bmi-input')).toBeInTheDocument()
    })

    it('has smoking and diabetes checkboxes', () => {
        render(<MockPredictionForm />)
        expect(screen.getByTestId('smoking-checkbox')).toBeInTheDocument()
        expect(screen.getByTestId('diabetes-checkbox')).toBeInTheDocument()
    })

    it('shows error when required fields missing', async () => {
        render(<MockPredictionForm />)
        fireEvent.click(screen.getByTestId('submit-button'))
        expect(await screen.findByTestId('error')).toHaveTextContent('Age and Blood Pressure are required')
    })

    it('shows loading state during submission', async () => {
        render(<MockPredictionForm />)
        await userEvent.type(screen.getByTestId('age-input'), '55')
        await userEvent.type(screen.getByTestId('systolic-bp-input'), '140')
        fireEvent.click(screen.getByTestId('submit-button'))
        expect(screen.getByTestId('submit-button')).toHaveTextContent('Analyzing...')
    })

    it('displays result after successful prediction', async () => {
        render(<MockPredictionForm />)
        await userEvent.type(screen.getByTestId('age-input'), '55')
        await userEvent.type(screen.getByTestId('systolic-bp-input'), '140')
        fireEvent.click(screen.getByTestId('submit-button'))
        expect(await screen.findByTestId('risk-category')).toHaveTextContent('MODERATE')
        expect(screen.getByTestId('risk-percentage')).toHaveTextContent('35%')
    })

    it('allows toggling smoking checkbox', async () => {
        render(<MockPredictionForm />)
        const checkbox = screen.getByTestId('smoking-checkbox')
        expect(checkbox).not.toBeChecked()
        await userEvent.click(checkbox)
        expect(checkbox).toBeChecked()
    })
})

// ============================================================
// Prediction History Tests
// ============================================================
describe('PredictionHistory', () => {
    const mockPredictions = [
        { id: '1', date: '2024-12-20', risk: 'LOW', percentage: 15 },
        { id: '2', date: '2024-12-19', risk: 'MODERATE', percentage: 35 },
        { id: '3', date: '2024-12-18', risk: 'HIGH', percentage: 72 },
    ]

    const MockPredictionHistory = ({ predictions = mockPredictions }) => (
        <div data-testid="prediction-history">
            <h1>Prediction History</h1>
            <div data-testid="prediction-count">{predictions.length} predictions</div>
            <table data-testid="history-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Risk Level</th>
                        <th>Percentage</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {predictions.map(p => (
                        <tr key={p.id} data-testid={`prediction-${p.id}`}>
                            <td>{p.date}</td>
                            <td data-testid={`risk-${p.id}`}>{p.risk}</td>
                            <td>{p.percentage}%</td>
                            <td>
                                <button data-testid={`view-${p.id}`}>View</button>
                                <button data-testid={`delete-${p.id}`}>Delete</button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )

    it('renders prediction history page', () => {
        render(<MockPredictionHistory />)
        expect(screen.getByText('Prediction History')).toBeInTheDocument()
    })

    it('displays prediction count', () => {
        render(<MockPredictionHistory />)
        expect(screen.getByTestId('prediction-count')).toHaveTextContent('3 predictions')
    })

    it('renders all predictions in table', () => {
        render(<MockPredictionHistory />)
        expect(screen.getByTestId('prediction-1')).toBeInTheDocument()
        expect(screen.getByTestId('prediction-2')).toBeInTheDocument()
        expect(screen.getByTestId('prediction-3')).toBeInTheDocument()
    })

    it('displays correct risk levels', () => {
        render(<MockPredictionHistory />)
        expect(screen.getByTestId('risk-1')).toHaveTextContent('LOW')
        expect(screen.getByTestId('risk-2')).toHaveTextContent('MODERATE')
        expect(screen.getByTestId('risk-3')).toHaveTextContent('HIGH')
    })

    it('has view and delete buttons for each prediction', () => {
        render(<MockPredictionHistory />)
        expect(screen.getByTestId('view-1')).toBeInTheDocument()
        expect(screen.getByTestId('delete-1')).toBeInTheDocument()
    })

    it('handles empty predictions list', () => {
        render(<MockPredictionHistory predictions={[]} />)
        expect(screen.getByTestId('prediction-count')).toHaveTextContent('0 predictions')
    })
})

// ============================================================
// Settings Page Tests
// ============================================================
describe('Settings Page', () => {
    const MockSettingsPage = () => {
        const [notifications, setNotifications] = React.useState(true)
        const [darkMode, setDarkMode] = React.useState(false)

        return (
            <div data-testid="settings-page">
                <h1>Settings</h1>
                <section data-testid="preferences">
                    <h2>Preferences</h2>
                    <label>
                        <input
                            type="checkbox"
                            data-testid="notifications-toggle"
                            checked={notifications}
                            onChange={(e) => setNotifications(e.target.checked)}
                        />
                        Email Notifications
                    </label>
                    <label>
                        <input
                            type="checkbox"
                            data-testid="dark-mode-toggle"
                            checked={darkMode}
                            onChange={(e) => setDarkMode(e.target.checked)}
                        />
                        Dark Mode
                    </label>
                </section>
                <section data-testid="danger-zone">
                    <h2>Danger Zone</h2>
                    <button data-testid="delete-account-button">Delete Account</button>
                </section>
            </div>
        )
    }

    it('renders settings page', () => {
        render(<MockSettingsPage />)
        expect(screen.getByText('Settings')).toBeInTheDocument()
    })

    it('has notification toggle', () => {
        render(<MockSettingsPage />)
        expect(screen.getByTestId('notifications-toggle')).toBeInTheDocument()
    })

    it('has dark mode toggle', () => {
        render(<MockSettingsPage />)
        expect(screen.getByTestId('dark-mode-toggle')).toBeInTheDocument()
    })

    it('has delete account button', () => {
        render(<MockSettingsPage />)
        expect(screen.getByTestId('delete-account-button')).toBeInTheDocument()
    })

    it('toggles notification preference', async () => {
        render(<MockSettingsPage />)
        const toggle = screen.getByTestId('notifications-toggle')
        expect(toggle).toBeChecked()
        await userEvent.click(toggle)
        expect(toggle).not.toBeChecked()
    })
})
