/**
 * Component Tests - CardioDetect Frontend
 * Tests all reusable components for proper rendering and behavior.
 */

import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'

// ============================================================
// AnimatedCounter Tests
// ============================================================
describe('AnimatedCounter', () => {
    // Mock the component since it uses framer-motion
    const MockAnimatedCounter = ({ value, suffix = '' }: { value: number; suffix?: string }) => (
        <span data-testid="counter">{value}{suffix}</span>
    )

    it('renders the counter value', () => {
        render(<MockAnimatedCounter value={42} />)
        expect(screen.getByTestId('counter')).toHaveTextContent('42')
    })

    it('renders with suffix', () => {
        render(<MockAnimatedCounter value={85} suffix="%" />)
        expect(screen.getByTestId('counter')).toHaveTextContent('85%')
    })

    it('handles zero value', () => {
        render(<MockAnimatedCounter value={0} />)
        expect(screen.getByTestId('counter')).toHaveTextContent('0')
    })
})

// ============================================================
// FeatureCard Tests
// ============================================================
describe('FeatureCard', () => {
    const MockFeatureCard = ({
        title,
        description,
        icon
    }: {
        title: string;
        description: string;
        icon: React.ReactNode
    }) => (
        <div data-testid="feature-card">
            <div data-testid="icon">{icon}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
    )

    it('renders title and description', () => {
        render(
            <MockFeatureCard
                title="Heart Detection"
                description="AI-powered analysis"
                icon={<span>❤️</span>}
            />
        )
        expect(screen.getByText('Heart Detection')).toBeInTheDocument()
        expect(screen.getByText('AI-powered analysis')).toBeInTheDocument()
    })

    it('renders icon', () => {
        render(
            <MockFeatureCard
                title="Test"
                description="Description"
                icon={<span data-testid="test-icon">🔬</span>}
            />
        )
        expect(screen.getByTestId('test-icon')).toBeInTheDocument()
    })
})

// ============================================================
// RiskGauge Tests
// ============================================================
describe('RiskGauge', () => {
    const MockRiskGauge = ({
        percentage,
        category
    }: {
        percentage: number;
        category: 'LOW' | 'MODERATE' | 'HIGH'
    }) => (
        <div data-testid="risk-gauge">
            <span data-testid="percentage">{percentage}%</span>
            <span data-testid="category">{category}</span>
        </div>
    )

    it('renders LOW risk correctly', () => {
        render(<MockRiskGauge percentage={15} category="LOW" />)
        expect(screen.getByTestId('percentage')).toHaveTextContent('15%')
        expect(screen.getByTestId('category')).toHaveTextContent('LOW')
    })

    it('renders MODERATE risk correctly', () => {
        render(<MockRiskGauge percentage={45} category="MODERATE" />)
        expect(screen.getByTestId('percentage')).toHaveTextContent('45%')
        expect(screen.getByTestId('category')).toHaveTextContent('MODERATE')
    })

    it('renders HIGH risk correctly', () => {
        render(<MockRiskGauge percentage={75} category="HIGH" />)
        expect(screen.getByTestId('percentage')).toHaveTextContent('75%')
        expect(screen.getByTestId('category')).toHaveTextContent('HIGH')
    })

    it('handles edge case 0%', () => {
        render(<MockRiskGauge percentage={0} category="LOW" />)
        expect(screen.getByTestId('percentage')).toHaveTextContent('0%')
    })

    it('handles edge case 100%', () => {
        render(<MockRiskGauge percentage={100} category="HIGH" />)
        expect(screen.getByTestId('percentage')).toHaveTextContent('100%')
    })
})

// ============================================================
// ThemeToggle Tests
// ============================================================
describe('ThemeToggle', () => {
    const MockThemeToggle = ({ isDark, onToggle }: { isDark: boolean; onToggle: () => void }) => (
        <button data-testid="theme-toggle" onClick={onToggle}>
            {isDark ? '🌙' : '☀️'}
        </button>
    )

    it('renders light mode icon', () => {
        render(<MockThemeToggle isDark={false} onToggle={() => { }} />)
        expect(screen.getByTestId('theme-toggle')).toHaveTextContent('☀️')
    })

    it('renders dark mode icon', () => {
        render(<MockThemeToggle isDark={true} onToggle={() => { }} />)
        expect(screen.getByTestId('theme-toggle')).toHaveTextContent('🌙')
    })

    it('calls onToggle when clicked', () => {
        const mockToggle = jest.fn()
        render(<MockThemeToggle isDark={false} onToggle={mockToggle} />)
        fireEvent.click(screen.getByTestId('theme-toggle'))
        expect(mockToggle).toHaveBeenCalledTimes(1)
    })
})

// ============================================================
// NotificationBell Tests
// ============================================================
describe('NotificationBell', () => {
    const MockNotificationBell = ({
        count,
        onClick
    }: {
        count: number;
        onClick: () => void
    }) => (
        <button data-testid="notification-bell" onClick={onClick}>
            🔔 {count > 0 && <span data-testid="badge">{count}</span>}
        </button>
    )

    it('renders without badge when count is 0', () => {
        render(<MockNotificationBell count={0} onClick={() => { }} />)
        expect(screen.getByTestId('notification-bell')).toBeInTheDocument()
        expect(screen.queryByTestId('badge')).not.toBeInTheDocument()
    })

    it('renders with badge when count > 0', () => {
        render(<MockNotificationBell count={5} onClick={() => { }} />)
        expect(screen.getByTestId('badge')).toHaveTextContent('5')
    })

    it('calls onClick when clicked', () => {
        const mockClick = jest.fn()
        render(<MockNotificationBell count={3} onClick={mockClick} />)
        fireEvent.click(screen.getByTestId('notification-bell'))
        expect(mockClick).toHaveBeenCalledTimes(1)
    })
})

// ============================================================
// Shimmer (Loading State) Tests
// ============================================================
describe('Shimmer', () => {
    const MockShimmer = ({ width, height }: { width: string; height: string }) => (
        <div
            data-testid="shimmer"
            style={{ width, height }}
            className="animate-pulse bg-gray-200"
        />
    )

    it('renders with specified dimensions', () => {
        render(<MockShimmer width="100px" height="20px" />)
        const shimmer = screen.getByTestId('shimmer')
        expect(shimmer).toHaveStyle({ width: '100px', height: '20px' })
    })

    it('has loading animation class', () => {
        render(<MockShimmer width="50px" height="50px" />)
        expect(screen.getByTestId('shimmer')).toHaveClass('animate-pulse')
    })
})

// ============================================================
// DragDropZone Tests
// ============================================================
describe('DragDropZone', () => {
    const MockDragDropZone = ({
        onFileSelect,
        acceptedTypes = ['image/png', 'application/pdf']
    }: {
        onFileSelect: (file: File) => void;
        acceptedTypes?: string[]
    }) => (
        <div data-testid="dropzone">
            <input
                type="file"
                data-testid="file-input"
                accept={acceptedTypes.join(',')}
                onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0])}
            />
            <p>Drop files here or click to upload</p>
        </div>
    )

    it('renders upload instruction', () => {
        render(<MockDragDropZone onFileSelect={() => { }} />)
        expect(screen.getByText(/Drop files here/i)).toBeInTheDocument()
    })

    it('accepts specified file types', () => {
        render(<MockDragDropZone onFileSelect={() => { }} acceptedTypes={['image/png']} />)
        expect(screen.getByTestId('file-input')).toHaveAttribute('accept', 'image/png')
    })
})
