import '@testing-library/jest-dom'
import React from 'react'

// Mock Next.js router
jest.mock('next/navigation', () => ({
    useRouter: () => ({
        push: jest.fn(),
        replace: jest.fn(),
        prefetch: jest.fn(),
        back: jest.fn(),
        forward: jest.fn(),
    }),
    usePathname: () => '/',
    useSearchParams: () => new URLSearchParams(),
}))

// Mock Next.js Image component
jest.mock('next/image', () => {
    const MockImage = (props: React.ComponentProps<'img'>) => {
        // eslint-disable-next-line @next/next/no-img-element, jsx-a11y/alt-text
        return React.createElement('img', props)
    }
    MockImage.displayName = 'MockImage'
    return {
        __esModule: true,
        default: MockImage,
    }
})

// Mock framer-motion to avoid animation issues in tests
jest.mock('framer-motion', () => {
    const createMockComponent = (tag: string) => {
        const MockComponent = ({ children, ...props }: { children?: React.ReactNode;[key: string]: unknown }) => {
            return React.createElement(tag, props, children)
        }
        MockComponent.displayName = `Motion${tag}`
        return MockComponent
    }

    return {
        motion: {
            div: createMockComponent('div'),
            span: createMockComponent('span'),
            button: createMockComponent('button'),
            p: createMockComponent('p'),
            h1: createMockComponent('h1'),
            h2: createMockComponent('h2'),
            h3: createMockComponent('h3'),
            section: createMockComponent('section'),
            nav: createMockComponent('nav'),
            ul: createMockComponent('ul'),
            li: createMockComponent('li'),
            a: createMockComponent('a'),
            form: createMockComponent('form'),
            input: createMockComponent('input'),
            label: createMockComponent('label'),
            svg: createMockComponent('svg'),
        },
        AnimatePresence: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
        useAnimation: () => ({ start: jest.fn(), stop: jest.fn() }),
        useInView: () => true,
        useMotionValue: (initial: number) => ({ get: () => initial, set: jest.fn() }),
        useTransform: () => 0,
        useSpring: () => 0,
    }
})

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: jest.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
    })),
})

// Mock localStorage
const localStorageMock = {
    getItem: jest.fn(),
    setItem: jest.fn(),
    removeItem: jest.fn(),
    clear: jest.fn(),
}
Object.defineProperty(window, 'localStorage', { value: localStorageMock })

// Mock fetch globally
global.fetch = jest.fn()

// Reset mocks between tests
beforeEach(() => {
    jest.clearAllMocks()
        ; (global.fetch as jest.Mock).mockReset()
})
