#!/bin/bash
# ╔═══════════════════════════════════════════════════════════════╗
# ║            CardioDetect - Complete Startup Script             ║
# ║         AI Heart Disease Detection & Prediction System        ║
# ╚═══════════════════════════════════════════════════════════════╝

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Config
DJANGO_PORT=8000
NEXTJS_PORT=3000
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${RED}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║${NC}     ${PURPLE}🫀 CardioDetect${NC} - AI Heart Disease Detection    ${RED}║${NC}"
echo -e "${RED}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 1: Kill existing processes
# ─────────────────────────────────────────────────────────────────
echo -e "${CYAN}[1/5]${NC} Stopping any existing servers..."
pkill -f "manage.py runserver" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
lsof -ti:$DJANGO_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$NEXTJS_PORT | xargs kill -9 2>/dev/null || true
sleep 1
echo -e "      ${GREEN}✓${NC} Ports cleared"

# ─────────────────────────────────────────────────────────────────
# Step 2: Check Python dependencies
# ─────────────────────────────────────────────────────────────────
echo -e "${CYAN}[2/5]${NC} Checking Python environment..."
cd "$PROJECT_ROOT"

# Check for virtual environment - prioritize Milestone_3/venv for Django deps
if [ -d "Milestone_3/venv" ]; then
    source Milestone_3/venv/bin/activate
    echo -e "      ${GREEN}✓${NC} Milestone_3 venv activated"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "      ${GREEN}✓${NC} Root .venv activated"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "      ${GREEN}✓${NC} Root venv activated"
else
    echo -e "      ${YELLOW}!${NC} No venv found, using system Python"
fi

# Verify Django is installed
if ! python3 -c "import django" 2>/dev/null; then
    echo -e "      ${RED}✗${NC} Django not found. Installing dependencies..."
    pip3 install -r Milestone_3/requirements.txt
fi
echo -e "      ${GREEN}✓${NC} Python dependencies OK"

# ─────────────────────────────────────────────────────────────────
# Step 3: Check Node.js dependencies
# ─────────────────────────────────────────────────────────────────
echo -e "${CYAN}[3/5]${NC} Checking Node.js environment..."
cd "$PROJECT_ROOT/Milestone_3/frontend"

if [ ! -d "node_modules" ]; then
    echo -e "      ${YELLOW}!${NC} node_modules not found. Installing..."
    npm install
fi
echo -e "      ${GREEN}✓${NC} Node.js dependencies OK"

# ─────────────────────────────────────────────────────────────────
# Step 4: Start Django Backend
# ─────────────────────────────────────────────────────────────────
echo -e "${CYAN}[4/5]${NC} Starting Django backend on port ${DJANGO_PORT}..."
cd "$PROJECT_ROOT/Milestone_3"

# Run migrations if needed
python3 manage.py migrate --check >/dev/null 2>&1 || python3 manage.py migrate --run-syncdb

# Start Django in background with DEBUG=True
DEBUG=True python3 manage.py runserver 0.0.0.0:$DJANGO_PORT > /tmp/django.log 2>&1 &

DJANGO_PID=$!


# Wait for Django to start
sleep 3
if kill -0 $DJANGO_PID 2>/dev/null; then
    echo -e "      ${GREEN}✓${NC} Django server running (PID: $DJANGO_PID)"
else
    echo -e "      ${RED}✗${NC} Django failed to start. Check /tmp/django.log"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────
# Step 5: Start Next.js Frontend
# ─────────────────────────────────────────────────────────────────
echo -e "${CYAN}[5/5]${NC} Starting Next.js frontend on port ${NEXTJS_PORT}..."
cd "$PROJECT_ROOT/Milestone_3/frontend"

npm run dev -- -H 0.0.0.0 > /tmp/nextjs.log 2>&1 &

NEXTJS_PID=$!

# Wait for Next.js to start
sleep 5
if kill -0 $NEXTJS_PID 2>/dev/null; then
    echo -e "      ${GREEN}✓${NC} Next.js server running (PID: $NEXTJS_PID)"
else
    echo -e "      ${RED}✗${NC} Next.js failed to start. Check /tmp/nextjs.log"
    kill $DJANGO_PID 2>/dev/null
    exit 1
fi

# Excel/CSV sync removed - feature deprecated

# ─────────────────────────────────────────────────────────────────
# Success!
# ─────────────────────────────────────────────────────────────────
echo ""

echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}          ${GREEN}✅ CardioDetect is running!${NC}              ${GREEN}║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}🌐 Frontend:${NC}  http://localhost:${NEXTJS_PORT}"
echo -e "  ${CYAN}🔧 Backend:${NC}   http://localhost:${DJANGO_PORT}"
echo -e "  ${CYAN}📊 Admin:${NC}     http://localhost:${DJANGO_PORT}/admin"
echo ""

echo ""
echo -e "  ${YELLOW}📝 Logs:${NC}"
echo -e "     Django:  tail -f /tmp/django.log"
echo -e "     Next.js: tail -f /tmp/nextjs.log"
echo ""
echo -e "  ${RED}Press Ctrl+C to stop all servers${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────
# Handle shutdown
# ─────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down servers...${NC}"
    kill $DJANGO_PID 2>/dev/null
    kill $NEXTJS_PID 2>/dev/null
    pkill -f "manage.py runserver" 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    echo -e "${GREEN}✓ Servers stopped${NC}"
    exit 0
}


trap cleanup INT TERM

# Keep script running
while true; do
    # Check if processes are still running
    if ! kill -0 $DJANGO_PID 2>/dev/null; then
        echo -e "${RED}Django server stopped unexpectedly!${NC}"
        cleanup
    fi
    if ! kill -0 $NEXTJS_PID 2>/dev/null; then
        echo -e "${RED}Next.js server stopped unexpectedly!${NC}"
        cleanup
    fi
    sleep 5
done
