#!/bin/bash

# ============================================
# CardioDetect Database Shell Script
# ============================================

echo "╔═══════════════════════════════════════════════════╗"
echo "║     🗄️  CardioDetect Database Shell               ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
cd /Users/prajanv/CardioDetect/Milestone_3
source venv/bin/activate

echo "📊 Quick Database Commands (copy & paste in psql):"
echo "─────────────────────────────────────────────────────"
echo ""
echo "  \\dt                                    -- List all tables"
echo "  \\q                                     -- Exit"
echo ""
echo "  SELECT * FROM predictions_systemnotification;     -- Notifications"
echo "  SELECT * FROM predictions_prediction;             -- Predictions"
echo "  SELECT * FROM accounts_pendingprofilechange;      -- Pending Changes"
echo "  SELECT id, email, role FROM accounts_user;        -- Users"
echo ""
echo "─────────────────────────────────────────────────────"
echo "Opening PostgreSQL shell..."
echo ""

# Open database shell
python manage.py dbshell
