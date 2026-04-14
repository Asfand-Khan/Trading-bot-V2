"""
dashboard.py — Live performance dashboard with real-time metrics.
Upgraded from v4.0: now shows Sharpe, drawdown, win-rate, profit factor, equity curve.
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template_string, request, redirect, url_for, jsonify
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("SUPABASE_DB_URL")

app = Flask(__name__)


def get_db():
    if not DB_URL or "[YOUR-PASSWORD]" in DB_URL:
        return None
    return psycopg2.connect(DB_URL)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Wall Street Oracle v5.0 Dashboard</title>
    <meta http-equiv="refresh" content="60">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0e1a; color: #e2e8f0; padding: 20px; }
        .header { text-align: center; padding: 20px 0; border-bottom: 1px solid #1e293b; margin-bottom: 30px; }
        .header h1 { color: #38bdf8; font-size: 1.8rem; }
        .header .subtitle { color: #64748b; margin-top: 5px; }

        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .metric-card { background: #1e293b; padding: 20px; border-radius: 10px; text-align: center; }
        .metric-card .value { font-size: 2rem; font-weight: bold; margin: 8px 0; }
        .metric-card .label { color: #64748b; font-size: 0.85rem; text-transform: uppercase; }
        .green { color: #22c55e; }
        .red { color: #ef4444; }
        .blue { color: #38bdf8; }
        .yellow { color: #f59e0b; }

        table { width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; margin-bottom: 30px; }
        th, td { padding: 12px 15px; border-bottom: 1px solid #334155; text-align: left; font-size: 0.9rem; }
        th { background: #0f172a; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.8rem; }
        .signal-buy { color: #22c55e; font-weight: bold; }
        .signal-sell { color: #ef4444; font-weight: bold; }
        .status-executed { color: #22c55e; }
        .status-skipped { color: #64748b; }
        .status-pending { color: #f59e0b; }
        .btn { padding: 5px 10px; border: none; cursor: pointer; border-radius: 4px; color: #fff; font-weight: 600; font-size: 0.75rem; margin: 2px; }
        .btn-win { background: #10b981; }
        .btn-lose { background: #ef4444; }
        .btn-ignore { background: #64748b; }
        h2 { color: #38bdf8; margin: 20px 0 15px; font-size: 1.2rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Wall Street Oracle v5.0</h1>
        <div class="subtitle">Elite Autonomous Scalper | Live Dashboard</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="label">Equity</div>
            <div class="value blue">${{ "%.2f"|format(equity) }}</div>
        </div>
        <div class="metric-card">
            <div class="label">Daily P&L</div>
            <div class="value {{ 'green' if daily_pnl >= 0 else 'red' }}">${{ "%.2f"|format(daily_pnl) }}</div>
        </div>
        <div class="metric-card">
            <div class="label">Drawdown</div>
            <div class="value {{ 'green' if drawdown < 5 else 'red' }}">{{ "%.2f"|format(drawdown) }}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Sharpe (30d)</div>
            <div class="value {{ 'green' if sharpe > 1.5 else 'yellow' if sharpe > 0 else 'red' }}">{{ "%.2f"|format(sharpe) }}</div>
        </div>
        <div class="metric-card">
            <div class="label">Win Rate</div>
            <div class="value {{ 'green' if win_rate > 50 else 'red' }}">{{ "%.1f"|format(win_rate) }}%</div>
        </div>
        <div class="metric-card">
            <div class="label">Profit Factor</div>
            <div class="value {{ 'green' if profit_factor > 1.5 else 'yellow' if profit_factor > 1 else 'red' }}">{{ "%.2f"|format(profit_factor) }}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Trades</div>
            <div class="value blue">{{ total_trades }}</div>
        </div>
        <div class="metric-card">
            <div class="label">Status</div>
            <div class="value {{ 'green' if not is_paused else 'red' }}">{{ 'PAUSED' if is_paused else 'ACTIVE' }}</div>
        </div>
    </div>

    <h2>Recent Signals</h2>
    <table>
        <tr>
            <th>Time</th>
            <th>Asset</th>
            <th>Signal</th>
            <th>Confidence</th>
            <th>ML Prob</th>
            <th>Price</th>
            <th>SL</th>
            <th>TP1 / TP2</th>
            <th>Regime</th>
            <th>Status</th>
            <th>Actions</th>
        </tr>
        {% for s in signals %}
        <tr>
            <td>{{ s.timestamp.strftime('%m-%d %H:%M') if s.timestamp else '' }}</td>
            <td><b>{{ s.asset }}</b></td>
            <td class="{{ 'signal-buy' if s.signal == 'BUY' else 'signal-sell' }}">{{ s.signal }}</td>
            <td>{{ s.confidence }}%</td>
            <td>{{ "%.0f%%"|format(s.ml_probability * 100) if s.ml_probability else 'N/A' }}</td>
            <td>${{ s.price }}</td>
            <td>{{ s.stop_loss }}</td>
            <td>{{ s.tp1 }} / {{ s.tp2 }}</td>
            <td>{{ s.regime or '' }}</td>
            <td class="status-{{ (s.status or 'pending')|lower }}">{{ s.status or 'PENDING' }}</td>
            <td>
                <form action="{{ url_for('update_status', sig_id=s.id) }}" method="POST" style="display:inline;">
                    <button type="submit" name="status" value="WIN" class="btn btn-win">WIN</button>
                    <button type="submit" name="status" value="LOSE" class="btn btn-lose">LOSE</button>
                    <button type="submit" name="status" value="IGNORED_BY_USER" class="btn btn-ignore">IGN</button>
                </form>
            </td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""


@app.route('/')
def index():
    conn = get_db()
    if not conn:
        return "<h2 style='color:red'>DB Connection Failed</h2>"

    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Get signals
    cur.execute("SELECT * FROM signals_log ORDER BY timestamp DESC LIMIT 50;")
    signals = cur.fetchall()

    # Get latest equity snapshot
    equity = 0
    daily_pnl = 0
    drawdown = 0
    sharpe = 0
    try:
        cur.execute("SELECT * FROM equity_curve ORDER BY timestamp DESC LIMIT 1;")
        snap = cur.fetchone()
        if snap:
            equity = float(snap.get('equity', 0) or 0)
            daily_pnl = float(snap.get('daily_pnl', 0) or 0)
            drawdown = float(snap.get('drawdown_pct', 0) or 0)
            sharpe = float(snap.get('sharpe_30d', 0) or 0)
    except Exception:
        pass

    # Calculate win rate and profit factor from recent trades
    win_count = sum(1 for s in signals if s.get('status') == 'WIN')
    lose_count = sum(1 for s in signals if s.get('status') == 'LOSE')
    total_decided = win_count + lose_count
    win_rate = (win_count / total_decided * 100) if total_decided > 0 else 0
    profit_factor = (win_count / (lose_count + 0.01)) if lose_count > 0 else win_count

    cur.close()
    conn.close()

    return render_template_string(HTML,
        signals=signals,
        equity=equity,
        daily_pnl=daily_pnl,
        drawdown=drawdown,
        sharpe=sharpe,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(signals),
        is_paused=False
    )


@app.route('/update/<int:sig_id>', methods=['POST'])
def update_status(sig_id):
    new_status = request.form.get('status')
    if new_status in ['WIN', 'LOSE', 'IGNORED_BY_USER']:
        conn = get_db()
        if conn:
            cur = conn.cursor()
            cur.execute("UPDATE signals_log SET status = %s WHERE id = %s", (new_status, sig_id))
            conn.commit()
            cur.close()
            conn.close()
    return redirect(url_for('index'))


@app.route('/api/metrics')
def api_metrics():
    """JSON API endpoint for external monitoring."""
    conn = get_db()
    if not conn:
        return jsonify({'error': 'no_db'}), 500

    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT * FROM equity_curve ORDER BY timestamp DESC LIMIT 1;")
        snap = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify(dict(snap) if snap else {})
    except Exception as e:
        cur.close()
        conn.close()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("  DASHBOARD: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
