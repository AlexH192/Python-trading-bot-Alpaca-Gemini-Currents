import json
from collections import defaultdict

records = json.load(open("trade_journal.json"))

ENTRY_ACTIONS = {"BUY", "SWING_ENTRY_SHORT", "SWING_ENTRY_LONG"}
EXIT_ACTIONS = {"SELL_TP1", "SELL_TP2", "SELL_STOP_INITIAL", "SELL_STOP_BE", "LIQUIDATION"}


def strategy_for(rec):
    snap = rec.get("market_snapshot", {})
    if rec["action"] in ("SWING_ENTRY_SHORT", "SWING_ENTRY_LONG"):
        return "SWING"
    st = snap.get("setup_type", "")
    if st.startswith("ORB"):
        return "ORB"
    if st:
        return "SWEEP"
    return "UNKNOWN"


def direction_for(entry_action: str) -> str:
    return "short" if entry_action == "SWING_ENTRY_SHORT" else "long"


entries = [r for r in records if r["action"] in ENTRY_ACTIONS]
exits = [r for r in records if r["action"] in EXIT_ACTIONS]

trade_groups = defaultdict(lambda: {"entries": [], "exits": []})

for e in entries:
    key = (e["ticker"], round(e["fill_price"], 4))
    trade_groups[key]["entries"].append(e)
    trade_groups[key]["strategy"] = strategy_for(e)
    trade_groups[key]["direction"] = direction_for(e["action"])

unattributed = []
for x in exits:
    ep = x.get("market_snapshot", {}).get("entry_price")
    if ep is None:
        unattributed.append(x)
        continue
    key = (x["ticker"], round(ep, 4))
    if key in trade_groups:
        trade_groups[key]["exits"].append(x)
    else:
        unattributed.append(x)

print(f"Total entry records: {len(entries)}   Total exit records: {len(exits)}")
print(f"Exit records with NO entry_price context (empty snapshot): {sum(1 for x in exits if x.get('market_snapshot',{}) == {})}")
print(f"Exit records that couldn't be matched to a known entry_price: {len(unattributed)}")
print()

results = []
for key, g in trade_groups.items():
    ticker, entry_price = key
    entries_list = g["entries"]
    exits_list = g["exits"]
    direction = g.get("direction", "long")

    entry_qty = sum(e["quantity"] for e in entries_list)
    exit_qty = sum(x["quantity"] for x in exits_list)
    exit_proceeds = sum(x["fill_price"] * x["quantity"] for x in exits_list)

    if direction == "short":
        pnl = (entry_price * exit_qty) - exit_proceeds
    else:
        pnl = exit_proceeds - (entry_price * exit_qty)

    outcomes = [x["action"] for x in exits_list]

    first_entry = entries_list[0]
    entry_snap = first_entry.get("market_snapshot", {})

    stop_loss = entry_snap.get("stop_loss")
    tp1 = entry_snap.get("tp1")
    tp2 = entry_snap.get("tp2")
    take_profit = entry_snap.get("take_profit")  # ORB's single-target field name

    exit_ts = exits_list[-1]["timestamp"] if exits_list else None

    results.append({
        "ticker": ticker,
        "entry_price": entry_price,
        "strategy": g.get("strategy", "UNKNOWN"),
        "direction": direction,
        "entry_qty": entry_qty,
        "exit_qty": exit_qty,
        "open_qty": entry_qty - exit_qty,
        "pnl": round(pnl, 2),
        "outcomes": outcomes,
        "entry_confidence": entry_snap.get("confidence_score"),
        "entry_setup": entry_snap.get("setup_type"),
        "entry_ts": first_entry["timestamp"],
        "exit_ts": exit_ts,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "take_profit": take_profit,
    })

results.sort(key=lambda r: r["entry_ts"])
for r in results:
    flag = "  <-- STILL OPEN / unresolved qty" if abs(r["open_qty"]) > 0.01 else ""
    print(f"{r['entry_ts'][:16]}  {r['ticker']:6s} {r['strategy']:6s} {r['direction']:5s} entry=${r['entry_price']:<8.3f} "
          f"qty={r['entry_qty']:<5.0f} closed={r['exit_qty']:<5.0f} pnl=${r['pnl']:<9.2f} "
          f"outcomes={r['outcomes']}{flag}")

print("\n--- Unattributed exits (no entry_price in snapshot -- can't tell which trade these belong to) ---")
for x in unattributed:
    print(f"  {x['timestamp'][:16]}  {x['ticker']:6s} {x['action']:12s} qty={x['quantity']:<6.0f} fill=${x['fill_price']}")

json.dump(results, open("reconciled_trades.json", "w"), indent=2, default=str)