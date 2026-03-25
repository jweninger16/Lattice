import json
f = 'live/gap_scanner_account.json'
a = json.load(open(f))
a['balance'] = 1995.86 + 9.48
a['total_pnl_usd'] = a['balance'] - a['starting_capital']
a['wins'] = 2
a['losses'] = 1
a['peak_balance'] = a['balance']
a['trade_history'][-1]['pnl_usd'] = 9.48
a['trade_history'][-1]['pnl_pct'] = 0.98
a['trade_history'][-1]['reason'] = 'trail'
a['trade_history'][-1]['exit'] = 245.05
a['trade_history'][-1]['balance_after'] = a['balance']
json.dump(a, open(f, 'w'), indent=2, default=str)
print(f"Fixed: balance=${a['balance']:,.2f}, record={a['wins']}W/{a['losses']}L")
