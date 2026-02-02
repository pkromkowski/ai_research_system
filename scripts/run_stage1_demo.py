#!/usr/bin/env python3
import os, json, sys
sys.path.insert(0, os.getcwd())
from model.orchestration.stock_analytics_orchestrator import StockAnalyticsOrchestrator
from pathlib import Path

OUTPUT_DIR = os.path.join(os.getcwd(), 'scripts', 'output')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print('Instantiating orchestrator for SNOW...')
orch = StockAnalyticsOrchestrator('SNOW')
print('Calling core Stage 1 methods...')
tech = orch.get_technical_metrics()
adv = orch.get_advanced_technical_metrics()
fin = orch.get_financial_metrics()
vol = orch.get_volume_positioning_metrics()
peers = orch.get_peer_metrics()
mac = orch.get_macro_metrics()
allm = orch.get_all_metrics()
ctx = orch.get_thesis_context()
info = orch.stock_data_provider.get_info()
index_prices = orch.index_provider.get_index_prices()
peer_prices = orch.peer_discovery_provider.get_peers_stock_data()

summary = {
    'technical_count': len(tech),
    'advanced_count': len(adv),
    'financial_count': len(fin),
    'volume_keys': list(vol.keys()),
    'peer_metric_keys': list(peers.keys()),
    'macro_keys': list(mac.keys()),
    'all_metrics_count': len(allm),
    'thesis_pe': ctx.pe_current,
    'info_keys': list(info.keys()) if info else None,
    'index_price_keys': list(index_prices.keys()) if index_prices else None,
    'peer_prices_keys': list(peer_prices.keys()) if peer_prices else None
}
print('Summary:', summary)
with open(os.path.join(OUTPUT_DIR, 'SNOW_stage1_live_snapshot.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)
print('Snapshot saved to', os.path.join(OUTPUT_DIR, 'SNOW_stage1_live_snapshot.json'))
