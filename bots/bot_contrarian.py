"""
bot_contrarian.py — Contrarian/low-ownership bot.

Strategy: Targets underowned players with upside.
Weights player selection inversely to projected ownership,
filtered by a minimum projection threshold to avoid pure punts.
Designed to win big in GPPs when chalk busts.
"""
