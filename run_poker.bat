@echo off
title Kronos Colab Keep-Alive Poker (10s Heartbeat)
cd /d "%~dp0"
echo ========================================================
echo Starting Kronos 10-Second Keep-Alive Heartbeat Daemon...
echo ========================================================
python keep_alive_poker.py
pause

