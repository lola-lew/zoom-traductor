@echo off
pip install pyinstaller soundcard websocket-client
pyinstaller --onefile --windowed --name ZoomTradutor capture_client.py
echo.
echo Ejecutable en: dist\ZoomTradutor.exe
pause
