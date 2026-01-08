@echo off
echo ============================================
echo Fixing TTS Dependencies
echo ============================================
echo.
echo This will reinstall transformers with the correct version
echo for TTS compatibility (4.37.x instead of 4.38+)
echo.
pause

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Uninstalling old transformers...
pip uninstall transformers -y

echo.
echo Installing transformers 4.37.x (compatible with TTS)...
pip install "transformers>=4.37.0,<4.38.0"

echo.
echo ============================================
echo Done! Transformers has been updated.
echo You can now restart the Rex AI Assistant.
echo ============================================
pause
