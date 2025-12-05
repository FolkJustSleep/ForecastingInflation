# Forecasting Inflation

## Set up the project

1. Create Virtual Environment for Python 3.11.13 version
   - Linux / macOS
       ```
       python3.11.13 -m venv .venv
       ```
   - Windows
        ```
        py -3.11.13 -m venv .venv
        ```

2. Activate Virtual Environment
   - Linux / macOS
       ```
       source .venv/bin/activate
       ```
   - Windows
        ```
        .\.venv\Scripts\Activate.ps1
        ```

3. Install Dependencies
    ```
    pip install -r requirements.txt
    ```

4. Run to test (Start forecasting)
    ```
    python src/forecasting/inflation_forecast.py
    ```