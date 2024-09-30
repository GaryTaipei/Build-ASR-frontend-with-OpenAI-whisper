1.系統架構為 ASR(Automatic Speech Recognition) frontend --> FastAPI --> Local LLM(Ollama 3.1)
2.這個 Repository 為 ASR Frontend，需要搭配另一個 Repo FastAPI + Local LLM
3.Git Clone Repo 程式到本機資料夾後，建議為本機資料夾建立虛擬環境 e.g python -m venv Build-ASR-frontend-with-OpenAI-whisper
4.在本機資料夾路徑的 command 視窗下指令安裝需要的套件:pip install -r requirements.txt
5.建立 .env 檔 裡面有兩個參數，第一個: API_URL = http://localhost:8000/chat 這個是 FastAPI Server 啟動後的網址與定義好的API /chat。
6.第二個參數 OPENAI_API_KEY=[Your OpenAI API Key]，需要申請 OpenAI API Key，網址如下： https://platform.openai.com/docs/overview
4.先執行 FastAPI Repository 啟動 API Server，然後再執行這個 Repo的 app.py 啟動前端的網頁開始測試語音查詢。