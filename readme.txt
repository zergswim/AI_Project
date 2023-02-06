[0. 설치]
apt-get install git-lfs
#git lfs install
git clone https://github.com/zergswim/boostcamp_proj_fin.git
apt install fontconfig

apt-get update
apt install libgl1-mesa-glx

pip install fastapi
pip install uvicorn
pip install ultralytics
pip install python-multipart
pip install streamlit
pip install jinja2==3.0.1

[1. API 실행]
python pred_api.py
http://49.50.167.222:30001
http://49.50.167.222:30001/docs

[2. Streamlit 실행]
streamlit run front.py --server.port 30002 --server.fileWatcherType none
http://49.50.167.222:30002